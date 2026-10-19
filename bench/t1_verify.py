"""
t1_verify.py — Tier 1: matched code-to-code verification (ours vs GADEN).

Both models: same streamed wind, same source, release 10/s, σ₀ = 1 m,
K_rel = 0.05 m²/s (γ = 1000 cm²/s), no stochastic velocity, no buoyancy.

    python bench/t1_verify.py [--only T1a,T1b] [--t-avg 300]

Writes bench/results/t1.json (+ per-scenario npz in bench/data/t1/).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paired as pr                      # noqa: E402
import gaden_client as gc                # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from gaussian_puff_env import GaussianPuffEnv   # noqa: E402

PLANES = [25.0, 50.0, 100.0, 150.0]
SCEN = {
    "T1a": dict(source=(20.0, 64.0, 30.0), U=3.0, steady_wind=True, shear_exponent=0.0, angle_range=0.0),
    "T1b": dict(source=(20.0, 64.0, 30.0), U=3.0, steady_wind=True, shear_exponent=None, angle_range=0.0),
    "T1c": dict(source=(20.0, 64.0, 30.0), U=3.0, wind_preset="meandering", angle_range=20.0),
    "T1d": dict(source=(10.0, 92.0, 15.0), U=3.0, wind_preset="meandering", angle_range=20.0, obstacles=True),
}


def plane_points(x, y_rng=(1.0, 127.0), z_rng=(1.0, 61.0), h=1.0):
    ys = np.arange(y_rng[0], y_rng[1] + 1e-9, h)
    zs = np.arange(z_rng[0], z_rng[1] + 1e-9, h)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1), ys, zs


def plane_metrics(C, ux, ys, zs, h=1.0):
    """C, ux: (ny, nz) time-mean C/Q (s/m³) and mean u_x; second-moment widths."""
    C = np.where(np.isfinite(C), C, 0.0)
    m0 = C.sum() * h * h
    if m0 <= 0:
        return {"m0": 0.0}
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    yc = (C * Y).sum() * h * h / m0; zc = (C * Z).sum() * h * h / m0
    sy = np.sqrt((C * (Y - yc) ** 2).sum() * h * h / m0)
    sz = np.sqrt((C * (Z - zc) ** 2).sum() * h * h / m0)
    return {"m0": float(m0), "y_c": float(yc), "z_c": float(zc), "sigma_y": float(sy), "sigma_z": float(sz),
            "C_max": float(C.max()), "flux_over_Q": float((C * ux).sum() * h * h)}


def run(name, t_spin, t_avg, seed=1):
    cfg = dict(SCEN[name])
    src = cfg.pop("source"); U = cfg.pop("U"); ar = cfg.pop("angle_range")
    obst = ([], [])
    if cfg.pop("obstacles", False):
        sp = [o for o in GaussianPuffEnv.FIXED_OBSTACLES if hasattr(o, "radius")]
        cu = [o for o in GaussianPuffEnv.FIXED_OBSTACLES if not hasattr(o, "radius")]
        obst = (sp, cu)
    sim = pr.make_ours(src, wind_angle=0.0, angle_range=ar, U=U, seed=seed, obstacles=obst, **cfg)
    os.makedirs(os.path.join(HERE, "data", "t1"), exist_ok=True)
    env_bin = pr.gaden_env(sim, os.path.join(HERE, "data", "t1", f"{name}_env.bin"))
    prm = gc.gaden_params(src, Q=pr.Q_TRACER, n_fil_sec=1.0 / sim.emit_interval, sigma0_m=1.0,
                          K_rel=0.05, noise_std=0.0, dt=3.0 / sim.fps, seed=0)
    allp, meta = [], []
    for d in PLANES:
        p, ys, zs = plane_points(src[0] + d)
        allp.append(p); meta.append((ys, zs))
    pts = np.concatenate(allp)
    valid = pr.free_in_both(sim, pts)
    # point series: plane centre column at source height and ±4 m off-axis
    ser_pts = []
    for d in PLANES:
        for dy in (0.0, 4.0, 8.0):
            ser_pts.append([src[0] + d, src[1] + dy, src[2]])
    ser_pts = np.array(ser_pts)
    allpts = np.concatenate([pts, ser_pts])
    sidx = np.arange(len(pts), len(allpts))
    with gc.GadenClient(env_bin, prm, threads=16) as g:
        out = pr.run_paired(sim, g, allpts, t_spin, t_avg, series_idx=sidx)
    ux = pr.trilinear(out["u_mean"], pts, sim.dx)[:, 0]
    res = {"config": {**SCEN[name], "obstacles": bool(obst[0] or obst[1])}, "planes": {},
           "wall_total_s": out["wall_total_s"], "wall_gaden_s": out["wall_gaden_s"],
           "t_spin": t_spin, "t_avg": t_avg}
    off = 0
    for d, (ys, zs), p in zip(PLANES, meta, allp):
        n = len(p); sl = slice(off, off + n); off += n
        shape = (len(ys), len(zs))
        v = valid[sl].reshape(shape)
        Co = np.where(v, out["mean_ours"][sl].reshape(shape), 0.0)
        Cg = np.where(v, out["mean_gaden"][sl].reshape(shape), 0.0)
        u = ux[sl].reshape(shape)
        mo, mg = plane_metrics(Co, u, ys, zs), plane_metrics(Cg, u, ys, zs)
        both = v & ((Co > 0.01 * Co.max()) | (Cg > 0.01 * Cg.max()))
        res["planes"][str(int(d))] = {
            "ours": mo, "gaden": mg,
            "ratio_gaden_over_ours": {k: (mg[k] / mo[k] if mo.get(k) else None)
                                      for k in ("sigma_y", "sigma_z", "C_max", "flux_over_Q") if k in mg},
            "field_r": float(np.corrcoef(Co[both], Cg[both])[0, 1]) if both.sum() > 3 else None,
            "field_nmse": float(np.mean((Co[both] - Cg[both]) ** 2) / (Co[both].mean() * Cg[both].mean())) if both.sum() > 3 else None,
            "n_invalid_points": int((~v).sum()),
        }
    so, sg = out["series_ours"], out["series_gaden"]
    ser = []
    for k, p in enumerate(ser_pts):
        a, b = so[:, k], sg[:, k]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        ser.append({"point": p.tolist(), "mean_ours": float(a.mean()), "mean_gaden": float(b.mean()),
                    "r": r, "nmse": float(np.mean((a - b) ** 2) / (a.mean() * b.mean())) if a.mean() > 0 and b.mean() > 0 else None})
    res["series"] = ser
    np.savez_compressed(os.path.join(HERE, "data", "t1", f"{name}.npz"), pts=pts, valid=valid,
                        mean_ours=out["mean_ours"][:len(pts)], mean_gaden=out["mean_gaden"][:len(pts)],
                        series_ours=so, series_gaden=sg, series_pts=ser_pts)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="T1a,T1b,T1c,T1d")
    ap.add_argument("--t-spin", type=float, default=120.0)
    ap.add_argument("--t-avg", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t1.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name in a.only.split(","):
        r = run(name, a.t_spin, a.t_avg)
        allres[name] = r
        with open(a.out, "w") as f:
            json.dump(allres, f, indent=1)
        print(name, json.dumps({d: v["ratio_gaden_over_ours"] for d, v in r["planes"].items()}), flush=True)


if __name__ == "__main__":
    main()
