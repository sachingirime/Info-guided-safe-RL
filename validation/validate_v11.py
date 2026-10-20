"""
validate_v11.py — validation of the v1.1 dispersion model (GaussianPuffEnv defaults:
surface-layer two-scale LSM, sub-grid near-ground wind, σ₀ = 0.3 m, K_rel = 0.01,
puffs every 0.05 s).  Reuses validate_sim's V3/V4 geometry, fits and Briggs bands.

  V3/V4  σ_y, σ_z vs Briggs (1973) open country, classes A–F, source at 10 m,
         U10 = 3 m/s, 3600 s averages; centreline/off-axis intermittency.  In v1.1
         Briggs is an external reference: nothing is fitted to it.
  V5     determinism (same seed → identical; different seed → different).
  V7     wall time per step and per query at the env's puff count.

V0, V1, V2, V6, V8 do not depend on the dispersion model and are unchanged.

    python validation/validate_v11.py [--only V3,V5,V7] [--avg 3600]
Writes validation/results_v11.json and figs (v11_*).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import warp as wp                                   # noqa: E402
import stable_fluid_3d as sf                        # noqa: E402
import validate_sim as vs                           # noqa: E402
from gaussian_puff_env import GaussianPuffEnv       # noqa: E402

E = GaussianPuffEnv
V11_KW = dict(E.SIM_DEFAULTS)
RESULTS = os.path.join(HERE, "results_v11.json")


def make_sim_v11(source, U, Q=1.0, emit=None, seed=1, angle=0.0, pg="D", max_puffs=None, **kw):
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=angle, wind_angle_range=20.0,
                                       wind_base_speed=U, wind_seed=seed, seed=seed,
                                       max_puffs=max_puffs or 4 * E.N_PUFFS, **{**V11_KW, **kw})
    sim.Q = Q; sim.emit_interval = E.PUFF_PERIOD; sim.sigma_0 = E.SIGMA_0
    sim.set_stability_class(pg)
    sim.source_pos = np.array(source, np.float32)
    sim.set_obstacles([], [])
    return sim


def v3_all(avg):
    vs.make_sim = make_sim_v11          # v3_run builds its sim through validate_sim.make_sim
    out = {}
    series_d = None
    for pg in "ABCDEF":
        d, series, sim = vs.v3_run(pg, avg=avg, record_series=(pg == "D"))
        d["setup"]["v11_kwargs"] = V11_KW
        d["setup"]["ustar"] = sim.ustar; d["setup"]["inv_L"] = sim.inv_L
        out[pg] = d
        if pg == "D":
            series_d = series
    vs.sigma_figure([out[c] for c in "ABC"], "v11_v3_sigma_ABC")
    vs.sigma_figure([out[c] for c in "DEF"], "v11_v3_sigma_DEF")
    # v4_stats saves figs/v4_pdfs (the v1.0 figure): keep v1.0's, store this one as v11_v4_pdfs
    import shutil
    figs = os.path.join(HERE, "figs")
    keep = {ext: os.path.join(figs, f"v4_pdfs.{ext}") for ext in ("png", "pdf")}
    backup = {ext: p + ".v10bak" for ext, p in keep.items() if os.path.exists(p)}
    for ext, b in backup.items():
        shutil.copy2(keep[ext], b)
    v4 = vs.v4_stats(series_d, out["D"]["rows"], avg)
    for ext, p in keep.items():
        if os.path.exists(p):
            shutil.move(p, os.path.join(figs, f"v11_v4_pdfs.{ext}"))
        if ext in backup:
            shutil.move(backup[ext], p)
    return out, v4


def v5():
    def run(seed):
        sim = make_sim_v11((40.0, 64.0, 5.0), 3.0, seed=seed, pg="B")
        for _ in range(600):
            sim.step()
        P = np.array([[60, 64, 5], [80, 66, 4], [100, 62, 3]], np.float32)
        return sim.puff_pos.numpy()[:sim.next_puff_idx].copy(), sim.puff_vs.numpy()[:sim.next_puff_idx].copy(), sim.query_concentration(P)
    a, b, c = run(11), run(11), run(12)
    same = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]))
    diff = bool(not np.allclose(a[0], c[0]))
    print(f"V5 determinism (v1.1): same seed identical {same}, different seed differs {diff}")
    return {"same_seed_identical": same, "different_seed_differs": diff, "pass": same and diff}


def v7():
    rows = []
    rng = np.random.default_rng(0)
    obst = E.FIXED_OBSTACLES
    sp = [o for o in obst if isinstance(o, vs.SphereObstacle)]; cu = [o for o in obst if isinstance(o, vs.CuboidObstacle)]
    for n_puffs in (E.N_PUFFS // 4, E.N_PUFFS):
        sim = make_sim_v11((60.0, 64.0, 5.0), 1.5, seed=1, max_puffs=n_puffs)
        sim.set_obstacles(sp, cu)
        sim.inject_puffs(rng.uniform([0, 0, 0], [256, 128, 60], (n_puffs, 3)), ages=rng.uniform(0, 100, n_puffs))
        sim._next_emit_time = 1e12
        for n_q in (0, 1, 1000):
            P = rng.uniform([0, 0, 4], [256, 128, 60], (max(n_q, 1), 3)).astype(np.float32)
            for _ in range(30):
                sim.step()
            wp.synchronize(); t0 = time.perf_counter()
            for _ in range(300):
                sim.step()
                if n_q:
                    sim.query_concentration(P)
            wp.synchronize(); dt = (time.perf_counter() - t0) / 300
            rows.append({"n_puffs": n_puffs, "n_query": n_q, "ms_per_step": dt * 1e3, "real_time_factor": sim.frame_dt / dt})
            print(f"  V7 v1.1 puffs={n_puffs:5d} query={n_q:5d}: {dt*1e3:6.3f} ms/step  RTF {sim.frame_dt/dt:6.1f}×")
    return {"device": wp.get_device().name, "rows": rows, "obstacles": len(obst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="V3,V5,V7")
    ap.add_argument("--avg", type=float, default=3600.0)
    a = ap.parse_args()
    wp.init(); wp.config.quiet = True
    res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    todo = [t.strip().upper() for t in a.only.split(",")]
    if "V3" in todo:
        res["V3"], res["V4"] = v3_all(a.avg)
    if "V5" in todo:
        res["V5"] = v5()
    if "V7" in todo:
        res["V7"] = v7()
    res["_meta"] = {"v11_kwargs": V11_KW, "sigma_0": E.SIGMA_0, "puff_period": E.PUFF_PERIOD,
                    "last_run": time.strftime("%Y-%m-%d %H:%M:%S"), "device": wp.get_device().name}
    json.dump(res, open(RESULTS, "w"), indent=2, default=float)
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
