"""
t2_run.py — Tier 2: Prairie Grass simulations (ours as shipped; GADEN variants).

Per run (wind rotated onto +x, source at (20, 64, h_s), arcs at 50/100/200 m,
receptors at z_r = 1.5 m every 0.5°, 10-min average sampled every 0.5 s):

  ours_s{1,2,3}   our simulator exactly as shipped (calibrated wind, LSM,
                  shared eddies), stability class and U10 from the MO fit.
  G-ourwind-n*    GADEN in lockstep with ours_s1 on the same time-varying
                  wind; example-project filament parameters (σ₀ 10 cm,
                  γ 10 cm²/s, 10 fil/s) with filamentNoise_std ∈ NOISES
                  (0.01 is the example-project value → "G-default-ourwind").
  G-default-steady GADEN on the time-mean of our wind (how GADEN is normally
                  used, with a steady CFD field), example-project parameters.

    python bench/t2_run.py --worker 0 --n-workers 3      (checkpoints per run)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import prairie_grass as pg               # noqa: E402
import gaden_client as gc                # noqa: E402
import paired as pr                      # noqa: E402
import stable_fluid_3d as sf             # noqa: E402

OUT = os.path.join(HERE, "data", "t2")
SRC_XY = (20.0, 64.0)
ARCS = (50, 100, 200)
AVG_S, SAMPLE_S = 600.0, 0.5
NOISES = (0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5)
NOISES_EXTRA = (2, 3, 5)          # supplement pass (--extra-noises): tuning hit the 1.5 bound
G_EXAMPLE = dict(sigma0_cm=10.0, gamma_cm2s=10.0, n_fil_sec=10.0, dt=0.1)   # example_project sim1
SEEDS = (1, 2, 3)


def arc_points(r, zr, dphi=0.5):
    phi = np.deg2rad(np.arange(-89.0, 89.0 + 1e-9, dphi))
    x = SRC_XY[0] + r * np.cos(phi); y = SRC_XY[1] + r * np.sin(phi)
    ok = (y >= 1.0) & (y <= 127.0) & (x <= 252.0)
    return np.stack([x[ok], y[ok], np.full(ok.sum(), zr)], 1), np.rad2deg(phi[ok])


def arc_summary(c, deg, r):
    st = pg.arc_stats(deg, c, float(r))
    if st is None:
        return {"cmax_Q": 0.0, "cwic_Q": 0.0, "sigma_y": None, "edge_max": None}
    return {"cmax_Q": st["cmax"], "cwic_Q": st["cwic"], "sigma_y": st["sigma_y"],
            "theta_c": st["theta_c"], "edge_max": st["edge_max"],
            "nearcl_Q": near_centreline(c, deg, r, st)}


def near_centreline(c, deg, r, st, k=0.67):
    """ASTM D6589 near-centreline value: mean C over samplers within ±k·σ_y
    of the arc centroid (σ_y, centroid from the same arc)."""
    rel = (np.asarray(deg) - deg[int(np.argmax(c))] + 180.0) % 360.0 - 180.0
    s = np.deg2rad(rel) * r
    sel = np.abs(s - st["theta_c"] * r) <= k * max(st["sigma_y"], 1e-6)
    return float(np.mean(c[sel])) if sel.any() else float(np.max(c))


def gaden_example_params(src, noise, seed=0):
    p = gc.gaden_params(src, Q=pr.Q_TRACER, n_fil_sec=G_EXAMPLE["n_fil_sec"],
                        sigma0_m=G_EXAMPLE["sigma0_cm"] / 100.0, K_rel=G_EXAMPLE["gamma_cm2s"] / 2e4,
                        noise_std=noise, dt=G_EXAMPLE["dt"], seed=seed)
    return p


def make_ours(run, seed):
    """Our simulator exactly as in validation V3 (validate_sim.make_sim) with
    the run's class and U10."""
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=0.0, wind_angle_range=20.0,
                                       wind_base_speed=float(run["U10"]), wind_seed=seed, seed=seed,
                                       max_puffs=4096)
    sim.Q = pr.Q_TRACER; sim.emit_interval = 0.2; sim.sigma_0 = 1.0
    sim.set_stability_class(run["pg"])
    sim.source_pos = np.array([*SRC_XY, run["hs"]], np.float32)
    sim.set_obstacles([], [])
    return sim


def spin_time(run):
    return float(max(150.0, 220.0 / max(run["U2"], 0.5) + 60.0))


def do_run(run, workdir, gaden_threads=1):
    zr = run["zr"]
    pts, degs = [], []
    for a in ARCS:
        p, d = arc_points(a, zr); pts.append(p); degs.append(d)
    P = np.concatenate(pts).astype(np.float32)
    splits = np.cumsum([len(p) for p in pts])[:-1]
    t_spin = spin_time(run)
    src = (*SRC_XY, run["hs"])
    res = {"exp": run["exp"], "pg": run["pg"], "U10": run["U10"], "t_spin": t_spin, "models": {}}

    for seed in SEEDS:
        sim = make_ours(run, seed)
        gads = []
        if seed == 1:
            env_bin = pr.gaden_env(sim, os.path.join(workdir, "env.bin"))
            for k, nz in enumerate(NOISES):
                gads.append(gc.GadenClient(env_bin, gaden_example_params(src, nz), threads=gaden_threads,
                                           workdir=os.path.join(workdir, f"g{k}")))
        fps = sim.fps
        n_frames = int(round((t_spin + AVG_S) * fps))
        k_s = int(round(SAMPLE_S * fps))
        acc = np.zeros(len(P)); accg = np.zeros((len(gads), len(P))); n = 0
        u_sum = np.zeros(sim.wind_shape + (3,)); n_u = 0
        U_hist = []
        t0 = time.time()
        for f in range(n_frames):
            if gads and f % 3 == 0:
                u = sim.u0.numpy()
                for g in gads:
                    g.set_wind(u); g.step(1)
            sim.step()
            if (f + 1) / fps <= t_spin + 1e-9 or (f + 1) % k_s:
                continue
            acc += sim.query_concentration(P) / sim.Q; n += 1
            U_hist.append(sim.U_src)
            if gads:
                for k, g in enumerate(gads):
                    accg[k] += g.query_kgm3(P) / pr.Q_TRACER
                u_sum += sim.u0.numpy(); n_u += 1
        wall = time.time() - t0
        mean = acc / n
        res["models"][f"ours_s{seed}"] = {
            "arcs": {str(a): arc_summary(c, d, a) for a, c, d in zip(ARCS, np.split(mean, splits), degs)},
            "U_src_mean": float(np.mean(U_hist)), "n_overwritten": int(sim.n_overwritten), "wall_s": wall}
        for k, g in enumerate(gads):
            m = accg[k] / n
            res["models"][f"G-ourwind-n{NOISES[k]}"] = {
                "arcs": {str(a): arc_summary(c, d, a) for a, c, d in zip(ARCS, np.split(m, splits), degs)}}
            g.close()
        if seed == 1:
            u_mean = u_sum / max(n_u, 1)
            np.save(os.path.join(workdir, "u_mean.npy"), u_mean.astype(np.float32))

    # GADEN on the steady time-mean wind (CPU only)
    u_mean = np.load(os.path.join(workdir, "u_mean.npy"))
    t0 = time.time()
    with gc.GadenClient(os.path.join(workdir, "env.bin"), gaden_example_params(src, 0.01),
                        threads=2, workdir=os.path.join(workdir, "gs")) as g:
        g.set_wind(u_mean)
        n_spin = int(round(t_spin / 0.1)); k_s = int(round(SAMPLE_S / 0.1))
        g.step(n_spin)
        acc = np.zeros(len(P)); n = 0
        for i in range(int(round(AVG_S / SAMPLE_S))):
            g.step(k_s)
            acc += g.query_kgm3(P) / pr.Q_TRACER; n += 1
    m = acc / n
    res["models"]["G-default-steady"] = {
        "arcs": {str(a): arc_summary(c, d, a) for a, c, d in zip(ARCS, np.split(m, splits), degs)},
        "wall_s": time.time() - t0}
    return res


def supplement(run, workdir, noises, gaden_threads=1):
    """Replay ours_s1 (deterministic: same seeds → same wind) with extra GADEN
    noise levels; returns the new G-ourwind models and the replayed ours_s1 arcs
    (checked against the stored ones by the caller)."""
    zr = run["zr"]
    pts, degs = [], []
    for a in ARCS:
        p, d = arc_points(a, zr); pts.append(p); degs.append(d)
    P = np.concatenate(pts).astype(np.float32)
    splits = np.cumsum([len(p) for p in pts])[:-1]
    t_spin = spin_time(run)
    src = (*SRC_XY, run["hs"])
    sim = make_ours(run, 1)
    env_bin = pr.gaden_env(sim, os.path.join(workdir, "env.bin"))
    gads = [gc.GadenClient(env_bin, gaden_example_params(src, nz), threads=gaden_threads,
                           workdir=os.path.join(workdir, f"gx{k}")) for k, nz in enumerate(noises)]
    fps = sim.fps
    n_frames = int(round((t_spin + AVG_S) * fps)); k_s = int(round(SAMPLE_S * fps))
    acc = np.zeros(len(P)); accg = np.zeros((len(gads), len(P))); n = 0
    for f in range(n_frames):
        if f % 3 == 0:
            u = sim.u0.numpy()
            for g in gads:
                g.set_wind(u); g.step(1)
        sim.step()
        if (f + 1) / fps <= t_spin + 1e-9 or (f + 1) % k_s:
            continue
        acc += sim.query_concentration(P) / sim.Q; n += 1
        for k, g in enumerate(gads):
            accg[k] += g.query_kgm3(P) / pr.Q_TRACER
    out = {}
    for k, g in enumerate(gads):
        out[f"G-ourwind-n{noises[k]}"] = {
            "arcs": {str(a): arc_summary(c, d, a) for a, c, d in zip(ARCS, np.split(accg[k] / n, splits), degs)}}
        g.close()
    replay = {str(a): arc_summary(c, d, a) for a, c, d in zip(ARCS, np.split(acc / n, splits), degs)}
    return out, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-noises", default="", help="supplement pass, e.g. 2,3,5 (needs run_*.json)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated exp numbers")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    table, dropped = pg.load()
    exps = sorted(table)
    if a.only:
        exps = [int(e) for e in a.only.split(",")]
    exps = exps[a.worker::a.n_workers]
    if a.extra_noises:
        noises = [float(v) for v in a.extra_noises.split(",")]
        for e in exps:
            path = os.path.join(OUT, f"run_{e:02d}.json")
            r = json.load(open(path))
            if all(f"G-ourwind-n{nz:g}" in r["models"] for nz in noises):
                continue
            wd = os.path.join(OUT, f"work_{e:02d}"); os.makedirs(wd, exist_ok=True)
            t0 = time.time()
            new, replay = supplement(table[e], wd, noises)
            stored = r["models"]["ours_s1"]["arcs"]
            dev = max(abs(replay[k]["cmax_Q"] / stored[k]["cmax_Q"] - 1) for k in stored if stored[k]["cmax_Q"] > 0)
            for k, v in new.items():
                r["models"][k.replace(".0", "") if k.endswith(".0") else k] = v
            r.setdefault("supplement", {})[a.extra_noises] = {"replay_max_rel_dev_cmax": dev}
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(r, f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
            print(f"[w{a.worker}] supplement exp {e}: replay dev {dev:.2e}, {time.time() - t0:.0f}s", flush=True)
        return
    for e in exps:
        path = os.path.join(OUT, f"run_{e:02d}.json")
        if os.path.exists(path):
            continue
        wd = os.path.join(OUT, f"work_{e:02d}")
        os.makedirs(wd, exist_ok=True)
        t0 = time.time()
        r = do_run(table[e], wd)
        r["wall_total_s"] = time.time() - t0
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(r, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"[w{a.worker}] exp {e} {table[e]['pg']} done in {r['wall_total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
