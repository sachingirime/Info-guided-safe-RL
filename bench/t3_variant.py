"""
t3_variant.py — T3 (whiff statistics along the real track) for variants of OUR
model, using t3_fluct's replay, streakline conditioning and calibration.

    python bench/t3_variant.py --tag s01k0005 --sigma0 0.1 --k-rel 0.0005 --emit 0.05 \
        --lsm-table bench/data/tl/tl_s0.1_k0.0005.json [--kwargs '{"subgrid_wind": true}'] \
        [--sets desert_low_wind]

Score (lower is better): mean |ln(model/data)| over bins 5–50 m of
intermittency, whiff frequency and whiff duration at V_θ ∈ {2, 4, 6} V, with
values floored at 1e-3 (γ), 1e-3 Hz and 0.05 s.  Selection uses the low-wind
run only; the high-wind run is the held-out check.
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
import t3_fluct as t3                    # noqa: E402

OUT = os.path.join(HERE, "data", "t3_variants")
FLOORS = {"gamma": 1e-3, "whiff_freq": 1e-3, "whiff_dur": 0.05}


def replay(D, kwargs, sigma0, emit, max_puffs, seed=1):
    import stable_fluid_3d as sf
    import paired as pr
    alpha = sf.PG_SHEAR_EXPONENTS[D["pg"]]
    U10 = D["S_mean"] * (10.0 / t3.Z_ANEMO) ** alpha
    track = np.stack([t3.SRC[0] + D["x"], t3.SRC[1] + D["y"], np.full(len(D["x"]), t3.H_SENS)], 1).astype(np.float32)
    track[:, 0] = np.clip(track[:, 0], 1, 255); track[:, 1] = np.clip(track[:, 1], 1, 127)
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=D["wind_dir_deg"], wind_angle_range=20.0,
                                       wind_base_speed=U10, wind_seed=seed, seed=seed, max_puffs=max_puffs, **kwargs)
    sim.Q = 1.0; sim.emit_interval = emit; sim.sigma_0 = sigma0
    sim.set_stability_class(D["pg"]); sim.set_obstacles([], [])
    sim.source_pos = np.array([*t3.SRC, t3.H_SRC], np.float32)
    ratio = int(round(sim.fps / t3.FS)); n = len(track); spin = 300.0
    c = np.zeros(n, np.float32); w = np.zeros((n, 2), np.float32); an = np.zeros((n, 2), np.float32)
    src_pt = np.array([[*t3.SRC, t3.H_SRC]])
    rng = np.random.default_rng(seed + 100); ss = np.zeros(3)
    for f in range(int(round((spin + n / t3.FS) * sim.fps))):
        if f % ratio == 0:
            i = f // ratio - int(round(spin * t3.FS))
            if 0 <= i < n:
                c[i] = sim.query_concentration(track[i:i + 1])[0]
                # plume-moving wind at the source (streakline): resolved + coherent eddies
                w[i] = pr.trilinear(sim.u0.numpy(), src_pt, sim.dx)[0, :2] + sim.coherent_velocity(src_pt)[0, :2]
                # model anemometer (T3b): + local small-scale turbulence at Z_ANEMO
                ss = sim.small_scale_ou(ss, t3.Z_ANEMO, 1.0 / t3.FS, rng)
                an[i] = w[i] + ss[:2]
        sim.step()
    return {f"ours_s{seed}": c, f"wind_s{seed}": w, f"anemo_s{seed}": an}, {"U10": U10, "n_overwritten": int(sim.n_overwritten)}


def score(res_set):
    """Mean |ln(model/data)| over bins 5–50 m, 3 stats × 3 thresholds."""
    errs = []
    for vth, T in res_set.items():
        m = T["models"].get("ours_s1", {})
        if not m.get("bins"):
            return np.inf
        for bd, bm in list(zip(T["data"], m["bins"]))[1:]:
            for k, fl in FLOORS.items():
                d, p = bd[k], bm[k]
                d = fl if not np.isfinite(d) else max(d, fl)
                p = fl if (p is None or not np.isfinite(p)) else max(p, fl)
                errs.append(abs(np.log(p / d)))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--k-rel", type=float, default=0.05)
    ap.add_argument("--emit", type=float, default=0.2)
    ap.add_argument("--max-puffs", type=int, default=8192)
    ap.add_argument("--lsm-table", default="")
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--sets", default="desert_low_wind,desert_high_wind")
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    kwargs = json.loads(a.kwargs); kwargs["K_rel"] = a.k_rel
    if a.lsm_table:
        tab = json.load(open(a.lsm_table))
        kwargs["lsm_table"] = {c: {k: v[k] for k in ("a_y", "a_z", "TL_y", "TL_z")} for c, v in tab["classes"].items()}
    out = os.path.join(OUT, a.tag); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "t3.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres["_config"] = {"sigma0": a.sigma0, "k_rel": a.k_rel, "emit": a.emit, "lsm_table": a.lsm_table, "kwargs": a.kwargs}
    for name in a.sets.split(","):
        if name in allres:
            continue
        D = t3.data_series(name)
        t0 = time.time()
        series, meta = replay(D, kwargs, a.sigma0, a.emit, a.max_puffs)
        rs = t3.analyse(D, series, cond="streak")
        allres[name] = {"score": score(rs), "near_streak": rs, **meta, "wall_s": time.time() - t0}
        np.savez_compressed(os.path.join(out, f"{name}.npz"), **series)
        with open(path, "w") as f:
            json.dump(allres, f, indent=1, default=float)
        print(f"[{a.tag}] {name}: score {allres[name]['score']:.3f}  ({time.time() - t0:.0f}s, overwritten {meta['n_overwritten']})", flush=True)


if __name__ == "__main__":
    main()
