"""
t3_fluct.py — Tier 3: concentration fluctuations along a real sensor track.

Reference: Nag & van Breugel (2024) desert runs (handheld 200 Hz odor sensor,
RTK-GPS position relative to the source, stationary anemometers).  The walker's
trajectory (xsrc, ysrc; x = east, y = north) is replayed through each model with
the model's mean wind set to the measured mean wind vector, and the same
statistics are computed on data and models at 10 Hz.

Models: ours as shipped (class from σ_θ, EPA σ_θ method; U10 from the mean
anemometer speed with the class power law, anemometer height assumed 2 m),
seeds 1 and 2; GADEN example params (noise 0.01) and tuned (noise from T2),
both in lockstep with ours seed 1.  Assumed (not in the dataset docs): source
and sensor at 1.5 m; steady mean wind in the models.

Calibration-free comparison (the sensor is uncalibrated and saturates at
10.29 V): for each data threshold V_θ ∈ {2, 4, 6} V the model threshold C_θ is
set so that the model's intermittency in the 0–5 m bin equals the data's; the
other bins (5–10, 10–20, 20–30, 30–50 m) and the whiff statistics are then
predictions.  Statistics per bin: intermittency γ, whiff frequency (onsets per
s in bin), mean whiff duration (s), and sKLD (Monroy et al. 2017: 10 bins)
of log(c/c_θ) within whiffs, clipped at the sensor's saturation ratio.

    python bench/t3_fluct.py --gaden-tuned-noise 1.5
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
import nag2024                           # noqa: E402
import gaden_client as gc                # noqa: E402
import paired as pr                      # noqa: E402
import stable_fluid_3d as sf             # noqa: E402
import t2_run                            # noqa: E402

OUT = os.path.join(HERE, "data", "t3")
FS = 10.0                                # Hz, common analysis rate
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50)]
V_TH = (2.0, 4.0, 6.0)
V_SAT = 10.28
H_SRC = H_SENS = 1.5
Z_ANEMO = 2.0
SRC = np.array([128.0, 64.0])
SIGMA_THETA_CLASS = [(22.5, "A"), (17.5, "B"), (12.5, "C"), (7.5, "D"), (3.8, "E"), (0.0, "F")]


def data_series(name):
    d = nag2024.load(name, cols=["time", "odor", "xsrc", "ysrc", "corrected_u", "corrected_v", "S2",
                                 "nearest_from_streakline"])
    k = int(round(200 / FS))
    n = len(d["time"]) // k * k
    blk = lambda a: a[:n].reshape(-1, k).mean(1)
    u, v = d["corrected_u"], d["corrected_v"]
    th = np.arctan2(v[::200], u[::200]); w = 900
    sig = [np.degrees(np.sqrt(-2 * np.log(np.hypot(np.cos(th[i:i + w]).mean(), np.sin(th[i:i + w]).mean()))))
           for i in range(0, len(th) - w + 1, w)]
    s_th = float(np.mean(sig))
    pg = next(c for lim, c in SIGMA_THETA_CLASS if s_th >= lim)
    return {"t": blk(d["time"]), "c": blk(d["odor"]), "x": blk(d["xsrc"]), "y": blk(d["ysrc"]),
            "u": blk(u), "v": blk(v), "streak_authors": blk(d["nearest_from_streakline"]),
            "wind_dir_deg": float(np.degrees(np.arctan2(v.mean(), u.mean()))), "S_mean": float(d["S2"].mean()),
            "sigma_theta_deg": s_th, "pg": pg}


def whiff_stats(c, dist, thr, valid=None):
    """Per-bin intermittency, whiff frequency (Hz), mean whiff duration (s) and
    within-whiff log(c/thr) samples.  A whiff = contiguous run above thr; it is
    attributed to the bin of its onset sample."""
    above = c > thr
    if valid is not None:
        above &= valid
    on = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    off = np.flatnonzero(~above[1:] & above[:-1]) + 1
    if above[0]:
        on = np.r_[0, on]
    if above[-1]:
        off = np.r_[off, len(c)]
    out = []
    for lo, hi in BINS:
        inb = (dist >= lo) & (dist < hi)
        if valid is not None:
            inb &= valid
        T = inb.sum() / FS
        sel = inb[on]
        dur = (off[sel] - on[sel]) / FS
        lr = np.log(np.maximum(c[inb & above], 1e-30) / thr)
        out.append({"bin": [lo, hi], "time_s": float(T), "gamma": float(above[inb].mean()) if inb.any() else np.nan,
                    "whiff_freq": float(sel.sum() / T) if T > 0 else np.nan,
                    "whiff_dur": float(dur.mean()) if len(dur) else np.nan, "logratio": lr})
    return out


STREAK_AGE = 120.0      # s, streakline length (particles older than this are dropped)
STREAK_R = 3.0          # m, "near the streakline" condition


def streak_offset(x, y, u, v, age=STREAK_AGE):
    """Distance from the sensor (x, y; source-relative) to the streakline of
    particles released at the source every 1/FS s and advected by the spatially
    uniform wind (u, v)(t) measured at the source — the same algorithm for the
    data (anemometer) and the models (model wind at the source)."""
    D = np.cumsum(np.stack([u, v], 1), 0) / FS          # D(t): displacement since t0
    n = len(x); A = int(age * FS)
    out = np.empty(n)
    P = np.stack([x, y], 1)
    for i0 in range(0, n, 2048):
        idx = np.arange(i0, min(i0 + 2048, n))
        lag = np.arange(A + 1)
        s_ = idx[:, None] - lag[None, :]                     # release indices
        ok = s_ >= 0
        rel = D[idx][:, None, :] - D[np.maximum(s_, 0)]      # particle positions (source-relative)
        d = np.linalg.norm(P[idx][:, None, :] - rel, axis=2)
        d[~ok] = np.inf
        out[idx] = d.min(1)
    return out


def skld(a, b, lo, hi, nb=10):
    """Symmetric KL on nb-bin histograms (Monroy et al. 2017), additive smoothing."""
    if len(a) < 5 or len(b) < 5:
        return np.nan
    e = np.linspace(lo, hi, nb + 1)
    p = np.histogram(np.clip(a, lo, hi), e)[0] + 0.5; q = np.histogram(np.clip(b, lo, hi), e)[0] + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def run_models(D, tuned_noise, seeds=(1, 2)):
    """Replay the track through ours (seeds) + GADEN worlds (with seed 1)."""
    alpha = sf.PG_SHEAR_EXPONENTS[D["pg"]]
    U10 = D["S_mean"] * (10.0 / Z_ANEMO) ** alpha
    track = np.stack([SRC[0] + D["x"], SRC[1] + D["y"], np.full(len(D["x"]), H_SENS)], 1).astype(np.float32)
    track[:, 0] = np.clip(track[:, 0], 1, 255); track[:, 1] = np.clip(track[:, 1], 1, 127)
    series = {}
    for seed in seeds:
        sim = sf.HybridFluidPuffSimulation(wind_center_angle=D["wind_dir_deg"], wind_angle_range=20.0,
                                           wind_base_speed=U10, wind_seed=seed, seed=seed, max_puffs=4096)
        sim.Q = 1.0; sim.emit_interval = 0.2; sim.sigma_0 = 1.0
        sim.set_stability_class(D["pg"]); sim.set_obstacles([], [])
        sim.source_pos = np.array([*SRC, H_SRC], np.float32)
        gads = {}
        if seed == seeds[0]:
            wd = os.path.join(OUT, "work"); os.makedirs(wd, exist_ok=True)
            env_bin = pr.gaden_env(sim, os.path.join(wd, "env.bin"))
            src = (*SRC, H_SRC)
            gads = {"G-default": gc.GadenClient(env_bin, t2_run.gaden_example_params(src, 0.01), threads=1,
                                                workdir=os.path.join(wd, "gd")),
                    "G-tuned": gc.GadenClient(env_bin, t2_run.gaden_example_params(src, tuned_noise), threads=1,
                                              workdir=os.path.join(wd, "gt"))}
        spin = 300.0
        ratio = int(round(sim.fps / FS))
        n = len(track)
        ser = {f"ours_s{seed}": np.zeros(n, np.float32), **{k: np.zeros(n, np.float32) for k in gads},
               f"wind_s{seed}": np.zeros((n, 2), np.float32)}
        src_pt = np.array([[*SRC, H_SRC]])
        t0 = time.time()
        for f in range(int(round((spin + n / FS) * sim.fps))):
            if f % ratio == 0:
                if gads:
                    u = sim.u0.numpy()
                    for g in gads.values():
                        g.set_wind(u); g.step(1)
                i = f // ratio - int(round(spin * FS))
                if 0 <= i < n:
                    p = track[i:i + 1]
                    ser[f"wind_s{seed}"][i] = pr.trilinear(sim.u0.numpy(), src_pt, sim.dx)[0, :2]
                    ser[f"ours_s{seed}"][i] = sim.query_concentration(p)[0]
                    for k, g in gads.items():
                        ser[k][i] = g.query_kgm3(p)[0] / pr.Q_TRACER
            sim.step()
        for g in gads.values():
            g.close()
        print(f"  seed {seed}: {time.time() - t0:.0f}s", flush=True)
        series.update(ser)
    return series, {"U10": U10, "alpha": alpha}


def analyse(D, series, cond=None):
    """cond: None (whole track) or "streak" (samples within STREAK_R of each side's
    own streakline; data and each model conditioned on their own wind)."""
    dist = np.hypot(D["x"], D["y"])
    half = len(dist) // 2
    res = {}
    if cond == "streak":
        vd = streak_offset(D["x"], D["y"], D["u"], D["v"]) < STREAK_R
        vm = {}
        for m in series:
            if m.startswith("wind_"):
                continue
            w = series["wind_s2"] if m == "ours_s2" else series["wind_s1"]
            vm[m] = streak_offset(D["x"], D["y"], w[:, 0], w[:, 1]) < STREAK_R
    else:
        vd = np.ones(len(dist), bool); vm = {m: vd for m in series}
    for vth in V_TH:
        dstat = whiff_stats(D["c"], dist, vth, vd)
        gamma0 = dstat[0]["gamma"]
        sat_ratio = np.log(V_SAT / vth)
        row = {"data": [{k: v for k, v in b.items() if k != "logratio"} for b in dstat], "models": {},
               "data_frac_time_near_streak": float(vd.mean())}
        # data half-vs-half (noise floor)
        v1 = np.zeros(len(dist), bool); v1[:half] = True
        h1, h2 = whiff_stats(D["c"], dist, vth, v1 & vd), whiff_stats(D["c"], dist, vth, ~v1 & vd)
        row["data_halves"] = [[{k: v for k, v in b.items() if k != "logratio"} for b in h] for h in (h1, h2)]
        for m, c in series.items():
            if m.startswith("wind_"):
                continue
            inb0 = (dist < BINS[0][1]) & vm[m]
            cm = c[inb0]
            if not np.any(cm > 0) or not np.isfinite(gamma0):
                row["models"][m] = {"c_threshold": None, "note": "no concentration in 0–5 m bin"}
                continue
            calib_ok = bool(np.mean(cm > 0) >= gamma0)
            thr = float(np.quantile(cm, 1 - gamma0)) if gamma0 > 0 else float(cm.max())
            thr = max(thr, 1e-30)
            ms = whiff_stats(c, dist, thr, vm[m])
            for b_d, b_m in zip(dstat, ms):
                b_m["skld"] = skld(np.minimum(b_m["logratio"], sat_ratio), b_d["logratio"], 0.0, sat_ratio)
            row["models"][m] = {"c_threshold": thr, "calibration_ok": calib_ok,
                                "frac_time_near_streak": float(vm[m].mean()), "bins": [{k: v for k, v in b.items() if k != "logratio"} for b in ms]}
        res[f"{vth:g}V"] = row
    return res


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaden-tuned-noise", type=float, required=True)
    ap.add_argument("--sets", default="desert_low_wind,desert_high_wind")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(HERE, "results", "t3.json")
    allres = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name in a.sets.split(","):
        D = data_series(name)
        npz = os.path.join(OUT, f"{name}_series.npz")
        if os.path.exists(npz):
            series = dict(np.load(npz)); meta = json.load(open(npz + ".json"))
        else:
            print(f"{name}: class {D['pg']} (σθ {D['sigma_theta_deg']:.1f}°), dir {D['wind_dir_deg']:.1f}°, "
                  f"S {D['S_mean']:.2f} m/s, {len(D['t'])/FS:.0f} s", flush=True)
            series, meta = run_models(D, a.gaden_tuned_noise)
            np.savez_compressed(npz, **series)
            json.dump(meta, open(npz + ".json", "w"))
        r = analyse(D, series)
        rs = analyse(D, series, cond="streak")
        so = streak_offset(D["x"], D["y"], D["u"], D["v"])
        chk = {"spearman_vs_authors_streak": float(_spearman(so, D["streak_authors"])),
               "median_abs_diff_m": float(np.median(np.abs(so - D["streak_authors"])))}
        allres[name] = {"streak_check": chk, "near_streak": rs,"class": D["pg"], "sigma_theta_deg": D["sigma_theta_deg"], "wind_dir_deg": D["wind_dir_deg"],
                        "S_mean": D["S_mean"], **meta, "gaden_tuned_noise": a.gaden_tuned_noise,
                        "assumptions": {"h_source_m": H_SRC, "h_sensor_m": H_SENS, "z_anemometer_m": Z_ANEMO},
                        "thresholds": r}
        with open(out_path, "w") as f:
            json.dump(allres, f, indent=1, default=float)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
