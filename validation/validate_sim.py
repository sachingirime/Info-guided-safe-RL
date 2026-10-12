"""
validate_sim.py — Headless validation of the fixed simulator (V0–V8).

Writes validation/results.json and figures (PNG + PDF) to validation/figs/.
Each test prints PASS/FAIL against its criterion (V4, V7, V8 are reports).

    python validation/validate_sim.py                 # all tests
    python validation/validate_sim.py --only V1,V3    # subset (merged into results.json)

Concentration is always sampled with query_concentration on arbitrary sensor
arrays (1 m grids), never the 4 m concentration grid.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle
import diag_state

wp.config.quiet = True

RESULTS = os.path.join(HERE, "results.json")
FIGS = os.path.join(HERE, "figs")

# Reference categorical order (blue, orange, aqua); neutral ink for references
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
C_REF = "#3d3d3a"
C_MUTED = "#8a8980"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": C_MUTED, "lines.linewidth": 1.6, "savefig.dpi": 300,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def save_fig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def make_sim(source, U, Q=1.0, emit=0.2, seed=1, angle=0.0, pg="D", max_puffs=1024, **kw):
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=angle, wind_angle_range=20.0,
                                       wind_base_speed=U, wind_seed=seed, seed=seed,
                                       max_puffs=max_puffs, **kw)
    sim.Q = Q; sim.emit_interval = emit; sim.sigma_0 = 2.0
    sim.set_stability_class(pg)
    sim.source_pos = np.array(source, np.float32)
    sim.set_obstacles([], [])
    return sim


def run(sim, seconds, every=None, cb=None):
    n = int(round(seconds / sim.frame_dt))
    k = int(round(every / sim.frame_dt)) if every else 0
    for i in range(n):
        sim.step()
        if cb and k and (i + 1) % k == 0:
            cb()


def plane(x, ys, zs):
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1).astype(np.float32), Y.shape


def moment_sigma(w, x):
    w = np.clip(w, 0, None)
    m = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))), float(m)


def gauss(y, a, mu, s):
    return a * np.exp(-0.5 * (y - mu) ** 2 / s ** 2)


def fit_gauss(y, c):
    s0, m0 = moment_sigma(c, y)
    p, _ = curve_fit(gauss, y, c, p0=[c.max(), m0, max(s0, 1.0)], maxfev=20000)
    return abs(float(p[2])), float(p[1])


def fit_reflected(z, c, h):
    f = lambda z, a, s: a * (np.exp(-0.5 * (z - h) ** 2 / s ** 2) + np.exp(-0.5 * (z + h) ** 2 / s ** 2))
    s0, _ = moment_sigma(c, z)
    p, _ = curve_fit(f, z, c, p0=[c.max(), max(s0, 1.0)], bounds=([0, 0.3], [np.inf, 200]), maxfev=20000)
    return float(p[1])


def slender_plume(Q, U, ys, zs, ysrc, h, sy, sz):
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return (Q / (2 * np.pi * U * sy * sz) * np.exp(-0.5 * (Y - ysrc) ** 2 / sy ** 2)
            * (np.exp(-0.5 * (Z - h) ** 2 / sz ** 2) + np.exp(-0.5 * (Z + h) ** 2 / sz ** 2)))


def briggs_D(x):
    return 0.08 * x / np.sqrt(1 + 0.0001 * x), 0.06 * x / np.sqrt(1 + 0.0015 * x)


# ── V0: before/after ─────────────────────────────────────────────────────────

def v0():
    sim = diag_state.make_sim()
    after = diag_state.diagnose(sim)
    after["code"] = "sim-fix"
    with open(os.path.join(HERE, "after.json"), "w") as f:
        json.dump(after, f, indent=2)
    with open(os.path.join(HERE, "before.json")) as f:
        before = json.load(f)
    dz_b = before["median_puff_height"] - before["source_z"]
    dz_a = after["median_puff_height"] - after["source_z"]
    ok = (after["grid_share"] == 0.0 and abs(after["w_max_source"]) < 0.5 and abs(dz_a) < 5.0)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    items = [("Grid share of sensed C", before["grid_share"] * 100, after["grid_share"] * 100, "%"),
             ("Vertical wind at source", abs(before["w_max_source"]), abs(after["w_max_source"]), "m/s"),
             ("Median puff height − source", dz_b, dz_a, "m")]
    for ax, (title, b, a, unit) in zip(axs, items):
        ax.bar([0, 1], [b, a], color=[C_MUTED, C_SERIES[0]], width=0.6)
        ax.set_xticks([0, 1], ["pre-fix", "fixed"])
        ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="x", visible=False)
        for i, v in enumerate([b, a]):
            ax.annotate(f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}", (i, v), ha="center",
                        va="bottom" if v >= 0 else "top",
                        xytext=(0, 2 if v >= 0 else -2), textcoords="offset points", fontsize=8)
        ax.margins(y=0.15)
    fig.tight_layout()
    save_fig(fig, "v0_before_after")
    print(f"V0 before/after: grid_share {before['grid_share']:.3f}→{after['grid_share']:.3f}, "
          f"|w| {abs(before['w_max_source']):.2f}→{abs(after['w_max_source']):.2f} m/s, "
          f"Δz {dz_b:.1f}→{dz_a:.1f} m  {verdict(ok)}")
    return {"before": before, "after": after, "pass": ok,
            "criterion": "grid_share == 0; |w| at source < 0.5 m/s (final snapshot); |median z − source z| < 5 m"}


# ── V1: analytic plume ───────────────────────────────────────────────────────

V1_SRC = (30.0, 64.0, 30.0)
V1_U = 2.0
V1_X = (25.0, 50.0, 100.0)
V1_YS = np.arange(0.0, 128.0 + 0.5, 1.0)
V1_ZS = np.arange(0.0, 100.0 + 0.5, 1.0)


def v1_field(f_rel, spin=150.0, avg=180.0, seed=1):
    sim = make_sim(V1_SRC, V1_U, Q=1.0, emit=0.2, seed=seed, f_rel=f_rel, steady_wind=True,
                   shear_exponent=0.0, ground_drag_rate=0.0, buoyancy_mode="none")
    pts = [plane(V1_SRC[0] + d, V1_YS, V1_ZS) for d in V1_X]
    P = np.concatenate([p for p, _ in pts])
    acc = np.zeros(len(P)); n = [0]

    def cb():
        acc[:] += sim.query_concentration(P); n[0] += 1
    run(sim, spin)
    run(sim, avg, every=0.5, cb=cb)
    shape = pts[0][1]
    fields = (acc / n[0]).reshape(len(V1_X), *shape)
    return fields, sim


def v1(avg=180.0):
    out = {"setup": {"source": V1_SRC, "U": V1_U, "Q": 1.0, "emit_interval": 0.2, "class": "D",
                     "spin_s": 150.0, "avg_s": avg, "sample_dt": 0.5,
                     "y_range": [0, 128], "z_range": [0, 100], "grid_m": 1.0},
           "criterion": "σ_y, σ_z within 5%; centerline C within 10%; flux within 5% of Q; n_overwritten == 0",
           "note": ("Plan specified y ± 30 m, z ≤ 60 m and a 180 s average. y/z ranges were widened so the "
                    "flux integral captures the plume (σ_y(100 m) ≈ 19.5 m). With 180 s, f_rel=0 at 100 m "
                    "had centreline error −12.8% (sampling noise: 2 m puffs, ~20 m meander); at 1800 s it is "
                    "+1.6%, so the default average is 1800 s. σ: second moments of the sim and the analytic "
                    "field on the same grid."),
           "runs": {}}
    dy = dz = 1.0
    ys, zs = V1_YS, V1_ZS
    jy = int(np.argmin(abs(ys - V1_SRC[1]))); kz = int(np.argmin(abs(zs - V1_SRC[2])))
    all_ok = True
    profiles = {}
    for f_rel in (0.0, 0.5, 1.0):
        fields, sim = v1_field(f_rel, avg=avg)
        K = np.array([sim.K_D_X, sim.K_D_Y, sim.K_D_Z])
        rows = []
        ok_f = sim.n_overwritten == 0
        for d, C in zip(V1_X, fields):
            sy = np.sqrt(sim.sigma_0 ** 2 + 2 * K[1] * d / V1_U)
            sz = np.sqrt(sim.sigma_0 ** 2 + 2 * K[2] * d / V1_U)
            A = slender_plume(1.0, V1_U, ys, zs, V1_SRC[1], V1_SRC[2], sy, sz)
            # Same functionals on sim and analytic fields (same grid/truncation)
            sig = lambda F: (moment_sigma(F.sum(1) * dz, ys)[0], moment_sigma(F.sum(0) * dy, zs)[0])
            (sy_s, sz_s), (sy_a, sz_a) = sig(C), sig(A)
            cl_s, cl_a = float(C[jy, kz]), float(A[jy, kz])
            flux = float(V1_U * C.sum() * dy * dz)
            r = {"x_minus_xs": d, "sigma_y_ref": float(sy), "sigma_z_ref": float(sz),
                 "sigma_y_sim": sy_s, "sigma_y_analytic_grid": sy_a, "err_sigma_y": sy_s / sy_a - 1,
                 "sigma_z_sim": sz_s, "sigma_z_analytic_grid": sz_a, "err_sigma_z": sz_s / sz_a - 1,
                 "centerline_sim": cl_s, "centerline_analytic": cl_a, "err_centerline": cl_s / cl_a - 1,
                 "flux": flux, "err_flux": flux / 1.0 - 1}
            r["pass"] = bool(abs(r["err_sigma_y"]) < 0.05 and abs(r["err_sigma_z"]) < 0.05
                             and abs(r["err_centerline"]) < 0.10 and abs(r["err_flux"]) < 0.05)
            ok_f &= r["pass"]
            rows.append(r)
            profiles[(f_rel, d)] = (C[:, kz], C[jy, :], A[:, kz], A[jy, :])
            print(f"  V1 f_rel={f_rel:.1f} x-xs={d:5.0f}: σy {sy_s:6.2f}/{sy_a:6.2f} ({100*r['err_sigma_y']:+.1f}%)  "
                  f"σz {sz_s:6.2f}/{sz_a:6.2f} ({100*r['err_sigma_z']:+.1f}%)  "
                  f"C_cl {100*r['err_centerline']:+.1f}%  flux {100*r['err_flux']:+.1f}%  {verdict(r['pass'])}")
        out["runs"][str(f_rel)] = {"rows": rows, "n_overwritten": sim.n_overwritten,
                                   "n_emitted": sim.n_emitted, "n_exited": sim.n_exited, "pass": bool(ok_f)}
        all_ok &= ok_f
    out["pass"] = bool(all_ok)
    print(f"V1 analytic plume: {verdict(all_ok)}")

    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for j, d in enumerate(V1_X):
        for i, f_rel in enumerate((0.0, 0.5, 1.0)):
            cy, cz, ay, az = profiles[(f_rel, d)]
            axs[0, j].plot(ys - V1_SRC[1], cy * 1e3, color=C_SERIES[i], label=f"sim, $f_{{rel}}$={f_rel:g}")
            axs[1, j].plot(cz * 1e3, zs, color=C_SERIES[i])
        axs[0, j].plot(ys - V1_SRC[1], ay * 1e3, "--", color=C_REF, lw=1.2, label="analytic")
        axs[1, j].plot(az * 1e3, zs, "--", color=C_REF, lw=1.2)
        axs[0, j].set_title(f"x − x$_s$ = {d:.0f} m")
        axs[0, j].set_xlim(-60, 60); axs[1, j].set_ylim(0, 80)
        axs[0, j].set_xlabel("y − y$_s$ (m)"); axs[1, j].set_xlabel("C (g/m³)")
    axs[0, 0].set_ylabel("C at z = h (g/m³)"); axs[1, 0].set_ylabel("z (m), at y = y$_s$")
    axs[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "v1_profiles")
    return out


# ── V2: kernel mass ──────────────────────────────────────────────────────────

def v2():
    import test_env
    m_free = test_env._integrate_puff(60.0, 3.0, -1e9)
    m_ground = test_env._integrate_puff(2.0, 3.0, 0.0)
    ok = abs(m_free - 1) < 0.01 and abs(m_ground - 1) < 0.01
    print(f"V2 kernel mass: free {m_free:.5f}, ground {m_ground:.5f}  {verdict(ok)}")
    return {"mass_ratio_free_z60": m_free, "mass_ratio_ground_z2": m_ground, "pass": bool(ok),
            "criterion": "∫C dV within 1% of m (σ=3 m, 0.5 m grid, ±6σ; z≥0 for the ground case)",
            "note": "n_overwritten == 0 is asserted in V1 and V3"}


# ── V3/V4: Pasquill–Gifford comparison and intermittency ────────────────────

V3_SRC = (20.0, 64.0, 10.0)
V3_U = 3.0
V3_X = (25.0, 50.0, 100.0, 150.0, 200.0)
V4_X = (25.0, 50.0, 100.0)
V4_OFF_DY = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def v3_v4(spin=150.0, avg=600.0, seed=1, sim_kwargs=None):
    sim_kwargs = sim_kwargs or {}
    # 4096 puffs: slow near-ground puffs outlive a 1024-slot ring buffer
    # (1024 × 0.2 s = 205 s) at default turbulence, and the lost mass would bias σ.
    sim = make_sim(V3_SRC, V3_U, Q=1.0, emit=0.2, seed=seed, max_puffs=4096, **sim_kwargs)
    ys, zs = V1_YS, V1_ZS
    pts = [plane(V3_SRC[0] + d, ys, zs) for d in V3_X]
    P = np.concatenate([p for p, _ in pts])
    acc = np.zeros(len(P)); n = [0]
    probes = [[V3_SRC[0] + d, V3_SRC[1], V3_SRC[2]] for d in V4_X]
    probes += [[V3_SRC[0] + 50.0, V3_SRC[1] + dy, V3_SRC[2]] for dy in V4_OFF_DY]
    probes = np.array(probes, np.float32)
    series = []
    k_avg = int(round(0.5 / sim.frame_dt)); k_ts = int(round(0.1 / sim.frame_dt))

    run(sim, spin)
    for i in range(int(round(avg / sim.frame_dt))):
        sim.step()
        if (i + 1) % k_avg == 0:
            acc[:] += sim.query_concentration(P); n[0] += 1
        if (i + 1) % k_ts == 0:
            series.append(sim.query_concentration(probes))
    fields = (acc / n[0]).reshape(len(V3_X), *pts[0][1])
    series = np.array(series)

    kz = int(np.argmin(abs(zs - V3_SRC[2])))
    rows = []
    for d, C in zip(V3_X, fields):
        sy_fit, yc = fit_gauss(ys, C[:, kz])
        jy = int(np.argmin(abs(ys - yc)))
        sz_fit = fit_reflected(zs, C[jy, :], V3_SRC[2])
        sy_mom, _ = moment_sigma(C[:, kz], ys)
        by, bz = briggs_D(d)
        rows.append({"x_minus_xs": d, "sigma_y_sim": sy_fit, "sigma_z_sim": sz_fit,
                     "sigma_y_moment": sy_mom, "plume_center_y": yc,
                     "sigma_y_briggs": float(by), "sigma_z_briggs": float(bz),
                     "ratio_y": sy_fit / by, "ratio_z": sz_fit / bz})
    in_range = [r for r in rows if 50.0 <= r["x_minus_xs"] <= 200.0]
    ok = all(0.67 <= r["ratio_y"] <= 1.5 and 0.67 <= r["ratio_z"] <= 1.5 for r in in_range)
    ok_ow = sim.n_overwritten == 0
    for r in rows:
        print(f"  V3 x-xs={r['x_minus_xs']:5.0f}: σy {r['sigma_y_sim']:6.2f} vs {r['sigma_y_briggs']:6.2f} "
              f"(×{r['ratio_y']:.2f})   σz {r['sigma_z_sim']:6.2f} vs {r['sigma_z_briggs']:6.2f} (×{r['ratio_z']:.2f})")
    print(f"V3 Pasquill–Gifford (50–200 m within [0.67, 1.5]): {verdict(ok and ok_ow)}  "
          f"(n_overwritten={sim.n_overwritten})")

    v3 = {"setup": {"source": V3_SRC, "U": V3_U, "class": "D", "spin_s": spin, "avg_s": avg,
                    "sim_kwargs": {k: v for k, v in sim_kwargs.items()},
                    "f_rel": sim.f_rel, "K": [sim.K_D_X, sim.K_D_Y, sim.K_D_Z],
                    "turb_ou_sigma": sim._turb_ou_sigma,
                    "wind_ou_sigma": sim.wind_model._sigma.tolist(),
                    "turb_layer_sigmas": list(sim.wind_model._turb_sigmas)},
          "rows": rows, "n_overwritten": sim.n_overwritten, "n_emitted": sim.n_emitted,
          "pass": bool(ok and ok_ow),
          "criterion": "sim/Briggs-D ratio of σ_y and σ_z within [0.67, 1.5] for 50–200 m; n_overwritten == 0",
          "note": "σ_y: Gaussian fit to C(y, z=h); σ_z: ground-reflected Gaussian fit (h fixed) at the "
                  "fitted plume centre. Constant-K puffs grow as √x while Briggs is ~linear at short range. "
                  "max_puffs=4096 (1024 overwrote 838 puffs: slow near-ground puffs live > 205 s). "
                  "Shear is normalised by the 128 m domain height, so U=3 m/s gives ≈1.6 m/s at z=10 m."}
    sweep_path = os.path.join(HERE, "v3_tuning_sweep.json")
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            v3["tuning_sweep"] = json.load(f)
        v3["tuning_conclusion"] = (
            "No setting of turb_scale ∈ {1, 0.5, 0.25, 0.1}, K (table, Briggs-200 m derived, two smaller "
            "sets) or f_rel ∈ {0.5, 1} puts both 50 m and 200 m inside [0.67, 1.5]: lowering K enough for "
            "50 m (ratio 1.6 at K_y=0.47, K_z=0.2) pushes 200 m to 0.74-0.79 and falling. Constant-K puff "
            "growth (σ² = σ0² + 2K·t) cannot follow Briggs' near-linear σ(x) over 50-200 m. Defaults left "
            "unchanged; not forced (per plan).")

    # σ vs x figure
    xs = np.linspace(10, 250, 200)
    by, bz = briggs_D(xs)
    K = v3["setup"]["K"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, key, ref, k, lab in [(axs[0], "sigma_y_sim", by, K[1], "σ$_y$"),
                                 (axs[1], "sigma_z_sim", bz, K[2], "σ$_z$")]:
        ax.plot(xs, ref, "-", color=C_REF, lw=1.2, label="Briggs open country, class D")
        ax.fill_between(xs, ref * 0.67, ref * 1.5, color=C_REF, alpha=0.08, lw=0, label="pass band ×[0.67, 1.5]")
        ax.plot(xs, np.sqrt(4 + 2 * k * xs / V3_U), ":", color=C_MUTED, lw=1.2,
                label="puff law √(σ₀²+2Kx/U), U = 3 m/s")
        ax.plot([r["x_minus_xs"] for r in rows], [r[key] for r in rows], "o", color=C_SERIES[0],
                ms=6, mec="white", mew=1, label="sim (mean plume)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("x − x$_s$ (m)"); ax.set_ylabel(f"{lab} (m)")
    axs[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "v3_sigma_vs_x")

    # V4 statistics
    sy50 = next(r["sigma_y_sim"] for r in rows if r["x_minus_xs"] == 50.0)
    j_off = int(np.argmin(abs(np.array(V4_OFF_DY) - sy50)))
    names = [f"centerline x−xs={d:.0f} m" for d in V4_X] + [f"off-axis x−xs=50 m, y−ys={V4_OFF_DY[j_off]:.0f} m"]
    cols = list(range(len(V4_X))) + [len(V4_X) + j_off]
    stats = []
    for name, c in zip(names, cols):
        ts = series[:, c].astype(np.float64)
        mean = ts.mean()
        stats.append({"point": name, "mean": float(mean),
                      "intermittency": float(np.mean(ts > 0.1 * mean)) if mean > 0 else 0.0,
                      "peak_to_mean_p99": float(np.percentile(ts, 99) / mean) if mean > 0 else float("nan"),
                      "cov": float(ts.std() / mean) if mean > 0 else float("nan")})
        print(f"  V4 {name:38s} mean {mean:.3e}  γ(C>0.1·mean) {stats[-1]['intermittency']:.2f}  "
              f"p99/mean {stats[-1]['peak_to_mean_p99']:.2f}  CoV {stats[-1]['cov']:.2f}")
    sanity = stats[-1]["intermittency"] < stats[1]["intermittency"]
    print(f"V4 intermittency (report; off-axis more intermittent than centerline at 50 m: {sanity})")
    v4 = {"sample_hz": 10, "duration_s": avg, "stats": stats, "off_axis_dy": V4_OFF_DY[j_off],
          "sigma_y_50m_sim": sy50, "sanity_offaxis_more_intermittent": bool(sanity),
          "note": "intermittency factor = fraction of time C > 0.1·mean (lower = more intermittent)"}

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, (s, c) in enumerate(zip(stats, cols)):
        ts = series[:, c] / max(s["mean"], 1e-30)
        bins = np.logspace(-2, 1, 46)
        h, e = np.histogram(ts, bins=bins, density=False)
        frac = h / len(series[:, c])
        color = (C_SERIES + [C_REF])[i]
        ax.step(np.sqrt(e[1:] * e[:-1]), frac, where="mid", color=color, lw=1.4,
                ls="-" if i < 3 else "--", label=s["point"].replace("centerline ", "").replace("off-axis ", "off-axis "))
    ax.set_xscale("log"); ax.set_xlim(1e-2, 10)
    ax.set_xlabel("C / mean C"); ax.set_ylabel("fraction of samples")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "v4_pdfs")
    return v3, v4


# ── V5: determinism ──────────────────────────────────────────────────────────

def v5():
    a, _ = v1_field(0.5, spin=60.0, avg=60.0, seed=11)
    b, _ = v1_field(0.5, spin=60.0, avg=60.0, seed=11)
    c, _ = v1_field(0.5, spin=60.0, avg=60.0, seed=12)
    same = bool(np.array_equal(a, b))
    diff = float(np.abs(a - c).max())
    ok = same and diff > 0
    print(f"V5 determinism: same seed identical={same}, different seed max|ΔC|={diff:.3e}  {verdict(ok)}")
    return {"same_seed_identical": same, "diff_seed_max_abs": diff, "pass": ok,
            "setup": "V1 configuration, f_rel=0.5, 60 s spin-up + 60 s average (shortened from V1)"}


# ── V6: line of sight ────────────────────────────────────────────────────────

def v6():
    import test_env
    try:
        test_env.test_los(); unit_ok = True
    except AssertionError:
        unit_ok = False
    src = (40.0, 64.0, 10.0)
    wall = CuboidObstacle(center=np.array([80.0, 64.0, 15.0]), half_extents=np.array([3.0, 16.0, 15.0]))
    xs = np.arange(20.0, 160.0 + 0.5, 1.0); ys = np.arange(14.0, 114.0 + 0.5, 1.0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, src[2])], 1).astype(np.float32)
    slices = {}
    for los in (False, True):
        sim = make_sim(src, 2.0, Q=1.0, seed=3, use_los=los)
        sim.set_obstacles([], [wall])
        acc = np.zeros(len(P)); n = [0]

        def cb():
            acc[:] += sim.query_concentration(P); n[0] += 1
        run(sim, 100.0)
        run(sim, 180.0, every=0.5, cb=cb)
        slices[los] = (acc / n[0]).reshape(X.shape)
    # mean C in the shadow box directly behind the wall
    behind = (X > 84) & (X < 100) & (abs(Y - 64) < 10)
    c_off, c_on = float(slices[False][behind].mean()), float(slices[True][behind].mean())
    print(f"V6 line of sight: unit test {verdict(unit_ok)}; mean C in wall shadow (z=10 m): "
          f"LOS off {c_off:.3e}, on {c_on:.3e} (×{c_on / max(c_off, 1e-30):.2f})")

    vmax = max(s.max() for s in slices.values())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, los in zip(axs, (False, True)):
        im = ax.pcolormesh(xs, ys, np.maximum(slices[los].T, vmax * 1e-4), cmap="Blues",
                           norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), shading="auto")
        c, h = wall.center, wall.half_extents
        ax.add_patch(plt.Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1],
                                   facecolor="#d9d8d2", edgecolor=C_REF, lw=0.8))
        ax.plot(src[0], src[1], "*", color=C_SERIES[1], ms=10, mec="white")
        ax.set_title(f"line of sight {'on' if los else 'off'}")
        ax.set_xlabel("x (m)"); ax.set_aspect("equal"); ax.grid(False)
    axs[0].set_ylabel("y (m)")
    fig.colorbar(im, ax=axs, label="mean C at z = 10 m (kg/m³)", shrink=0.9)
    save_fig(fig, "v6_los_slice")
    return {"unit_test_pass": unit_ok, "shadow_mean_C_los_off": c_off, "shadow_mean_C_los_on": c_on,
            "setup": "wind +x 2 m/s (turbulence on), source (40,64,10), wall centre (80,64,15) "
                     "half-extents (3,16,15); slice z=10 m; 100 s spin-up, 180 s mean",
            "pass": bool(unit_ok)}


# ── V7: performance ──────────────────────────────────────────────────────────

def v7():
    rows = []
    obst = GaussianPuffEnv.FIXED_OBSTACLES
    spheres = [o for o in obst if isinstance(o, SphereObstacle)]
    cuboids = [o for o in obst if isinstance(o, CuboidObstacle)]
    rng = np.random.default_rng(0)
    for los in (False, True):
        sim = make_sim((60.0, 64.0, 10.0), 1.5, seed=1, use_los=los)
        sim.set_obstacles(spheres, cuboids)
        run(sim, 10.0)
        sim.inject_puffs(rng.uniform([0, 0, 0], [256, 128, 60], (1024, 3)), ages=rng.uniform(0, 100, 1024))
        for n_q in (0, 1, 1000):
            P = rng.uniform([0, 0, 4], [256, 128, 60], (max(n_q, 1), 3)).astype(np.float32)
            n_steps = 600
            wp.synchronize(); t0 = time.perf_counter()
            for _ in range(n_steps):
                sim.step()
                if n_q:
                    sim.query_concentration(P)
            wp.synchronize(); dt = (time.perf_counter() - t0) / n_steps
            rows.append({"use_los": los, "n_query": n_q, "ms_per_step": dt * 1e3,
                         "steps_per_s": 1 / dt, "real_time_factor": sim.frame_dt / dt})
            print(f"  V7 los={str(los):5s} query={n_q:5d}: {dt*1e3:6.3f} ms/step  {1/dt:7.0f} steps/s  "
                  f"RTF {sim.frame_dt/dt:6.1f}×")
    gpu = wp.get_device().name
    print(f"V7 performance on {gpu} (report)")
    return {"device": gpu, "n_puffs": 1024, "n_obstacles": len(obst), "frame_dt": 1 / 30, "rows": rows,
            "note": "one sim step = one 1/30 s frame (wind solve, 25 SOR iterations, puff advection); "
                    "query = query_concentration incl. host copy"}


# ── V8: pressure solve ───────────────────────────────────────────────────────

def v8():
    out = {"sor_iters": None, "rows": []}
    obst = GaussianPuffEnv.FIXED_OBSTACLES
    spheres = [o for o in obst if isinstance(o, SphereObstacle)]
    cuboids = [o for o in obst if isinstance(o, CuboidObstacle)]
    for with_obst in (False, True):
        sim = make_sim((60.0, 64.0, 10.0), 1.5, seed=1)
        if with_obst:
            sim.set_obstacles(spheres, cuboids)
        out["sor_iters"] = sim.sor_iters
        post, pre = [], []

        def cb():
            post.append(sim.divergence_residual()); pre.append(sim.last_div_pre)
        run(sim, 30.0)
        run(sim, 30.0, every=1.0, cb=cb)
        r = {"obstacles": with_obst, "residual_post": float(np.mean(post)), "residual_pre": float(np.mean(pre))}
        out["rows"].append(r)
        print(f"  V8 obstacles={str(with_obst):5s}: mean|div|/(U/dx) after projection {r['residual_post']:.4f} "
              f"(before {r['residual_pre']:.4f})")
    print(f"V8 pressure solve: Red-Black SOR, ω=1.7, {out['sor_iters']} iterations (report)")
    out["note"] = ("class D, U=1.5 m/s, 30-60 s. Residual does not drop with more iterations "
                   "(0.0076 at 200): the floor is the collocated central-difference div/grad vs compact "
                   "7-point Laplacian mismatch, not SOR convergence.")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="V0,V1,V2,V3,V5,V6,V7,V8",
                    help="comma list; V3 also produces V4")
    ap.add_argument("--v1-avg", type=float, default=1800.0)
    ap.add_argument("--v3-avg", type=float, default=600.0)
    args = ap.parse_args()
    wp.init()

    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    todo = [t.strip().upper() for t in args.only.split(",")]
    t_start = time.time()
    if "V0" in todo: results["V0"] = v0()
    if "V1" in todo: results["V1"] = v1(avg=args.v1_avg)
    if "V2" in todo: results["V2"] = v2()
    if "V3" in todo or "V4" in todo:
        results["V3"], results["V4"] = v3_v4(avg=args.v3_avg)
    if "V5" in todo: results["V5"] = v5()
    if "V6" in todo: results["V6"] = v6()
    if "V7" in todo: results["V7"] = v7()
    if "V8" in todo: results["V8"] = v8()
    results["_meta"] = {"device": wp.get_device().name, "warp": wp.__version__,
                        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "last_run_tests": todo, "last_run_wall_s": round(time.time() - t_start, 1)}

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSummary:")
    for k in ("V0", "V1", "V2", "V3", "V5", "V6"):
        if k in results:
            print(f"  {k}: {verdict(results[k]['pass'])}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
