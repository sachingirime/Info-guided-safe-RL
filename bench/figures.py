"""
figures.py — benchmark figures (bench/figs/*.png|pdf) from bench/results/*.json
and bench/data/t1/*.npz.  Style and colours match validation/validate_sim.py.

    python bench/figures.py [t1] [t2] [t3] [t4]
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
RES = os.path.join(HERE, "results")

C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # ours, GADEN (tuned/matched), GADEN (default)
C_REF = "#3d3d3a"
C_MUTED = "#8a8980"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": C_MUTED, "lines.linewidth": 1.6, "savefig.dpi": 300,
})


def fmt(v):
    """Compact number: 2 decimals below 100, else 10^k."""
    if not np.isfinite(v):
        return "–"
    return f"{v:.2f}" if v < 100 else f"10$^{{{int(np.floor(np.log10(v)))}}}$"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_t1():
    """Crosswind profiles at source height, ours vs GADEN, per scenario."""
    planes = (25, 50, 100, 150)
    scen = [("T1a", "steady, uniform"), ("T1c", "meandering wind"), ("T1d", "meandering + obstacles")]
    fig, axs = plt.subplots(len(scen), len(planes), figsize=(7.0, 1.9 * len(scen)), sharey="row")
    for i, (s, title) in enumerate(scen):
        d = np.load(os.path.join(HERE, "data", "t1", f"{s}.npz"))
        pts, v = d["pts"], d["valid"]
        src_z = json.load(open(os.path.join(RES, "t1.json")))[s]["config"]["source"][2]
        src_x = json.load(open(os.path.join(RES, "t1.json")))[s]["config"]["source"][0]
        src_y = json.load(open(os.path.join(RES, "t1.json")))[s]["config"]["source"][1]
        for j, pl in enumerate(planes):
            sel = np.isclose(pts[:, 0], src_x + pl) & np.isclose(pts[:, 2], np.round(src_z)) & v
            y = pts[sel, 1] - src_y
            ax = axs[i, j]
            ax.plot(y, d["mean_ours"][sel] * 1e3, color=C_SERIES[0], label="ours (matched)")
            ax.plot(y, d["mean_gaden"][sel] * 1e3, color=C_SERIES[1], lw=1.2, ls="--", label="GADEN")
            if i == 0:
                ax.set_title(f"x − x$_s$ = {pl} m")
            if j == 0:
                ax.set_ylabel(f"{title}\nC/Q (10$^{{-3}}$ s/m³)")
            if i == len(scen) - 1:
                ax.set_xlabel("y − y$_s$ (m)")
            ax.set_xlim(-40, 40) if i else ax.set_xlim(-8, 8)
    axs[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "t1_profiles")


def fig_t2():
    r = json.load(open(os.path.join(RES, "t2.json")))
    sys.path.insert(0, HERE)
    import t2_stats as ts
    table, _, runs = ts.load_runs()
    tuned = f"G-ourwind-n{r['tuning']['chosen_noise_std']}"
    models = [("ours", "ours (as shipped)", C_SERIES[0]), (tuned, f"GADEN tuned ({r['tuning']['chosen_noise_std']})", C_SERIES[1]),
              ("G-ourwind-n0.01", "GADEN example params", C_SERIES[2])]
    # scatter: arc-max C/Q, observed vs predicted
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True, sharey=True)
    lo, hi = 1e-5, 1e-1
    for ax, (m, lab, col) in zip(axs, models):
        pl = ts.pairs(table, runs, m, "cmax_Q")
        co = np.array([p[2] for p in pl]); cp = np.maximum([p[3] for p in pl], lo)
        arcs = np.array([p[1] for p in pl])
        for a, mk in ((50, "o"), (100, "s"), (200, "^")):
            s = arcs == a
            ax.scatter(co[s], cp[s], s=10, marker=mk, color=col, alpha=0.75, linewidths=0,
                       label=f"{a} m")
        xx = np.array([lo, hi])
        ax.plot(xx, xx, color=C_REF, lw=0.8); ax.plot(xx, 2 * xx, color=C_MUTED, lw=0.6, ls="--")
        ax.plot(xx, 0.5 * xx, color=C_MUTED, lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        st = r["stats"]["cmax_Q"].get(m, {}).get("point", {})
        ax.set_title(lab)
        ax.text(0.03, 0.97, f"FAC2 {st.get('FAC2', np.nan):.2f}\nMG {fmt(st.get('MG', np.nan))}\nVG {fmt(st.get('VG', np.nan))}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=C_REF)
        ax.set_xlabel("observed arc-max C/Q (s/m³)")
    axs[0].set_ylabel("predicted arc-max C/Q (s/m³)")
    axs[0].legend(frameon=False, loc="lower right", markerscale=1.2)
    fig.tight_layout()
    save(fig, "t2_scatter_cmax")

    # MG–VG with bootstrap CIs, per quantity
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, q, title in zip(axs, ("cmax_Q", "cwic_Q", "sigma_y"), ("arc-max C/Q", "CWIC/Q", "σ$_y$")):
        d = r["stats"][q].get("G-ourwind-n0.01", {}).get("point", {})
        ax.text(0.03, 0.97, f"GADEN example params\n(off scale): MG {fmt(d.get('MG', np.nan))}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, color=C_MUTED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        for m, lab, col in models[:2]:
            v = r["stats"][q].get(m)
            if not v:
                continue
            p, c = v["point"], v["ci"]
            ax.errorbar(p["MG"], p["VG"], xerr=[[p["MG"] - c["MG"][0]], [c["MG"][1] - p["MG"]]],
                        yerr=[[p["VG"] - c["VG"][0]], [c["VG"][1] - p["VG"]]],
                        fmt="o", ms=5, color=col, capsize=2, lw=1.0, label=lab)
        mg = np.logspace(np.log10(0.25), np.log10(4.0), 200)
        ax.plot(mg, np.exp(np.log(mg) ** 2), color=C_MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1, color=C_REF, lw=0.6)
        ax.set_xlabel("MG (>1: under-prediction)"); ax.set_title(title)
        ax.set_xlim(0.25, 4.0); ax.set_ylim(1.0, 12.0)
        ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["1", "2", "4", "8"]); ax.minorticks_off()
    axs[0].set_ylabel("VG")
    axs[2].legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    save(fig, "t2_mg_vg")


def fig_t3(vth="4V"):
    r = json.load(open(os.path.join(RES, "t3.json")))
    sets = [k for k in ("desert_low_wind", "desert_high_wind") if k in r]
    stats = [("gamma", "intermittency γ"), ("whiff_freq", "whiff frequency (Hz)"), ("whiff_dur", "whiff duration (s)")]
    fig, axs = plt.subplots(len(sets), 3, figsize=(7.0, 2.2 * len(sets)), squeeze=False)
    models = [("ours_s1", "ours", C_SERIES[0], "-"),
              ("G-default", "GADEN example", C_SERIES[2], ":")]
    for i, name in enumerate(sets):
        T = r[name]["near_streak"][vth]
        xc = [np.mean(b["bin"]) for b in T["data"]]
        for j, (k, lab) in enumerate(stats):
            ax = axs[i, j]
            ax.plot(xc, [b[k] for b in T["data"]], "o-", color=C_REF, lw=1.2, ms=4, label="data")
            for h in T["data_halves"]:
                ax.plot(xc, [b[k] for b in h], color=C_MUTED, lw=0.6, alpha=0.8)
            for m, ml, col, ls in models:
                mm = T["models"].get(m, {})
                if not mm.get("bins"):
                    continue
                ax.plot(xc, [b[k] for b in mm["bins"]], ls, color=col, marker="s", ms=3, lw=1.3, label=ml)
            ax.axvspan(0, 5, color="#e6e5e0", alpha=0.6, lw=0)
            if i == len(sets) - 1:
                ax.set_xlabel("distance from source (m)")
            ax.set_title(lab if i == 0 else "")
            if j == 0:
                cls = r[name]["class"]
                ax.set_ylabel(f"{name.replace('_', ' ')}\n(class {cls})")
    axs[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    import textwrap
    note = (f"Threshold {vth}; samples within 3 m of each side's own streakline. Shaded 0–5 m bin calibrates each "
            "model's threshold; thin grey lines: data halves. GADEN tuned omitted: its plume reaches the track less "
            "often than the data whiff rate, so its threshold cannot be calibrated.")
    fig.text(0.01, -0.06, textwrap.fill(note, 140), fontsize=7, color=C_MUTED)
    fig.tight_layout()
    save(fig, "t3_whiff_stats")


def fig_t4(name="t4"):
    r = json.load(open(os.path.join(RES, f"{name}.json")))
    W = r["worlds"]
    names = {"ours": "ours", "G-default": "GADEN\nexample", "G-tuned": "GADEN\ntuned"}
    fig, axs = plt.subplots(1, len(r["families"]), figsize=(6.4, 2.7))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f2f6fc", "#2a78d6", "#0d3b75"])
    for ax, (fam, F) in zip(axs, r["families"].items()):
        M = np.array([[F["matrix"][f"{A}->{B}"]["success"] for B in W] for A in W])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
        for a in range(len(W)):
            for b in range(len(W)):
                c = F["matrix"][f"{W[a]}->{W[b]}"]
                ax.text(b, a, f"{c['success']:.2f}\n[{c['success_ci'][0]:.2f}, {c['success_ci'][1]:.2f}]",
                        ha="center", va="center", fontsize=6.5, color="white" if c["success"] > 0.55 else C_REF)
        ax.set_xticks(range(len(W))); ax.set_xticklabels([names[w] for w in W], fontsize=7)
        ax.set_yticks(range(len(W))); ax.set_yticklabels([names[w] for w in W], fontsize=7)
        ax.set_xlabel("tested in"); ax.set_ylabel("tuned in"); ax.set_title(f"{fam}: success rate"); ax.grid(False)
    fig.tight_layout()
    save(fig, "t4_transfer" if name == "t4" else f"{name}_transfer")


def fig_t2_v11():
    r = json.load(open(os.path.join(RES, "t2_v11.json")))
    sys.path.insert(0, HERE)
    import t2_stats as ts
    import t2_final as tf
    table, _, base = ts.load_runs()
    runs = {}
    for e, rr in base.items():
        v = json.load(open(os.path.join(HERE, "data", "t2_variants", r["tag"], f"run_{e:02d}.json")))["models"]
        m = dict(rr["models"]); m["v1.1"] = tf.ensemble(v, "s", (1, 2, 3)); m["v1.0"] = tf.ensemble(rr["models"], "ours_s", (1, 2, 3))
        runs[e] = {"models": m}
    models = [("v1.1", "v1.1 (surface-layer LSM)", C_SERIES[0]), ("v1.0", "v1.0 (Briggs-fitted)", C_SERIES[2]),
              ("G-ourwind-n0.7", "GADEN tuned (0.7)", C_SERIES[1])]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.8))
    for j, (q, lab, lo, hi) in enumerate((("cmax_Q", "arc-max C/Q (s/m³)", 1e-5, 1e-1),
                                          ("cwic_Q", "CWIC/Q (s/m²)", 1e-3, 1.0), ("sigma_y", "σ$_y$ (m)", 1.0, 100.0))):
        ax = axs[0, j]
        for m, ml, col in models:
            pl = ts.pairs(table, runs, m, q)
            co = np.array([p[2] for p in pl]); cp = np.maximum([p[3] for p in pl], lo)
            ax.scatter(co, cp, s=7, color=col, alpha=0.6, linewidths=0, label=ml)
        xx = np.array([lo, hi]); ax.plot(xx, xx, color=C_REF, lw=0.8)
        for f in (0.5, 2.0):
            ax.plot(xx, f * xx, color=C_MUTED, lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("observed " + lab, fontsize=7.5); ax.set_title(lab)
        ax2 = axs[1, j]
        for m, ml, col in models:
            st = r["stats"]["even_heldout"][q][m]; p, c = st["point"], st["ci"]
            ax2.errorbar(p["MG"], p["VG"], xerr=[[p["MG"] - c["MG"][0]], [c["MG"][1] - p["MG"]]],
                         yerr=[[p["VG"] - c["VG"][0]], [c["VG"][1] - p["VG"]]], fmt="o", ms=5, color=col, capsize=2, lw=1.0, label=ml)
        mg = np.logspace(np.log10(0.25), np.log10(4.0), 200)
        ax2.plot(mg, np.exp(np.log(mg) ** 2), color=C_MUTED, lw=0.8, ls=":")
        ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.axvline(1, color=C_REF, lw=0.6)
        ax2.set_xlim(0.25, 4.0); ax2.set_ylim(1.0, 8.0)
        ax2.set_xticks([0.25, 0.5, 1, 2, 4]); ax2.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax2.set_yticks([1, 2, 4, 8]); ax2.set_yticklabels(["1", "2", "4", "8"]); ax2.minorticks_off()
        ax2.set_xlabel("MG (>1: under-prediction)", fontsize=7.5)
    axs[0, 0].set_ylabel("predicted"); axs[1, 0].set_ylabel("VG (held-out even runs)")
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left", markerscale=1.5)
    fig.tight_layout()
    save(fig, "t2_v11")


def fig_t3a():
    r = json.load(open(os.path.join(HERE, "data", "t3_rescore.json")))
    sets = ("desert_low_wind", "desert_high_wind")
    series = [("sl_sg_s03k01", "ours_s1", "v1.1", C_SERIES[0], "-"), ("v1.0-bench", "ours_s1", "v1.0", C_SERIES[2], ":"),
              ("v1.0-bench", "G-default", "GADEN example", C_SERIES[1], "--")]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.4), squeeze=False)
    for i, sname in enumerate(sets):
        d = r["sl_sg_s03k01"][sname]["matched"]["data"]
        xc = [np.mean(b["bin"]) for b in d]
        for j, (k, lab) in enumerate((("dur_mean", "mean whiff duration (s)"), ("blank_median", "median blank (s)"))):
            ax = axs[i, j]
            ax.plot(xc, [b[k] for b in d], "o-", color=C_REF, ms=4, lw=1.2, label="data")
            for tag, mk, ml, col, ls in series:
                mm = r[tag][sname]["matched"]["models"][mk]
                ys = [b.get(k, np.nan) if b.get("matched") else np.nan for b in mm]
                ax.plot(xc, ys, ls, color=col, marker="s", ms=3, lw=1.3, label=ml)
            ax.set_yscale("log")
            if i == 1:
                ax.set_xlabel("distance from source (m)")
            ax.set_title(lab if i == 0 else "")
            if j == 0:
                ax.set_ylabel(f"{sname.replace('_', ' ')}")
    axs[0, 0].legend(frameon=False, fontsize=7)
    import textwrap
    from matplotlib.ticker import FuncFormatter, LogLocator
    for ax in axs.ravel():
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    note = ("At matched intermittency per distance bin (samples within 3 m of each side's streakline): invariant to "
            "the sensor's unknown, saturating transfer function. Gaps: the model plume reaches the sensor less often "
            "than the data whiff rate in that bin.")
    fig.text(0.01, -0.05, textwrap.fill(note, 120), fontsize=6.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "t3a_whiff_timing")


if __name__ == "__main__":
    which = sys.argv[1:] or ["t1", "t2"]
    for w in which:
        {"t1": fig_t1, "t2": fig_t2, "t3": fig_t3, "t4": fig_t4, "t2_v11": fig_t2_v11, "t3a": fig_t3a,
         "t4_v11": lambda: fig_t4("t4_v11")}[w]()
        print("figure", w)
