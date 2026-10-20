"""
t2_stats.py — Tier 2 statistics (Prairie Grass) from bench/data/t2/run_*.json.

Measures (Hanna et al. 1993; Chang & Hanna 2004; Hanna & Chang 2012), with
Co observed and Cp predicted, over all (run, arc) pairs:
    FB   = (mean Co − mean Cp) / (0.5 (mean Co + mean Cp))      (> 0: under-prediction)
    NMSE = mean (Co − Cp)² / (mean Co · mean Cp)
    MG   = exp(mean ln Co − mean ln Cp)
    VG   = exp(mean (ln Co − ln Cp)²)
    FAC2 = fraction with 0.5 ≤ Cp/Co ≤ 2
    NAD  = mean |Co − Cp| / (mean Co + mean Cp)
    R    = Pearson correlation
MG/VG use a floor (FLOOR) on Co and Cp so that zero predictions are defined.
95% CIs: bootstrap over runs (all arcs of a run resampled together), 2000
resamples.  Model comparison: paired bootstrap of Δ = stat(A) − stat(B) on
the same resamples; significant if the CI excludes 0.

ASTM D6589-style test: regimes = stability group (A–C, D, E–F) × arc; near-
centreline concentration (samplers within ±0.67 σ_y of the arc centroid);
bootstrap mean per regime for observations and for each model; reports the
fractional difference model/obs with CI.

GADEN tuning: filamentNoise_std chosen to minimise mean (ln Co − ln Cp)² of
arc-max C/Q over odd-numbered runs; scored on even-numbered runs only.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prairie_grass as pg     # noqa: E402

FLOOR = {"cmax_Q": 1e-5, "cwic_Q": 1e-4, "sigma_y": 0.1, "nearcl_Q": 1e-5}
GROUPS = {"A-C": "ABC", "D": "D", "E-F": "EF"}
N_BOOT = 2000
SIGMA_Y_MAX_200 = 20.0   # 200 m CWIC/σ_y used only if 3σ_y,obs < 60 m (domain half-width)


def measures(co, cp, floor):
    co = np.asarray(co, float); cp = np.asarray(cp, float)
    mo, mp = co.mean(), cp.mean()
    lo, lp = np.log(np.maximum(co, floor)), np.log(np.maximum(cp, floor))
    ratio = np.where(co > 0, cp / np.where(co > 0, co, 1), np.inf)
    return {"FB": (mo - mp) / (0.5 * (mo + mp)) if mo + mp > 0 else np.nan,
            "NMSE": np.mean((co - cp) ** 2) / (mo * mp) if mo * mp > 0 else np.inf,
            "MG": float(np.exp(np.mean(lo - lp))),
            "VG": float(np.exp(np.mean((lo - lp) ** 2))),
            "FAC2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
            "NAD": float(np.mean(np.abs(co - cp)) / (mo + mp)) if mo + mp > 0 else np.nan,
            "R": float(np.corrcoef(co, cp)[0, 1]) if co.std() > 0 and cp.std() > 0 else np.nan,
            "n": int(len(co))}


def load_runs():
    table, dropped = pg.load()
    runs = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "t2", "run_*.json"))):
        r = json.load(open(f))
        runs[r["exp"]] = r
    return table, dropped, runs


def model_value(run, model, arc, q):
    """Ours: mean over seeds (ensemble mean, as D6589 compares to)."""
    if model == "ours":
        vals = [run["models"][f"ours_s{s}"]["arcs"][str(arc)][q] for s in (1, 2, 3)
                if f"ours_s{s}" in run["models"]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    v = run["models"].get(model, {}).get("arcs", {}).get(str(arc), {}).get(q)
    return v


def pairs(table, runs, model, q, exps=None, arcs=(50, 100, 200), groups=None):
    """[(exp, arc, Co, Cp)] with the pre-registered 200 m rule for CWIC/σ_y."""
    out = []
    for e, run in runs.items():
        if exps is not None and e not in exps:
            continue
        ob = table[e]
        if groups and ob["pg"] not in groups:
            continue
        for a in arcs:
            o = ob["arcs"].get(a)
            if o is None:
                continue
            if a == 200 and q in ("cwic_Q", "sigma_y") and o["sigma_y"] > SIGMA_Y_MAX_200:
                continue
            cp = model_value(run, model, a, q)
            if cp is None and q == "sigma_y":
                cp = 0.0
            if cp is None:
                continue
            co = o[q] if q != "nearcl_Q" else o.get("nearcl_Q")
            out.append((e, a, float(co), float(cp)))
    return out


def boot(pr_list, floor, rng, n_boot=N_BOOT):
    exps = sorted({p[0] for p in pr_list})
    by = {e: [(p[2], p[3]) for p in pr_list if p[0] == e] for e in exps}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(exps, len(exps), replace=True)
        co = [c for e in pick for c, _ in by[e]]; cp = [p for e in pick for _, p in by[e]]
        stats.append(measures(co, cp, floor))
    return stats, exps


def ci(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def evaluate(table, runs, models, q, exps=None, groups=None, arcs=(50, 100, 200), seed=0):
    """Point estimates + CIs per model, and paired differences vs the first model."""
    floor = FLOOR[q]
    res = {}
    # common (exp, arc) set across models so the comparison is paired
    sets = {m: {(p[0], p[1]): p for p in pairs(table, runs, m, q, exps, arcs, groups)} for m in models}
    keys = set.intersection(*[set(s) for s in sets.values()]) if sets else set()
    if not keys:
        return {}
    exps_k = sorted({k[0] for k in keys})
    boot_stats = {}
    for m in models:
        pl = [sets[m][k] for k in sorted(keys)]
        pt = measures([p[2] for p in pl], [p[3] for p in pl], floor)
        rng = np.random.default_rng(seed)       # same resamples for every model → paired
        bs, _ = boot(pl, floor, rng)
        boot_stats[m] = bs
        res[m] = {"point": pt, "ci": {k: ci([b[k] for b in bs]) for k in ("FB", "NMSE", "MG", "VG", "FAC2", "NAD", "R")}}
    ref = models[0]
    for m in models[1:]:
        d = {}
        for k in ("FB", "NMSE", "MG", "VG", "FAC2", "NAD"):
            if k in ("FB",):
                diffs = [abs(a[k]) - abs(b[k]) for a, b in zip(boot_stats[ref], boot_stats[m])]
                name = "|FB|"
            elif k in ("MG",):
                diffs = [abs(np.log(a[k])) - abs(np.log(b[k])) for a, b in zip(boot_stats[ref], boot_stats[m])]
                name = "|ln MG|"
            elif k == "FAC2":
                diffs = [a[k] - b[k] for a, b in zip(boot_stats[ref], boot_stats[m])]
                name = "FAC2"
            else:
                diffs = [a[k] - b[k] for a, b in zip(boot_stats[ref], boot_stats[m])]
                name = k
            c = ci(diffs)
            sig = c[0] is not None and (c[0] > 0 or c[1] < 0)
            d[name] = {"ci": c, "significant": bool(sig)}
        res[m]["diff_vs_" + ref] = d
    res["_n_pairs"] = len(keys); res["_n_runs"] = len(exps_k)
    return res


def tune_gaden(table, runs, noises, exps):
    """filamentNoise_std minimising mean (ln Co − ln Cp)² of arc-max C/Q on exps."""
    scores = {}
    for n in noises:
        pl = pairs(table, runs, f"G-ourwind-n{n}", "cmax_Q", exps)
        if not pl:
            continue
        co = np.log(np.maximum([p[2] for p in pl], FLOOR["cmax_Q"]))
        cp = np.log(np.maximum([p[3] for p in pl], FLOOR["cmax_Q"]))
        scores[n] = float(np.mean((co - cp) ** 2))
    best = min(scores, key=scores.get)
    return best, scores


def d6589(table, runs, models, seed=1):
    """Regime-averaged near-centreline C/Q, bootstrap over runs within regime."""
    rng = np.random.default_rng(seed)
    out = {}
    for gname, gl in GROUPS.items():
        for a in (50, 100, 200):
            key = f"{gname}@{a}"
            exps = [e for e in runs if table[e]["pg"] in gl and a in table[e]["arcs"]]
            if len(exps) < 3:
                continue
            obs = np.array([table[e]["arcs"][a]["nearcl_Q"] for e in exps])
            row = {"n_runs": len(exps), "obs_mean": float(obs.mean())}
            idx = [rng.integers(0, len(exps), len(exps)) for _ in range(N_BOOT)]
            row["obs_ci"] = ci([obs[i].mean() for i in idx])
            for m in models:
                pv = np.array([model_value(runs[e], m, a, "nearcl_Q") or 0.0 for e in exps])
                ratio = [pv[i].mean() / obs[i].mean() for i in idx]
                row[m] = {"mean": float(pv.mean()), "ratio": float(pv.mean() / obs.mean()), "ratio_ci": ci(ratio)}
            out[key] = row
    return out


def main():
    table, dropped, runs = load_runs()
    # observed near-centreline values (same estimator as the models)
    import t2_run as tr
    raw = pg.read_arcs(); corr = pg.read_corrections()
    for e, ob in table.items():
        for a, st in ob["arcs"].items():
            deg, c = raw[e]["arcs"][a]
            f = corr.get(e, {}).get(a)
            cc = (c / f if f else c) * 1e-3 / ob["Q_gs"]
            s2 = pg.arc_stats(deg, cc, float(a))
            st["nearcl_Q"] = tr.near_centreline(cc, deg, a, s2)

    exps_all = sorted(runs)
    odd = [e for e in exps_all if e % 2 == 1]
    even = [e for e in exps_all if e % 2 == 0]
    best, scores = tune_gaden(table, runs, tr.NOISES + tr.NOISES_EXTRA, odd)
    tuned = f"G-ourwind-n{best}"
    models = ["ours", "G-default-steady", "G-ourwind-n0.01", tuned]
    labels = {"ours": "ours (as shipped)", "G-default-steady": "GADEN example params, steady wind",
              "G-ourwind-n0.01": "GADEN example params, our wind", tuned: f"GADEN tuned (noise {best}), our wind"}

    res = {"n_runs": len(runs), "runs": exps_all, "dropped": {str(k): v for k, v in dropped.items()},
           "classes": {e: table[e]["pg"] for e in exps_all},
           "tuning": {"train_runs": odd, "test_runs": even, "scores_train_lnmse": scores, "chosen_noise_std": best},
           "models": labels, "floor": FLOOR, "n_boot": N_BOOT, "stats": {}, "stats_test_only": {}, "by_group": {},
           "d6589": None}
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        res["stats"][q] = evaluate(table, runs, models, q)
        res["stats_test_only"][q] = evaluate(table, runs, models, q, exps=even)
        res["by_group"][q] = {g: evaluate(table, runs, ["ours", tuned], q, groups=gl) for g, gl in GROUPS.items()}
        res["by_arc_" + q] = {str(a): evaluate(table, runs, ["ours", tuned], q, arcs=(a,)) for a in (50, 100, 200)}
    res["d6589"] = d6589(table, runs, ["ours", tuned])
    # our seed-to-seed noise floor: stats of seed 2 and 3 against seed 1 as "observations"
    nf = {}
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        co, cp = [], []
        for e, r in runs.items():
            for a in (50, 100, 200):
                v1 = r["models"]["ours_s1"]["arcs"][str(a)][q]
                for s in (2, 3):
                    v = r["models"][f"ours_s{s}"]["arcs"][str(a)][q]
                    if v1 is not None and v is not None:
                        co.append(v1); cp.append(v)
        nf[q] = measures(co, cp, FLOOR[q])
    res["noise_floor_ours_seed_vs_seed"] = nf
    out = os.path.join(HERE, "results", "t2.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print_summary(res)
    print("wrote", out)


def print_summary(res):
    print(f"runs {res['n_runs']}; GADEN tuned noise = {res['tuning']['chosen_noise_std']} "
          f"(train ln-MSE {res['tuning']['scores_train_lnmse']})")
    for scope in ("stats", "stats_test_only"):
        print(f"== {scope}")
        for q, r in res[scope].items():
            print(f"  {q}  ({r.get('_n_pairs')} pairs, {r.get('_n_runs')} runs)")
            for m, v in r.items():
                if m.startswith("_"):
                    continue
                p, c = v["point"], v["ci"]
                print(f"    {m:22s} FB {p['FB']:+.2f} [{c['FB'][0]:+.2f},{c['FB'][1]:+.2f}]  NMSE {p['NMSE']:.2f}  "
                      f"MG {p['MG']:.2f} [{c['MG'][0]:.2f},{c['MG'][1]:.2f}]  VG {p['VG']:.2f}  "
                      f"FAC2 {p['FAC2']:.2f} [{c['FAC2'][0]:.2f},{c['FAC2'][1]:.2f}]  NAD {p['NAD']:.2f}  R {p['R']:.2f}")


if __name__ == "__main__":
    main()
