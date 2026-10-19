"""
t4_stats.py — Tier 4 analysis: cross-simulator transfer of source-seeking agents.

For each agent family (cast-surge, infotaxis) and tuning world A, the variant
with the highest success rate on the tuning episodes (0–39) in A is chosen
(ties: lower mean time-to-source, failures counted as T_EP).  That variant is
evaluated on the test episodes (40–119) in every world B.

Per cell: success rate with a Wilson 95% interval, median time to source
(successes), median time to first detection, mean final distance, mean path
length.  Transfer gap: success(B) − success(A) for the A-tuned variant, with a
paired bootstrap 95% CI over test episodes (same scenarios in both worlds).
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T_EP = 300.0
TUNE = range(0, 40)
TEST = range(40, 120)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(float(c - h), 0.0), min(float(c + h), 1.0)]


def load():
    eps = {}
    for f in glob.glob(os.path.join(HERE, "data", "t4", "ep_*.json")):
        r = json.load(open(f)); eps[r["ep"]] = r
    return eps


def table_for(eps, ids, world, variant):
    rows = [eps[e]["results"][world][variant] for e in ids if e in eps]
    succ = np.array([r["t_success"] is not None for r in rows])
    t = np.array([r["t_success"] if r["t_success"] is not None else T_EP for r in rows])
    return rows, succ, t


def main():
    eps = load()
    worlds = list(next(iter(eps.values()))["results"])
    variants = list(eps[next(iter(eps))]["results"][worlds[0]])
    fams = {"cast-surge": [v for v in variants if v.startswith("cs_")],
            "infotaxis": [v for v in variants if v.startswith("info_")]}
    tune = [e for e in TUNE if e in eps]; test = [e for e in TEST if e in eps]
    res = {"n_tune": len(tune), "n_test": len(test), "worlds": worlds, "families": {}}
    rng = np.random.default_rng(0)
    for fam, vs in fams.items():
        F = {"chosen": {}, "matrix": {}}
        for A in worlds:
            score = {v: (table_for(eps, tune, A, v)[1].mean(), -table_for(eps, tune, A, v)[2].mean()) for v in vs}
            best = max(score, key=score.get)
            F["chosen"][A] = {"variant": best, "tune_success": float(score[best][0])}
            _, sA, _ = table_for(eps, test, A, best)
            for B in worlds:
                rows, s, t = table_for(eps, test, B, best)
                det = [r["t_first_det"] for r in rows if r["t_first_det"] is not None]
                cell = {"variant": best, "n": int(len(s)), "success": float(s.mean()), "success_ci": wilson(int(s.sum()), len(s)),
                        "median_t_success": float(np.median(t[s])) if s.any() else None,
                        "median_t_first_det": float(np.median(det)) if det else None,
                        "frac_detected": float(len(det) / len(rows)),
                        "mean_final_dist": float(np.mean([r["final_dist"] for r in rows])),
                        "mean_path": float(np.mean([r["path"] for r in rows]))}
                if B != A:
                    idx = [rng.integers(0, len(s), len(s)) for _ in range(2000)]
                    gaps = [s[i].mean() - sA[i].mean() for i in idx]
                    lo, hi = np.percentile(gaps, [2.5, 97.5])
                    cell["gap_vs_tuning_world"] = {"value": float(s.mean() - sA.mean()), "ci": [float(lo), float(hi)],
                                                   "significant": bool(lo > 0 or hi < 0)}
                F["matrix"][f"{A}->{B}"] = cell
        # all-variant table on test episodes (oracle view)
        F["all_variants_test"] = {B: {v: float(table_for(eps, test, B, v)[1].mean()) for v in vs} for B in worlds}
        res["families"][fam] = F
    out = os.path.join(HERE, "results", "t4.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for fam, F in res["families"].items():
        print(f"== {fam} (tune {res['n_tune']}, test {res['n_test']})")
        for A in worlds:
            print(f"  tuned in {A:10s} → {F['chosen'][A]['variant']}")
            for B in worlds:
                c = F["matrix"][f"{A}->{B}"]
                g = c.get("gap_vs_tuning_world")
                gs = f"  gap {g['value']:+.2f} [{g['ci'][0]:+.2f},{g['ci'][1]:+.2f}]{' *' if g['significant'] else ''}" if g else ""
                print(f"     on {B:10s} success {c['success']:.2f} [{c['success_ci'][0]:.2f},{c['success_ci'][1]:.2f}] "
                      f"detected {c['frac_detected']:.2f}  final dist {c['mean_final_dist']:.0f} m{gs}")
    print("wrote", out)


if __name__ == "__main__":
    main()
