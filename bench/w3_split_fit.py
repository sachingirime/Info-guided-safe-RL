"""
w3_split_fit.py — split-sample calibration on Prairie Grass (independence check).

Fits the lateral LSM parameters per stability class to the observed arc σ_y of
the ODD-numbered Prairie Grass runs only (the even runs are never seen), using
the same Taylor model as validation/fit_tl.py:

    σ_y²(x) = σ₀² + 2 K_rel t + 2 σ_v² T_L² (t/T_L − 1 + e^{−t/T_L}),  t = x/U, σ_v = a_y U

(U cancels: T_L ∝ 1/U makes σ_y a function of x only).  Vertical parameters
(a_z, T_L,z) stay at the Briggs fit: the arcs give no σ_z.  The 200 m arc is
used only where 3σ_y,obs < 60 m (the pre-registered domain rule of T2).

    python bench/w3_split_fit.py --sigma0 1.0 --k-rel 0.05 --briggs bench/data/tl/x.json \
        --out bench/data/tl/pg_odd_s1_k0.05.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import prairie_grass as pg       # noqa: E402
import fit_tl                    # noqa: E402

SIGMA_Y_MAX_200 = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--k-rel", type=float, default=0.05)
    ap.add_argument("--briggs", default="", help="fit_tl output whose a_z/TL_z are kept (default: refit)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table, _ = pg.load()
    briggs = json.load(open(a.briggs))["classes"] if a.briggs else fit_tl.fit(a.sigma0, a.k_rel)
    odd = [e for e in table if e % 2 == 1]
    out = {"sigma0": a.sigma0, "K_rel": a.k_rel, "train_runs": sorted(odd), "classes": {}}
    for c in "ABCDEF":
        xs, sy = [], []
        for e in odd:
            if table[e]["pg"] != c:
                continue
            for arc, st in table[e]["arcs"].items():
                if arc == 200 and st["sigma_y"] > SIGMA_Y_MAX_200:
                    continue
                xs.append(float(arc)); sy.append(st["sigma_y"])
        xs, sy = np.array(xs), np.array(sy)
        b = briggs[c]
        row = {"a_z": b["a_z"], "TL_z": b["TL_z"], "n_points": int(len(xs)),
               "briggs_a_y": b["a_y"], "briggs_TL_y": b["TL_y"]}
        if len(xs) < 3:
            row.update({"a_y": b["a_y"], "TL_y": b["TL_y"], "note": "too few odd-run points: Briggs kept"})
        else:
            def f(p):
                ay, ltl = np.exp(p[0]), p[1]
                m = fit_tl.model_sigma(xs, ay, np.exp(ltl), a.sigma0, a.k_rel)
                return np.sum(np.log(m / sy) ** 2)
            r = minimize(f, [np.log(b["a_y"]), np.log(b["TL_y"])], method="L-BFGS-B",
                         bounds=[(np.log(0.01), np.log(1.0)), (np.log(5.0), np.log(2000.0))])
            ay, TL = float(np.exp(r.x[0])), float(np.exp(r.x[1]))
            pred = fit_tl.model_sigma(xs, ay, TL, a.sigma0, a.k_rel)
            pb = fit_tl.model_sigma(xs, b["a_y"], b["TL_y"], a.sigma0, a.k_rel)
            row.update({"a_y": round(ay, 4), "TL_y": round(TL, 1),
                        "rms_log_ratio_fit": float(np.sqrt(np.mean(np.log(pred / sy) ** 2))),
                        "rms_log_ratio_briggs": float(np.sqrt(np.mean(np.log(pb / sy) ** 2)))})
        out["classes"][c] = row
        print(f"{c}: n={row['n_points']:2d}  a_y {row['briggs_a_y']:.3f}→{row['a_y']:.3f}  TL_y {row['briggs_TL_y']:7.1f}→{row['TL_y']:7.1f}"
              + (f"  rms ln σ_y: Briggs {row['rms_log_ratio_briggs']:.2f} → fit {row['rms_log_ratio_fit']:.2f}" if "rms_log_ratio_fit" in row else f"  ({row.get('note')})"))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
