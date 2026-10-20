"""
select_v11.py — pre-registered selection of the v1.1 configuration.

Selection data: odd-numbered Prairie Grass runs (T2) and the low-wind desert
run (T3a).  Held out: even runs and the high-wind desert run.

    E2 = mean over q ∈ {arc-max C/Q, CWIC/Q, σ_y} of (|ln MG_q| + ln VG_q)   (odd runs, seed 1)
    E3 = T3a matched-intermittency whiff-timing score (low wind)
    choose argmin E2 + E3
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CANDIDATES = ["sl_meas:sl", "sl_sg:sl_sg", "sl_s03k01:sl_s03k01", "sl_sg_s03k01:sl_sg_s03k01"]


def t2_error(tag, which):
    subprocess.run([PY, os.path.join(HERE, "t2_variant.py"), "--tag", tag, "--stats", "--exps", which, "--seeds", "1"],
                   check=True, capture_output=True)
    r = json.load(open(os.path.join(HERE, "data", "t2_variants", tag, f"stats_{which}.json")))
    errs, base = [], []
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        m = r[q][tag]; b = r[q]["v1.0"]
        errs.append(abs(np.log(m["MG"])) + np.log(m["VG"]))
        base.append(abs(np.log(b["MG"])) + np.log(b["VG"]))
    return float(np.mean(errs)), float(np.mean(base)), r


def main():
    sys.path.insert(0, HERE)
    import t3_fluct as t3
    D = {s: t3.data_series(s) for s in ("desert_low_wind", "desert_high_wind")}
    rows = {}
    for c in CANDIDATES:
        t2tag, t3tag = c.split(":")
        e2, e2_base, _ = t2_error(t2tag, "odd")
        e2_even, e2_even_base, rev = t2_error(t2tag, "even")
        row = {"E2_odd": e2, "E2_odd_v1.0": e2_base, "E2_even": e2_even, "E2_even_v1.0": e2_even_base,
               "even_stats": {q: rev[q][t2tag] for q in ("cmax_Q", "cwic_Q", "sigma_y")}}
        for s in D:
            f = os.path.join(HERE, "data", "t3_variants", t3tag, f"{s}.npz")
            if os.path.exists(f):
                r3 = t3.analyse_matched(D[s], dict(np.load(f)))
                row[f"E3_{s}"] = t3.score_matched(r3)
        row["select_score"] = row["E2_odd"] + row.get("E3_desert_low_wind", np.inf)
        rows[c] = row
        print(f"{c:28s} E2 odd {e2:.3f} (v1.0 {e2_base:.3f})  E3 low {row.get('E3_desert_low_wind', np.nan):.3f}  "
              f"→ select {row['select_score']:.3f}   | held out: E2 even {e2_even:.3f} (v1.0 {e2_even_base:.3f}), "
              f"E3 high {row.get('E3_desert_high_wind', np.nan):.3f}", flush=True)
    best = min(rows, key=lambda k: rows[k]["select_score"])
    print("SELECTED:", best)
    json.dump({"rule": __doc__, "rows": rows, "selected": best},
              open(os.path.join(HERE, "results", "v11_selection.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
