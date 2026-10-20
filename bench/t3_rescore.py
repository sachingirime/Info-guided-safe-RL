"""t3_rescore.py — T3a (matched-intermittency whiff timing) and T3b (wind at the
sensor) for every stored T3 series (v1.0 benchmark + variants)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t3_fluct as t3

out = {}
D = {s: t3.data_series(s) for s in ("desert_low_wind", "desert_high_wind")}
for s, d in D.items():
    out.setdefault("data_wind", {})[s] = t3.wind_stats(d["u"], d["v"])
srcs = {f"v1.0-bench": {s: os.path.join(HERE, "data", "t3", f"{s}_series.npz") for s in D}}
for tagdir in sorted(glob.glob(os.path.join(HERE, "data", "t3_variants", "*"))):
    tag = os.path.basename(tagdir).replace("_hw", "")
    for s in D:
        f = os.path.join(tagdir, f"{s}.npz")
        if os.path.exists(f):
            srcs.setdefault(tag, {})[s] = f
for tag, files in srcs.items():
    for s, f in files.items():
        ser = dict(np.load(f))
        r = t3.analyse_matched(D[s], ser)
        row = {"score_matched": {m: t3.score_matched(r, m) for m in r["models"]}, "matched": r}
        key = "anemo_s1" if "anemo_s1" in ser else "wind_s1"
        if key in ser:
            row["wind"] = t3.wind_stats(ser[key][:, 0], ser[key][:, 1]); row["wind_source"] = key
        out.setdefault(tag, {})[s] = row
        sm = row["score_matched"]
        print(f"{tag:12s} {s:17s} " + "  ".join(f"{m}: {v:.2f}" for m, v in sm.items())
              + (f"  | wind σθ {row['wind']['sigma_theta_deg']:.1f}° TI {row['wind']['turb_intensity']:.2f} T {row['wind']['T_dir_s']:.0f}s" if "wind" in row else ""), flush=True)
for s, w in out["data_wind"].items():
    print(f"DATA wind {s}: σθ {w['sigma_theta_deg']:.1f}° TI {w['turb_intensity']:.2f} T {w['T_dir_s']:.0f}s")
json.dump(out, open(os.path.join(HERE, "data", "t3_rescore.json"), "w"), indent=1, default=float)
