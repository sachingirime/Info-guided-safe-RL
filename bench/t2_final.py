"""
t2_final.py — Tier 2 statistics for v1.1 against v1.0 and GADEN (same runs, arcs, CIs).

v1.1 = bench/data/t2_variants/<tag> (seeds 1–3, measured met per run);
v1.0 and GADEN variants from bench/data/t2/run_*.json (GADEN was driven by the
resolved wind, which v1.1 does not change).  Reports all runs, the odd (used for
selection) and even (held-out) halves, stability groups and the D6589 test.

    python bench/t2_final.py --tag sl_sg_s03k01
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prairie_grass as pg     # noqa: E402
import t2_run as tr            # noqa: E402
import t2_stats as ts          # noqa: E402


def ensemble(models, prefix, seeds):
    out = {"arcs": {}}
    for a in ("50", "100", "200"):
        d = {}
        for q in ("cmax_Q", "cwic_Q", "sigma_y", "nearcl_Q"):
            v = [models[f"{prefix}{s}"]["arcs"][a][q] for s in seeds if f"{prefix}{s}" in models]
            v = [x for x in v if x is not None]
            d[q] = float(np.mean(v)) if v else None
        out["arcs"][a] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sl_sg_s03k01")
    a = ap.parse_args()
    table, dropped, base = ts.load_runs()
    raw = pg.read_arcs(); corr = pg.read_corrections()
    for e, ob in table.items():
        for arc, st in ob["arcs"].items():
            deg, c = raw[e]["arcs"][arc]; f = corr.get(e, {}).get(arc)
            cc = (c / f if f else c) * 1e-3 / ob["Q_gs"]
            st["nearcl_Q"] = tr.near_centreline(cc, deg, arc, pg.arc_stats(deg, cc, float(arc)))
    runs = {}
    for e, r in base.items():
        f = os.path.join(HERE, "data", "t2_variants", a.tag, f"run_{e:02d}.json")
        v = json.load(open(f))["models"]
        m = dict(r["models"])
        m["v1.1"] = ensemble(v, "s", (1, 2, 3))
        m["v1.0"] = ensemble(r["models"], "ours_s", (1, 2, 3))
        m["v1.1_seeds"] = sorted(v)
        runs[e] = {"models": m}
    seeds_present = sorted({s for r in runs.values() for s in r["models"]["v1.1_seeds"]})
    tuned = "G-ourwind-n0.7"
    models = ["v1.1", "v1.0", tuned, "G-ourwind-n0.01", "G-default-steady"]
    exps = sorted(runs)
    res = {"tag": a.tag, "v1.1_seeds": seeds_present, "n_runs": len(exps), "models": models, "stats": {}}
    for scope, sel in (("all", exps), ("odd_selection", [e for e in exps if e % 2]), ("even_heldout", [e for e in exps if e % 2 == 0])):
        res["stats"][scope] = {q: ts.evaluate(table, runs, models, q, exps=sel) for q in ("cmax_Q", "cwic_Q", "sigma_y")}
    res["by_group"] = {q: {g: ts.evaluate(table, runs, ["v1.1", "v1.0", tuned], q, groups=gl) for g, gl in ts.GROUPS.items()}
                       for q in ("cmax_Q", "cwic_Q", "sigma_y")}
    res["d6589"] = ts.d6589(table, runs, ["v1.1", "v1.0", tuned])
    # v1.1 seed-to-seed noise floor
    nf = {}
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        co, cp = [], []
        for e in exps:
            vv = json.load(open(os.path.join(HERE, "data", "t2_variants", a.tag, f"run_{e:02d}.json")))["models"]
            for arc in ("50", "100", "200"):
                x1 = vv["s1"]["arcs"][arc][q]
                for s in ("s2", "s3"):
                    if s in vv and x1 is not None and vv[s]["arcs"][arc][q] is not None:
                        co.append(x1); cp.append(vv[s]["arcs"][arc][q])
        nf[q] = ts.measures(co, cp, ts.FLOOR[q]) if co else None
    res["noise_floor_v1.1_seed_vs_seed"] = nf
    out = os.path.join(HERE, "results", "t2_v11.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    for scope in res["stats"]:
        print(f"== {scope}")
        for q, r in res["stats"][scope].items():
            print(f"  {q} ({r['_n_pairs']} pairs, {r['_n_runs']} runs)")
            for m in models:
                p, c = r[m]["point"], r[m]["ci"]
                d = r[m].get("diff_vs_v1.1", {})
                sig = " ".join(k for k, x in d.items() if x["significant"])
                print(f"    {m:18s} FB {p['FB']:+.2f} [{c['FB'][0]:+.2f},{c['FB'][1]:+.2f}] NMSE {p['NMSE']:6.2f} MG {p['MG']:6.2f} "
                      f"VG {ts_fmt(p['VG'])} FAC2 {p['FAC2']:.2f} [{c['FAC2'][0]:.2f},{c['FAC2'][1]:.2f}] NAD {p['NAD']:.2f} R {p['R']:.2f}"
                      + (f"  sig≠v1.1: {sig}" if sig else ""))
    print("wrote", out)


def ts_fmt(v):
    return f"{v:6.2f}" if v < 1e3 else f"{v:6.0e}"


if __name__ == "__main__":
    main()
