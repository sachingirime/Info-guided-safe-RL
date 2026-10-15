"""
a4_gamma.py — A4: sensitivity of V3 (class D) and V4 to γ in T_E = T_L·i/γ.

    python validation/a4_gamma.py --gamma 0.6          # one run → validation/a4/gamma_0.6.json
    python validation/a4_gamma.py --analyse            # → validation/a4_gamma.json
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "a4")

def stats(series, rows):
    import validate_sim as V
    sy50 = next(r["sigma_y_sim"] for r in rows if r["x_minus_xs"] == 50.0)
    j = int(np.argmin(abs(np.array(V.V4_OFF_DY) - sy50)))
    cols = list(range(len(V.V4_X))) + [len(V.V4_X) + j]
    names = [f"centreline {d:.0f} m" for d in V.V4_X] + [f"off-axis 50 m, dy={V.V4_OFF_DY[j]:.0f} m"]
    out = []
    for n, c in zip(names, cols):
        ts = series[:, c].astype(float); m = ts.mean()
        out.append({"point": n, "gamma_int": float(np.mean(ts > 0.1 * m)), "p99_over_mean": float(np.percentile(ts, 99) / m),
                    "cov": float(ts.std() / m)})
    return out

def run(g, avg, seed):
    import warp as wp; wp.config.quiet = True
    import validate_sim as V
    os.makedirs(OUT, exist_ok=True)
    d, series, sim = V.v3_run("D", avg=avg, seed=seed, record_series=True, sim_kwargs=dict(hp_gamma=g))
    res = {"gamma": g, "seed": seed, "avg_s": avg, "U_src_mean": d["setup"]["U_src_mean"], "lsm_TE": list(sim.lsm_TE),
           "rows": [{k: r[k] for k in ("x_minus_xs", "ratio_y", "ratio_z", "sigma_y_sim", "sigma_z_sim")} for r in d["rows"]],
           "pass": d["pass"], "v4": stats(series, d["rows"])}
    json.dump(res, open(os.path.join(OUT, f"gamma_{g}_seed{seed}.json"), "w"), indent=1, default=float)
    print("done", g, seed, flush=True)

def analyse():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "gamma_*.json")))]
    json.dump({"runs": runs}, open(os.path.join(HERE, "a4_gamma.json"), "w"), indent=1, default=float)
    for r in sorted(runs, key=lambda r: (r["gamma"], r["seed"])):
        print(f"γ={r['gamma']} seed={r['seed']} U_src={r['U_src_mean']:.2f} T_E={np.round(r['lsm_TE'][1:],1)} pass={r['pass']}")
        print("   ratio_y", [round(x["ratio_y"], 2) for x in r["rows"]], " ratio_z", [round(x["ratio_z"], 2) for x in r["rows"]])
        print("   V4 γ_int", [round(s["gamma_int"], 2) for s in r["v4"]], " p99/mean", [round(s["p99_over_mean"], 1) for s in r["v4"]],
              " CoV", [round(s["cov"], 2) for s in r["v4"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gamma", type=float); ap.add_argument("--avg", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--analyse", action="store_true"); a = ap.parse_args()
    analyse() if a.analyse else run(a.gamma, a.avg, a.seed)
