"""
a6_classA.py — A6: class A spot check with ground-image-aware σ_z fits,
fitted on the mean field of several seeds (class A T_E is 250–500 s, so one
3600 s average is not converged).

    python validation/a6_classA.py --seed 1            # one run → validation/a6/A_seed1.npz
    python validation/a6_classA.py --analyse           # → validation/a6_classA.json
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "a6")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--cls", default="A")
    ap.add_argument("--avg", type=float, default=3600.0); ap.add_argument("--analyse", action="store_true"); a = ap.parse_args()
    import validate_sim as V
    if not a.analyse:
        import warp as wp; wp.config.quiet = True
        os.makedirs(OUT, exist_ok=True)
        d, _, sim, F = V.v3_run(a.cls, avg=a.avg, seed=a.seed, return_fields=True)
        np.savez(os.path.join(OUT, f"{a.cls}_seed{a.seed}.npz"), fields=F, U10=d["setup"]["U10_realised"])
        print("done", a.cls, a.seed, flush=True)
    else:
        res = {}
        for cls in sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(OUT, "*_seed*.npz"))}):
            files = sorted(glob.glob(os.path.join(OUT, f"{cls}_seed*.npz")))
            data = [np.load(f) for f in files]
            ys, zs = V.V1_YS, np.arange(0.0, 80.0 + 0.25, 0.5)
            per = [V.v3_rows(dd["fields"], ys, zs, cls) for dd in data]
            ens = V.v3_rows(np.mean([dd["fields"] for dd in data], 0), ys, zs, cls)
            ok = all(V.V3_BAND[0] <= r["ratio_y"] <= V.V3_BAND[1] and V.V3_BAND[0] <= r["ratio_z"] <= V.V3_BAND[1] for r in ens)
            res[cls] = {"n_seeds": len(data), "U10_realised": [float(dd["U10"]) for dd in data], "ensemble_rows": ens,
                        "per_seed_ratio_z": [[round(r["ratio_z"], 3) for r in p] for p in per],
                        "per_seed_ratio_y": [[round(r["ratio_y"], 3) for r in p] for p in per], "ensemble_pass": ok}
            print(f"class {cls}: {len(data)} seeds, U10 {np.round(res[cls]['U10_realised'],2)}  ensemble pass={ok}")
            for r in ens:
                print(f"   x={r['x_minus_xs']:5.0f}: σy ×{r['ratio_y']:.2f}  σz(h fixed) ×{r['ratio_z']:.2f}  σz(h free) ×{r['ratio_z_free_h']:.2f} (h={r['h_fit']:.1f} m)")
            print("   per-seed σz ratios:", res[cls]["per_seed_ratio_z"])
        json.dump(res, open(os.path.join(HERE, "a6_classA.json"), "w"), indent=1, default=float)
