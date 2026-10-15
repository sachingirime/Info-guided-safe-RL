"""
a1_v1_seeds.py — A1: is the V1 σ_y offset with shared-eddy release sampling noise or bias?

Runs the V1 configuration (steady wind, class D, K_rel=0.05, shared eddies on,
150 s spin-up, 7200 s average) for the given seeds and saves each mean field
to validation/a1/seed_XX.npz.  `--analyse` computes the ensemble error vs N.

    python validation/a1_v1_seeds.py --seeds 1 2 3 4
    python validation/a1_v1_seeds.py --analyse
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "a1")

def run(seeds, avg):
    import warp as wp; wp.config.quiet = True
    import validate_sim as V
    os.makedirs(OUT, exist_ok=True)
    for sd in seeds:
        f, sim = V.v1_field(0.05, avg=avg, seed=sd, shared_eddies=True)
        np.savez(os.path.join(OUT, f"seed_{sd:02d}.npz"), fields=f, avg=avg,
                 lsm_sigma=np.array(sim.lsm_sigma), lsm_TL=np.array(sim.lsm_TL), sigma_0=sim.sigma_0, k_rel=sim.k_rel)
        print("done seed", sd, flush=True)

def errors(F, d_list, meta):
    import validate_sim as V
    class S: pass
    s = S(); s.lsm_sigma = tuple(meta["lsm_sigma"]); s.lsm_TL = tuple(meta["lsm_TL"]); s.sigma_0 = float(meta["sigma_0"]); s.k_rel = float(meta["k_rel"])
    ys, zs = V.V1_YS, V.V1_ZS
    jy = int(np.argmin(abs(ys - V.V1_SRC[1]))); kz = int(np.argmin(abs(zs - V.V1_SRC[2])))
    out = []
    for d, C in zip(d_list, F):
        sy, sz = V.lsm_sigma_analytic(s, d, V.V1_U)
        A = V.slender_plume(1.0, V.V1_U, ys, zs, V.V1_SRC[1], V.V1_SRC[2], sy, sz)
        m = lambda G: (V.moment_sigma(G.sum(1), ys)[0], V.moment_sigma(G.sum(0), zs)[0])
        (a, b), (c, e) = m(C), m(A)
        out.append({"x": d, "sy": a / c - 1, "sz": b / e - 1, "ccl": C[jy, kz] / A[jy, kz] - 1, "flux": V.V1_U * C.sum() - 1})
    return out

def analyse():
    import validate_sim as V
    files = sorted(glob.glob(os.path.join(OUT, "seed_*.npz")))
    data = [np.load(f) for f in files]
    meta = {k: data[0][k] for k in ("lsm_sigma", "lsm_TL", "sigma_0", "k_rel")}
    per_seed = [errors(d["fields"], V.V1_X, meta) for d in data]
    res = {"n_seeds_available": len(data), "avg_s_per_seed": float(data[0]["avg"]), "per_seed": per_seed, "ensemble": {}}
    for N in (1, 2, 4, 8, 16):
        if N > len(data): continue
        F = np.mean([d["fields"] for d in data[:N]], axis=0)
        res["ensemble"][N] = errors(F, V.V1_X, meta)
    # single-seed scatter → expected ensemble SE ∝ 1/√N
    sd = {k: float(np.std([p[2][k] for p in per_seed], ddof=1)) for k in ("sy", "sz", "ccl")}
    mean = {k: float(np.mean([p[2][k] for p in per_seed])) for k in ("sy", "sz", "ccl")}
    res["x100_single_seed_std"] = sd; res["x100_mean_over_seeds"] = mean
    res["x100_standard_error_of_mean"] = {k: sd[k] / np.sqrt(len(data)) for k in sd}
    json.dump(res, open(os.path.join(HERE, "a1_v1_seeds.json"), "w"), indent=1, default=float)
    for N, rows in res["ensemble"].items():
        print(f"N={N:2d}: " + "  ".join(f"x={r['x']:.0f}: σy {100*r['sy']:+.1f}% σz {100*r['sz']:+.1f}% Ccl {100*r['ccl']:+.1f}%" for r in rows))
    print("x=100 m over seeds: mean", {k: round(100*v, 2) for k, v in mean.items()}, " single-seed std", {k: round(100*v, 2) for k, v in sd.items()},
          " SE of mean", {k: round(100*v, 2) for k, v in res["x100_standard_error_of_mean"].items()})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*"); ap.add_argument("--avg", type=float, default=7200.0)
    ap.add_argument("--analyse", action="store_true"); a = ap.parse_args()
    analyse() if a.analyse else run(a.seeds, a.avg)
