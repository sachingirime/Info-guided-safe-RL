"""
t2_variant.py — Prairie Grass runs for a variant of OUR model (no GADEN), for
screening simulator changes against the v1.0 baseline.

    python bench/t2_variant.py --tag subgrid --kwargs '{"subgrid_wind": true}' \
        [--seeds 1] [--lsm-table tl.json] [--sigma0 1.0] [--emit 0.2] [--exps odd|even|all]
    python bench/t2_variant.py --tag subgrid --stats [--exps even]

Runs land in bench/data/t2_variants/<tag>/run_XX.json (checkpointed); --stats
prints the BOOT measures of the variant next to the stored v1.0 baseline
(ours, same seeds) over the same run–arc pairs.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import prairie_grass as pg               # noqa: E402
import t2_run as tr                      # noqa: E402
import t2_stats as ts                    # noqa: E402

OUT = os.path.join(HERE, "data", "t2_variants")


def run_one(run, seed, kwargs, sigma0, emit, max_puffs, met=""):
    import stable_fluid_3d as sf
    kwargs = dict(kwargs)
    if met == "measured":      # run's own u*, 1/L (profile fit), site z0
        kwargs.update(met={"ustar": float(run["ustar"]), "inv_L": float(run["inv_L"]), "zi": 1000.0}, z0=pg.Z0)
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=0.0, wind_angle_range=20.0,
                                       wind_base_speed=float(run["U10"]), wind_seed=seed, seed=seed,
                                       max_puffs=max_puffs, **kwargs)
    sim.Q = 1.0; sim.emit_interval = emit; sim.sigma_0 = sigma0
    sim.set_stability_class(run["pg"])
    sim.source_pos = np.array([*tr.SRC_XY, run["hs"]], np.float32)
    sim.set_obstacles([], [])
    pts, degs = [], []
    for a in tr.ARCS:
        p, d = tr.arc_points(a, run["zr"]); pts.append(p); degs.append(d)
    P = np.concatenate(pts).astype(np.float32)
    splits = np.cumsum([len(p) for p in pts])[:-1]
    t_spin = tr.spin_time(run)
    n_frames = int(round((t_spin + tr.AVG_S) * sim.fps)); k_s = int(round(tr.SAMPLE_S * sim.fps))
    acc = np.zeros(len(P)); n = 0
    for f in range(n_frames):
        sim.step()
        if (f + 1) / sim.fps <= t_spin + 1e-9 or (f + 1) % k_s:
            continue
        acc += sim.query_concentration(P) / sim.Q; n += 1
    mean = acc / n
    return {"arcs": {str(a): tr.arc_summary(c, d, a) for a, c, d in zip(tr.ARCS, np.split(mean, splits), degs)},
            "n_overwritten": int(sim.n_overwritten)}


def select(exps, which):
    return {"odd": [e for e in exps if e % 2], "even": [e for e in exps if e % 2 == 0]}.get(which, exps)


def stats(tag, which, seeds):
    table, _, base = ts.load_runs()
    raw = pg.read_arcs(); corr = pg.read_corrections()
    for e, ob in table.items():
        for a, st in ob["arcs"].items():
            deg, c = raw[e]["arcs"][a]; f = corr.get(e, {}).get(a)
            cc = (c / f if f else c) * 1e-3 / ob["Q_gs"]
            st["nearcl_Q"] = tr.near_centreline(cc, deg, a, pg.arc_stats(deg, cc, float(a)))
    var = {}
    for f in glob.glob(os.path.join(OUT, tag, "run_*.json")):
        r = json.load(open(f)); var[r["exp"]] = r
    exps = select(sorted(set(var) & set(base)), which)
    # assemble both as "runs" with model keys base_sX / var_sX
    merged = {}
    for e in exps:
        m = {}
        for s in seeds:
            m[f"base_s{s}"] = base[e]["models"][f"ours_s{s}"]
            m[f"var_s{s}"] = var[e]["models"][f"s{s}"]
        merged[e] = {"models": m}

    def ens(prefix):
        out = {}
        for e, r in merged.items():
            out[e] = {"models": {"m": {"arcs": {}}}}
            for a in ("50", "100", "200"):
                d = {}
                for q in ("cmax_Q", "cwic_Q", "sigma_y", "nearcl_Q"):
                    v = [r["models"][f"{prefix}_s{s}"]["arcs"][a][q] for s in seeds]
                    v = [x for x in v if x is not None]
                    d[q] = float(np.mean(v)) if v else None
                out[e]["models"]["m"]["arcs"][a] = d
        return out
    res = {}
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        row = {}
        for name, prefix in (("v1.0", "base"), (tag, "var")):
            R = ens(prefix)
            pl = ts.pairs(table, R, "m", q, exps)
            row[name] = ts.measures([p[2] for p in pl], [p[3] for p in pl], ts.FLOOR[q])
        res[q] = row
        by = {}
        for g, gl in ts.GROUPS.items():
            by[g] = {}
            for name, prefix in (("v1.0", "base"), (tag, "var")):
                pl = ts.pairs(table, ens(prefix), "m", q, exps, groups=gl)
                by[g][name] = ts.measures([p[2] for p in pl], [p[3] for p in pl], ts.FLOOR[q]) if pl else None
        res[q + "_by_group"] = by
    print(f"== {tag} vs v1.0, runs {which} ({len(exps)}), seeds {seeds}")
    for q in ("cmax_Q", "cwic_Q", "sigma_y"):
        for name, m in res[q].items():
            print(f"  {q:7s} {name:10s} FB {m['FB']:+.2f}  NMSE {m['NMSE']:.2f}  MG {m['MG']:.2f}  VG {m['VG']:.2f}  FAC2 {m['FAC2']:.2f}  NAD {m['NAD']:.2f}  R {m['R']:.2f}")
        for g, d in res[q + "_by_group"].items():
            print(f"      {g:4s} " + "  ".join(f"{n}: MG {v['MG']:.2f} FAC2 {v['FAC2']:.2f}" for n, v in d.items() if v))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--lsm-table", default="")
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--emit", type=float, default=0.2)
    ap.add_argument("--max-puffs", type=int, default=4096)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--exps", default="all")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--met", default="", help="'measured': per-run u*, L from the profile fit (surface_layer)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.stats:
        res = stats(a.tag, a.exps, seeds)
        json.dump(res, open(os.path.join(OUT, a.tag, f"stats_{a.exps}.json"), "w"), indent=1, default=float)
        return
    import warp as wp
    wp.config.quiet = True
    kwargs = json.loads(a.kwargs)
    if a.lsm_table:
        tab = json.load(open(a.lsm_table))
        kwargs["lsm_table"] = {c: {k: v[k] for k in ("a_y", "a_z", "TL_y", "TL_z")} for c, v in tab["classes"].items()}
    out = os.path.join(OUT, a.tag); os.makedirs(out, exist_ok=True)
    json.dump({"kwargs": a.kwargs, "met": a.met, "lsm_table": a.lsm_table, "sigma0": a.sigma0, "emit": a.emit,
               "max_puffs": a.max_puffs}, open(os.path.join(out, "config.json"), "w"))
    table, _ = pg.load()
    exps = select(sorted(table), a.exps)[a.worker::a.n_workers]
    for e in exps:
        path = os.path.join(out, f"run_{e:02d}.json")
        r = json.load(open(path)) if os.path.exists(path) else {"exp": e, "models": {}}
        todo = [s for s in seeds if f"s{s}" not in r["models"]]
        if not todo:
            continue
        t0 = time.time()
        for s in todo:
            r["models"][f"s{s}"] = run_one(table[e], s, kwargs, a.sigma0, a.emit, a.max_puffs, a.met)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(r, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"[{a.tag} w{a.worker}] exp {e} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
