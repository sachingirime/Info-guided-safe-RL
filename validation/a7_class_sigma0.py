"""
a7_class_sigma0.py — A7: class-dependent release size (σ0 = 0.5 m for E,
0.3 m for F; T_L refitted, validation/tl_fit_class_sigma0.json) vs the default
σ0 = 1 m, V3 spot check for classes E and F (3600 s).

    python validation/a7_class_sigma0.py            # → validation/a7_class_sigma0.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import warp as wp; wp.config.quiet = True
import validate_sim as V

fit = json.load(open(os.path.join(HERE, "tl_fit_class_sigma0.json")))["classes"]
table = {c: {"TL_y": fit[c]["TL_y"], "TL_z": fit[c]["TL_z"]} for c in ("E", "F")}
res = {}
for cls in sys.argv[1:] or ["F", "E"]:
    for tag, kw in (("default_sigma0_1m", {}),
                    ("class_sigma0", dict(class_sigma0={"E": 0.5, "F": 0.3}, lsm_table=table))):
        d, _, sim = V.v3_run(cls, avg=3600.0, seed=1, sim_kwargs=kw)
        res[f"{cls}/{tag}"] = {"sigma_0": sim.sigma_0, "TL": list(sim.lsm_TL), "U10": d["setup"]["U10_realised"],
                               "ratio_y": [r["ratio_y"] for r in d["rows"]], "ratio_z": [r["ratio_z"] for r in d["rows"]],
                               "pass": d["pass"]}
        print(cls, tag, "σ0", sim.sigma_0, "ratio_y", np.round(res[f"{cls}/{tag}"]["ratio_y"], 2),
              "ratio_z", np.round(res[f"{cls}/{tag}"]["ratio_z"], 2), "pass", d["pass"], flush=True)
json.dump(res, open(os.path.join(HERE, f"a7_class_sigma0_{'_'.join(sys.argv[1:]) or 'EF'}.json"), "w"), indent=1, default=float)
