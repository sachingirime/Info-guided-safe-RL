"""
t5_cost.py — Tier 5: compute cost, ours (GPU, Warp) vs GADEN (CPU, OpenMP).

Same plume (T1c set-up: meandering wind, matched release, 10 filaments/s),
measured after a 150 s spin-up:
  step cost   wall time per simulated second (ours: wind solver + puffs at
              30 Hz; GADEN: filament step at 10 Hz, excluding wind upload)
  query cost  wall time of one concentration query for M = 1 and 1000 points
Writes bench/results/t5.json.
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paired as pr                  # noqa: E402
import gaden_client as gc            # noqa: E402

wp.config.quiet = True


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main(threads=8):
    src = (20.0, 64.0, 30.0)
    sim = pr.make_ours(src, angle_range=20.0, U=3.0, seed=1, wind_preset="meandering")
    env_bin = pr.gaden_env(sim, os.path.join(HERE, "data", "t5_env.bin"))
    prm = gc.gaden_params(src, Q=pr.Q_TRACER, n_fil_sec=10, sigma0_m=1.0, K_rel=0.05, dt=0.1)
    out = {"cpu": platform.processor() or platform.machine(), "gaden_threads": threads}
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    rng = np.random.default_rng(0)
    with gc.GadenClient(env_bin, prm, threads=threads) as g:
        for f in range(int(150 * sim.fps)):
            if f % 3 == 0:
                g.set_wind(sim.u0.numpy()); g.step(1)
            sim.step()
        wp.synchronize()
        out["n_puffs_ours"] = int(len(sim.get_puff_data()[0])); out["n_filaments_gaden"] = int(len(g.filaments()))
        # step cost per simulated second
        def ours_1s():
            for _ in range(int(sim.fps)):
                sim.step()
            wp.synchronize()
        u = sim.u0.numpy()
        g.set_wind(u)
        out["ours_wall_per_sim_s"] = timeit(ours_1s, 10)
        out["gaden_wall_per_sim_s"] = timeit(lambda: g.step(10), 10)
        out["gaden_wind_upload_s"] = timeit(lambda: g.set_wind(u), 20)
        for M in (1, 1000):
            P = np.column_stack([rng.uniform(25, 200, M), rng.uniform(40, 88, M), rng.uniform(20, 40, M)]).astype(np.float32)
            out[f"ours_query_{M}_s"] = timeit(lambda: sim.query_concentration(P), 50)
            out[f"gaden_query_{M}_s"] = timeit(lambda: g.query(P), 50)
    out["ours_realtime_factor"] = 1.0 / out["ours_wall_per_sim_s"]
    out["gaden_realtime_factor"] = 1.0 / out["gaden_wall_per_sim_s"]
    path = os.path.join(HERE, "results", "t5.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
