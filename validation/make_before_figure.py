"""
make_before_figure.py — "before" 2D concentration/wind panels from the pre-fix code.

Draws the same panels as realtime_viz.py (draw_conc_panel / draw_wind_panel)
but with GaussianPuffEnv / HybridFluidPuffSimulation from tag `pre-fix`
(grid gas + jet on, as shipped).  Captured at 120 s of sim time.

    git worktree add /tmp/prefix pre-fix
    python validation/make_before_figure.py --prefix /tmp/prefix
writes figures/sim_2d_conc_before.{png,pdf} and figures/sim_2d_wind_before.{png,pdf}.
"""
import argparse
import importlib.util
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True, help="checkout of tag pre-fix")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sim-time", type=float, default=120.0)
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(args.prefix))     # pre-fix gaussian_puff_env / stable_fluid_3d
sys.argv = [sys.argv[0], "--headless"]               # realtime_viz: Agg backend
spec = importlib.util.spec_from_file_location("realtime_viz_current", os.path.join(ROOT, "realtime_viz.py"))
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

import numpy as np
import matplotlib.pyplot as plt
import gaussian_puff_env as gpe
assert os.path.abspath(args.prefix) in os.path.abspath(gpe.__file__), gpe.__file__

env = gpe.GaussianPuffEnv()
obs, _ = env.reset(seed=args.seed)
trail = [env.p.copy()]
while env.fluid_sim.sim_time < args.sim_time:
    obs, *_ = env.step(rv.gradient_policy(obs, env))
    trail.append(env.p.copy())
fig_c, fig_w = plt.figure(figsize=(7, 5)), plt.figure(figsize=(7, 5))
rv.draw_conc_panel(fig_c, env, trail, title=f"Pre-fix: column-integrated concentration  t={env.fluid_sim.sim_time:.0f} s")
rv.draw_wind_panel(fig_w, env, title="Pre-fix: mean wind speed")
os.chdir(ROOT)
rv.save_2d(fig_c, fig_w, suffix="_before")
print("wrote figures/sim_2d_conc_before.{png,pdf}, figures/sim_2d_wind_before.{png,pdf}")
