"""B13: performance baseline (recorded in tests/report.json; no pass/fail)."""
import time

import numpy as np
import pytest
import warp as wp

from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle
from tests import helpers as H


def _rate(sim, n_query, steps=300):
    P = np.random.default_rng(0).uniform([0, 0, 4], [256, 128, 60], (max(n_query, 1), 3)).astype(np.float32)
    for _ in range(30):
        sim.step()
    wp.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
        if n_query:
            sim.query_concentration(P)
    wp.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return {"steps_per_s": 1 / dt, "ms_per_step": 1e3 * dt, "real_time_factor": sim.frame_dt / dt}


@pytest.mark.slow
def test_performance_baseline(record):
    env = GaussianPuffEnv(seed=0)
    env.reset(seed=1)
    out = {"env_defaults_step_only": _rate(env.fluid_sim, 0),
           "env_defaults_query_1": _rate(env.fluid_sim, 1)}
    t0 = time.perf_counter()
    for _ in range(200):
        env.step(np.zeros(3, np.float32))
    out["env_step_per_s"] = 200 / (time.perf_counter() - t0)
    sph = [o for o in env.obstacles if isinstance(o, SphereObstacle)]
    box = [o for o in env.obstacles if isinstance(o, CuboidObstacle)]
    rng = np.random.default_rng(1)
    for los in (False, True):
        sim = H.make_sim(source=tuple(env.theta), U=1.5, max_puffs=env.N_PUFFS, use_los=los, obstacles=(sph, box))
        sim.inject_puffs(rng.uniform([0, 0, 0], [256, 128, 60], (env.N_PUFFS, 3)), ages=rng.uniform(0, 300, env.N_PUFFS))
        for nq in (0, 1, 1000):
            out[f"max_puffs_{env.N_PUFFS}_los_{'on' if los else 'off'}_query_{nq}"] = _rate(sim, nq)
    out["device"] = wp.get_device().name
    record(**out)
