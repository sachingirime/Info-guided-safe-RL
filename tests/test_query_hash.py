"""v1.1: hash-grid concentration query equals the brute-force puff sum."""
import numpy as np

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv


def _compare(sim, P):
    sim.use_hash_query = True
    a = sim.query_concentration(P)
    sim.use_hash_query = False
    b = sim.query_concentration(P)
    sim.use_hash_query = True
    scale = max(float(np.abs(b).max()), 1e-30)
    return float(np.max(np.abs(a - b)) / scale), int((b > 0).sum())


def test_hash_equals_bruteforce_env(record):
    env = GaussianPuffEnv(seed=0, warmup_s=60.0)
    env.reset(seed=3)
    sim = env.fluid_sim
    rng = np.random.default_rng(0)
    src = np.asarray(sim.source_pos, float)
    P = np.vstack([src + rng.normal(0, [30, 30, 4], (3000, 3)),
                   rng.uniform([0, 0, 0], [256, 128, 40], (3000, 3))]).astype(np.float32)
    P[:, 2] = np.abs(P[:, 2])
    err, nz = _compare(sim, P)
    record(max_rel_err=err, nonzero_points=nz, radius=sim._influence_radius())
    assert nz > 100 and err < 1e-5


def test_hash_equals_bruteforce_injected_old(record):
    sim = sf.HybridFluidPuffSimulation(max_puffs=512, surface_layer=True)
    sim.set_obstacles([], [])
    rng = np.random.default_rng(1)
    pos = rng.uniform([20, 20, 0], [236, 108, 30], (512, 3))
    sim.inject_puffs(pos, ages=rng.uniform(0, 3000, 512))      # older than the ring-buffer bound
    sim.emit_interval = 0.05
    P = rng.uniform([0, 0, 0], [256, 128, 40], (4000, 3)).astype(np.float32)
    err, nz = _compare(sim, P)
    record(max_rel_err=err, nonzero_points=nz, radius=sim._influence_radius())
    assert nz > 100 and err < 1e-5
