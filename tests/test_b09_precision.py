"""B9: numerical precision of the concentration kernel (float32 GPU vs float64)."""
import numpy as np

from gaussian_puff_env import SphereObstacle, CuboidObstacle
from tests import helpers as H


def test_gpu_vs_float64_reference(record):
    """1000 random configurations (≤ 2048 puffs, obstacles, sensors near and far):
    max relative error < 1e-4 wherever C > 1e-6 of the configuration's peak."""
    rng = np.random.default_rng(11)
    sim = H.make_sim(max_puffs=2048, steady_wind=True)
    worst, worst_cfg, n_eval = 0.0, None, 0
    for cfg in range(1000):
        n = int(rng.integers(1, 2049))
        pos = np.c_[rng.uniform(20, 236, n), rng.uniform(10, 118, n), np.abs(rng.normal(0, 15, n))]
        ages = rng.uniform(0.05, 200, n); mass = rng.uniform(0.01, 2, n)
        sim.sigma_0 = float(rng.uniform(0.3, 3)); sim.k_rel = float(rng.uniform(0, 0.5))
        sph = [SphereObstacle(rng.uniform([30, 20, 0], [220, 108, 30]), float(rng.uniform(2, 10))) for _ in range(rng.integers(0, 3))]
        box = [CuboidObstacle(rng.uniform([30, 20, 5], [220, 108, 30]), rng.uniform(1, 10, 3)) for _ in range(rng.integers(0, 3))]
        sim.set_obstacles(sph, box)
        sim.sim_time = 1000.0
        sim.inject_puffs(pos, ages=ages, masses=mass)
        near = pos[rng.integers(0, n, 15)] + rng.normal(0, 2, (15, 3))
        far = np.c_[rng.uniform(0, 256, 10), rng.uniform(0, 128, 10), rng.uniform(0, 60, 10)]
        S = np.r_[near, far]; S[:, 2] = np.abs(S[:, 2])
        # read back what the GPU holds (float32) so both sides see identical inputs
        P = sim.puff_pos.numpy()[:n].astype(float); A = sim.sim_time - sim.puff_emit_time.numpy()[:n].astype(float)
        M = sim.puff_mass.numpy()[:n].astype(float); S32 = S.astype(np.float32)
        c_gpu = sim.query_concentration(S32).astype(float)
        c_ref = H.conc_reference64(S32.astype(float), P, M, A, float(np.float32(sim.sigma_0)),
                                   float(np.float32(sim.k_rel)), sph, box)
        peak = c_ref.max()
        if peak <= 0:
            continue
        m = c_ref > 1e-6 * peak
        rel = np.abs(c_gpu[m] - c_ref[m]) / c_ref[m]
        n_eval += int(m.sum())
        if rel.max() > worst:
            worst, worst_cfg = float(rel.max()), cfg
    record(max_rel_err=worst, worst_config=worst_cfg, n_sensor_values_compared=n_eval)
    assert worst < 1e-4, worst
