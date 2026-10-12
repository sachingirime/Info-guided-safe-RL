"""
test_env.py — Unit tests for GaussianPuffEnv / HybridFluidPuffSimulation.

Run:  python test_env.py        (also runs under pytest)
Requires a CUDA GPU.
"""

import numpy as np
import warp as wp

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv, SphereObstacle

wp.config.quiet = True


def _make_sim(**kw):
    """Small helper: class-D sim, wind +x, source (60, 64, 10), no obstacles."""
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=0.0, wind_angle_range=20.0,
                                       wind_base_speed=1.5, wind_seed=42, **kw)
    sim.Q = 5.0; sim.emit_interval = 0.2; sim.sigma_0 = 2.0
    sim.set_stability_class("D")
    sim.source_pos = np.array([60.0, 64.0, 10.0], np.float32)
    sim.set_obstacles([], [])
    return sim


def _probe_points(theta, w):
    """27 points around and downwind of the source."""
    d = np.array([w[0], w[1], 0.0]); d /= max(np.linalg.norm(d), 1e-9)
    pts = [theta + d * r + np.array([0, 0, dz]) + np.cross([0, 0, 1], d) * dy
           for r in (5.0, 15.0, 30.0) for dy in (-5.0, 0.0, 5.0) for dz in (-3.0, 0.0, 3.0)]
    return np.array(pts, np.float32)


def test_obs_dimensions():
    env = GaussianPuffEnv(seed=0, warmup_s=1.0)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (10,), f"Bad obs shape: {obs.shape}"
    assert env.action_space.shape == (3,), "Bad action shape"
    obs2, r, term, trunc, info = env.step(np.zeros(3, np.float32))
    assert obs2.shape == (10,), f"Bad obs shape after step: {obs2.shape}"
    assert isinstance(r, float), "Reward must be float"
    assert np.all((obs2 >= 0) & (obs2 <= 1)), "obs out of [0, 1]"
    print("✓ obs dimensions")


def test_determinism():
    """Same reset seed → identical concentration sequences; different seeds differ."""
    env = GaussianPuffEnv(seed=0)
    actions = np.random.default_rng(123).uniform(-5, 5, (100, 3)).astype(np.float32)

    def episode(seed):
        env.reset(seed=seed)
        probes = _probe_points(env.theta, env.fluid_sim.wind_model._wind_mean)
        seq = []
        for a in actions:
            _, _, _, _, info = env.step(a)
            seq.append(np.concatenate([[info["c_true"]], env.fluid_sim.query_concentration(probes)]))
        return np.array(seq)

    a, b, c = episode(7), episode(7), episode(8)
    assert a.max() > 0, "probes saw no gas — test is vacuous"
    assert np.array_equal(a, b), f"same seed differs: max|Δ|={np.abs(a - b).max()}"
    assert not np.array_equal(a, c), "different seeds gave identical sequences"
    print(f"✓ determinism (max probe C = {a.max():.3e})")


def test_no_grid_gas():
    """Defaults: no Eulerian gas after 60 s (env reset warms up for 60 s)."""
    env = GaussianPuffEnv(seed=0)
    env.reset(seed=3)
    assert env.fluid_sim.sim_time >= 59.9
    rho_max = float(env.fluid_sim.rho0.numpy().max())
    assert rho_max == 0.0, f"rho0.max() = {rho_max}"
    print("✓ no grid gas")


def test_emission_count():
    """60 s at emit_interval 0.2 s → 300 ± 1 puffs, none overwritten."""
    sim = _make_sim(max_puffs=1024)
    for _ in range(int(round(60.0 / sim.frame_dt))):
        sim.step()
    n_active = len(sim.get_puff_data()[0])
    assert abs(sim.n_emitted - 300) <= 1, sim.n_emitted
    assert sim.n_overwritten == 0, sim.n_overwritten
    assert sim.n_exited + n_active == sim.n_emitted
    assert np.allclose(sim.puff_mass.numpy()[:sim.n_emitted], sim.Q * sim.emit_interval)
    print(f"✓ emission count ({sim.n_emitted} emitted, {sim.n_overwritten} overwritten)")


def test_rng():
    """Seeded centroid noise: reproducible, seed-dependent, uncorrelated axes."""
    def run(seed):
        sim = _make_sim(seed=seed)
        for _ in range(900):
            sim.step()
        return sim.puff_pos.numpy()[:sim.next_puff_idx].copy()
    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)

    N = 100_000
    sim = sf.HybridFluidPuffSimulation(wind_base_speed=0.0, steady_wind=True, max_puffs=N, seed=3)
    p0 = np.tile(np.array([128.0, 64.0, 64.0], np.float32), (N, 1))
    sim.inject_puffs(p0)
    sim.step_puffs()
    dp = sim.puff_pos.numpy() - p0
    r = np.corrcoef(dp.T)
    assert max(abs(r[0, 1]), abs(r[0, 2]), abs(r[1, 2])) < 0.01, r
    print(f"✓ RNG (|corr| ≤ {max(abs(r[0,1]), abs(r[0,2]), abs(r[1,2])):.4f})")


def test_los():
    """V6: a sphere between puff and sensor blocks the puff."""
    def conc(use_los, with_sphere):
        sim = sf.HybridFluidPuffSimulation(max_puffs=1, use_los=use_los, f_rel=0.0)
        sim.sigma_0 = 8.0   # σ=2 m would be culled at 10σ regardless of the sphere
        sim.set_obstacles([SphereObstacle(np.array([60.0, 64.0, 20.0]), 4.0)] if with_sphere else [], [])
        sim.inject_puffs([[50.0, 64.0, 20.0]])
        return float(sim.query_concentration(np.array([[70.0, 64.0, 20.0]], np.float32))[0])
    blocked, open_, no_los = conc(True, True), conc(True, False), conc(False, True)
    assert blocked == 0.0, blocked
    assert open_ > 0.0 and np.isclose(no_los, open_)
    print(f"✓ LOS (blocked 0, open {open_:.3e})")


def _integrate_puff(z0, sigma, lo_z):
    sim = sf.HybridFluidPuffSimulation(max_puffs=1, f_rel=0.0)
    sim.sigma_0 = sigma
    sim.set_obstacles([], [])
    c0 = np.array([128.0, 64.0, z0])
    sim.inject_puffs([c0], masses=2.5)
    h = 0.5
    ax = np.arange(-6 * sigma, 6 * sigma + h / 2, h)
    zs = np.arange(max(lo_z, z0 - 6 * sigma), z0 + 6 * sigma + h / 2, h)
    X, Y, Z = np.meshgrid(c0[0] + ax, c0[1] + ax, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    c = sim.query_concentration(pts).astype(np.float64).reshape(X.shape)
    # trapezoid weights so the z ≥ 0 cut is integrated exactly at the boundary
    w = lambda n: np.r_[0.5, np.ones(n - 2), 0.5] * h
    return np.einsum("ijk,i,j,k->", c, w(len(ax)), w(len(ax)), w(len(zs))) / 2.5


def test_kernel_mass():
    """V2: one puff integrates to its mass, in free air and next to the ground."""
    m_free = _integrate_puff(60.0, 3.0, -1e9)
    m_ground = _integrate_puff(2.0, 3.0, 0.0)
    assert abs(m_free - 1) < 0.01, m_free
    assert abs(m_ground - 1) < 0.01, m_ground
    print(f"✓ kernel mass (free {m_free:.4f}, ground {m_ground:.4f})")


TESTS = [test_obs_dimensions, test_emission_count, test_rng, test_los,
         test_kernel_mass, test_no_grid_gas, test_determinism]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e!r}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
