"""B11: environment (episode level)."""
import numpy as np
import pytest

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv

ACTIONS = np.random.default_rng(99).uniform(-5, 5, (500, 3)).astype(np.float32)


def _episode(env, seed, n=100):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in ACTIONS[:n]:
        out.append(env.step(a)[0])
    return np.array(out)


def test_reset_hygiene(record):
    """Episode A then reset(seed) == fresh env reset(seed): nothing carries over."""
    env = GaussianPuffEnv(seed=1, info_map_size=4, warmup_s=20.0)
    env.reset(seed=10)
    env.set_theta_hat(np.array([50.0, 50.0, 5.0])); env.set_info_map(np.full(4, 0.7)); env.set_zeta_inv(0.9)
    for a in ACTIONS[:200]:
        env.step(a)
    a_obs = _episode(env, 21)
    fresh = GaussianPuffEnv(seed=777, info_map_size=4, warmup_s=20.0)
    b_obs = _episode(fresh, 21)
    sa, sb = env.fluid_sim, fresh.fluid_sim
    same = {
        "obs": bool(np.array_equal(a_obs, b_obs)),
        "puff_pos": bool(np.array_equal(sa.puff_pos.numpy(), sb.puff_pos.numpy())),
        "puff_vel": bool(np.array_equal(sa.puff_vel.numpy(), sb.puff_vel.numpy())),
        "shared_eddy_state": bool(np.array_equal(sa._release_zeta, sb._release_zeta)),
        "source": bool(np.array_equal(env.theta, fresh.theta)),
        "anemometer_eddy_history": bool(np.array_equal(sa._eddy_hist_v, sb._eddy_hist_v)
                                        and np.array_equal(sa._eddy_hist_t, sb._eddy_hist_t)),
        "sensor_lag": env._y_lagged == fresh._y_lagged,
        "step_count": env.step_count == fresh.step_count,
        "theta_hat_cleared": env._theta_hat is None,
        "info_map_cleared": bool(np.all(a_obs[0, 10:] == 0)),
        "u0": bool(np.array_equal(sa.u0.numpy(), sb.u0.numpy())),
    }
    record(**same)
    assert all(same.values()), same


def test_determinism_500_steps(record):
    env = GaussianPuffEnv(seed=0, warmup_s=20.0)
    a, b, c = _episode(env, 5, 500), _episode(env, 5, 500), _episode(env, 6, 500)
    record(identical=bool(np.array_equal(a, b)), differ=bool(not np.array_equal(a, c)))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_spawn_validity_1000_resets(record):
    env = GaussianPuffEnv(seed=0, warmup_s=0.0)
    src_obs = drone_obs = drone_src = np.inf
    spawn_xy = []
    for s in range(1000):
        env.reset(seed=s)
        src_obs = min(src_obs, min(o.distance(env.theta) for o in env.obstacles))
        drone_obs = min(drone_obs, min(o.distance(env.p) for o in env.obstacles))
        drone_src = min(drone_src, float(np.linalg.norm(env.p - env.theta)))
        spawn_xy.append(env.p[:2].copy())
    spawn_xy = np.array(spawn_xy)
    record(min_source_obstacle_dist=src_obs, min_drone_obstacle_dist=drone_obs, min_drone_source_dist=drone_src,
           distinct_spawn_points=int(len(np.unique(spawn_xy.round(3), axis=0))))
    assert src_obs > 0 and drone_obs > 0 and drone_src >= 30.0
    assert len(np.unique(spawn_xy.round(3), axis=0)) > 900        # spawn is actually random


def test_obstacles_set_before_warmup(record):
    env = GaussianPuffEnv(seed=0, warmup_s=1.0)
    sim = env.fluid_sim
    seen = []
    orig = sim.step

    def spy():
        if not seen:
            seen.append(sim.n_spheres + sim.n_cuboids)
        orig()
    sim.step = spy
    for s in range(5):
        seen.clear(); env.reset(seed=s)
        assert seen and seen[0] == len(env.obstacles) > 0
    record(obstacles_at_first_warmup_step=seen[0])


def test_warmup_plume_reach(record):
    """After the default warm-up the plume reaches ≥ 100 m downwind (or the domain edge)."""
    env = GaussianPuffEnv(seed=0)
    rows = []
    for s in range(10):
        env.reset(seed=s)
        sim = env.fluid_sim
        w = np.array(sim.wind_model._wind_mean); w /= np.linalg.norm(w)
        act = sim.puff_active.numpy() == 1
        reach = float(((sim.puff_pos.numpy()[act] - env.theta) @ w).max())
        # distance from the source to the domain boundary along w
        t = [((env.DOMAIN_MAX[i] if w[i] > 0 else 0.0) - env.theta[i]) / w[i] for i in range(2) if abs(w[i]) > 1e-9]
        rows.append({"seed": s, "reach_m": reach, "edge_m": float(min(t)), "U10": sim.wind_base_speed})
    ok = [r["reach_m"] >= min(100.0, r["edge_m"] - 5.0) for r in rows]
    record(rows=rows, warmup_s=env.WARMUP_S, n_ok=int(sum(ok)))
    assert all(ok), rows


@pytest.mark.slow
def test_observation_ranges_100_episodes(record):
    env = GaussianPuffEnv(seed=0)
    rng = np.random.default_rng(3)
    obs_all = []
    for ep in range(100):
        obs, _ = env.reset(seed=1000 + ep); obs_all.append(obs)
        done = False
        while not done:
            obs, r, te, tr, _ = env.step(rng.uniform(-env.V_MAX, env.V_MAX, 3).astype(np.float32))
            obs_all.append(obs); done = te or tr
    O = np.array(obs_all)
    frac = {name: {"at_0": float(np.mean(O[:, i] == 0)), "at_1": float(np.mean(O[:, i] == 1)),
                   "gt_0.5": float(np.mean(O[:, i] > 0.5))} for i, name in ((6, "c_norm"), (7, "c_detected"))}
    record(n_steps=len(O), finite=bool(np.isfinite(O).all()), min=float(O.min()), max=float(O.max()), channels=frac)
    assert np.isfinite(O).all() and O.min() >= 0 and O.max() <= 1


def test_sensor_model(record):
    env = GaussianPuffEnv(seed=0, warmup_s=1.0)
    env.reset(seed=1)
    tau, dt, b, se = env.SENSOR_TAU, env.DT, env.BIAS, env.SIGMA_E
    # step response (noise off)
    env.SIGMA_E = 0.0
    env._c_true = 0.0; env._y_lagged = b
    C = 1e-2; ys = []
    env._c_true = C
    for _ in range(40):
        ys.append(env._get_sensor_reading())
    frac = (np.array(ys) - b) / C
    t63 = dt * (np.argmax(frac >= 1 - np.exp(-1)) + 1)
    # noise std and bias (lag removed by a huge alpha → 0)
    env.SIGMA_E = se; env._sensor_alpha = 0.0
    raw = np.array([env._get_sensor_reading() for _ in range(200_000)])
    env._c_true = 0.0
    floor = np.array([env._get_sensor_reading() for _ in range(200_000)])
    record(t63_s=t63, tau_s=tau, noise_std=float(raw.std()), spec_std=se, mean_minus_C=float(raw.mean() - C),
           bias=b, clip_bias_at_C0=float(floor.mean() - b))
    assert abs(t63 - tau) <= 0.05 * tau + 1e-9, t63
    assert abs(raw.std() / se - 1) < 0.01
    assert abs((raw.mean() - C) / b - 1) < 0.01            # bias applied once


def test_anemometer_coherent(record):
    """v1.1: the anemometer adds the coherent (shared-eddy) velocity that moves the
    plume, lagged by the advection time (Taylor), instead of independent OU noise."""
    env = GaussianPuffEnv(seed=0, warmup_s=30.0)
    env.reset(seed=2)
    sim = env.fluid_sim
    n = min(sim._eddy_hist_i, sf.EDDY_HIST_N)
    order = np.argsort(sim._eddy_hist_t[:n])
    ts, vs = sim._eddy_hist_t[:n][order], sim._eddy_hist_v[:n][order]
    src = np.asarray(sim.source_pos, float)
    d = np.array([sim.wind_model.u_mean, sim.wind_model.v_mean]); d /= np.linalg.norm(d)
    at_src = sim.coherent_velocity(src[None])[0]
    x_dn = 30.0
    p_dn = src + np.array([*(x_dn * d), 0.0])
    t_dn = sim.sim_time - x_dn / sim.U_src
    expect_dn = vs[np.searchsorted(ts, t_dn, side="right") - 1]
    p_up = src - np.array([*(20.0 * d), 0.0])          # upwind: no lag
    clean = env._sample_wind_at(env.p)
    obs = env._observed_wind()
    off = GaussianPuffEnv(seed=0, warmup_s=5.0, wind_noise=False)
    off.reset(seed=2)
    exact_off = bool(np.array_equal(off._observed_wind(), off._sample_wind_at(off.p)))
    sim_ind = sf.HybridFluidPuffSimulation(shared_eddies=False, max_puffs=64)
    sim_ind.emit_puff()
    record(at_source=at_src.tolist(), latest=vs[-1].tolist(), downwind=sim.coherent_velocity(p_dn[None])[0].tolist(),
           expected_downwind=expect_dn.tolist(), off_is_exact=exact_off)
    assert np.array_equal(at_src, vs[-1])
    assert np.array_equal(sim.coherent_velocity(p_dn[None])[0], expect_dn)
    assert np.array_equal(sim.coherent_velocity(p_up[None])[0], vs[-1])
    assert np.allclose(obs - clean, sim.coherent_velocity(env.p[None])[0])
    assert exact_off
    assert np.all(sim_ind.coherent_velocity(np.zeros((1, 3))) == 0)


def test_units_round_trip(record):
    c = np.logspace(-9, 0, 50)
    for T, P in ((273.15, 101325.0), (293.15, 101325.0), (310.0, 90000.0)):
        rt = sf.ppm_to_kgm3(sf.kgm3_to_ppm(c, T, P), T, P)
        assert np.allclose(rt, c, rtol=1e-12, atol=0)
    ppm = float(sf.kgm3_to_ppm(1e-3))                      # 1 g/m³ CH4 at 20 °C → ≈ 1500 ppm
    record(ppm_of_1_g_per_m3=ppm)
    assert 1490 < ppm < 1510
