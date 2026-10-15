"""A2: per-puff velocity statistics with shared-eddy release.

Each puff's velocity must be N(0, σ²) at release and at all ages, and its
single-particle spread must follow Taylor's solution; only the correlation
between puffs may change.
"""
import numpy as np
import stable_fluid_3d as sf

N_CHAINS, CHAIN_LEN = 40000, 10        # independent sequences × puffs; SE(var) ≈ √(2/N_CHAINS) = 0.7%
X0 = np.array([128.0, 64.0, 64.0])


def _release_chains(shared):
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=0.0, wind_base_speed=3.0, steady_wind=True,
                                       shear_exponent=0.0, seed=21, shared_eddies=shared,
                                       max_puffs=N_CHAINS * CHAIN_LEN)
    sim.emit_interval = 0.2
    sim.source_pos = X0.astype(np.float32)
    sim.set_obstacles([], [])
    V = np.empty((N_CHAINS, CHAIN_LEN, 3))
    for c in range(N_CHAINS):
        sim._release_zeta = None                     # new independent sequence
        for k in range(CHAIN_LEN):
            V[c, k] = np.array(sim._release_velocity(), float)
    return sim, V


def test_release_marginal_and_taylor(record):
    sim, V = _release_chains(True)
    sig = np.array(sim.lsm_sigma); TL = np.array(sim.lsm_TL); TE = np.array(sim.lsm_TE)
    # 1) marginal at release: variance over independent chains at every position k
    var_k = V.var(axis=0) / sig ** 2                         # (CHAIN_LEN, 3)
    assert np.all(np.abs(var_k.mean(0) - 1) < 0.02), var_k.mean(0)
    assert np.all(np.abs(var_k - 1) < 0.03), np.abs(var_k - 1).max()   # ≈ 4 SE per position
    # 2) between-puff correlation: adjacent puffs ρ = e^{-τe/T_E}
    rho = np.array([np.corrcoef(V[:, 1:, a].ravel(), V[:, :-1, a].ravel())[0, 1] for a in range(3)])
    rho_ref = np.exp(-sim.emit_interval / TE)
    assert np.all(np.abs(rho - rho_ref) < 0.01), (rho, rho_ref)

    # 3) evolve all puffs (no resolved wind: step_puffs only) and check v and spread vs age
    n = N_CHAINS * CHAIN_LEN
    sim.inject_puffs(np.tile(X0, (n, 1)), velocities=V.reshape(-1, 3))
    v0 = sim.puff_vel.numpy().astype(float)
    ages, var_v, spread, taylor = [], [], [], []
    t, steps = 0.0, 0
    for target in (0.5 * TL[2], 1.0 * TL[2], 30.0, 60.0):
        while t < target - 1e-9:
            sim.step_puffs(); sim.sim_time += sim.sim_dt; t += sim.sim_dt; steps += 1
        act = sim.puff_active.numpy() == 1
        v = sim.puff_vel.numpy().astype(float)[act]
        x = sim.puff_pos.numpy().astype(float)[act] - X0
        ages.append(t)
        var_v.append(v.var(0) / sig ** 2)
        spread.append(x.var(0))
        taylor.append(2 * sig ** 2 * TL ** 2 * (t / TL - 1 + np.exp(-t / TL)))
    var_v, spread, taylor = map(np.array, (var_v, spread, taylor))
    ratio = spread / taylor
    record(sigma=sig, TL=TL, TE=TE, rho=rho, rho_ref=rho_ref, ages=ages,
           var_v_over_sigma2=var_v, spread_over_taylor=ratio)
    # effective samples ≈ N_CHAINS (puffs within a chain are correlated)
    assert np.all(np.abs(var_v - 1) < 0.03), var_v
    assert np.all(np.abs(ratio - 1) < 0.03), ratio


def test_independent_release_same_marginal(record):
    sim, V = _release_chains(False)
    sig = np.array(sim.lsm_sigma)
    var = V.reshape(-1, 3).var(0) / sig ** 2
    rho = np.array([np.corrcoef(V[:, 1:, a].ravel(), V[:, :-1, a].ravel())[0, 1] for a in range(3)])
    record(var_over_sigma2=var, rho=rho)
    assert np.all(np.abs(var - 1) < 0.02), var
    assert np.all(np.abs(rho) < 0.01), rho
