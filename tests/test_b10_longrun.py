"""B10 (+ B6.22, B9.34): one hour of simulated time at env defaults."""
import time
import warnings

import numpy as np
import pytest

from gaussian_puff_env import GaussianPuffEnv


@pytest.fixture(scope="module")
def hour_run():
    env = GaussianPuffEnv(seed=0)
    env.reset(seed=1)                                  # 60 s warm-up, env defaults (U = 1.5 m/s, class D, obstacles)
    sim = env.fluid_sim
    log = {"t": [], "ke": [], "n_active": [], "steps_per_s": [], "nan": False, "residual": [], "t_ke": [], "ke_1s": []}
    n_window = int(round(30.0 / sim.frame_dt))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        t_end = sim.sim_time + 3600.0
        while sim.sim_time < t_end - 1e-9:
            t0 = time.perf_counter()
            for k in range(n_window):
                sim.step()
                if k % 30 == 0:          # KE every 1 s: gusts (2 s) spike KE, so 30 s sampling aliases them
                    uu = sim.u0.numpy()
                    log["t_ke"].append(sim.sim_time); log["ke_1s"].append(float(0.5 * (uu ** 2).sum(-1).mean()))
            u = sim.u0.numpy()
            act = sim.puff_active.numpy() == 1
            log["steps_per_s"].append(n_window / (time.perf_counter() - t0))
            log["t"].append(sim.sim_time); log["ke"].append(float(0.5 * (u ** 2).sum(-1).mean()))
            log["n_active"].append(int(act.sum()))
            log["nan"] |= bool(not np.isfinite(u).all() or not np.isfinite(sim.puff_pos.numpy()[act]).all()
                               or not np.isfinite(sim.puff_vel.numpy()[act]).all())
            if len(log["t"]) % 10 == 0:                  # every 5 min
                log["residual"].append(sim.divergence_residual())
        overflow_warnings = [str(x.message) for x in w if "overflow" in str(x.message)]
    expected = int(np.floor(sim.sim_time / sim.emit_interval + 1e-9)) + 1
    return {"sim": sim, "log": log, "expected_emitted": expected, "warnings": overflow_warnings}


@pytest.mark.slow
def test_buffer_sizing_one_hour(hour_run, record):
    sim = hour_run["sim"]
    record(n_overwritten=sim.n_overwritten, warnings=hour_run["warnings"], max_active=max(hour_run["log"]["n_active"]),
           max_puffs=sim.max_puffs)
    assert sim.n_overwritten == 0 or hour_run["warnings"]


@pytest.mark.slow
def test_clock_drift_one_hour(hour_run, record):
    sim = hour_run["sim"]
    record(sim_time=sim.sim_time, emitted=sim.n_emitted, expected=hour_run["expected_emitted"])
    assert abs(sim.n_emitted - hour_run["expected_emitted"]) <= 1


@pytest.mark.slow
def test_long_run_stability(hour_run, record):
    log = hour_run["log"]
    t = np.array(log["t"]); n = np.array(log["n_active"]); sps = np.array(log["steps_per_s"])
    half = t >= t[0] + 1800
    tk = np.array(log["t_ke"]); ke = np.array(log["ke_1s"]); hk = tk >= tk[0] + 1800
    slope = np.polyfit(tk[hk], ke[hk], 1)[0]
    ke_trend = abs(slope) * 1800 / ke[hk].mean()
    q = len(ke) // 4
    quarters = [float(ke[i * q:(i + 1) * q].mean()) for i in range(4)]
    res = np.array(log["residual"])
    first, last = sps[2:12].mean(), sps[-10:].mean()           # skip the first minute (JIT/warm caches)
    record(nan=log["nan"], ke_mean=float(ke.mean()), ke_max=float(ke.max()), ke_trend_rel_per_30min=float(ke_trend),
           ke_quarter_means=quarters,
           residual=res.tolist(), puff_count_cv_second_half=float(n[half].std() / n[half].mean()),
           puff_count_mean=float(n[half].mean()), steps_per_s_first_5min=float(first), steps_per_s_last_5min=float(last))
    assert not log["nan"]
    assert ke_trend < 0.10, ke_trend
    assert max(quarters) / min(quarters) < 1.10, quarters
    assert res.max() / res.min() < 1.5, res
    assert n[half].std() / n[half].mean() < 0.10
    assert last >= 0.9 * first, (first, last)
