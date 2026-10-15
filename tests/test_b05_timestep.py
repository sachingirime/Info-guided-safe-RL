"""B5: time-step and frame-rate invariance."""
import numpy as np
import pytest

from tests import helpers as H

FPS = (15, 30, 60)


def test_emission_rate_vs_fps(record):
    out = {}
    for fps in FPS:
        sim = H.make_sim(U=3.0, fps=fps)
        for _ in range(int(round(60.0 * fps))):
            sim.step()
        mass = float(sim.puff_mass.numpy()[:sim.n_emitted].sum())
        out[fps] = {"emitted": sim.n_emitted, "puffs_per_s": sim.n_emitted / sim.sim_time,
                    "mass_per_s": mass / sim.sim_time, "sim_time": sim.sim_time}
    record(**{str(k): v for k, v in out.items()})
    n = [v["emitted"] for v in out.values()]
    assert max(n) - min(n) <= 1, n
    m = [v["mass_per_s"] for v in out.values()]
    assert max(m) / min(m) - 1 < 0.01, m


def test_ou_variance_vs_fps(record):
    """Exact OU update: stationary velocity variance and the autocorrelation at
    a fixed lag do not depend on Δt."""
    res = {}
    for fps in FPS:
        sim = H.make_sim(U=3.0, fps=fps, max_puffs=40000, steady_wind=True)
        N = 40000
        sim.inject_puffs(np.tile([128.0, 64.0, 64.0], (N, 1)))
        v0 = sim.puff_vel.numpy().astype(float)
        for _ in range(int(round(20.0 * fps))):
            sim.step_puffs(); sim.sim_time += sim.sim_dt
        v1 = sim.puff_vel.numpy().astype(float)
        sig = np.array(sim.lsm_sigma)
        res[fps] = {"var_ratio": (v1.var(0) / sig ** 2).tolist(),
                    "rho_20s": [float(np.corrcoef(v0[:, a], v1[:, a])[0, 1]) for a in range(3)],
                    "rho_ref": np.exp(-20.0 / np.array(sim.lsm_TL)).tolist()}
    record(**{str(k): v for k, v in res.items()})
    for fps, r in res.items():
        assert np.all(np.abs(np.array(r["var_ratio"]) - 1) < 0.03), (fps, r)
        assert np.all(np.abs(np.array(r["rho_20s"]) - np.array(r["rho_ref"])) < 0.02), (fps, r)


@pytest.mark.slow
def test_mean_field_vs_fps(record):
    """V1 (steady, independent puff noise) and V3 class D (defaults; shared
    eddies off to separate Δt effects from meander noise) change < 5%."""
    jobs = [("v1_metrics", {"fps": f}) for f in FPS]
    jobs += [("v3_class", {"pg": "D", "seed": 1, "avg": 3600.0,
                           "sim_kwargs": {"fps": f, "shared_eddies": False}}) for f in FPS]
    res = H.run_parallel(jobs)
    v1, v3 = res[:3], res[3:]
    dev_v1 = max(abs(r["rows"][i][k] / v1[1]["rows"][i][k] - 1)
                 for r in v1 for i in range(3) for k in ("sigma_y", "sigma_z", "c_cl"))
    dev_v3 = max(abs(r["rows"][i][k] / v3[1]["rows"][i][k] - 1)
                 for r in v3 for i in range(5) for k in ("sigma_y_sim", "sigma_z_sim"))
    record(v1=v1, v3=v3, max_rel_dev_v1=dev_v1, max_rel_dev_v3=dev_v3)
    assert dev_v1 < 0.05, dev_v1
    assert dev_v3 < 0.05, dev_v3


@pytest.mark.slow
def test_env_clock_non_integer_ratio(record):
    res = H.run_parallel([("env_clock", {"fps": f}) for f in (30, 25, 24)])
    record(results=res)
    for r in res:
        assert abs(r["drift_s"]) <= r["frame_dt"] + 1e-9, r
