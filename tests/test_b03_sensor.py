"""B3: sensor queries at edge positions."""
import numpy as np

from gaussian_puff_env import SphereObstacle
from tests import helpers as H


def _sim_with_plume(**kw):
    sim = H.make_sim(source=(60.0, 64.0, 10.0), U=3.0, **kw)
    for _ in range(int(60 / sim.frame_dt)):
        sim.step()
    return sim


def test_sensor_at_source(record):
    sim = _sim_with_plume()
    pos, vel, mass, ages = H.puffs(sim)
    S = np.array([sim.source_pos], float)
    c = sim.query_concentration(S.astype(np.float32)).astype(float)
    ref = H.conc_reference64(S, pos, mass, ages, sim.sigma_0, sim.k_rel)
    record(C_gpu=c[0], C_ref=ref[0], rel_err=abs(c[0] - ref[0]) / ref[0])
    assert np.isfinite(c).all() and c[0] > 0
    assert abs(c[0] - ref[0]) / ref[0] < 1e-4


def test_sensor_outside_returns_zero(record):
    sph = SphereObstacle(np.array([90.0, 64.0, 10.0]), 6.0)
    sim = _sim_with_plume(obstacles=((sph,), ()))
    S = np.array([[70.0, 64.0, -0.5], [70.0, 64.0, -10.0],          # below ground
                  [-1.0, 64.0, 10.0], [300.0, 64.0, 10.0], [70.0, 200.0, 10.0], [70.0, 64.0, 200.0],  # outside
                  [90.0, 64.0, 10.0], [92.0, 64.0, 12.0]], np.float32)  # inside the sphere
    c = sim.query_concentration(S)
    record(values=c)
    assert np.isfinite(c).all() and np.all(c == 0.0), c


def test_sensor_at_ground(record):
    sim = _sim_with_plume()
    pos, vel, mass, ages = H.puffs(sim)
    S = np.c_[np.linspace(60, 140, 41), np.full(41, 64.0), np.zeros(41)]
    c = sim.query_concentration(S.astype(np.float32)).astype(float)
    ref = H.conc_reference64(S, pos, mass, ages, sim.sigma_0, sim.k_rel)
    m = ref > 1e-6 * ref.max()
    rel = np.abs(c[m] - ref[m]) / ref[m]
    record(max_rel_err=float(rel.max()), n_points=int(m.sum()))
    assert np.isfinite(c).all() and c.max() > 0
    assert rel.max() < 1e-4
