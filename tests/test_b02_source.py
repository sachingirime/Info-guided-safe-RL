"""B2: source placement."""
import numpy as np
import pytest

from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle
from tests import helpers as H


@pytest.mark.parametrize("z", [0.0, 0.1])
def test_source_at_ground(z, record):
    sim = H.make_sim(source=(60.0, 64.0, z), U=3.0)
    zmin = np.inf
    for _ in range(int(120 / sim.frame_dt)):
        sim.step()
        act = sim.puff_active.numpy() == 1
        zmin = min(zmin, float(sim.puff_pos.numpy()[act, 2].min()))
    S = np.c_[np.linspace(60, 120, 31), np.full(31, 64.0), np.zeros(31)].astype(np.float32)
    C0 = sim.query_concentration(S)
    record(z_source=z, min_puff_z=zmin, ground_C_max=float(C0.max()))
    assert zmin >= 0.0
    assert np.isfinite(C0).all() and C0.max() > 0


SOURCES = {"x_lo": (1.0, 64.0, 10.0), "x_hi": (255.0, 64.0, 10.0), "y_lo": (128.0, 1.0, 10.0),
           "y_hi": (128.0, 127.0, 10.0), "top": (128.0, 64.0, 127.0), "corner_lo": (1.0, 1.0, 1.0),
           "corner_hi": (255.0, 127.0, 127.0)}


@pytest.mark.parametrize("name", list(SOURCES))
def test_source_near_faces(name, record):
    """1 m inside each face, the top and two corners: no error, puffs that
    leave are counted as exited."""
    sim = H.make_sim(source=SOURCES[name], U=3.0, angle=45.0)
    for _ in range(int(60 / sim.frame_dt)):
        sim.step()
    pos, vel, _, _ = H.puffs(sim)
    record(source=SOURCES[name], emitted=sim.n_emitted, active=len(pos), exited=sim.n_exited,
           overwritten=sim.n_overwritten)
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert sim.n_emitted == len(pos) + sim.n_exited + sim.n_overwritten


def test_env_never_spawns_source_in_obstacle(record):
    env = GaussianPuffEnv(seed=0, warmup_s=0.0)
    worst = np.inf
    for s in range(1000):
        env.reset(seed=s)
        d = min(o.distance(env.theta) for o in env.obstacles)
        worst = min(worst, d)
    record(min_source_obstacle_distance=worst)
    assert worst > 0.0


def test_sim_rejects_source_inside_obstacle():
    sim = H.make_sim(obstacles=((SphereObstacle(np.array([60.0, 64.0, 10.0]), 5.0),), ()), source=(30.0, 64.0, 10.0))
    with pytest.raises(ValueError, match="inside an obstacle"):
        sim.source_pos = np.array([61.0, 64.0, 10.0], np.float32)
    with pytest.raises(ValueError, match="outside the domain"):
        sim.source_pos = np.array([-1.0, 64.0, 10.0], np.float32)


def test_source_next_to_wall(record):
    """Source 0.5 m from a wall face: no puff inside the wall over 300 s, and the
    far-side concentration equals the line-of-sight reference."""
    wall = CuboidObstacle(center=np.array([80.0, 64.0, 15.0]), half_extents=np.array([2.0, 15.0, 15.0]))
    sim = H.make_sim(source=(77.5, 64.0, 10.0), U=3.0, obstacles=((), (wall,)))
    inside = 0
    for i in range(int(300 / sim.frame_dt)):
        sim.step()
        if i % 30 == 0:
            pos, *_ = H.puffs(sim)
            inside += int((H.sdf_box(pos, wall.center, wall.half_extents) < 0).sum())
    pos, vel, mass, ages = H.puffs(sim)
    # far side: 2–40 m behind the wall, beside and above its shadow (gas that went around/over)
    X, Y, Z = np.meshgrid([84.0, 100.0, 120.0], np.linspace(30, 98, 18), [5.0, 15.0, 32.0], indexing="ij")
    S = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    C_gpu = sim.query_concentration(S.astype(np.float32)).astype(float)
    C_ref = H.conc_reference64(S, pos, mass, ages, sim.sigma_0, sim.k_rel, (), (wall,))
    rel = np.abs(C_gpu - C_ref) / max(C_ref.max(), 1e-30)
    record(puffs_inside_logged=inside, far_side_C_max=float(C_gpu.max()), max_rel_err_vs_ref=float(rel.max()),
           n_far_side_nonzero=int((C_ref > 0).sum()), n_blocked_shadow=int(((C_ref == 0) & (S[:, 2] < 30)).sum()))
    assert inside == 0
    assert (C_ref > 0).sum() > 10          # the comparison is not vacuous
    assert rel.max() < 1e-4


def test_mobile_source(record):
    env = GaussianPuffEnv(seed=0, source_mobile=True, warmup_s=5.0)
    env.reset(seed=3)
    dmin, out = np.inf, 0
    for _ in range(1000):
        env.step(np.zeros(3, np.float32))
        dmin = min(dmin, min(o.distance(env.theta) for o in env.obstacles))
        out += int(np.any(env.theta[:2] < env.SOURCE_XY_MIN) or np.any(env.theta[:2] > env.SOURCE_XY_MAX)
                   or not env.SOURCE_Z_MIN <= env.theta[2] <= env.SOURCE_Z_MAX)
        assert np.allclose(env.fluid_sim.source_pos, env.theta, atol=1e-4)
    # puffs are released from the current source position
    sim = env.fluid_sim
    for p in ([100.0, 50.0, 5.0], [140.0, 70.0, 8.0]):
        sim.source_pos = np.array(p, np.float32)
        i0 = sim.next_puff_idx
        sim._next_emit_time = sim.sim_time          # force a release now
        sim.emit_puff()
        assert np.allclose(sim.puff_pos.numpy()[i0 % sim.max_puffs], p, atol=1e-4)
    record(min_source_obstacle_distance=dmin, steps_outside_region=out)
    assert dmin > 0.0 and out == 0
