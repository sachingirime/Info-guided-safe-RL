"""B12: invalid configuration raises clear errors."""
import numpy as np
import pytest

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle


def _sim(**kw):
    return sf.HybridFluidPuffSimulation(**kw)


@pytest.mark.parametrize("attr,val,msg", [("sigma_0", -1.0, "sigma_0"), ("sigma_0", 0.0, "sigma_0"), ("Q", -0.1, "Q must")])
def test_negative_attributes(attr, val, msg):
    sim = _sim()
    with pytest.raises(ValueError, match=msg):
        setattr(sim, attr, val)


@pytest.mark.parametrize("kw,msg", [({"K_rel": -0.01}, "K_rel"), ({"wind_base_speed": -1.0}, "wind_base_speed"),
                                    ({"stability_class": "G"}, "Unknown PG class"), ({"fps": 0}, "fps"),
                                    ({"sigma_min": -0.1}, "sigma_min"), ({"class_sigma0": {"F": -0.3}}, "class_sigma0"),
                                    ({"wind_preset": "stormy"}, "wind_preset"), ({"buoyancy_mode": "float"}, "buoyancy_mode")])
def test_bad_constructor_args(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _sim(**kw)


def test_env_negative_q_sigma0():
    class E1(GaussianPuffEnv):
        Q_INIT = -1.0
    class E2(GaussianPuffEnv):
        SIGMA_0 = -0.5
    with pytest.raises(ValueError, match="Q must"):
        E1(seed=0)
    with pytest.raises(ValueError, match="sigma_0"):
        E2(seed=0)


def test_env_buffer_too_small_for_warmup():
    class E(GaussianPuffEnv):
        N_PUFFS = 100                      # 20 s of puffs < 60 s warm-up
    with pytest.raises(ValueError, match="N_PUFFS"):
        E(seed=0)


def test_env_grid_mismatch():
    class E(GaussianPuffEnv):
        DOMAIN_MAX = np.array([200.0, 128.0, 128.0])
    with pytest.raises(ValueError, match="does not match the fluid grid"):
        E(seed=0)


def test_obstacles_outside_domain():
    sim = _sim()
    with pytest.raises(ValueError, match="outside the domain"):
        sim.set_obstacles([SphereObstacle(np.array([400.0, 64.0, 10.0]), 5.0)], [])
    with pytest.raises(ValueError, match="outside the domain"):
        sim.set_obstacles([], [CuboidObstacle(np.array([60.0, -30.0, 10.0]), np.array([5.0, 5.0, 5.0]))])
    with pytest.raises(ValueError, match="invalid obstacle"):
        sim.set_obstacles([SphereObstacle(np.array([60.0, 64.0, 10.0]), -2.0)], [])
