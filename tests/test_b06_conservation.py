"""B6: conservation and bookkeeping."""
import warnings

import numpy as np
import pytest

from tests import helpers as H


def test_mass_balance_every_step(record):
    """emitted = in_domain + exited + overwritten (counts exactly, mass to float
    tolerance), with a varying Q and a buffer small enough to overflow."""
    sim = H.make_sim(source=(30.0, 64.0, 10.0), U=3.0, max_puffs=450)   # 90 s buffer: fast puffs exit, slow near-ground puffs are overwritten
    worst_mass = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(int(240 / sim.frame_dt)):
            sim.Q = 1.0 + 0.5 * np.sin(sim.sim_time / 20.0)
            sim.step()
            if i % 30 == 0:
                act = int((sim.puff_active.numpy() == 1).sum())
                assert sim.n_emitted == act + sim.n_exited + sim.n_overwritten
                bal = sim.mass_in_domain + sim.mass_exited + sim.mass_overwritten
                worst_mass = max(worst_mass, abs(bal - sim.mass_emitted) / sim.mass_emitted)
    record(emitted=sim.n_emitted, exited=sim.n_exited, overwritten=sim.n_overwritten,
           mass_emitted=sim.mass_emitted, worst_rel_mass_error=worst_mass)
    assert sim.n_overwritten > 0 and sim.n_exited > 0      # both paths exercised
    assert worst_mass < 1e-5, worst_mass


@pytest.mark.parametrize("sigma", [0.3, 1.0, 5.0])
@pytest.mark.parametrize("hfac", [0.1, 1.0, 10.0])
def test_kernel_mass(sigma, hfac, record):
    """∫_{z≥0} C dV of one puff = its mass within 1% (image term near the ground)."""
    h = hfac * sigma
    sim = H.make_sim(K_rel=0.0, max_puffs=1, steady_wind=True)
    sim.sigma_0 = sigma
    c0 = np.array([128.0, 64.0, h])
    sim.inject_puffs([c0], masses=2.0)
    d = sigma / 5.0
    ax = np.arange(-6 * sigma, 6 * sigma + d / 2, d)
    zs = np.arange(max(0.0, h - 6 * sigma), h + 6 * sigma + d / 2, d)
    X, Y, Z = np.meshgrid(c0[0] + ax, c0[1] + ax, zs, indexing="ij")
    C = sim.query_concentration(np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(np.float32)).astype(float).reshape(X.shape)
    w = lambda n: np.r_[0.5, np.ones(n - 2), 0.5] * d
    ratio = np.einsum("ijk,i,j,k->", C, w(len(ax)), w(len(ax)), w(len(zs))) / 2.0
    record(sigma=sigma, height=h, mass_ratio=ratio)
    assert abs(ratio - 1) < 0.01, ratio


def test_overflow_warning_names_needed_size(record):
    sim = H.make_sim(source=(30.0, 64.0, 10.0), U=1.0, max_puffs=100)
    with pytest.warns(RuntimeWarning, match=r"max_puffs \(env N_PUFFS\) >= (\d+)") as w:
        for _ in range(int(60 / sim.frame_dt)):
            sim.step()
    msg = str(w[0].message)
    need = int(msg.split(">= ")[1].rstrip("."))
    record(message=msg, needed=need)
    assert need > sim.max_puffs
