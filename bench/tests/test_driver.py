"""Driver unit checks (plan Step 1): transport, kernel, growth law, mass, seeding."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gaden_client as gc

DIMS = (32, 16, 16)
DX = 4.0
SRC = (20.0, 32.0, 30.0)
Q = 1e-6   # kg/s: tiny tracer rate so GADEN's ppm-proportional buoyancy is negligible


@pytest.fixture
def env_bin(tmp_path):
    p = str(tmp_path / "env.bin")
    gc.write_env(p, DIMS, gc.env_cells(np.zeros(DIMS, np.uint8)), DX)
    return p


def uniform(U=3.0):
    u = np.zeros((*DIMS, 3), np.float32)
    u[..., 0] = U
    return u


def make(env_bin, **kw):
    prm = gc.gaden_params(SRC, Q=Q, n_fil_sec=10, sigma0_m=1.0, K_rel=0.05, **kw)
    g = gc.GadenClient(env_bin, prm, threads=4)
    g.set_wind(uniform())
    return g, prm


def test_transport_exact(env_bin):
    g, prm = make(env_bin)
    with g:
        t, n = g.step(50)
        f = g.filaments()
    assert n == 50
    # first filament released at step 1 and moved in that same step: 50 moves
    x = np.sort(f[:, 0])[::-1]
    assert np.allclose(x[0], SRC[0] + 3.0 * 0.1 * 50, atol=1e-3)
    assert np.allclose(f[:, 1], SRC[1], atol=1e-4) and np.allclose(f[:, 2], SRC[2], atol=1e-4)


def test_growth_law(env_bin):
    g, prm = make(env_bin)
    with g:
        g.step(100)
        f = g.filaments()
    oldest = f[np.argmax(f[:, 0])]
    t = 100 * 0.1
    s_ref = np.sqrt(prm["filamentInitialSigma"] ** 2 + prm["filamentGrowthGamma"] * t)
    assert abs(oldest[3] / s_ref - 1) < 2e-3   # Euler on σ: O(γdt/σ²) error


def test_kernel_and_cutoff(env_bin):
    prm = gc.gaden_params(SRC, Q=Q, n_fil_sec=10, sigma0_m=1.0, K_rel=0.05)
    with gc.GadenClient(env_bin, prm, threads=4) as g:
        g.set_wind(uniform())
        g.step(1)
        f = g.filaments()
        assert len(f) == 1
        c0 = f[0, :3].astype(np.float64)
        s = f[0, 3] / 100.0
        r = np.array([0.0, 0.5, 1.0, 2.0, 2.9, 3.1, 4.0]) * s
        pts = c0 + np.stack([np.zeros_like(r), r, np.zeros_like(r)], 1)
        c = g.query(pts)
        ppm_c = prm["filamentPPMcenter_initial"] * (prm["filamentInitialSigma"] / f[0, 3]) ** 3
        ref = ppm_c * np.exp(-0.5 * (r / s) ** 2)
        inside = r < 3 * s
        assert np.allclose(c[inside], ref[inside], rtol=1e-4)
        assert np.all(c[~inside] == 0)


def test_mass_per_filament(env_bin):
    """∫C dV of one filament (kg) = Q/n_fil × P(χ²₃ < 9) (3σ cutoff)."""
    prm = gc.gaden_params(SRC, Q=Q, n_fil_sec=10, sigma0_m=1.0, K_rel=0.05)
    with gc.GadenClient(env_bin, prm, threads=8) as g:
        g.set_wind(uniform())
        g.step(1)
        c0 = g.filaments()[0, :3].astype(np.float64)
        h = 0.1
        ax = np.arange(-3.2, 3.2 + 1e-9, h)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        pts = c0 + np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        m = gc.ppm_to_kgm3(g.query(pts)).sum() * h ** 3
    from scipy.stats import chi2
    expect = Q / 10 * chi2.cdf(9.0, 3)
    assert abs(m / expect - 1) < 0.01, (m, expect)


def test_seed(env_bin):
    def run(seed):
        prm = gc.gaden_params(SRC, Q=Q, n_fil_sec=10, sigma0_m=1.0, K_rel=0.05,
                              noise_std=0.5, seed=seed)
        with gc.GadenClient(env_bin, prm, threads=4) as g:
            g.set_wind(uniform())
            g.step(40)
            return g.filaments()
    a, b, c = run(1), run(1), run(2)
    assert np.array_equal(a, b)
    assert not np.allclose(a[:, 1], c[:, 1])
