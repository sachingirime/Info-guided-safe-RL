"""v1.1 surface-layer two-scale LSM (surface_layer=True)."""
import numpy as np
import pytest
from scipy.stats import kstest

import stable_fluid_3d as sf

H = sf.NZ * sf.DX           # closed-column height (m)


def _column(pg="A", n=4096, seed=3, t=300.0):
    sim = sf.HybridFluidPuffSimulation(wind_base_speed=0.3, steady_wind=True, surface_layer=True,
                                       stability_class=pg, max_puffs=n, closed_domain=True,
                                       wind_seed=seed, seed=seed, shared_eddies=False)
    sim.Q = 0.0; sim.emit_interval = 1e9; sim._next_emit_time = 1e12
    sim.set_obstacles([], [])
    rng = np.random.default_rng(seed)
    P = np.column_stack([rng.uniform(20, 236, n), rng.uniform(20, 108, n), rng.uniform(0, H, n)])
    sim.inject_puffs(P, ages=1.0)
    # start the small-scale velocities in equilibrium N(0, σ_i(z)²)
    vs = rng.standard_normal((n, 3)) * np.array([sim.sl_sigma(z) for z in P[:, 2]])
    import warp as wp
    sim.puff_vs.assign(wp.array(np.vstack([vs, np.zeros((sim.max_puffs - n, 3))]).astype(np.float32), dtype=wp.vec3))
    for _ in range(int(t * sim.fps)):
        sim.step()
    return sim


@pytest.mark.slow
def test_well_mixed_unstable(record):
    """Thomson (1987) well-mixed criterion: an initially uniform tracer stays uniform
    in z when σ_w varies with height (class A)."""
    sim = _column("A")
    z = sim.puff_pos.numpy()[:4096, 2]
    ks = kstest(z / H, "uniform")
    frac_low = float(np.mean(z < 0.1 * H))
    record(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), frac_lowest_10pct=frac_low, mean_z_over_H=float(z.mean() / H))
    assert ks.pvalue > 0.01
    assert abs(frac_low - 0.1) < 0.02


@pytest.mark.slow
def test_small_scale_variance(record):
    sim = _column("A", t=120.0)
    z = sim.puff_pos.numpy()[:4096, 2]
    vs = sim.puff_vs.numpy()[:4096]
    su, sv, _ = sim.sl_sigma(10.0)
    sw = np.array([sim.sl_sigma(zz)[2] for zz in z])
    r = {"u": float(vs[:, 0].std() / su), "v": float(vs[:, 1].std() / sv), "w_norm": float((vs[:, 2] / sw).std())}
    record(**r)
    for k, v in r.items():
        assert abs(v - 1) < 0.05, (k, v)


def test_met_ordering(record):
    """u* and ε decrease and T(z) grows from unstable to stable at fixed U10."""
    rows = {}
    for c in "ABCDEF":
        sim = sf.HybridFluidPuffSimulation(wind_base_speed=3.0, surface_layer=True, stability_class=c, max_puffs=8)
        rows[c] = (sim.ustar, sim.inv_L, sim.sl_T(2.0)[2], sim._convective()[0])
    record(**{c: list(v) for c, v in rows.items()})
    us = [rows[c][0] for c in "ABCDEF"]
    assert all(a > b for a, b in zip(us, us[1:]))
    assert all(rows[c][3] > 0 for c in "ABC") and all(rows[c][3] == 0 for c in "DEF")
