"""B1: wind extremes."""
import numpy as np
import pytest

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv
from tests import helpers as H


@pytest.mark.parametrize("U", [0.0, 0.05, 0.2])
def test_calm_wind(U, record):
    """No NaN/Inf; σ floor and finite T_E; plume grows; mass balance."""
    env = GaussianPuffEnv(seed=0, warmup_s=60.0)
    env.fluid_sim.wind_base_speed = U
    env.reset(seed=1)
    sim = env.fluid_sim
    obs_ok = True
    for k in range(600):                                   # 60 s of env time
        obs, *_ = env.step(np.zeros(3, np.float32))
        obs_ok &= bool(np.isfinite(obs).all() and (obs >= 0).all() and (obs <= 1).all())
    pos, vel, _, ages = H.puffs(sim)
    # growth: total spread of same-age puffs (centroid scatter + puff size,
    # r² = |x − x̄|² + 3σ_puff²) increases with age.  Empty bins (puffs that left
    # the domain: the calm-wind floor of the wind model is 0.3 m/s) are skipped.
    spreads = []
    for lo in (0.0, 30.0, 60.0, 90.0):
        m = (ages >= lo) & (ages < lo + 10.0)
        if m.sum() < 5:
            continue
        r2 = ((pos[m] - pos[m].mean(0)) ** 2).sum(1) + 3.0 * sim.puff_sigma(ages[m])[:, 0] ** 2
        spreads.append(float(np.sqrt(r2.mean())))
    C = sim.query_concentration(np.c_[np.full(50, sim.source_pos[0]), np.full(50, sim.source_pos[1]),
                                      np.linspace(0, 40, 50)].astype(np.float32))
    sig, TL, TE = np.array(sim.lsm_sigma), np.array(sim.lsm_TL), np.array(sim.lsm_TE)
    record(U=U, U_src=sim.U_src, lsm_sigma=sig, lsm_TL=TL, lsm_TE=TE, spread_xyz=spreads,
           n_active=len(pos), emitted=sim.n_emitted, exited=sim.n_exited, overwritten=sim.n_overwritten)
    assert obs_ok
    for a in (sim.u0.numpy(), pos, vel, C):
        assert np.isfinite(a).all()
    assert np.all(sig >= sim.sigma_min) and np.isfinite(TL).all() and np.isfinite(TE).all()
    assert len(spreads) >= 3 and all(b > a for a, b in zip(spreads, spreads[1:])), spreads   # grows with age
    assert sim.n_emitted == len(pos) + sim.n_exited + sim.n_overwritten


@pytest.mark.slow
def test_strong_wind(record):
    """U = 8, 12, 15 m/s: stable 600 s, no tunnelling through a 2 m wall,
    centreline C·U at 50 m constant within 10%."""
    jobs = [("strong_wind_tunnel", {"U": U}) for U in (8.0, 12.0, 15.0)]
    jobs += [("plume_along", {"U": U, "avg": 1800.0, "source": [60.0, 64.0, 10.0]}) for U in (8.0, 12.0, 15.0)]
    res = H.run_parallel(jobs)
    tun, plume = res[:3], res[3:]
    cu = np.array([p["c_centre"] * p["U"] for p in plume])
    record(tunnel=tun, plume=plume, c_times_u=cu, cu_spread=float(cu.max() / cu.min() - 1))
    for t in tun:
        assert not t["nan"] and t["tunnels"] == 0 and t["inside_logged"] == 0, t
        assert t["ke_last"] < 2.0 * t["ke_first"], t                     # bounded
    assert cu.max() / cu.min() - 1 < 0.10, cu


@pytest.mark.slow
def test_direction_invariance(record):
    """σ_y, σ_z, centreline C at 50 m agree across 8 directions within 10%
    (steady wind, independent puff noise: isolates grid anisotropy)."""
    kw = {"sim_kwargs": {"steady_wind": True, "shared_eddies": False}, "avg": 1200.0}
    res = H.run_parallel([("plume_along", dict(kw, angle_deg=a)) for a in range(0, 360, 45)])
    sy = np.array([r["sigma_y"] for r in res]); sz = np.array([r["sigma_z"] for r in res])
    cc = np.array([r["c_centre"] for r in res])
    spread = {k: float(v.max() / v.min() - 1) for k, v in (("sigma_y", sy), ("sigma_z", sz), ("c_centre", cc))}
    record(results=res, spread=spread)
    assert all(v < 0.10 for v in spread.values()), spread


def test_sudden_direction_change(record):
    """Rotate the mean wind 90° mid-run: no NaN, no teleporting, plume re-forms."""
    sim = H.make_sim(source=(128.0, 64.0, 10.0), U=3.0, angle=0.0)
    dt = sim.frame_dt
    sig = max(sim.lsm_sigma)
    max_ratio = 0.0; prev = None
    for i in range(int(240 / dt)):
        if i == int(120 / dt):
            sim.set_wind_direction(90.0)
        sim.step()
        act = sim.puff_active.numpy() == 1
        pos = sim.puff_pos.numpy().astype(float)
        assert np.isfinite(pos[act]).all()
        if prev is not None:
            m = act & prev[1]
            if m.any():
                umax = np.linalg.norm(sim.u0.numpy(), axis=-1).max()
                disp = np.linalg.norm(pos[m] - prev[0][m], axis=1).max()
                max_ratio = max(max_ratio, disp / ((umax + 5 * sig) * dt))
        prev = (pos, act)
    pos, vel, _, ages = H.puffs(sim)
    young = ages < 30.0
    d = pos[young] - sim.source_pos
    ang = np.degrees(np.arctan2(d[:, 1].mean(), d[:, 0].mean()))
    record(max_disp_over_bound=max_ratio, young_puff_direction_deg=ang)
    assert max_ratio <= 1.0, max_ratio
    assert abs(ang - 90.0) < 20.0, ang
