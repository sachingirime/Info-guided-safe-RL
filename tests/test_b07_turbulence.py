"""B7: turbulence model statistics."""
import numpy as np
import pytest
from scipy.stats import chisquare

from tests import helpers as H

FAST_TL = {"D": {"TL_y": 20.0, "TL_z": 10.0}}       # short T_L so lags fit mid-domain


def test_stationary_ou_statistics(record):
    """Per-axis variance = σ² within 2%; autocorrelation at lags {0.5, 1, 2}·T_L
    = e^{−lag/T_L} within 0.03."""
    N = 100_000
    sim = H.make_sim(U=3.0, max_puffs=N, steady_wind=True, lsm_table=FAST_TL)
    sig, TL = np.array(sim.lsm_sigma), np.array(sim.lsm_TL)
    sim.inject_puffs(np.tile([128.0, 64.0, 64.0], (N, 1)))
    v0 = sim.puff_vel.numpy().astype(float)
    lags = sorted({round(f * T, 6) for f in (0.5, 1.0, 2.0) for T in TL})
    t, var_err, rho_err, rows = 0.0, 0.0, 0.0, []
    for lag in lags:
        while t < lag - 1e-9:
            sim.step_puffs(); sim.sim_time += sim.sim_dt; t += sim.sim_dt
        assert (sim.puff_active.numpy() == 1).all()
        v = sim.puff_vel.numpy().astype(float)
        vr = v.var(0) / sig ** 2
        rho = np.array([np.corrcoef(v0[:, a], v[:, a])[0, 1] for a in range(3)])
        ref = np.exp(-t / TL)
        on = np.isin(np.round(t / TL, 2), (0.5, 1.0, 2.0))        # axes for which this lag is a test lag
        var_err = max(var_err, float(np.abs(vr - 1).max()))
        rho_err = max(rho_err, float(np.abs(rho - ref)[on].max()) if on.any() else 0.0)
        rows.append({"lag": t, "var_ratio": vr, "rho": rho, "rho_ref": ref})
    record(sigma=sig, TL=TL, rows=rows, max_var_err=var_err, max_rho_err=rho_err)
    assert var_err < 0.02 and rho_err < 0.03


def test_well_mixed_closed_box(record):
    """Thomson (1987) well-mixed test: a uniform puff field in a closed,
    reflecting box (no wind, no emission) stays uniform after 10·T_L."""
    N = 200_000
    table = {"D": {"a_y": 0.5, "a_z": 0.5, "TL_y": 5.0, "TL_z": 5.0}}   # σ = 1.5 m/s, T_L = 5 s
    sim = H.make_sim(U=3.0, max_puffs=N, steady_wind=True, lsm_table=table, closed_domain=True)
    rng = np.random.default_rng(2)
    sim.inject_puffs(rng.uniform(0, 1, (N, 3)) * H.DOMAIN)
    TL = max(sim.lsm_TL)
    for _ in range(int(round(10 * TL / sim.sim_dt))):
        sim.step_puffs(); sim.sim_time += sim.sim_dt
    pos = sim.puff_pos.numpy().astype(float)
    hist, _ = np.histogramdd(pos, bins=(10, 10, 10), range=[(0, d) for d in H.DOMAIN])
    stat, p = chisquare(hist.ravel())
    face = hist[[0, -1], :, :].mean() / hist.mean(), hist[:, [0, -1], :].mean() / hist.mean(), hist[:, :, [0, -1]].mean() / hist.mean()
    record(chi2=float(stat), p_value=float(p), active=int((sim.puff_active.numpy() == 1).sum()),
           boundary_bin_ratio_xyz=face, rms_displacement_m=float(np.sqrt(2 * 1.5 ** 2 * TL * 10 * TL)))
    assert (sim.puff_active.numpy() == 1).all()
    assert p > 0.01, (stat, p, face)


@pytest.mark.slow
def test_class_ordering(record):
    """At x = 100 m and the same realised U10: σy, σz decrease from A to F.
    Compared on 3-seed means: single 3600 s runs of the long-T_E classes scatter
    too much (σy CoV ≈ 29% for A, 19% for B) to order A and B reliably."""
    seeds = (1, 2, 3)
    res = H.run_parallel([("v3_class", {"pg": c, "seed": s, "avg": 3600.0}) for c in "ABCDEF" for s in seeds])
    at = lambda r, k: next(x[k] for x in r["rows"] if x["x_minus_xs"] == 100.0)
    per = {c: [r for r in res if r["class"] == c] for c in "ABCDEF"}
    sy = [float(np.mean([at(r, "sigma_y_sim") for r in per[c]])) for c in "ABCDEF"]
    sz = [float(np.mean([at(r, "sigma_z_sim") for r in per[c]])) for c in "ABCDEF"]
    gam = [float(np.mean([r["centreline_intermittency"][2] for r in per[c]])) for c in "ABCDEF"]
    record(classes=list("ABCDEF"), seeds=list(seeds), sigma_y_100m=sy, sigma_z_100m=sz, intermittency_100m=gam,
           sigma_y_100m_per_seed={c: [at(r, "sigma_y_sim") for r in per[c]] for c in "ABCDEF"},
           U10=[r["U10"] for r in res])
    assert all(a > b for a, b in zip(sy, sy[1:])), sy
    assert all(a > b for a, b in zip(sz, sz[1:])), sz


@pytest.mark.slow
def test_seed_spread_class_D(record):
    """V3 class D over 10 seeds: coefficient of variation of σy, σz per distance
    (noise floor for the GADEN benchmark).  Report only."""
    res = H.run_parallel([("v3_class", {"pg": "D", "seed": s, "avg": 3600.0}) for s in range(1, 11)])
    xs = [r["x_minus_xs"] for r in res[0]["rows"]]
    sy = np.array([[r["sigma_y_sim"] for r in x["rows"]] for x in res])
    sz = np.array([[r["sigma_z_sim"] for r in x["rows"]] for x in res])
    ry = np.array([[r["ratio_y"] for r in x["rows"]] for x in res]); rz = np.array([[r["ratio_z"] for r in x["rows"]] for x in res])
    record(x=xs, cov_sigma_y=(sy.std(0, ddof=1) / sy.mean(0)).tolist(), cov_sigma_z=(sz.std(0, ddof=1) / sz.mean(0)).tolist(),
           mean_ratio_y=ry.mean(0).tolist(), mean_ratio_z=rz.mean(0).tolist(),
           n_seeds_pass=int(sum(r["pass"] for r in res)), U10=[r["U10"] for r in res])
    assert np.isfinite(sy).all() and np.isfinite(sz).all()
