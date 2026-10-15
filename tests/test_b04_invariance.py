"""B4: linearity and invariances (no obstacles, steady uniform wind)."""
import numpy as np
import pytest

from tests import helpers as H

STEADY = dict(steady_wind=True)


def _run(src, Q=1.0, U=3.0, T=60.0, seed=4, **kw):
    sim = H.make_sim(source=src, Q=Q, U=U, seed=seed, **{**STEADY, **kw})
    for _ in range(int(T / sim.frame_dt)):
        sim.step()
    return sim


def _grid(src, dx=(0, 60), dy=(-15, 15), z=(0, 25)):
    xs = np.linspace(src[0] + dx[0], src[0] + dx[1], 31); ys = np.linspace(src[1] + dy[0], src[1] + dy[1], 16)
    zs = np.linspace(*z, 11)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(np.float32)


def test_linearity_in_Q(record):
    src = (60.0, 64.0, 10.0); S = _grid(src)
    c1 = _run(src, Q=1.0).query_concentration(S)
    c2 = _run(src, Q=2.0).query_concentration(S)
    c0 = _run(src, Q=0.0).query_concentration(S)
    record(max_C=float(c1.max()), max_abs_dev=float(np.abs(c2 - 2 * c1).max()))
    np.testing.assert_array_equal(c2, 2 * c1)
    assert np.all(c0 == 0.0)


def test_superposition_kernel(record):
    """The sim has one source; superposition is checked on the concentration
    kernel: puffs of two single-source runs injected together = sum of each."""
    a = _run((60.0, 50.0, 10.0)); b = _run((80.0, 80.0, 5.0), seed=9)
    pa, _, ma, aa = H.puffs(a); pb, _, mb, ab = H.puffs(b)
    S = _grid((60.0, 64.0, 10.0), dy=(-30, 30))
    ca, cb = a.query_concentration(S).astype(float), b.query_concentration(S).astype(float)
    u = H.make_sim(max_puffs=len(pa) + len(pb), **STEADY)
    u.sim_time = a.sim_time
    u.inject_puffs(np.r_[pa, pb], ages=np.r_[aa, ab], masses=np.r_[ma, mb])
    cu = u.query_concentration(S).astype(float)
    m = (ca + cb) > 1e-6 * (ca + cb).max()
    rel = np.abs(cu - (ca + cb))[m] / (ca + cb)[m]
    record(max_rel_err=float(rel.max()), note="single-source sim: superposition tested on the kernel")
    assert rel.max() < 1e-5


def test_translation_invariance(record):
    src = np.array([60.0, 50.0, 10.0]); d = np.array([40.0, 20.0, 0.0])
    S = _grid(src)
    c1 = _run(tuple(src)).query_concentration(S).astype(float)
    c2 = _run(tuple(src + d)).query_concentration((S + d).astype(np.float32)).astype(float)
    m = c1 > 1e-6 * c1.max()
    rel = np.abs(c2 - c1)[m] / c1[m]
    record(max_rel_err=float(rel.max()), median_rel_err=float(np.median(rel)))
    assert rel.max() < 1e-3, rel.max()


@pytest.mark.slow
def test_rotation_90(record):
    """Wind +x vs +y with the geometry rotated: mean plume within 5%."""
    kw = {"sim_kwargs": {"steady_wind": True, "shared_eddies": False}, "avg": 1800.0}
    r0, r90 = H.run_parallel([("plume_along", dict(kw, angle_deg=0.0, source=[60.0, 64.0, 10.0])),
                              ("plume_along", dict(kw, angle_deg=90.0, source=[128.0, 20.0, 10.0]))])
    dev = {k: abs(r90[k] / r0[k] - 1) for k in ("sigma_y", "sigma_z", "c_centre")}
    record(plus_x=r0, plus_y=r90, rel_dev=dev)
    assert all(v < 0.05 for v in dev.values()), dev
