"""Long-running jobs executed in subprocesses by helpers.run_parallel.
Usage: python tests/jobs.py <function> '<json kwargs>'  → prints one JSON line."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation"))
import warp as wp  # noqa: E402

wp.config.quiet = True
import stable_fluid_3d as sf  # noqa: E402
from tests import helpers as H  # noqa: E402


def _gauss_fit(s, c):
    import validate_sim as V
    return V.fit_gauss(s, c)


def plume_along(angle_deg=0.0, U=3.0, dist=50.0, avg=1200.0, spin=150.0, seed=1, source=(128.0, 64.0, 10.0),
                pg="D", sim_kwargs=None):
    """Mean plume at `dist` m downwind along the mean-wind direction: σ_y (Gaussian fit),
    σ_z (ground-reflected fit, h fixed) and C at the fitted centre at source height."""
    import validate_sim as V
    sim = H.make_sim(source=source, U=U, angle=angle_deg, seed=seed, pg=pg, max_puffs=4096, **(sim_kwargs or {}))
    a = np.radians(angle_deg)
    w = np.array([np.cos(a), np.sin(a), 0.0]); n = np.array([-np.sin(a), np.cos(a), 0.0])
    ss = np.arange(-30.0, 30.0 + 0.25, 0.5); zs = np.arange(0.0, 50.0 + 0.25, 0.5)
    S, Z = np.meshgrid(ss, zs, indexing="ij")
    P = (np.array(source) + dist * w + S[..., None] * n + Z[..., None] * np.array([0, 0, 1.0])).reshape(-1, 3).astype(np.float32)
    acc = np.zeros(len(P)); k = [0]
    V.run(sim, spin)

    def cb():
        acc[:] += sim.query_concentration(P); k[0] += 1
    V.run(sim, avg, every=0.5, cb=cb)
    C = (acc / k[0]).reshape(S.shape)
    kz = int(np.argmin(abs(zs - source[2])))
    sy, sc = V.fit_gauss(ss, C[:, kz])
    js = int(np.argmin(abs(ss - sc)))
    sz = V.fit_reflected(zs, C[js, :], source[2])
    return {"angle": angle_deg, "U": U, "sigma_y": sy, "sigma_z": sz, "c_centre": float(C[js, kz]),
            "centre_offset": sc, "U_src": sim.U_src, "n_overwritten": sim.n_overwritten}


def strong_wind_tunnel(U=8.0, T=600.0, seed=1):
    """Thin wall across the plume; per-step tunnelling check, stability over T seconds."""
    from gaussian_puff_env import CuboidObstacle
    wall = CuboidObstacle(center=np.array([100.0, 64.0, 15.0]), half_extents=np.array([1.0, 20.0, 15.0]))
    sim = H.make_sim(source=(60.0, 64.0, 10.0), U=U, seed=seed, obstacles=((), (wall,)))
    c, h = wall.center, wall.half_extents
    prev = None; tunnels = inside = 0; nan = False; ke = []
    for i in range(int(T / sim.frame_dt)):
        sim.step()
        act = sim.puff_active.numpy() == 1
        pos = sim.puff_pos.numpy().astype(float)
        if prev is not None:
            m = act & prev[1]
            p0, p1 = prev[0][m], pos[m]
            cross = H.seg_hits_box64(p0, p1, c, h) & (np.sign(p0[:, 0] - c[0]) != np.sign(p1[:, 0] - c[0]))
            tunnels += int(cross.sum())
        inside += int((H.sdf_box(pos[act], c, h) < 0).sum())
        prev = (pos, act)
        if i % 300 == 0:
            u = sim.u0.numpy()
            nan |= bool(~np.isfinite(u).all() | ~np.isfinite(pos[act]).all())
            ke.append(float((u ** 2).sum(-1).mean()))
    return {"U": U, "tunnels": tunnels, "inside_logged": inside, "nan": nan, "ke_first": ke[1], "ke_last": ke[-1],
            "ke_max": max(ke), "n_exited": sim.n_exited, "n_emitted": sim.n_emitted}


def v3_class(pg="D", seed=1, avg=3600.0, sim_kwargs=None):
    import validate_sim as V
    d, series, sim = V.v3_run(pg, avg=avg, seed=seed, record_series=True, sim_kwargs=sim_kwargs or {})
    rows = [{k: r[k] for k in ("x_minus_xs", "sigma_y_sim", "sigma_z_sim", "ratio_y", "ratio_z")} for r in d["rows"]]
    ts = series[:, :len(V.V4_X)].astype(float)
    gam = [float(np.mean(ts[:, j] > 0.1 * ts[:, j].mean())) for j in range(ts.shape[1])]
    return {"class": pg, "seed": seed, "rows": rows, "U10": d["setup"]["U10_realised"], "pass": d["pass"],
            "centreline_intermittency": gam}


if __name__ == "__main__":
    fn, kw = sys.argv[1], json.loads(sys.argv[2])
    res = globals()[fn](**kw)
    print(json.dumps(res, default=float))
