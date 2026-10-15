"""B8: obstacles, bounce and line of sight."""
import numpy as np
import pytest
import warp as wp

import stable_fluid_3d as sf
from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle
from tests import helpers as H

MARGIN = 2.0
ZERO_TURB = {"D": {"a_y": 0.0, "a_z": 0.0}}


@pytest.fixture(scope="module")
def obstacle_run():
    """Env obstacle layout (fixed + random), env source, 300 s; positions every step."""
    env = GaussianPuffEnv(seed=0, warmup_s=0.0)
    env.reset(seed=5)
    sph = [o for o in env.obstacles if isinstance(o, SphereObstacle)]
    box = [o for o in env.obstacles if isinstance(o, CuboidObstacle)]
    sim = H.make_sim(source=tuple(env.theta), U=3.0, angle=float(np.degrees(np.arctan2(64 - env.theta[1], 128 - env.theta[0]))),
                     obstacles=(sph, box))
    inside = crossings = 0
    prev = None; hist = {}; umean = np.zeros(sim.wind_shape + (3,)); nu = 0
    for i in range(int(300 / sim.frame_dt)):
        sim.step()
        act = sim.puff_active.numpy() == 1
        pos = sim.puff_pos.numpy().astype(float)
        inside += int((H.sdf_all(pos[act], sph, box) < 0).sum())
        if prev is not None:
            m = act & prev[1] & (sim.puff_emit_time.numpy() == prev[2])
            p0, p1 = prev[0][m], pos[m]
            hit = np.zeros(len(p0), bool)
            for s in sph:   # a single step whose segment passes through a solid
                c = np.asarray(s.center, float); d = p1 - p0
                t = np.clip(((c - p0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-18), 0, 1)
                hit |= np.linalg.norm(p0 + t[:, None] * d - c, axis=1) < float(s.radius) - 1e-3
            for b in box:
                hit |= H.seg_hits_box64(p0, p1, np.asarray(b.center, float), np.asarray(b.half_extents, float) - 1e-3)
            crossings += int(hit.sum())
        prev = (pos, act, sim.puff_emit_time.numpy().copy())
        if i % 30 == 0 and i * sim.frame_dt >= 60.0:          # time-mean wind after 60 s
            umean += sim.u0.numpy(); nu += 1
        if i % 30 == 0:                                       # 1 s log for the stuck test
            et = sim.puff_emit_time.numpy()
            for k in np.where(act)[0]:
                hist.setdefault((int(k), float(et[k])), []).append(pos[k].copy())
    return {"sim": sim, "sph": sph, "box": box, "inside": inside, "crossings": crossings, "hist": hist,
            "u": sim.u0.numpy(), "u_mean": umean / nu, "mask": sim.obstacle_mask.numpy()}


def test_no_puffs_inside_obstacles(obstacle_run, record):
    record(puffs_inside_any_step=obstacle_run["inside"], n_obstacles=len(obstacle_run["sph"]) + len(obstacle_run["box"]))
    assert obstacle_run["inside"] == 0


def test_no_puff_crosses_an_obstacle(obstacle_run, record):
    record(crossing_steps=obstacle_run["crossings"])
    assert obstacle_run["crossings"] == 0


def test_no_stuck_puffs(obstacle_run, record):
    """Stuck = within margin+0.1 m of a surface with < 0.1 m net displacement over 10 s."""
    sph, box = obstacle_run["sph"], obstacle_run["box"]
    stuck = near = 0
    for traj in obstacle_run["hist"].values():
        P = np.array(traj)
        if len(P) < 11:
            continue
        d = H.sdf_all(P, sph, box)
        for j in range(len(P) - 10):
            if np.all(d[j:j + 11] <= MARGIN + 0.1):
                near += 1
                if np.linalg.norm(P[j + 10] - P[j]) < 0.1:
                    stuck += 1
                    break
    record(stuck_puffs=stuck, near_surface_10s_windows=near)
    assert stuck == 0


@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_bounce_reverses_normal_velocity(shape, record):
    sim = H.make_sim(max_puffs=1, U=3.0, steady_wind=True, lsm_table=ZERO_TURB, sigma_min=0.0, source=(5.0, 5.0, 5.0),
                     obstacles=(((SphereObstacle(np.array([60.0, 64.0, 20.0]), 5.0),), ()) if shape == "sphere"
                                else ((), (CuboidObstacle(np.array([60.0, 64.0, 20.0]), np.array([5.0, 5.0, 5.0])),))))
    n = np.array([-1.0, 0.0, 0.0])                         # outward normal of the face/side at x = 55
    p = np.array([55.0 - 1.5, 64.0, 20.0])                  # 1.5 m outside (< 2 m margin)
    v = np.array([1.0, 0.3, 0.0])                           # moving into the obstacle
    sim.inject_puffs([p], velocities=[v])
    sim.step_puffs()
    v1 = sim.puff_vel.numpy()[0].astype(float)
    # expected: OU decay (σ = 0), move, then reflect about the normal at the moved position
    vd = v * np.exp(-sim.sim_dt / np.array(sim.lsm_TL))
    x1 = p + vd * sim.sim_dt
    if shape == "sphere":
        n = (x1 - np.array([60.0, 64.0, 20.0])); n /= np.linalg.norm(n)
    v_exp = vd - 2 * (vd @ n) * n
    record(v_before=v, v_after=v1, v_expected=v_exp)
    assert vd @ n < 0 < v1 @ n                                # was moving in, now moving out
    assert np.allclose(v1, v_exp, atol=1e-5), (v1, v_exp)


@pytest.mark.parametrize("case", ["sphere_centre", "box_centre", "overlap_crevice", "two_spheres"])
def test_degenerate_geometry(case, record):
    S = SphereObstacle(np.array([60.0, 64.0, 20.0]), 5.0)
    geo = {"sphere_centre": (([S], []), [60.0, 64.0, 20.0]),
           "box_centre": (([], [CuboidObstacle(np.array([100.0, 64.0, 20.0]), np.array([4.0, 4.0, 4.0]))]), [100.0, 64.0, 20.0]),
           "overlap_crevice": (([SphereObstacle(np.array([60.0, 64.0, 30.0]), 5.0)],
                                [CuboidObstacle(np.array([60.0, 64.0, 22.0]), np.array([10.0, 10.0, 4.0]))]), [60.0, 64.0, 25.5]),
           "two_spheres": (([S, SphereObstacle(np.array([66.0, 64.0, 20.0]), 5.0)], []), [63.0, 64.0, 20.0])}
    (sph, box), p = geo[case]
    sim = H.make_sim(max_puffs=1, U=3.0, steady_wind=True, lsm_table=ZERO_TURB, sigma_min=0.0, source=(5.0, 5.0, 5.0),
                     obstacles=(sph, box))
    sim.inject_puffs([p], velocities=[(0.0, 0.0, 0.0)])
    sim.step_puffs()
    q = sim.puff_pos.numpy()[0].astype(float)
    d = float(H.sdf_all(q, sph, box)[0])
    record(final_position=q, signed_distance=d)
    assert np.isfinite(q).all() and d >= 0.0, (q, d)


@wp.kernel
def _seg_tests(p0: wp.array(dtype=wp.vec3), p1: wp.array(dtype=wp.vec3), c: wp.array(dtype=wp.vec3),
               r: wp.array(dtype=float), h: wp.array(dtype=wp.vec3), out_s: wp.array(dtype=int), out_b: wp.array(dtype=int)):
    i = wp.tid()
    out_s[i] = 1 if sf.seg_hits_sphere(p0[i], p1[i], c[i], r[i]) else 0
    out_b[i] = 1 if sf.seg_hits_box(p0[i], p1[i], c[i], h[i]) else 0


def test_los_vs_float64(record):
    """GPU segment–sphere/box tests vs float64 on 8000 random configurations
    plus 1000 exactly tangent sphere and 1000 corner-grazing box segments:
    mismatches only at tangency (the float64 result flips within ±1e-4 of the
    radius/half-extent), and < 0.1% of the random configurations."""
    rng = np.random.default_rng(7)
    N = 10_000
    c = rng.uniform(20, 100, (N, 3)); r = rng.uniform(1, 10, N); h = rng.uniform(1, 10, (N, 3))
    p0 = c + rng.normal(0, 15, (N, 3)); p1 = c + rng.normal(0, 15, (N, 3))
    # tangent spheres: segment perpendicular to a radius, touching at distance r
    k = 1000
    u = rng.normal(size=(k, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(u, rng.normal(size=(k, 3))); w /= np.linalg.norm(w, axis=1, keepdims=True)
    touch = c[:k] + r[:k, None] * u
    p0[:k] = touch - 8 * w; p1[:k] = touch + 8 * w
    # corner/edge-grazing boxes: segment through a box corner
    corner = c[k:2 * k] + h[k:2 * k] * np.sign(rng.normal(size=(k, 3)))
    d = rng.normal(size=(k, 3)); p0[k:2 * k] = corner - 6 * d; p1[k:2 * k] = corner + 6 * d
    A = lambda x: wp.array(x.astype(np.float32), dtype=wp.vec3)
    out_s = wp.zeros(N, dtype=int); out_b = wp.zeros(N, dtype=int)
    wp.launch(_seg_tests, dim=N, inputs=[A(p0), A(p1), A(c), wp.array(r.astype(np.float32), dtype=float), A(h), out_s, out_b])
    gs, gb = out_s.numpy().astype(bool), out_b.numpy().astype(bool)
    # float64 reference on the float32-rounded inputs
    f = lambda x: x.astype(np.float32).astype(float)
    rs, rb = H.seg_hits_sphere64(f(p0), f(p1), f(c), f(r)), H.seg_hits_box64(f(p0), f(p1), f(c), f(h))
    eps = 1e-4
    tang_s = H.seg_hits_sphere64(f(p0), f(p1), f(c), f(r) - eps) != H.seg_hits_sphere64(f(p0), f(p1), f(c), f(r) + eps)
    tang_b = H.seg_hits_box64(f(p0), f(p1), f(c), f(h) - eps) != H.seg_hits_box64(f(p0), f(p1), f(c), f(h) + eps)
    ms, mb = gs != rs, gb != rb
    rnd = np.arange(N) >= 2 * k                     # random configurations (the first 2k are constructed grazing cases)
    rate = (ms[rnd].sum() + mb[rnd].sum()) / (2 * rnd.sum())
    record(n=N, n_random=int(rnd.sum()), random_mismatch_rate=float(rate),
           tangent_sphere_mismatch=int(ms[:k].sum()), corner_box_mismatch=int(mb[k:2 * k].sum()),
           sphere_mismatch_not_tangent=int((ms & ~tang_s).sum()), box_mismatch_not_tangent=int((mb & ~tang_b).sum()),
           sphere_hits=int(rs.sum()), box_hits=int(rb.sum()))
    assert (ms & ~tang_s).sum() == 0 and (mb & ~tang_b).sum() == 0     # mismatches only at tangency
    assert rate < 1e-3, rate


def test_wind_around_obstacles(obstacle_run, record):
    u, mask = obstacle_run["u"], obstacle_run["mask"] == 1
    sim = obstacle_run["sim"]
    speed = np.linalg.norm(u, axis=-1)
    # net flux into solid cells through faces shared with fluid cells (collocated grid:
    # face velocity = average of the two cells, solid cell = 0)
    flux = area = 0.0
    for ax in range(3):
        a = np.moveaxis(mask, ax, 0); va = np.moveaxis(u[..., ax], ax, 0)
        for sgn in (+1, -1):          # fluid cell at index j, solid at j+sgn
            fl = ~a[:-1] if sgn > 0 else ~a[1:]
            so = a[1:] if sgn > 0 else a[:-1]
            vf = va[:-1] if sgn > 0 else va[1:]
            sel = fl & so
            flux += float((0.5 * vf[sel] * sgn).sum()); area += float(sel.sum())
    rel_flux = abs(flux) / (sim.wind_base_speed * max(area, 1))
    # wake: time-mean speed 4–12 m downwind of each obstacle at its mid-height vs
    # the freestream (cells > 20 m from any obstacle) at the same height
    um = obstacle_run["u_mean"]; spm = np.linalg.norm(um, axis=-1)
    w = np.array([um[..., 0][~mask].mean(), um[..., 1][~mask].mean(), 0.0]); w /= np.linalg.norm(w)
    dx = sim.dx
    centres = (np.indices(mask.shape).transpose(1, 2, 3, 0) + 0.5) * dx
    dist = H.sdf_all(centres.reshape(-1, 3), obstacle_run["sph"], obstacle_run["box"]).reshape(mask.shape)
    wake = []
    for o in obstacle_run["sph"] + obstacle_run["box"]:
        zc = float(o.center[2]) if hasattr(o, "radius") else 0.5 * float(o.center[2] + o.half_extents[2])
        k = int(zc // dx)
        ext = (float(np.sqrt(max(o.radius ** 2 - (zc - o.center[2]) ** 2, 0.0))) if hasattr(o, "radius")
               else float(np.abs(np.asarray(o.half_extents)[:2]) @ np.abs(w[:2])))
        pts = [np.asarray(o.center, float)[:2] + (ext + s_) * w[:2] for s_ in (4.0, 8.0, 12.0)]
        ij = [(int(p[0] // dx), int(p[1] // dx)) for p in pts if 0 <= p[0] < 256 and 0 <= p[1] < 128]
        if not ij:
            continue
        v_wake = float(np.mean([spm[i, j, k] for i, j in ij]))
        free = spm[:, :, k][dist[:, :, k] > 20.0].mean()
        wake.append((v_wake, float(free)))
    record(max_speed_inside=float(speed[mask].max()), net_flux_into_solids_rel=rel_flux, solid_faces=area,
           wake_vs_free=wake)
    assert speed[mask].max() == 0.0
    assert rel_flux < 0.05, rel_flux
    assert wake and all(vw < vf for vw, vf in wake), wake
