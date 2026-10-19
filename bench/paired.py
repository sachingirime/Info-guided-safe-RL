"""
paired.py — run our simulator and GADEN in lockstep on the same (streamed) wind.

Our sim steps at fps (default 30 Hz); every `ratio` frames GADEN receives our
current wind field u0 and advances one step of deltaTime = ratio/fps.  Both
release at the same rate from the same source; concentrations are compared as
C/Q (kg/m³ per kg/s).
"""

import os
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import stable_fluid_3d as sf          # noqa: E402
import gaden_client as gc             # noqa: E402

Q_TRACER = 1e-6        # kg/s for both models (GADEN buoyancy ∝ ppm → negligible)
NO_LSM = {c: {"a_y": 0.0, "a_z": 0.0, "TL_y": 100.0, "TL_z": 100.0} for c in "ABCDEF"}


def make_ours(source, wind_angle=0.0, angle_range=0.0, U=3.0, seed=1, matched=True,
              stability="D", emit_interval=0.1, max_puffs=4096, obstacles=None, **kw):
    """Our simulator.  matched=True → GADEN-equivalent configuration (no LSM
    velocity, no shared eddies, no buoyancy); otherwise the shipped defaults."""
    if matched:
        kw.setdefault("lsm_table", NO_LSM)
        kw.setdefault("sigma_min", 0.0)
        kw.setdefault("shared_eddies", False)
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=wind_angle, wind_angle_range=angle_range,
                                       wind_base_speed=U, wind_seed=seed, max_puffs=max_puffs,
                                       stability_class=stability, **kw)
    sim.Q = Q_TRACER
    sim.sigma_0 = 1.0
    sim.emit_interval = emit_interval
    sp, cu = obstacles or ([], [])
    sim.set_obstacles(sp, cu)
    sim.source_pos = np.array(source, np.float32)
    return sim


def gaden_env(sim, path):
    """Export our obstacle mask → GADEN env (outlets on x+, y±, z+)."""
    mask = sim.obstacle_mask.numpy()
    gc.write_env(path, mask.shape, gc.env_cells(mask), float(sim.dx))
    return path


def free_in_both(sim, pts):
    """Points outside our exact obstacles and in a Free GADEN voxel (not an outlet face)."""
    pts = np.asarray(pts, float)
    ok = np.array([sim.obstacle_sdf(p) > 0 for p in pts]) if (sim.n_spheres + sim.n_cuboids) else np.ones(len(pts), bool)
    mask = sim.obstacle_mask.numpy()
    nx, ny, nz = mask.shape
    idx = np.floor(pts / sim.dx).astype(int)
    inb = np.all((idx >= 0) & (idx < [nx, ny, nz]), 1)
    idx = np.clip(idx, 0, [nx - 1, ny - 1, nz - 1])
    cells = gc.env_cells(mask)
    free = cells[idx[:, 0], idx[:, 1], idx[:, 2]] == 0
    return ok & inb & free


def run_paired(sim, gaden, pts, t_spin, t_avg, sample_dt=1.0, ratio=3, gaden_on=True,
               series_idx=None, series_dt=0.1):
    """Advance both models; return time-mean C/Q at pts and optional series.

    series_idx: indices into pts recorded every series_dt (point series)."""
    fps = sim.fps
    n_frames = int(round((t_spin + t_avg) * fps))
    every_sample = int(round(sample_dt * fps))
    every_series = int(round(series_dt * fps))
    s_ours = np.zeros(len(pts)); s_gad = np.zeros(len(pts)); n_s = 0
    ser_o, ser_g = [], []
    pts32 = np.asarray(pts, np.float32)
    sp = pts32[series_idx] if series_idx is not None else None
    u_sum = np.zeros(sim.wind_shape + (3,)); n_u = 0
    t0 = time.time(); t_g = 0.0
    for f in range(n_frames):
        if gaden_on and f % ratio == 0:
            tg = time.time()
            gaden.set_wind(sim.u0.numpy())
            gaden.step(1)
            t_g += time.time() - tg
        sim.step()
        t = (f + 1) / fps
        if t <= t_spin + 1e-9:
            continue
        if (f + 1) % every_sample == 0:
            s_ours += sim.query_concentration(pts32) / sim.Q
            if gaden_on:
                tg = time.time()
                s_gad += gaden.query_kgm3(pts32) / Q_TRACER
                t_g += time.time() - tg
            u_sum += sim.u0.numpy(); n_u += 1
            n_s += 1
        if sp is not None and (f + 1) % every_series == 0:
            ser_o.append(sim.query_concentration(sp) / sim.Q)
            if gaden_on:
                ser_g.append(gaden.query_kgm3(sp) / Q_TRACER)
    wp.synchronize()
    out = {"mean_ours": s_ours / n_s, "mean_gaden": s_gad / n_s if gaden_on else None,
           "u_mean": u_sum / max(n_u, 1), "n_samples": n_s,
           "wall_total_s": time.time() - t0, "wall_gaden_s": t_g}
    if sp is not None:
        out["series_ours"] = np.array(ser_o)
        out["series_gaden"] = np.array(ser_g) if gaden_on else None
    return out


def trilinear(u, pts, dx):
    """Our cell-centred trilinear sample of a (nx, ny, nz, 3) field at pts (m)."""
    g = np.asarray(pts, float) / dx - 0.5
    n = np.array(u.shape[:3])
    g = np.clip(g, 0, n - 1 - 1e-6)
    i0 = np.floor(g).astype(int); fr = g - i0; i1 = np.minimum(i0 + 1, n - 1)
    out = 0
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                ix = i1[:, 0] if cx else i0[:, 0]; iy = i1[:, 1] if cy else i0[:, 1]; iz = i1[:, 2] if cz else i0[:, 2]
                w = (fr[:, 0] if cx else 1 - fr[:, 0]) * (fr[:, 1] if cy else 1 - fr[:, 1]) * (fr[:, 2] if cz else 1 - fr[:, 2])
                out = out + w[:, None] * u[ix, iy, iz]
    return out
