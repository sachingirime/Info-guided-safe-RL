"""Shared helpers for the edge-case test suite: sim factory, float64 reference
implementations (SDF, line of sight, concentration), and a small parallel runner
for long simulations (alternates the two GPUs)."""
import json
import os
import subprocess
import sys

import numpy as np

import stable_fluid_3d as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DOMAIN = np.array([sf.NX * sf.DX, sf.NY * sf.DX, sf.NZ * sf.DX])


def make_sim(source=(60.0, 64.0, 10.0), U=3.0, angle=0.0, Q=1.0, emit=0.2, seed=1, pg="D",
             max_puffs=2048, obstacles=((), ()), **kw):
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=angle, wind_angle_range=20.0, wind_base_speed=U,
                                       wind_seed=seed, seed=seed, max_puffs=max_puffs, stability_class=pg, **kw)
    sim.Q = Q
    sim.emit_interval = emit
    sim.set_obstacles(list(obstacles[0]), list(obstacles[1]))
    sim.source_pos = np.array(source, np.float32)
    return sim


def puffs(sim):
    """Active puff positions, velocities, masses and ages (float64)."""
    act = sim.puff_active.numpy() == 1
    return (sim.puff_pos.numpy()[act].astype(float), sim.puff_vel.numpy()[act].astype(float),
            sim.puff_mass.numpy()[act].astype(float), sim.sim_time - sim.puff_emit_time.numpy()[act].astype(float))


# ── float64 geometry references ──────────────────────────────────────────────

def sdf_sphere(p, c, r):
    return np.linalg.norm(p - c, axis=-1) - r


def sdf_box(p, c, h):
    d = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


def sdf_all(p, spheres, cuboids):
    p = np.atleast_2d(p)
    out = np.full(len(p), np.inf)
    for s in spheres:
        out = np.minimum(out, sdf_sphere(p, np.asarray(s.center, float), float(s.radius)))
    for b in cuboids:
        out = np.minimum(out, sdf_box(p, np.asarray(b.center, float), np.asarray(b.half_extents, float)))
    return out


def seg_hits_sphere64(p0, p1, c, r):
    d = p1 - p0
    t = np.clip(((c - p0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(p0 + t[..., None] * d - c, axis=-1) < r


def seg_hits_box64(p0, p1, c, h):
    d = p1 - p0
    tmin = np.zeros(d.shape[:-1]); tmax = np.ones(d.shape[:-1]); hit = np.ones(d.shape[:-1], bool)
    for a in range(3):
        small = np.abs(d[..., a]) < 1e-9
        ca, ha = np.asarray(c)[..., a], np.asarray(h)[..., a]
        hit &= ~(small & (np.abs(p0[..., a] - ca) > ha))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(small, 0.0, 1.0 / np.where(small, 1.0, d[..., a]))
            t1 = (ca - ha - p0[..., a]) * inv
            t2 = (ca + ha - p0[..., a]) * inv
        tmin = np.where(small, tmin, np.maximum(tmin, np.minimum(t1, t2)))
        tmax = np.where(small, tmax, np.minimum(tmax, np.maximum(t1, t2)))
    return hit & (tmin <= tmax)


def conc_reference64(sensors, pos, mass, ages, sigma0, k_rel, spheres=(), cuboids=(), use_los=True):
    """float64 reference of puff_sum_at (+ query masking): isotropic Gaussians with
    ground image, 4σ cull, line of sight from the puff centre, 0 inside obstacles,
    below ground or outside the domain."""
    S = np.atleast_2d(np.asarray(sensors, float))
    out = np.zeros(len(S))
    sig2 = sigma0 ** 2 + 2.0 * k_rel * ages
    sig = np.sqrt(sig2)
    for k, p in enumerate(S):
        if p[2] < 0 or np.any(p < 0) or np.any(p > DOMAIN):
            continue
        if use_los and (spheres or cuboids) and sdf_all(p, spheres, cuboids)[0] < 0:
            continue
        diff = p - pos
        keep = (np.abs(diff[:, 0]) <= 4 * sig) & (np.abs(diff[:, 1]) <= 4 * sig)
        dz_r = diff[:, 2]; dz_i = p[2] + pos[:, 2]
        keep &= ~((np.abs(dz_r) > 4 * sig) & (np.abs(dz_i) > 4 * sig))
        if use_los and (spheres or cuboids) and keep.any():
            idx = np.where(keep)[0]
            blocked = np.zeros(len(idx), bool)
            P0 = pos[idx]; P1 = np.broadcast_to(p, P0.shape)
            for s in spheres:
                blocked |= seg_hits_sphere64(P0, P1, np.asarray(s.center, float), float(s.radius))
            for b in cuboids:
                blocked |= seg_hits_box64(P0, P1, np.asarray(b.center, float), np.asarray(b.half_extents, float))
            keep[idx[blocked]] = False
        d = diff[keep]; s2 = sig2[keep]; s1 = sig[keep]
        g = np.exp(-0.5 * (d[:, 0] ** 2 + d[:, 1] ** 2) / s2) * (
            np.exp(-0.5 * d[:, 2] ** 2 / s2) + np.exp(-0.5 * (p[2] + pos[keep, 2]) ** 2 / s2))
        out[k] = np.sum(mass[keep] / ((2 * np.pi) ** 1.5 * s1 ** 3) * g)
    return out


# ── parallel runner for long jobs ────────────────────────────────────────────

RETRIES = []   # (job, error tail) of jobs that crashed and were retried once


def run_parallel(jobs, per_gpu=2, n_gpu=2):
    """jobs: list of (function_name, kwargs). Functions live in tests/jobs.py and
    return JSON-serialisable results. Runs up to per_gpu·n_gpu at once.
    A job that crashes is retried once (recorded in RETRIES): the 2026-10-15
    fresh-clone run hit a transient TypeError inside Warp's own type check
    (types.is_array) that did not reproduce alone or concurrently."""
    procs, results, queue = [], [None] * len(jobs), list(enumerate(jobs))
    tried = set()
    slots = per_gpu * n_gpu
    running = []
    while queue or running:
        while queue and len(running) < slots:
            i, (fn, kw) = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n_gpu), PYTHONPATH=ROOT)
            p = subprocess.Popen([PY, os.path.join(ROOT, "tests", "jobs.py"), fn, json.dumps(kw)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
            running.append((i, p))
        i, p = running.pop(0)
        out, err = p.communicate()
        if p.returncode != 0:
            if i not in tried:
                tried.add(i)
                RETRIES.append({"job": [jobs[i][0], jobs[i][1]], "error": err[-600:]})
                queue.insert(0, (i, jobs[i]))
                continue
            raise RuntimeError(f"job {jobs[i]} failed twice:\n{err[-3000:]}")
        results[i] = json.loads(out.strip().splitlines()[-1])
    return results
