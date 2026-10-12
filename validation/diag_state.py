"""
diag_state.py — Before/after state diagnostic for HybridFluidPuffSimulation.

Advances a sim for 120 s of sim time (PG class D, env-like settings, no
obstacles) and reports:

  grid_share          sum(c_grid) / sum(c_grid + c_puff) over grid cells with
                      z in [4, 60] m and total C > 3e-4 kg/m^3
  w_max_source        peak vertical wind (m/s) in cells within 10 m of source
                      (final snapshot; w_max_source_tmax = max over the run)
  median_puff_height  median z of active puffs (m), with source_z
  n_active_puffs
  divergence_residual (fixed code only)

The grid term is read directly from rho0 * RHO_SCALE (exactly what the
concentration-grid kernel adds per cell), so the same script runs unchanged
on the pre-fix tag and on the fixed code.

Usage:
    python validation/diag_state.py --out validation/after.json
    # pre-fix: run with PYTHONPATH pointing at a checkout of tag pre-fix
"""

import argparse
import json
import os
import sys

import numpy as np
import warp as wp

if "stable_fluid_3d" not in sys.modules and not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stable_fluid_3d as sf

SOURCE = (60.0, 64.0, 10.0)
Q = 5.0
EMIT_INTERVAL = 0.2
SIGMA_0 = 2.0
WIND_SPEED = 1.5
WIND_ANGLE = 0.0
WIND_RANGE = 20.0
SEED = 42


def make_sim(**kwargs):
    """Env-like sim (GaussianPuffEnv defaults), wind along +x, no obstacles."""
    sim = sf.HybridFluidPuffSimulation(
        wind_center_angle=WIND_ANGLE, wind_angle_range=WIND_RANGE,
        wind_base_speed=WIND_SPEED, wind_seed=SEED, max_puffs=1024, **kwargs)
    sim.Q = Q
    sim.sigma_0 = SIGMA_0
    sim.emit_interval = EMIT_INTERVAL
    sim.set_stability_class("D")
    sim.source_pos = np.array(SOURCE, dtype=np.float32)
    sim.set_obstacles([], [])
    return sim


def _w_near_source(sim, radius=10.0):
    u = sim.u0.numpy()
    dx = sim.dx
    nx, ny, nz = u.shape[:3]
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    centers = np.stack([(ii + 0.5) * dx, (jj + 0.5) * dx, (kk + 0.5) * dx], axis=-1)
    near = np.linalg.norm(centers - np.asarray(sim.source_pos), axis=-1) <= radius
    return float(u[..., 2][near].max())


def diagnose(sim, duration=120.0, z_band=(4.0, 60.0), c_thresh=3e-4):
    n_steps = int(round(duration / sim.frame_dt))
    w_tmax = -np.inf
    for i in range(n_steps):
        sim.step()
        if i % 30 == 0:
            w_tmax = max(w_tmax, _w_near_source(sim))
    wp.synchronize()

    # Grid share on the concentration grid
    sim.compute_conc_grid()
    total = sim.conc_grid.numpy()
    use_grid = getattr(sim, "use_grid_gas", True)
    c_grid = sim.rho0.numpy() * float(sim.RHO_SCALE) if use_grid else np.zeros_like(total)
    nz = total.shape[2]
    zc = (np.arange(nz) + 0.5) * sim.dx
    zmask = (zc >= z_band[0]) & (zc <= z_band[1])
    sel = (total > c_thresh) & zmask[None, None, :]
    denom = float(total[sel].sum())
    grid_share = float(c_grid[sel].sum() / denom) if denom > 0 else 0.0

    pos, ages = sim.get_puff_data()
    out = {
        "sim_time": float(sim.sim_time),
        "grid_share": grid_share,
        "n_cells_above_thresh": int(sel.sum()),
        "w_max_source": _w_near_source(sim),
        "w_max_source_tmax": float(w_tmax),
        "median_puff_height": float(np.median(pos[:, 2])) if len(pos) else float("nan"),
        "source_z": float(sim.source_pos[2]),
        "n_active_puffs": int(len(pos)),
        "rho_max": float(sim.rho0.numpy().max()),
    }
    if hasattr(sim, "divergence_residual"):
        out["divergence_residual"] = float(sim.divergence_residual())
        out["sor_iters"] = int(getattr(sim, "sor_iters", -1))
    for key in ("n_emitted", "n_overwritten", "n_exited"):
        if hasattr(sim, key):
            v = getattr(sim, key)
            out[key] = int(v() if callable(v) else v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--legacy", action="store_true",
                    help="fixed code only: use_grid_gas=True, f_rel=1, old drag/jitter/emission")
    args = ap.parse_args()

    wp.init()
    kwargs = {}
    if args.legacy:
        kwargs = dict(use_grid_gas=True, f_rel=1.0, ground_drag_rate=2.4,
                      release_jitter=1.5, legacy_emission=True)
    sim = make_sim(**kwargs)
    res = diagnose(sim, duration=args.duration)
    res["code"] = os.path.abspath(sf.__file__)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
