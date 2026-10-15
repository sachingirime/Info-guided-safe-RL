# NVIDIA Warp Gas Plume Gym Environment

A GPU-accelerated [Gymnasium](https://gymnasium.farama.org/) environment for **drone-based gas source localization** in 3D turbulent wind fields.

Built on [NVIDIA Warp](https://developer.nvidia.com/warp), the environment couples an Eulerian Stable Fluids wind solver with a Lagrangian Gaussian Puff dispersion model to produce physically realistic methane plume dynamics at interactive rates.

## Features

- **GPU-accelerated 3D fluid simulation** via NVIDIA Warp (64x32x32 grid, 4 m cells, 256x128x128 m domain)
- **Lagrangian Gaussian puff model** (up to 2048 puffs in the env): RK2 advection on the resolved wind plus a per-puff OU turbulent velocity (Lagrangian stochastic model calibrated to Briggs), ground image, line-of-sight occlusion by obstacles
- **Pasquill-Gifford stability classes** (A-G) for turbulent diffusivity, shear exponent and lapse rate
- **Methane sensor model** with bias, noise, first-order lag and soft sigmoid detection
- **Mobile gas source** with Ornstein-Uhlenbeck drift
- **Obstacle support** (sphere and cuboid primitives with SDF-based boundary conditions)
- **Standard Gymnasium API** (`reset`, `step`, `observation_space`, `action_space`)

See a simulation at: https://youtu.be/etPAydxxQGs

## Observation Space (10D + optional info map)

| Index | Name | Description |
|-------|------|-------------|
| 0-2 | `p_norm` | Drone position / domain size |
| 3-5 | `w_norm` | Local wind vector, `clip(w / 5 m/s, -1, 1) * 0.5 + 0.5` |
| 6 | `c_norm` | Soft log1p concentration (sensor reading minus bias) |
| 7 | `c_detected` | Sigmoid detection confidence around the 3σ noise floor |
| 8 | `zeta_inv` | EMGR observability metric (injected with `set_zeta_inv`) |
| 9 | `d_mle` | Distance to the MLE source estimate / domain diagonal (injected with `set_theta_hat`; 1.0 until set). Not a concentration. |
| 10… | info map | Optional EMGR information-map channels (`info_map_size`) |

## Action Space

Continuous velocity command in R^3, clipped to [-V_MAX, V_MAX] with **V_MAX = 10 m/s** (DT = 0.1 s).

## Installation

```bash
pip install -r requirements.txt     # pyvista only for realtime_viz.py
```

Requires a CUDA-capable GPU for the Warp fluid solver.

## Quick Start

```python
from gaussian_puff_env import GaussianPuffEnv

env = GaussianPuffEnv()
obs, info = env.reset(seed=42)    # advances WARMUP_S = 60 s so the plume is developed

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

`CONVERGE_DIST = 0.0`: the RL setting never terminates on distance to the source. Planner runs use their own convergence radius.

### Simulator options

Extra keyword arguments to `GaussianPuffEnv(...)` are passed to `HybridFluidPuffSimulation`. Defaults are the fixed physics.

| Argument | Default | Meaning |
|---|---|---|
| `seed` | `wind_seed` | Puff-noise RNG seed (the env derives one per episode from its RNG) |
| `K_rel` | `0.05` | m²/s, relative puff growth σ² = σ₀² + 2·K_rel·age (σ₀ = `SIGMA_0` = 1 m) |
| `shared_eddies` | `True` | Meandering-plume model (Gifford 1959): successive puffs' release velocities correlated over the Eulerian time scale T_E, so the plume meanders coherently. `False` = independent puff noise (GADEN-equivalent) |
| `hp_gamma` | `0.6` | γ in the Hay–Pasquill ratio T_L/T_E = γ/i |
| `stability_class` | `"D"` | PG class A–F (set via the env's `stability_class`) |
| `drag_compensation` | `True` | Target scaled in the ground-drag layers so the realised mean wind follows the power law (realised 10 m wind = `wind_base_speed`) |
| `wind_preset` | `"calibrated"` | `"calibrated"` (small resolved fluctuations, ≲3° meander) or `"meandering"` (pre-LSM amplitudes, harder RL) |
| `turb_scale` | from preset | Multiplies all stochastic resolved-wind amplitudes |
| `buoyancy_mode` | `"none"` | `"none"` (neutral gas) or `"rise"` |
| `w_rise0` | `0.0` | Initial rise speed (m/s) for `"rise"`; decays as (σ₀/σ)³ |
| `ground_drag_rate` | `2.4` | 1/s, ground drag in the lowest 3 cells |
| `release_jitter` | `0.0` | m, std of release-position noise |
| `enable_turbulence` | `True` | Coarse OU field and the three OU turbulence layers |
| `enable_gusts` | `True` | Gust model |
| `steady_wind` | `False` | Remove all wind randomness (validation) |
| `shear_exponent` | `None` | Override the PG power-law shear exponent |
| `use_los` | `True` | Line-of-sight occlusion of puffs by obstacles |
| `use_grid_gas` | `False` | Legacy Eulerian gas field + source jet (pre-fix figures only) |
| `legacy_emission` | `False` | Legacy two puffs per interval (pre-fix figures only) |

Env-only arguments: `warmup_s` (default 60 s), `wind_noise` (default `True`: OU anemometer turbulence with the class σ and `wind_noise_tau` = 1 s).

## Testing and validation

```bash
python test_env.py                        # unit tests (~10 s)
python validation/diag_state.py           # before/after state diagnostic
python validation/validate_sim.py         # V0-V8 (~25 min on a 4090), writes validation/results.json + validation/figs/
python validation/fit_tl.py               # refit the T_L table (validation/tl_fit.json)
python realtime_viz.py --headless         # paper figures in figures/ (gitignored)
python validation/make_before_figure.py --prefix <checkout of tag pre-fix>
```

Validation summary (class D, RTX 4090): V1 analytic plume within 5%; V3 mean-plume σ within ×[0.76, 1.49] of Briggs D over 25-200 m; V4 centreline intermittency γ 0.71-0.83; 30-47× real time. See `validation/results.json` and `CHANGELOG.md`.

## Environment Details

### Wind Field
Inviscid Stable Fluids on a 64x32x32 grid (4 m cells): semi-Lagrangian advection and a pressure projection (Red-Black SOR, 25 iterations), relaxed toward a stochastic target wind with time constant 1/α ≈ 0.33 s. The target is an OU mean wind, three OU turbulence layers, a coarse OU spatial perturbation field and Poisson gusts, shaped by a power-law shear profile u(z) = U_ref·(z/10 m)^α (U_ref = base speed at 10 m). Obstacles are solid cells (u = 0) with Neumann pressure; the ground has w = 0 and dt-scaled drag; open faces have p = 0.

### Gas Dispersion
Lagrangian Gaussian puffs, one per `PUFF_PERIOD` with mass Q·`PUFF_PERIOD`:
- RK2 advection on the trilinearly interpolated wind
- Absolute dispersion: each puff carries a turbulent velocity v following an OU process (Thomson 1987) with σ_u = σ_v = a_y·U, σ_w = a_z·U from the Briggs near-field slopes and T_L fitted to Briggs σ(x) (`validation/fit_tl.py`); reflections flip v's normal component
- Coherent meander (meandering-plume model, Gifford 1959): release velocities of successive puffs are correlated over T_E = T_L·i/γ (Hay–Pasquill form, Anfossi et al. 2006), so the instantaneous plume meanders and is intermittent
- Relative dispersion: puff growth σ² = σ₀² + 2·K_rel·age
- Ground reflection by the method of images; removal at the open domain faces
- Obstacles: puffs pushed 2 m out of solids; line-of-sight occlusion from puff centre to the sample point
- Buoyancy off by default; optional decaying rise velocity for methane

### Sensor Model
TDLAS-style methane sensor: bias 1e-4 kg/m³, Gaussian noise σ = 1e-4 kg/m³, first-order lag τ = 1 s, detection threshold 3e-4 kg/m³ (3σ), normalisation ceiling 0.1 kg/m³. Anemometer: resolved wind + OU turbulence (class σ, τ = 1 s).

## Physics changes since `pre-fix`

Tag `pre-fix` = commit `1d30cdb`. Branch `sim-fix` changes:

1. **Sensor reads puffs only.** The Eulerian grid gas (25-60% of sensed concentration depending on the definition) and the source momentum jet (6.5 m/s updraft at the source) are off; `use_grid_gas=True` restores them.
2. **Grid convention.** All samplers use cell centres (`p/dx − 0.5`), matching the obstacle mask and shear profile (pre-fix puffs and the env saw the wind shifted 2 m on every axis), and clamp to the grid edge correctly.
3. **Emission.** One puff per interval (was two → 2Q), no release jitter (was 1.5 m), fixed emission schedule (float drift emitted every 7 frames at 0.2 s → Q 14% low), per-puff mass, overwrite/exit counters.
4. **Seeded RNG.** Warp `rand_init`/`randn` per puff and step replaces a hash with correlated axes.
5. **Dispersion.** Pre-fix used the full K for both puff growth and a centroid random walk (σ₀² + 4K·t), and the class-D K row did not follow its own formula. Now: per-puff OU turbulent velocity calibrated to Briggs (A–F) with release-correlated meander, relative growth K_rel = 0.05 m²/s, σ₀ = 1 m; resolved-wind turbulence ×0.1 (`wind_preset="calibrated"`); shear referenced to 10 m instead of the 128 m domain top.
6. **Buoyancy.** Off by default; `"rise"` is a decaying velocity. Pre-fix added the 4.55 m/s² methane acceleration as a velocity (median puff height 37 m for a 10 m source).
7. **Ground drag** scaled by dt (was per frame).
8. **Line-of-sight occlusion**: gas no longer leaks through obstacles.
9. **Gusts** are a Poisson process with the stated mean interval (pre-fix fired every ~3 s, active 69% of the time).
10. **Env**: per-episode puff seed, obstacles placed before a 60 s warm-up (was 0.33 s, after the plume started), `N_PUFFS` 1024 → 2048 (1024 overwrote up to 16% of emitted puffs per episode), `CONC_SIG_CEIL` 5e-3 → 0.1 kg/m³, anemometer turbulence, corrected docstrings.

Full list with old/new values: [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this environment in your research, please cite:

```bibtex
@inproceedings{giri2026infoguided,
  title={Information-Guided Safe Reinforcement Learning for Autonomous Gas Source Localization using sUAS},
  author={Giri, Sachin and Zhao, Thomas and Huynh, Matthew and Chen, YangQuan},
  booktitle={IEEE Conference on Decision and Control (CDC)},
  year={2026}
}
```

## License

MIT License
