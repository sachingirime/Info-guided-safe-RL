# Changelog

## v1.0-sim (branch `sim-fix`, from tag `pre-fix` = `1d30cdb`)

Every behaviour change, with the old and new value. Validation numbers are in
`validation/results.json`; commit messages carry the per-step checks.

### Concentration and emission

| Change | Old (`pre-fix`) | New | Why |
|---|---|---|---|
| Grid gas in the sensor | Eulerian ρ grid × 1e-3 added to every reading (25% of sensed C by the plan's definition, 60% median per cell) | Off (`use_grid_gas=False`) | Non-physical second gas field |
| Source momentum jet | (0,0,2)·w·0.5 m/s added to the wind every frame → 6.5 m/s updraft at the source | Off (only with `use_grid_gas`) | Non-physical |
| Puffs per interval | 2 (released 2Q) | 1 | Bookkeeping bug |
| Emission timing | `t − t_last < τe` with float drift: every 7 frames at τe = 0.2 s (Q 14% low) | Fixed schedule tₙ = n·τe | Bookkeeping bug |
| Puff mass | Current Q × τe applied to all puffs at query time | Stored per puff at release (Q·τe) | Wrong with `Q_GROWTH` |
| Release jitter | Hidden 1.5 m | `release_jitter=0` | Undocumented |
| Ring buffer | Overwrites silent | `n_emitted`, `n_overwritten`, `n_exited` counters | Detect mass loss |
| Env `N_PUFFS` | 1024 (20–213 puffs overwritten per 200 s episode, ≤16% of mass) | 2048 (0 overwritten) | Mass loss |
| Line of sight | Puffs summed through obstacles | Segment–sphere/box test from puff centre; points inside obstacles read 0 (`use_los=True`) | Gas leaked through walls |

### Puff dispersion

| Change | Old | New | Why |
|---|---|---|---|
| Model | Puff growth σ² = σ₀² + 2K·age **and** centroid random walk ℓ = √(2K): mean plume σ₀² + 4K·t | Per-puff OU turbulent velocity (Thomson 1987), relative growth σ² = σ₀² + 2·K_rel·age | Double counting; constant K cannot follow Briggs |
| Diffusivity table | `PASQUILL_GIFFORD_KD` (class D K_y = 3.8, K_z = 1.3 m²/s; does not follow its own formula, which gives 1.9 / 0.8) | `PG_LSM`: σ_u = σ_v = a_y·U, σ_w = a_z·U (Briggs slopes, A–F) and fitted T_L (`validation/fit_tl.py`) | V3: old model 2.4–7.5× Briggs D |
| T_L scaling | — | T_L = T_L,fit·3 m/s / U_src | Taylor term depends on x only, like Briggs |
| U for intensities | — | Running mean (τ = 30 s) of the resolved wind at the source | Target profile is ~20% too fast at 10 m (ground drag) |
| σ₀ (`SIGMA_0`) | 2.0 m | 1.0 m | σ₀ = 2 m > Briggs D σ_z(25 m) = 1.47 m |
| Relative growth | f_rel·K (f_rel = 0.5 in the first fix) | `K_rel` = 0.05 m²/s | Relative dispersion only (Farrell/GADEN) |
| Coherent meander | — (independent puffs: centreline intermittency γ = 1.0) | Shared eddies (meandering-plume model, Gifford 1959): release velocities correlated over T_E = T_L·i/γ, γ = 0.6 (Hay–Pasquill form, Anfossi et al. 2006; `shared_eddies`, `hp_gamma`). Not GADEN's mechanism: GADEN's filament noise is independent and its coherence comes from the CFD wind | Plume meanders as a ribbon; γ 0.71–0.83 |
| Centroid noise RNG | Integer hash of (i, int(100t)), correlated axes | Warp `rand_init(seed, i)` per step; `seed` argument | Reproducibility |
| Ground bounce | Position reflected | Position and v_w reflected | Well-mixed condition |
| Obstacle bounce | Pushed 2 m out | Pushed 2 m out, normal component of v reflected | Consistency |
| Stability classes | A–G | A–F (Briggs has no G) | Calibration table |

### Wind

| Change | Old | New | Why |
|---|---|---|---|
| Grid convention | Puffs/sensor/env sampled at p/dx; mask and shear at cell centres (2 m shift on every axis); index clamped after the fraction | All samplers p/dx − 0.5, clamp-to-edge | Half-cell offset bug |
| Shear | (z/128 m)^α, U_ref = wind at the domain top | U_ref·(z/10 m)^α | U_ref was not the 10 m wind |
| Gusts | New Exp(15 s) draw every frame vs the timer: a gust every ~3 s, active 69% | Poisson onsets, mean gap 15 s (active 12%) | Timing bug |
| Ground drag | ×(1 − 0.08(1 − k/3)) per frame | ×exp(−2.4·dt·(1 − k/3)) | Frame-rate independence (profile within 2%) |
| Resolved turbulence | Coarse OU σ_stat ≈ 2.7 m/s, mean-wind meander ≈ 14° at 3 m/s | `wind_preset="calibrated"`: ×0.1 (meander ≲ 3° at 1.5 m/s); `"meandering"` keeps ×1 | Turbulence is in the calibrated LSM |
| Buoyancy | 4.55 m/s² methane *acceleration* added as a velocity (median puff 27 m above a 10 m source) | Off; `buoyancy_mode="rise"`: w_rise0·(σ₀/σ)³ m/s | Unit error |
| `METHANE_BUOYANCY_ACCEL` | Used by puffs | Legacy grid-gas path only | — |
| Pressure solve | 25 SOR iterations, no diagnostic | 25 iterations, `divergence_residual()` = 0.0016 | V8 |

### Environment

| Change | Old | New |
|---|---|---|
| Warm-up | 10 frames (0.33 s) | `WARMUP_S` = 60 s |
| Obstacle placement | After warm-up started | Before warm-up |
| Puff seed | Fixed | Drawn from the env RNG each reset |
| `CONC_SIG_CEIL` | 5e-3 kg/m³ (saturated c_norm for 41–55% of detections with the new plume) | 0.1 kg/m³ (6–10%) |
| Anemometer | Smooth resolved wind | + OU turbulence, class σ, τ = 1 s (`wind_noise`) |
| Docstrings | 9-D obs; index 9 called concentration | 10-D; index 9 is the MLE distance |

Not changed: `CONC_NOISE_FLOOR` (3e-4), `DETECT_TEMPERATURE` (1e-4), `SIGMA_E`, `BIAS`, `Q_INIT` (5 kg/s), `V_MAX` (10 m/s), `CONVERGE_DIST` (0, RL setting).
Consequence for RL: with the narrower plume, a random drone-zone point is above the 3e-4 kg/m³ detection floor 0.5% of the time (pre-fix 7.5%).

### Tooling

- `test_env.py` rewritten: 9 tests (obs, determinism, no grid gas, emission count, RNG, LSM, shared-eddy release, LOS, kernel mass).
- `validation/`: `diag_state.py`, `validate_sim.py` (V0–V8), `fit_tl.py`, `make_before_figure.py`, `results.json`, figures.
- `realtime_viz.py`: `--headless` paper figures, puff spheres sized by σ_puff, dead first copy removed.
- `requirements.txt` pinned.

### Known limitations

- Briggs formulas are stated for 100 m–10 km; the 25–100 m comparison is an extrapolation.
- Class F (and A σ_z at 50 m) are outside the V3 band at short range; σ₀ = 1 m exceeds Briggs F σ_z (0.4 m at 25 m).
- Hay–Pasquill γ = 0.6 is an order-one constant, not verified against a specific dataset.
- Pressure projection on the collocated grid removes ~55% of the divergence; more SOR iterations do not help (discretisation mismatch).
- Grid size is fixed at 64×32×32 × 4 m (module-level constants).
