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
| Env `N_PUFFS` | 1024 (20–213 puffs overwritten per 200 s episode, ≤16% of mass) | 2560 (2048 first; a 1 h run still overwrote 24 puffs) | Mass loss |
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
| Realised 10 m wind | Relaxation + ground drag left the resolved 10 m wind ~20% below the target; gusts added a +0.09 m/s mean | Target divided by the relaxation/drag equilibrium factor d·n/(1−d(1−n)) in the drag layers (`drag_compensation`); gust mean subtracted (zero-mean gusts). Realised U10 / requested = 1.006, 1.003, 1.001 at 1.5, 3, 8 m/s (3 seeds × 1200 s, `validation/a5_u10.json`) | A5 |
| Gusts | New Exp(15 s) draw every frame vs the timer: a gust every ~3 s, active 69% | Poisson onsets, mean gap 15 s (active 12%) | Timing bug |
| Ground drag | ×(1 − 0.08(1 − k/3)) per frame | ×exp(−2.4·dt·(1 − k/3)) | Frame-rate independence (profile within 2%) |
| Resolved turbulence | Coarse OU σ_stat ≈ 2.7 m/s, mean-wind meander ≈ 14° at 3 m/s | `wind_preset="calibrated"`: ×0.1 (meander ≲ 3° at 1.5 m/s); `"meandering"` keeps ×1 | Turbulence is in the calibrated LSM |
| Buoyancy | 4.55 m/s² methane *acceleration* added as a velocity (median puff 27 m above a 10 m source) | Off; `buoyancy_mode="rise"`: w_rise0·(σ₀/σ)³ m/s | Unit error |
| `METHANE_BUOYANCY_ACCEL` | Used by puffs | Legacy grid-gas path only | — |
| Pressure solve | 25 SOR iterations, no diagnostic | 25 iterations, `divergence_residual()` = 0.0016 | V8 |

### Environment

| Change | Old | New |
|---|---|---|
| Warm-up | 10 frames (0.33 s) | `WARMUP_S` = 120 s (60 s first; plume reached < 100 m in 5/10 episodes) |
| Obstacle placement | After warm-up started | Before warm-up |
| Puff seed | Fixed | Drawn from the env RNG each reset |
| `CONC_SIG_CEIL` | 5e-3 kg/m³ (saturated c_norm for 41–55% of detections with the new plume) | 0.1 kg/m³ (6–10%) |
| Anemometer | Smooth resolved wind | + OU turbulence, class σ, τ = 1 s (`wind_noise`) |
| Docstrings | 9-D obs; index 9 called concentration | 10-D; index 9 is the MLE distance |

Not changed: `CONC_NOISE_FLOOR` (3e-4), `DETECT_TEMPERATURE` (1e-4), `SIGMA_E`, `BIAS`, `Q_INIT` (5 kg/s), `V_MAX` (10 m/s), `CONVERGE_DIST` (0, RL setting).
Consequence for RL: with the narrower plume, a random drone-zone point is above the 3e-4 kg/m³ detection floor 0.5% of the time (pre-fix 7.5%).

### Pre-merge round (Parts A–B of the pre-merge plan)

| ID | Change | Old | New |
|---|---|---|---|
| A3 | Shared-eddy flag | `release_correlation` | `shared_eddies` (default True); attributed to the meandering-plume model (Gifford 1959; Anfossi et al. 2006), not GADEN |
| A5 | Realised 10 m wind | 0.80–0.82 × requested (relaxation + drag), gusts added +0.09 m/s mean | `drag_compensation` (default True) and zero-mean gusts: 1.001–1.006 × requested |
| A6 | V3 σ_z fit | ground-reflected fit, h fixed | same (primary) + free-h fit reported |
| A7 | Class-dependent release size | — | optional `class_sigma0`, `lsm_table`; off by default (does not make E/F pass) |
| B1 | Calm wind | σ = a·U → 0 as U → 0 | `sigma_min` = 0.05 m/s floor (new physics parameter; affects only a·U < 0.05 m/s: class F σ_w below 3.1 m/s, E σ_w below 1.7 m/s, D σ_w below 0.8 m/s). The wind model's existing 0.3 m/s minimum speed is unchanged |
| B1 | Wind rotation | — | `set_wind_direction()` |
| B2 | Source inside an obstacle | env could spawn it (2.71% of resets, fixed sphere at (46, 92, 21)); sim accepted it | env resamples with 2 m clearance; `source_pos` raises ValueError inside an obstacle or outside the domain; mobile source never steps into an obstacle |
| B3 | Query below ground / outside the domain | image term mirrored the plume below z = 0; out-of-domain points summed nearby puffs | returns 0 |
| B5 | Frame rate | fixed 30 fps; env ran round(DT/frame_dt) frames per step (drifted for non-integer ratios: −200 s per 10,000 steps at 25 fps) | `fps` argument; env steps the sim to the elapsed env time (drift < 1e-9 s) |
| B6 | Mass bookkeeping | counts only | exited/overwritten mass counters; one-time overflow warning naming the needed `N_PUFFS` |
| B7 | Closed box | — | `closed_domain` (tests only) |
| B8 | Obstacle push-out | puff at a sphere centre stayed inside; puff in a crevice between overlapping obstacles could end inside one | +z fallback normal; two passes; lift until clear |
| B11 | Drone spawn | always the unchecked fallback (230, 115, 9): the source-window exclusion covered the whole spawn area | `DRONE_SPAWN = "random"` (≥ 30 m from the source, ≥ 5 m from obstacles); `"corner"` reproduces the old position, validated. **Changes the RL episode distribution** |
| B11 | Reset | MLE estimate (obs[9]) leaked across episodes | cleared in `reset()` |
| B11 | Units | — | `kgm3_to_ppm`, `ppm_to_kgm3` |
| B12 | Input validation | none | ValueError for negative Q, σ₀, K_rel, wind speed, σ_min, fps ≤ 0, unknown class/preset, invalid or out-of-domain obstacles, > 10 obstacles, env N_PUFFS too small for the warm-up, env domain ≠ fluid grid |

Follow-up after review: `WARMUP_S` 60 → 120 s (plume reached < 100 m in 5/10 episodes at 1.5 m/s) and `N_PUFFS` 2048 → 2560 (a 1 h run overwrote 24 puffs); B7.25 class ordering compares 3-seed means (single runs of classes A/B scatter by 19–29%).

### Tooling

- `test_env.py` rewritten: 9 tests (obs, determinism, no grid gas, emission count, RNG, LSM, shared-eddy release, LOS, kernel mass).
- `validation/`: `diag_state.py`, `validate_sim.py` (V0–V8), `fit_tl.py`, `make_before_figure.py`, `results.json`, figures.
- `realtime_viz.py`: `--headless` paper figures, puff spheres sized by σ_puff, dead first copy removed.
- `requirements.txt` pinned.
- `tests/` (pytest) edge-case and reliability suite, `run_all_tests.sh [--slow]`, `tests/report.json`, `tests/SUMMARY.md`.

### Known limitations

- Briggs formulas are stated for 100 m–10 km; the 25–100 m comparison is an extrapolation.
- Classes E and F are outside the V3 band at short range (σ₀ = 1 m and K_rel·t exceed Briggs F σ_z = 0.4 m at 25 m); a class-dependent σ₀ alone does not fix it.
- Class A σ_z at 50 m is at the band edge (×0.68 on a 3-seed ensemble, low in every seed): homogeneous Gaussian Langevin model vs convective turbulence. Classes A/B need multi-seed averages (single-run σ_y CoV ≈ 19–29%).
- Calm wind: σ floor 0.05 m/s and the wind model's 0.3 m/s minimum speed.
- Hay–Pasquill γ = 0.6 is an order-one constant, not verified against a specific dataset.
- Pressure projection on the collocated grid removes ~55% of the divergence; more SOR iterations do not help (discretisation mismatch).
- Grid size is fixed at 64×32×32 × 4 m (module-level constants).
