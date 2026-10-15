# Pre-merge test summary (branch `sim-fix`)

NVIDIA RTX 4090 (driver 550.163.01), Warp 1.11.1, Python 3.10.20.

**Current status: 96/96 pass.**
- Rerun after review (fresh clone of `7dfe34b`, WARMUP_S = 120 s, N_PUFFS = 2560, class ordering on 3-seed means): `./run_all_tests.sh --slow` → 95 passed, 1 failed (34 min).
- The failure was a transient crash inside Warp (TypeError in its own type check, `warp._src.types.is_array`) in one of the ten class-D seed-spread jobs. It did not reproduce alone or with four concurrent jobs.
- `run_parallel` now retries a crashed job once and records it in `report.json` (`meta.job_retries`). With that change the test passed in the fresh clone with 0 retries (`tests/report_fresh_clone_seed_spread_rerun.json`).
- First run (before the follow-up): fresh clone of `57e2274`, 94 passed, 2 failed (B7.25 class ordering, B11.38 warm-up reach). Both are fixed; see below. Report: `tests/report_fresh_clone.json`.
- `validation/validate_sim.py`: **V0, V1, V2, V3, V5, V6 PASS** (validation/results.json). The follow-up changed env constants only, which validation does not use.

Nothing has been merged, tagged or pushed.

## Part A

| ID | Result | Key numbers |
|---|---|---|
| A1 V1 shared-eddy offset | **Noise, not bias** | σ_y error at 100 m vs number of seeds N: 1 → +6.0%, 4 → +4.9%, 8 → +3.7%, 16 → +0.9%. Single-seed std 5.6%. The 16-seed mean (σ_y +0.3%, σ_z +1.3%, centreline −0.1%) matches the independent-release V1. |
| A2 Per-puff velocity statistics | PASS | Release variance within 2% of σ² at every sequence position. After 22–60 s, var(v)/σ² = 0.996–1.002 and spread/Taylor = 0.997–1.004. Only adjacent-puff ρ changes (0.9925 vs e^{−τe/T_E} = 0.9925). |
| A3 Attribution, `shared_eddies` flag | Done | Code, README, CHANGELOG and tex: meandering-plume model (Gifford 1959; Anfossi et al. 2006), not GADEN. |
| A4 γ sensitivity | **Not sensitive**; γ = 0.6 kept | Across γ ∈ {0.4, 0.6, 0.8}, σ_y ratios change by ≤ 0.08 (seed-to-seed spread ≤ 0.20) and σ_z by ≤ 0.02. Centreline intermittency 0.67–0.86, rising slightly with γ. |
| A5 Realised U10 = requested | PASS | ×1.006 / 1.003 / 1.001 at 1.5 / 3 / 8 m/s (3 seeds × 1200 s; was ×0.80–0.82). |
| A6 Image-aware σ_z fit, class A | PASS (ensemble) | 3-seed mean field: σ_y ×0.93–1.03, σ_z ×0.68–1.00. σ_z at 50 m (×0.68) sits at the band edge in every seed, a model limitation. |
| A7 Class-dependent σ₀ | **Does not fix E/F**; default kept | F (σ₀ = 0.3 m): σ_z at 25 m ×3.54 → 2.60. E (σ₀ = 0.5 m): ×2.08 → 1.73. K_rel·t dominates at short range. |
| A8 Tex | Done | Compiles (Tectonic), no overfull boxes. All references checked on Crossref (`validation/a8_references.md`); Stull page numbers could not be verified and were removed. Only `wind_field_method.tex` is staged in the CDC26 repo; `figures/validation/` there is untracked. |

## Part B

| ID | Result | Key numbers |
|---|---|---|
| B1.1 Calm wind | PASS | U ∈ {0, 0.05, 0.2}: no NaN, σ = σ_min, finite T_L/T_E, total spread grows with age, mass balances. Realised wind ≈ 0.3 m/s (the wind model's minimum speed). |
| B1.2 Strong wind | PASS | 8/12/15 m/s: 0 tunnelling steps, 0 puffs inside a 2 m wall, KE steady. C·U at 50 m agrees within 2.4%. |
| B1.3 Direction invariance | PASS | 8 directions: σ_y, σ_z and C agree within 2.1%, 4.2% and 3.6%. |
| B1.4 Sudden 90° rotation | PASS | Max step displacement 0.56 of the bound; young puffs travel at 96°. |
| B2.5–7 Source at ground, faces, corners, top | PASS | min puff z = 0; emitted = active + exited + overwritten at every position. |
| B2.8 Source inside obstacle | PASS (after fix) | Env never spawns one in 1000 resets (min clearance 2.02 m). The sim raises a clear ValueError. |
| B2.9 Source 0.5 m from a wall | PASS | 0 puffs inside over 300 s. 76 far-side values match the float64 LOS reference to 6.9e-8; the shadow reads 0. |
| B2.10 Mobile source | PASS | Never inside an obstacle or outside its region; puffs are released at the current position. |
| B3.11–13 Sensor queries | PASS | At the source and at z = 0, GPU = float64 reference (< 1e-4). Below ground, outside the domain or inside an obstacle → exactly 0. |
| B4.14 Linearity in Q | PASS | 2Q gives exactly 2C; Q = 0 gives exactly 0. |
| B4.15 Superposition | PASS (kernel level) | The sim supports a single source, so superposition is checked on the kernel: exact. |
| B4.16 Translation | PASS | Max relative difference 3.5e-4 (float32 positions). |
| B4.17 Rotation 90° | PASS | σ_y 0.4%, σ_z 2.6%, C 1.3%. |
| B5.18 dt halved/doubled | PASS | 300 puffs and 1.000 kg/s at 15/30/60 fps. OU variance within 1%. V1 changes ≤ 1.4%; V3 class D ≤ 4.6% (shared eddies off). |
| B5.19 Env DT vs frame_dt | PASS (after fix) | Drift ≤ 4e-10 s over 10,000 steps at 30/25/24 fps. Before the fix: −200 s at 25 fps. |
| B6.20 Mass balance | PASS | Counts exact; mass within 6.6e-7 (767 exited, 66 overwritten). |
| B6.21 Kernel mass | PASS | 0.99985–0.99990 for σ ∈ {0.3, 1, 5} m at heights {0.1σ, σ, 10σ}. |
| B6.22 Buffer sizing, 1 h | PASS (warning) | N_PUFFS = 2048: 24 puffs (24 kg) overwritten. After the follow-up, **N_PUFFS = 2560: 1 puff** (1 kg of 18,750 kg emitted, 0.005%; oldest active puff 414 s). The warning now names ≥ 3200. |
| B7.23 OU statistics | PASS | Variance within 0.8%; autocorrelation within 0.0055. |
| B7.24 Well-mixed (Thomson) | PASS | χ² = 1000.7 on 1000 bins, p = 0.48. Boundary bins at 0.99 of the mean. |
| B7.25 Class ordering | PASS (3-seed means) | σ_y at 100 m: 20.1 / 17.2 / 12.3 / 9.1 / 7.1 / 5.5 m (A→F). σ_z: 15.8 / 10.8 / 7.9 / 6.0 / 3.3 / 2.3 m. Intermittency: 0.26 / 0.35 / 0.50 / 0.68 / 0.80 / 0.90. Single runs failed at A < B (19.3 vs 20.1 m) because classes A/B scatter by 29% and 19% per run. |
| B7.26 Seed spread, class D | PASS (report) | CoV σ_y 5.7–6.7%, σ_z 1.8–3.1%; 10/10 seeds pass V3. |
| B8.27–28 No puffs inside, none stuck | PASS | 0 inside, 0 stuck, 0 steps crossing a solid (7 obstacles, 300 s). |
| B8.29 Bounce velocity flip | PASS | Matches the expected reflection to 1e-5 (sphere and box face). |
| B8.30 Degenerate geometry | PASS (after fix) | Sphere/box centre, crevice, overlapping spheres → signed distance +2.0 m. |
| B8.31 LOS vs float64 | PASS | 0 of 8000 random configurations differ. All 392 differences among 2000 tangent/corner cases are at exact tangency. |
| B8.32 Wind around obstacles | PASS | |u| = 0 inside; net flux into solids 3.3% of U·area. Wakes are 5–22% slower than the freestream. |
| B9.33 GPU vs float64 | PASS | Max relative error 1.25e-6 over 23,919 values (1000 configurations). |
| B9.34 1 h clock | PASS | 18,450 emitted = expected. |
| B10.35 1 h stability | PASS | No NaN; KE trend 0.2% per 30 min; residual 0.0015–0.0021; puff-count CV 6%; steps/s 1361 → 1356. |
| B11.36 Reset hygiene | PASS (after fix) | Bit-identical to a fresh env (puffs, shared eddies, anemometer, sensor lag, MLE, info map). |
| B11.37 Determinism | PASS | 500 steps identical with the same seed; different seeds differ. |
| B11.38 Spawn validity | PASS (after fixes) | 1000 resets: source ≥ 2.0 m and drone ≥ 5.0 m from obstacles, drone ≥ 30.2 m from source, 1000 distinct spawns. Obstacles placed before warm-up. Plume reach after the 120 s warm-up: ≥ 100 m (or the domain edge) in 10/10 episodes, 110–184 m. With 60 s it was 69–90 m in 5/10. |
| B11.39 Observation ranges | PASS (report) | 100 episodes (155k steps), all finite and in [0, 1]. c_norm is 0 on 33.5% of steps and 1 on 0%; c_detected > 0.5 on 1.8%. |
| B11.40 Sensor model | PASS | 63% response at 1.0 s; noise σ within 0.2%; bias applied once. Clipping bias at C = 0: +8e-6 kg/m³. |
| B11.41 Anemometer | PASS | σ within 0.5%; ρ(τ) within 0.009 of e^{−1}; off = exact resolved wind. |
| B11.42 Units | PASS | Round trip to 1e-12; 1 g/m³ CH₄ = 1499.7 ppm at 20 °C. |
| B12.43 Input validation | PASS | 17 cases. |
| B13.44 Performance | Baseline recorded | Env defaults 0.74 ms/step (45×). 2560 puffs + 1000 queries: 1.4 ms (LOS off), 2.1 ms (LOS on). |

## Failures and causes

All failures from the first run are resolved:

1. **B7.25 class ordering:** sampling noise. Classes A and B have long meander time scales, so the test now compares 3-seed means, which are ordered on both axes.
2. **B11.38 warm-up reach:** `WARMUP_S` raised from 60 s to 120 s.
3. **Rerun, seed-spread job crash:** a transient error inside Warp that did not reproduce. The runner retries once and records it.

Open, reported: with N_PUFFS = 2560, a 1-hour run still overwrites 1 puff (0.005% of the mass). Puffs that linger near the ground live up to about 400 s; N_PUFFS = 3200 would remove this. Episodes of 200 s (+120 s warm-up) never overflow.

## Code changes this round

Each is in its commit and in `CHANGELOG.md`, under "Pre-merge round".

- `shared_eddies` (renamed from `release_correlation`).
- `drag_compensation` and zero-mean gusts.
- `class_sigma0` and `lsm_table`.
- `sigma_min` (0.05 m/s) — **new physics parameter**.
- `set_wind_direction`.
- Validated `source_pos`, `Q` and `sigma_0`.
- Query returns 0 below ground and outside the domain.
- `fps`.
- Env steps the sim by elapsed time.
- Mass counters and the overflow warning.
- `closed_domain` (tests only).
- Obstacle push-out fixes (degenerate normal, crevice escape).
- Drone spawn fix with `DRONE_SPAWN` — **changes the RL episode distribution**.
- MLE estimate reset.
- Units helpers.
- Input validation.

Follow-up after review: `WARMUP_S` 60 → 120 s, `N_PUFFS` 2048 → 2560, B7.25 on 3-seed means, `run_parallel` retries a crashed job once.

## Limitations to state in the paper

- Briggs formulas are stated for 100 m–10 km; the 25–100 m comparison is an extrapolation.
- Classes E and F are wider than Briggs at short range: σ₀ = 1 m and K_rel·t exceed Briggs F σ_z = 0.4 m at 25 m. A class-dependent σ₀ alone does not fix this.
- Class A σ_z at 50 m is at the band edge (×0.68): a homogeneous Gaussian Langevin model cannot reproduce convective, skewed turbulence. Classes A and B need multi-seed averages.
- Shared-eddy release is a one-particle approximation of meander, using Hay–Pasquill γ = 0.6. γ is unverified against data, but results are insensitive across 0.4–0.8.
- Calm wind: σ floor 0.05 m/s and a 0.3 m/s minimum wind speed.
- The collocated projection removes about half of the divergence (residual 0.0016).
- Line of sight is tested from the puff centre only.
- Single source; fixed 64×32×32 grid (4 m cells).
- With the calibrated plume, gas is detectable at about 0.5% of random drone-zone points (pre-fix 7.5%), so RL episodes are harder.
