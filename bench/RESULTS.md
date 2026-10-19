# GADEN benchmark: results

**Models compared:**
- **Ours:** `v1.0-sim` (as shipped unless stated).
- **GADEN:** `gaden_core` at `1a20e35`, built CPU-only and driven by `bench/gaden_driver`, a persistent stdin/stdout process.

**Inputs:** GADEN always receives our wind field (streamed every 0.1 s) and our obstacle mask.

**Concentrations** are compared as C/Q. GADEN runs at a tracer rate of 1e-6 kg/s, which keeps its ppm-proportional buoyancy negligible.

**Sources:**
- numbers: `bench/results/t{1..5}.json`;
- figures: `bench/figs/`;
- design: `GADEN_BENCHMARK_PLAN.md`.

## GADEN configurations

| Name | Filament parameters | Wind |
|---|---|---|
| GADEN matched (T1) | σ₀ 1 m, γ = 2·K_rel = 1000 cm²/s, noise 0 | ours, streamed |
| GADEN example | example_project/sim1: σ₀ 10 cm, γ 10 cm²/s, noise 0.01, 10 filaments/s | ours, streamed (or its time-mean, "steady") |
| GADEN tuned | example parameters with filamentNoise_std = **0.7** | ours, streamed |

The tuned noise was chosen on odd-numbered Prairie Grass runs: the arc-max ln-MSE has an interior minimum at 0.7 on the grid {0.01 … 5}.

## Quirks found in GADEN's source (relevant to any comparison)

1. **3σ cutoff in the concentration sum.** A filament keeps only P(χ²₃ < 9) = 97.07% of its mass, and plume second moments shrink by √(P(χ²₅<9)/P(χ²₃<9)) = 0.958.
2. **Noise displacement scales with dt, not √dt.** The effective diffusivity therefore depends on the time step. The configured std is also multiplied by 10 internally.
3. **Identical noise across OpenMP threads.** Each thread's engine is default-seeded, so all threads share one 1000-entry noise table. Runs are reproducible only for a fixed thread count. The driver adds a seed that advances each thread's engine.
4. **Buoyancy ∝ (1 − SG)·C_centre in ppm.** At realistic release rates this dominates: ethanol at a 9700 ppm filament centre sinks at about 12 m/s.
5. **No ground image.** Filaments that reach z = 0 slide along the ground.

## T1: matched verification (`figs/t1_profiles`)

Both codes ran the same model (no turbulent velocity, no buoyancy) on the same streamed wind, with 300 s averages.

| Scenario | σ_y, σ_z ratio (GADEN/ours) | C_max ratio | Flux/Q (GADEN; ours) | Field r |
|---|---|---|---|---|
| T1a steady uniform | 0.955–0.961 | 0.996–0.997 | 0.969–0.972; 1.000 | 1.0000 |
| T1b steady, class-D shear | 0.959–0.961 | 0.996–0.997 | 0.970–0.972; 0.999 | 1.0000 |
| T1c meandering wind | 0.987–1.008 | 0.961–1.033 | 0.955–0.976 | 0.987–0.999 |
| T1d meandering + obstacles | 0.77–1.09 | 0.91–**4.79** | 0.91–2.20 | 0.64–0.82 |

**T1a–T1c pass** the plan's criteria:
- σ within 5%, centreline within 10%, flux within 5% of Q;
- point time-series r ≥ 0.97 in T1c.

The remaining 4% σ gap and 3% flux gap match the analytic 3σ-cutoff values above. Our flux/Q = 1.000 in steady wind is the analytic value.

**T1d (investigated):** the obstacle differences come from two design differences, not a transport error:
- **Geometry representation.** GADEN uses 4 m voxels, while our puffs use exact sphere and box SDFs.
- **Wall treatment.** GADEN filaments slide along walls and collect at the upstream stagnation face, giving 2–4× more filaments within 4 m of a surface. Up to 311 of 767 sit inside the exact sphere. Ours are pushed 2 m off the surface.

The 4.8× spike at 25 m is that pile-up.

## T2: Prairie Grass mean field (`figs/t2_scatter_cmax`, `figs/t2_mg_vg`)

**Data:**
- 64 runs (Irwin digitisation): runs 3, 4, 13 and 14 excluded as standard, plus the two S runs.
- Arcs at 50/100/200 m, receptors at 1.5 m.
- Evaporative-loss corrected.

**Stability:** Monin–Obukhov fit to each run's profiles, then Golder classes:
- A 5, B 4, C 14, D 26, E 7, F 8 runs.

**Ours:** 3 seeds per run, ensemble mean.

**Statistics:**
- FB, NMSE, MG, VG, FAC2, NAD and R (Chang & Hanna 2004; Hanna & Chang 2012).
- 95% CIs by bootstrap over runs (2000 resamples).
- Model difference: paired bootstrap over runs.

Results over all 192 run–arc pairs:

| Quantity | Model | FB [95% CI] | NMSE | MG [95% CI] | VG | FAC2 [95% CI] |
|---|---|---|---|---|---|---|
| Arc-max C/Q | **ours** | +0.08 [−0.05, +0.20] | 0.40 | 0.97 [0.86, 1.09] | 1.33 | **0.84** [0.77, 0.91] |
| | GADEN tuned | +0.71 [+0.44, +0.92] | 4.06 | 1.05 [0.82, 1.37] | 4.15 | 0.40 [0.33, 0.48] |
| | GADEN example | — | 43 | 51 | ≫ | 0.03 |
| CWIC/Q | **ours** | +0.28 [+0.20, +0.34] | 0.23 | 1.35 [1.26, 1.46] | 1.21 | **0.93** [0.87, 0.98] |
| | GADEN tuned | +1.19 [+1.09, +1.27] | 4.05 | 3.27 [2.95, 3.61] | 5.25 | 0.13 [0.09, 0.18] |
| σ_y | **ours** | +0.35 [+0.24, +0.47] | 0.58 | 1.32 [1.20, 1.46] | 1.28 | **0.84** [0.75, 0.91] |
| | GADEN tuned | +0.02 [−0.14, +0.18] | 0.53 | 0.89 [0.77, 1.04] | 1.52 | 0.68 [0.59, 0.77] |

**Held-out check (even runs only, never used for GADEN tuning):** ours scores FAC2 0.90 on arc-max, 0.96 on CWIC and 0.88 on σ_y; tuned GADEN scores 0.38 on arc-max.

**Ours vs tuned GADEN (paired bootstrap):** ours is significantly better on |FB|, NMSE, VG, FAC2 and NAD for both arc-max and CWIC. The two are statistically tied on |ln MG| for arc-max.

**By stability class:**
- GADEN tuned fails worst in stable conditions (E–F arc-max FAC2 0.11, MG 4.8). One isotropic noise level cannot fit both lateral and vertical spread across stabilities.
- Ours stays at FAC2 0.77–0.90 across class groups.

**ASTM D6589 near-centreline test** (ratio model/observed, 95% CI):
- ours 0.72–1.51, CIs overlapping 1 in 5 of 9 regimes;
- GADEN tuned 0.07–2.37.

**Known bias in ours:** plumes about 25% too narrow (σ_y MG 1.32), worst in A–C. The 200 m arc in A–C over-predicts near-centreline C by ×1.5.

**Noise floor:** our seed-vs-seed scatter is FAC2 0.95–0.99 and VG 1.05–1.22, so model-to-data differences of this size are not seed noise.

**Independence caveat:** our T_L table was fitted to Briggs (1973), whose open-country curves draw partly on Prairie Grass. **For our model T2 is a consistency check, not an independent validation.** For GADEN it is independent.

## T3: fluctuations along a real sensor track (`figs/t3_whiff_stats`)

**Data:** Nag & van Breugel (2024), desert runs:
- low wind: 6448 s, class A by σ_θ = 32.5°;
- high wind: 2427 s, class D by σ_θ = 11.4°.

**Method:**
- **Track replay:** the walker's GPS track (source-relative) is replayed through each model at 10 Hz, with the model's mean wind set to the measured mean wind.
- **Streakline conditioning:** statistics are restricted to samples within 3 m of each side's own streakline. The streakline is computed from that side's wind at the source. My data-side streakline matches the authors' column with Spearman 0.998–0.999.
- **Uncalibrated sensor:** it saturates at 10.29 V, so each model's threshold is set to match the data's intermittency in the 0–5 m bin. Bins at 5–50 m are predictions.

**Assumptions:**
- Source and sensor at 1.5 m (not given in the dataset documentation).
- Steady mean wind in the models.

Findings (4 V threshold; 2 V and 6 V give the same picture):
- **Ours** loses whiffs beyond about 10 m. Intermittency at 10–30 m is ≤ 0.005 against 0.07–0.11 in the data, and whiff frequency falls to ≈ 0. Within-whiff PDF sKLD is 0.6–3.1.
  - **Interpretation:** our 1 m Gaussian puffs (σ₀ = 1 m, K_rel growth) smooth out the thin, high-concentration filaments that carry far-field whiffs. The model reproduces mean fields (T2), but not sensor-scale peak-to-mean structure beyond about 10 m.
- **GADEN example** (10 cm filaments) keeps whiff structure better. Intermittency at 10–30 m is 0.014–0.026, whiff durations are within the data's range, and sKLD is 0.02–0.89. It still decays faster than the data.
- **GADEN tuned** cannot be calibrated: its plume reaches the track points less often than the data's whiff rate.
- **Data noise floor:** the two halves of each record differ by up to 2–4× per bin.

## T4: source-seeking transfer (`figs/t4_transfer`)

**Episodes:** 120 env-like episodes:
- random wind direction, U₁₀ 1.5 m/s, class D, open field;
- agent at source height, 2 m/s, 300 s;
- success means getting within 2 m.

**Worlds:** ours, GADEN example and GADEN tuned run on the same wind and the same scenarios.

**Agents:**
- cast-and-surge, 6 variants;
- infotaxis (Vergassola et al. 2007), 3 thresholds.

**Protocol:** tune on episodes 0–39 in world A, test on episodes 40–119 in every world.

| Infotaxis, tuned in → | tested in ours | GADEN example | GADEN tuned |
|---|---|---|---|
| ours (θ = 1e-3) | **0.56** [0.45, 0.67] | 0.36 [0.27, 0.47] | 0.29 [0.20, 0.39] |
| GADEN example (θ = 1e-5) | 0.36 [0.27, 0.47] | 0.36 | 0.29 |
| GADEN tuned (θ = 1e-5) | 0.36 | 0.36 | 0.29 |

- **Transfer gap.** The ours-tuned agent drops 0.20 [0.09, 0.31] in GADEN example and 0.28 [0.14, 0.40] in GADEN tuned, both significant.
- **Detection.** The plume is encountered in 74% of episodes in ours, against 46–49% in the GADEN worlds.
- **Threshold choice.** In both GADEN worlds all three thresholds give identical results (187 of 240 world-episodes end identically across thresholds): GADEN's compact filaments make concentration effectively binary. So the simulator changes both expected performance and which parameter looks optimal. In ours the threshold matters (0.36 / 0.41 / 0.56).
- **Cast-and-surge** (my implementation) rarely succeeds anywhere (0.00–0.21), so conclusions rest on infotaxis.

## T5: compute cost

**Measurement:** T1c set-up, about 670 puffs/filaments, RTX 4090, GADEN with 8 OpenMP threads.

| | Ours | GADEN |
|---|---|---|
| Wall time per simulated second | 21 ms (47× real time), **including the 3-D wind solve at 30 Hz** | 0.15 ms (filament step only; wind precomputed) |
| Query, 1 point | 0.22 ms (GPU launch-bound) | 0.02 ms |
| Query, 1000 points | 0.25 ms | 0.12 ms |

## Bottom line

- **Mean field:** ours is far closer to field data than GADEN at any tested setting (Prairie Grass FAC2 0.84–0.93 against ≤ 0.68), subject to the Briggs/Prairie Grass caveat.
- **Fluctuations:** GADEN's small filaments preserve far-field whiff structure that our puff model smooths away. This is the main limitation of our simulator found here. Candidate fix to test: smaller σ₀ and K_rel, with a puff-size-aware LSM, or a filament sub-model.
- **Matched verification:** the two transport codes agree to within the analytic effect of GADEN's 3σ cutoff in open terrain. They differ near obstacles by design.
- **Task level:** the choice of simulator significantly changes measured agent performance and the optimal agent parameters.
