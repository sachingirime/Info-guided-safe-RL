# GADEN benchmark plan (v2: literature-standard evaluation)

**Goal:** evaluate our simulator (`v1.0-sim`) against GADEN (`MAPIRlab/gaden_core` @ `1a20e35`) using the protocol the dispersion-modelling literature uses for model intercomparison. That means both models against common reference data, with standard statistics, bootstrap confidence intervals and significance tests between the two models, and a robot-olfaction task metric on top.

**Ground rules (unchanged):**
- One commit per step. Stop and report on a failed check.
- Never tune a parameter of *our* model to pass. Any GADEN tuning follows the held-out protocol in T2.
- Branch `gaden-bench`. Code lives under `bench/`; `bench/external/` and `bench/data/` are gitignored.

---

## 0. Why v1 of this plan was not enough

A v1-style code-to-code comparison only shows that the two implementations agree when configured to be the same model. That is a *verification* result. In the literature, model *evaluation* has three parts:

1. Models are compared with **observations**, using a fixed set of statistics with published acceptance criteria. Sources: Chang & Hanna 2004, *Meteorol. Atmos. Phys.* 87:167; Hanna & Chang 2012, *Meteorol. Atmos. Phys.* 116:133; ASTM D6589.
2. Two models are compared by testing whether the **difference in those statistics** is significant, using a bootstrap (the BOOT software approach).
3. Concentration **fluctuations** are evaluated separately from the mean, using intensity, intermittency, PDF shape and peak-to-mean ratios. These are what a gas sensor on a robot actually experiences.

GADEN's own validation was a symmetric KL divergence on 10-bin histograms of normalised TGS-2620 MOX responses, against the Vergara et al. UCSD wind-tunnel data. Results were reported as checkerboard plots only, with good agreement at 0.34 m/s and poor agreement at 0.1 m/s (Monroy et al. 2017, *Sensors* 17:1479). We reuse its sKLD metric for comparability, but within a stronger protocol.

So the benchmark has four tiers. Only T1 is apples-to-apples by construction; every result is labelled with its tier.

| Tier | Question | Reference | Apples-to-apples? |
|---|---|---|---|
| T1 | Do the two codes solve the same equations identically when configured alike? | analytic solution + each other | **yes**: same wind, map, source, release, growth, no noise |
| T2 | How well does each reproduce **mean** near-field dispersion? | Prairie Grass field data (50/100/200 m arcs) | same inputs, different physics (by design) |
| T3 | How well does each reproduce **concentration fluctuations** seen by a sensor? | outdoor fast-sensor plume data + published fluctuation statistics | same inputs, different physics |
| T4 | Does the choice of simulator change source-seeking conclusions? | cross-simulator task performance | same task, swapped simulator |

---

## 1. Shared infrastructure (≈1 h, the part estimated earlier)

### Step 1.0: GADEN build
- Clone at `1a20e35` into `bench/external/`.
- Build CPU-only with `-DCMAKE_DISABLE_FIND_PACKAGE_OpenCL=ON`.
- Record compiler, flags and commit in `bench/results/env.json`.
- **Check:** `example_project` runs for 60 s.

### Step 1.1: C++ driver `bench/gaden_driver/`
- **Inputs:**
  - `env.bin`: grid and uint8 cell states;
  - `wind/` snapshots in GADEN's binary format, or a single snapshot for steady wind;
  - `params.json`;
  - `sensors.npy` (M×3 positions) plus a sample interval.
- **Outputs:**
  - `conc.npy` in ppm, shape (n_samples, M), from `CalculateConcentration`;
  - `log.json` with wall time and filament counts.

### Step 1.2: Exporter `bench/export.py`
- **Mapping:**
  - our wind → GADEN snapshots;
  - obstacles → `Obstacle`, lateral and top faces → `Outlet`;
  - σ₀ = 100 cm; γ = 2·K_rel in cm²/s; `numFilaments_sec` = 1/τₑ;
  - `filamentPPMcenter_initial` from Q·τₑ using GADEN's mole bookkeeping;
  - gas ethanol (SG 1.038) unless a scenario says otherwise.
- **Check:** mass released over 600 s agrees within 1%, and GADEN's wind lookup equals our cell values at 20 random points.

### Step 1.3: Our "GADEN-equivalent" configuration
- a_y = a_z = 0, σ_min = 0, `shared_eddies=False`, `buoyancy_mode="none"`, K_rel = γ/2.
- This is configuration only; no code change.

### Unit checks (`bench/tests/`)
- **Single filament in uniform wind:** the centre moves U·t exactly.
- **Concentration:** equals the analytic Gaussian within 1e-5 inside 3σ.
- **Growth:** σ² = σ₀² + γt.

---

## 2. T1: Verification (code-to-code, deterministic)

### Scenarios
| ID | Wind | Geometry | Noise |
|---|---|---|---|
| T1a | steady uniform U = 3 m/s | open, source at z = 30 m (ground image < 1e-6) | 0 both |
| T1b | steady, power-law shear, class D | open | 0 both |
| T1c | one exported time-varying `meandering` realisation, 0.1 s snapshots | open | 0 both |
| T1d | as T1c | env obstacles | 0 both |

### Metrics
- **Per plane x − xₛ ∈ {25, 50, 100, 150} m:** σ_y, σ_z, centreline C, and mass flux ∬U·C dA / Q.
- **Per point:** time-series Pearson r and NMSE, since both runs are a single deterministic realisation.
- **Reference:** T1a is also checked against the closed-form slender-plume solution (as in V1).

### Acceptance (T1a/T1b)
- σ within 5%, centreline within 10%, flux within 5% of Q for **both** codes.
- If this fails, attribute the gap with one-factor sensitivity runs and do not proceed until it is explained:
  - nearest-cell vs trilinear sampling;
  - Euler vs RK2;
  - wind interval 0.1 vs 1 s;
  - 3σ vs 4σ cutoff;
  - voxel vs exact line of sight (T1d only).

### Acceptance (T1c/T1d)
- Report the numbers.
- Investigate if a mean-field difference exceeds 15% or the median point-wise r < 0.8.

---

## 3. T2: Mean-field evaluation against Prairie Grass

### Data
- **Project Prairie Grass** (O'Neill, Nebraska, 1956): continuous near-ground SO₂ release at 0.46 m (runs 63–68: 1.5 m); 10-min average samplers at 1.5 m on arcs at 50, 100, 200, 400 and 800 m.
- Source: the digitised set distributed with the ASTM D6589 evaluation package (J. Irwin, harmo.org) or the Aarhus OML subset.
- Exclude runs 3, 4, 13 and 14 (anomalous low wind and roughness), as is standard. Report results with and without the exclusion.

### Domain fit (fixed in advance, not chosen after the results)
- Our domain is 256 × 128 m with the source at x = 20 m, so the 50 and 100 m arcs are usable for all runs.
- The 200 m arc is usable only where 3σ_y(200 m) < 60 m, i.e. classes C–F. Runs are flagged, not dropped silently.
- 400 and 800 m are out of domain and are not evaluated.

### Per-run inputs
- Measured wind speed and direction (wind rotated onto +x), z₀ = 0.006 m, and Monin–Obukhov L.
- Our stability class from L via Golder (1972).
- Our model is used exactly as shipped (`calibrated` preset, LSM, shared eddies). Our T_L table was fitted to Briggs, *not* to Prairie Grass runs.

**Independence caveat.** Briggs' open-country curves draw partly on Prairie Grass, so for *our* model T2 is a consistency check, not an independent validation. This must be stated in the paper. GADEN has no such dependence.

### GADEN configurations (all three reported, so tuning cannot hide)
| Config | Description |
|---|---|
| G-default | steady mean wind (how GADEN is normally used, with a steady CFD field) and the `filamentNoise_std` / γ from GADEN's shipped example config |
| G-tuned | `filamentNoise_std` fitted on odd-numbered runs and evaluated on even-numbered runs only (held-out; split fixed now) |
| G-ourwind | GADEN driven by our time-varying `meandering` wind (resolved turbulence supplies the spread) |

### Quantities per run and arc
1. Arc-maximum C/Q, unpaired in space. This is the standard "max on arc" comparison.
2. Crosswind-integrated concentration (CWIC/Q).
3. σ_y from the arc profile, using the same second-moment estimator for observations and models.
4. Paired-in-space sampler concentrations for NAD.

### Statistics
- **BOOT set** (Chang & Hanna 2004): FB, NMSE, MG, VG, FAC2, plus NAD (Hanna & Chang 2012). All are computed per model, overall and stratified by stability class (A–C, D, E–F) and by arc.
- **Confidence intervals:** 95% CIs by bootstrap over runs (1000 resamples, resampling runs not samplers, to respect within-run correlation).
- **Model vs model:** paired bootstrap of ΔFB, ΔNMSE, ΔMG and ΔVG (ours − GADEN). A difference is significant if its 95% CI excludes 0. This is the BOOT model-comparison test.
- **ASTM D6589 centreline test:** group runs into regimes (stability × arc) and compare the regime-averaged observed near-centreline C (bootstrap) with each model's ensemble mean. This handles the fact that each observation is one realisation while the model output is an ensemble mean.
- **Our model's seed noise:** 3 seeds per run. Report the across-seed spread of each statistic as the noise floor.

### Acceptance criteria (reported, not tuned to)
- Rural criteria are quoted from the papers: FAC2 ≥ 0.5 and |FB| ≤ 0.3, with the NMSE limit taken from the paper itself.
- Secondary sources disagree on that limit (1.5, 3 or 4). **The original Chang & Hanna 2004 and Hanna & Chang 2012 texts must be checked before any threshold is written into results or the tex.**
- Urban criteria (FAC2 ≥ 0.3, |FB| ≤ 0.67, NMSE ≤ 6, NAD ≤ 0.5) are shown only for context.

### Compute
- About 60 usable runs × 3 seeds × ~12 min sim (spin-up plus a 10-min average). At 47× real time this is about 45 min GPU for ours.
- GADEN G-default needs a single wind snapshot per run and is cheap. G-ourwind needs ~0.6 GB of wind per run, written and deleted one run at a time.

---

## 4. T3: Fluctuation evaluation (what a robot's sensor sees)

Mean fields don't decide source-seeking performance; intermittency and whiff structure do. GADEN's own evaluation was PDF-based, so this tier is directly comparable to their paper.

### Reference data
1. **Nag & van Breugel (2024)**, outdoor propylene plume, Dryad/Zenodo doi:10.5061/dryad.2547d7wvr, CC0, 3.3 GB.
   - 200 Hz odor sensor on an RTK-GPS-tracked handheld stack; 8 stationary anemometers (7 at ~30 m, 1 at ~1 m from the source); 60 × 60 m area.
   - Same spatial scale as our domain, so it is the primary T3 reference.
   - Sensor units are probably uncalibrated, so only **shape statistics** are compared: normalised by the local mean, or threshold-relative.
   - To verify after download: sensor type and response time, and whether the wind series are long enough to drive our solver's inflow.
2. **Published fluctuation statistics** (literature values, cited, not re-measured): centreline intensity i = σ_c/C̄, and intermittency vs distance, for near-ground point sources in neutral surface layers (e.g. Mylne & Mason 1991, Yee et al. field studies). Used as range checks only.
3. **Not used, with reason stated:** the UCSD Vergara wind tunnel (2.5 × 1.2 × 0.4 m) that GADEN validated on. It cannot be resolved on our 4 m grid, and our ABL turbulence closure does not apply to a tunnel. FFT-07 (Dugway) would be ideal for source-term estimation, but public availability is unconfirmed; request it, don't depend on it.

### Protocol
- Drive both models with wind matching the reference: mean speed and direction from the anemometers, and our stability class from σ_θ.
- Sample at the reference sensor positions. For mobile tracks, convert to distance from source and crosswind offset relative to the mean wind.
- **Sensor model:** apply the *same* first-order lag (τ from the reference sensor's spec) to both models' output before computing statistics. Also report the unfiltered statistics.

### Statistics (per distance bin, centreline and off-axis)
- **Moments:** C̄ (normalised), intensity i, skewness and kurtosis.
- **Intermittency:** γ(C > θ) at three thresholds θ ∈ {0.05, 0.1, 0.2}·C̄_max, so threshold sensitivity is visible.
- **Peak-to-mean:** C₉₀/C̄ and C₉₉/C̄.
- **PDF shape:** MLE fit of a gamma distribution. Report the shape parameter and the Anderson–Darling statistic. The gamma form is recommended for the upper tail in the fluctuation literature.
- **Whiff/bout statistics** (Schmuker et al. 2016 bout pipeline; also used by Nag & van Breugel):
  - whiff frequency, whiff duration and blank duration;
  - their trend with source distance (bout count falls with distance, inter-bout interval grows).
- **Time scales:** integral time scale of C(t) and the frequency spectrum slope.
- **Distribution distances:**
  - symmetric KL on 10-bin histograms, identical to Monroy et al. 2017, for comparability with GADEN's paper;
  - plus Jensen–Shannon (bounded) and Wasserstein-1 on log C (no bin dependence).
- **Noise floor:** the same distances between two seeds of the same model, and between two halves of the reference record. A model-to-data distance is meaningful only relative to these.

### Acceptance
- Report only. Fluctuation evaluation has no community-standard thresholds.
- The claim we can make is a ranking: which model is closer to data on each statistic, with bootstrap CIs.

---

## 5. T4: Task-level evaluation (does it matter for source seeking?)

This is what reviewers of a source-seeking paper will ask for.

### Agents
- **Deterministic baselines:**
  - cast-and-surge;
  - gradient/chemotaxis;
  - infotaxis (Vergassola et al. 2007, using each simulator's own detection likelihood).
- **Optionally the CDC26 RL policy**, if it can be evaluated without retraining.

### Cross-simulator matrix
- Each agent is tuned (or trained) on simulator A and evaluated on A and on B, for A, B ∈ {ours, GADEN G-default, GADEN G-ourwind}.
- The off-diagonal drop is the "sim gap".

### Metrics
These follow the gas-source-localisation review metrics (Francis et al. 2022):
- success rate (reached within 2 m of the source in under 300 s);
- final localisation error;
- time to first detection;
- time to source;
- path length.

Each comes with a Wilson CI for success rate and a bootstrap CI for continuous metrics.

### Episodes
- 200 per cell, with seeds and source and start positions fixed and shared across simulators.
- Run with the same `GaussianPuffEnv` drone dynamics: the concentration call is routed to GADEN through the driver (batch-replay of GADEN concentration fields along logged trajectories is *not* valid for closed-loop agents, so this needs online queries; see risks).

---

## 6. T5: Cost

- Wall time per simulated second and concentration-query cost (1 and 1000 points) for both, on T1c and T2 runs.
- Hardware stated: ours on GPU, GADEN on CPU with OpenMP and its core count.

---

## 7. Deliverables

- **Results:** `bench/results/gaden_results.json` with every statistic, its CI and noise floor, the parameter mapping, run lists and exclusions.
- **Figures:**
  - T1 profile overlays;
  - T2 scatter (C_obs vs C_pred with factor-2 lines), FB-vs-NMSE and MG-vs-VG plots with CIs, per model and stability group;
  - T3 PDFs and whiff statistics vs distance;
  - T4 transfer matrix.
- **Tex:** a "Comparison with GADEN and field data" subsection in `wind_field_method.tex`. All numbers come from the JSON. The Prairie Grass independence caveat and the G-tuned split are stated explicitly.

---

## 8. Effort and compute (honest)

| Tier | Work | Compute |
|---|---|---|
| Infrastructure + T1 | ≈1 h | minutes |
| T2 Prairie Grass | ≈3 h: data parsing is the main cost | ≈45 min GPU + GADEN CPU |
| T3 fluctuations | ≈3 h: download and inspect the 3.3 GB set first | ≈30 min |
| T4 task level | ≈3–4 h: online GADEN queries from Python | 1–2 h |
| Write-up | ≈1 h | none |

T1 is a hard gate for T2–T4. T2 and T3 are independent and can run in either order.

---

## 9. Risks and open items

- **Unverified details:**
  - the exact Chang & Hanna acceptance thresholds;
  - Prairie Grass digital data format and run list;
  - the Nag & van Breugel sensor type;
  - FFT-07 availability.
  - All are to be checked from primary sources before use; none is assumed in results.
- **GADEN online queries for T4:** the C++ driver is batch. T4 needs either a persistent driver process (stdin/stdout or socket, stepping on request) or the cppyy bindings. Prototype this in T1; if it isn't working within 1 h, T4 uses GADEN G-default only through a precomputed concentration-field cache at 1 s / 1 m. Note that this is open-loop-exact only for steady wind.
- **Disk:** G-ourwind wind exports are about 0.6–1.4 GB each. Write, run, delete.
- **Hardware:** the i9-14900K has shown MCE errors and a kernel oops. Update the BIOS (Intel 0x12B microcode) before long runs. Every long job checkpoints per run, so a crash loses at most one run.
- **Conda env `gaden-bench`:** a partial prefix is left from the crash. Remove it and re-run `bench/setup_env.sh --force` before Step 1.0.
