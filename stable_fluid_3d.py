
"""
stable_fluid_3d.py — 3-D Stable Fluids + Hybrid Gaussian-Puff concentration model.

Target gas: Methane (CH₄) — M=16.04 g/mol, ρ=0.657 kg/m³ at STP.
Concentration units: kg/m³.  Emission rate Q: kg/s.

Physics:
  - Jos Stam stable fluids (advect, pressure-project, diffuse) on 64×32×32 grid
  - Obstacle-aware velocity/pressure/density (SDF-based Neumann BCs)
  - Lagrangian Gaussian puffs with RK2 advection + Langevin noise
  - Ground reflection via method of images at z=0
  - Puff dispersion: per-puff OU turbulent velocity (Lagrangian stochastic
    model, Thomson 1987) calibrated to Briggs (1973) classes A–F, plus
    relative puff growth σ² = σ₀² + 2·K_rel·age
  - Optional buoyant rise velocity w_rise0·(σ₀/σ)³ (off by default)
"""

import warp as wp
import math
import numpy as np

# ── Methane (CH₄) physical constants ─────────────────────────────────────────
# M = 16.04 g/mol, ρ_STP = 0.657 kg/m³, D_mol = 2.2e-5 m²/s (negligible vs K_D)
# Buoyancy: w_b ≈ 0.5–1.5 m/s (entrainment-limited dilute plume)
# Q in kg/s; each puff carries Q · emit_interval (kg)
# Concentration in kg/m³; to ppm: c × (R·T)/(M·P) × 1e6

# ── Puff dispersion: Lagrangian stochastic model (Thomson 1987) ─────────────
# Each puff carries a turbulent velocity v (per axis, OU / Langevin):
#   v ← v·e^{−dt/T_L} + σ·√(1 − e^{−2dt/T_L})·ξ,   ξ ~ N(0,1) (seeded)
# added to the resolved wind in both RK2 stages.  Turbulence is homogeneous
# (σ, T_L constant with height), so the OU model satisfies Thomson's
# well-mixed condition without a drift term.  Reflections at the ground and
# at obstacles flip the normal component of v.
# Intensities from the Briggs (1973) open-country near-field slopes:
#   σ_u = σ_v = a_y·U,  σ_w = a_z·U,   U = mean wind at source height.
# T_L (s) fitted offline (validation/fit_tl.py → tl_fit.json) so that
#   σ²(x) = σ₀² + 2·K_rel·t + 2σ_v²T_L²(t/T_L − 1 + e^{−t/T_L}),  t = x/U
# matches Briggs σ_y, σ_z over x = 25–200 m at U = LSM_U_FIT, with σ₀ = 1 m,
# K_rel = 0.05 m²/s, T_L ∈ [5, 2000] s.  The sim uses T_L·LSM_U_FIT/U, so the
# Taylor term depends on x only (surface-layer scaling T_L ∝ z/u*, u* ∝ U).
# Fit residuals (sim/Briggs over 25–200 m, from fit_tl.py):
#   A y [1.00,1.03] z [1.00,1.04]   B y [0.99,1.05] z [0.99,1.09]
#   C y [0.98,1.11] z [0.97,1.20]   D y [0.97,1.20] z [0.95,1.35]
#   E y [0.95,1.34] z [0.86,2.05]   F y [0.91,1.67] z [1.00,3.50]
#   (F σ_z hits the 5 s lower bound: Briggs F σ_z(25 m) = 0.4 m < σ₀.  Briggs'
#   formulas are stated for 100 m–10 km; 25–100 m is extrapolation.)
# (First fit used σ₀ = 2 m, K_rel = 0.2: class D failed at 25 m because
# σ₀ > Briggs D σ_z(25 m) = 1.47 m; see validation/results.json.)
#
# Shared eddies (meandering-plume model: Gifford 1959; Anfossi et al. 2006 for
# the time-scale ratio): the velocity of each new puff is
# σ·ζ with ζ an OU process in release time, corr(ζ_n, ζ_{n−1}) = e^{−τ_e/T_E}.
# Successive releases sample the velocity at a *fixed point* (the source), so
# the correlation time is the Eulerian scale T_E = T_L/β with the Hay–Pasquill
# ratio β = γ/i, i = σ/U (= a_y or a_z); Anfossi et al. (2006, Atmos. Environ.
# 40:326) confirm β ∝ 1/i.  γ = HP_GAMMA = 0.6 is an order-one constant (the
# literature value varies with regime; not tuned here).
# Consecutive puffs ride the same large eddy, so the instantaneous plume
# meanders coherently; each puff's own velocity is still N(0, σ²) and OU in
# time, so the mean plume (V1/V3) is unchanged.  This is not GADEN's
# approach: GADEN's per-filament noise is independent and coherence between
# filaments comes only from its time-varying CFD wind.  shared_eddies=False
# gives independent release velocities (GADEN-equivalent puff noise).
# Puff growth σ_puff² = σ₀² + 2·K_rel·age represents relative dispersion only
# (Farrell et al. 2002; GADEN, Monroy et al. 2017).
LSM_U_FIT = 3.0          # m/s, reference wind of the T_L fit
HP_GAMMA = 0.6           # Hay–Pasquill constant in β = T_L/T_E = γ/i
PG_LSM = {
    "A": {"a_y": 0.22, "a_z": 0.20,  "TL_y": 686.7, "TL_z": 1509.2, "desc": "Very unstable"},
    "B": {"a_y": 0.16, "a_z": 0.12,  "TL_y": 512.8, "TL_z":  548.6, "desc": "Unstable"},
    "C": {"a_y": 0.11, "a_z": 0.08,  "TL_y": 321.9, "TL_z":  166.1, "desc": "Slightly unstable"},
    "D": {"a_y": 0.08, "a_z": 0.06,  "TL_y": 198.4, "TL_z":   44.0, "desc": "Neutral (default)"},
    "E": {"a_y": 0.06, "a_z": 0.03,  "TL_y": 121.4, "TL_z":   21.3, "desc": "Slightly stable"},
    "F": {"a_y": 0.04, "a_z": 0.016, "TL_y":  55.3, "TL_z":    5.0, "desc": "Stable"},
}

# Previous constant-K table, replaced by PG_LSM (kept for reference).  Its
# class-D row does not follow its own stated formula: K = σ²(200 m)/(2t) with
# Briggs D gives K_y ≈ 1.9, K_z ≈ 0.8, not 3.8 and 1.3.  Constant K also gives
# σ ∝ √x, which cannot follow Briggs' near-linear σ(x) over 50–200 m (V3, pre-LSM).
# # Pasquill-Gifford turbulent diffusivities K_D (m²/s) for σ² = σ₀² + 2·K_D·t.
# # Derived from Briggs (1973) rural σ_y(x), σ_z(x) power-law formulas
# # at reference distance x=200 m, mean wind U=3 m/s (travel time t≈67 s):
# #   K_D = σ²(x) / (2·t)
# # Cross-wind (Y) and vertical (Z) from Briggs curves; along-wind (X) ≈ 0.3·K_D_Y
# # (along-wind spreading dominated by shear, not turbulent diffusion).
# # References: Gifford (1961), Briggs (1973), Seinfeld & Pandis (2016) Ch. 18.
# PASQUILL_GIFFORD_KD = {
#     "A": {"K_D_X": 4.3,  "K_D_Y": 14.2, "K_D_Z": 11.9,  "desc": "Very unstable"},
#     "B": {"K_D_X": 3.0,  "K_D_Y":  8.5, "K_D_Z":  5.5,  "desc": "Unstable"},
#     "C": {"K_D_X": 2.2,  "K_D_Y":  5.5, "K_D_Z":  2.8,  "desc": "Slightly unstable"},
#     "D": {"K_D_X": 1.5,  "K_D_Y":  3.8, "K_D_Z":  1.3,  "desc": "Neutral (default)"},
#     "E": {"K_D_X": 1.0,  "K_D_Y":  2.2, "K_D_Z":  0.6,  "desc": "Slightly stable"},
#     "F": {"K_D_X": 0.5,  "K_D_Y":  1.1, "K_D_Z":  0.15, "desc": "Stable"},
#     "G": {"K_D_X": 0.25, "K_D_Y":  0.5, "K_D_Z":  0.07, "desc": "Extremely stable"},
# }
#
# # Dispersion split (Gifford 1959 meandering-plume picture; Thomson 1987 for the
# # random-walk ↔ diffusion relation).  The total single-puff-ensemble spread is
# #   σ_α² = σ₀² + 2·K_α·t,
# # shared between puff growth (relative dispersion) and centroid noise:
# #   puff growth   σ_puff² = σ₀² + 2·f_rel·K_α·age
# #   centroid walk ℓ_α = √(2·(1 − f_rel)·K_α)  (m/√s), dX = ℓ_α·dW
# # so the ensemble-mean plume spreads as σ₀² + 2·K_α·t for any f_rel.
# # (Pre-fix: puffs grew with the full K *and* walked with ℓ = √(2K), so the
# # mean plume spread as σ₀² + 4·K·t — dispersion counted twice.)
# # The table below is the f_rel = 0 limit (ℓ for the full K), kept for
# # reference only; the sim computes ℓ from K and f_rel at launch time.
# import math as _math
# PASQUILL_GIFFORD_LANGEVIN = {
#     cls: {
#         "L_X": round(_math.sqrt(2.0 * kd["K_D_X"]), 2),
#         "L_Y": round(_math.sqrt(2.0 * kd["K_D_Y"]), 2),
#         "L_Z": round(_math.sqrt(2.0 * kd["K_D_Z"]), 2),
#     }
#     for cls, kd in PASQUILL_GIFFORD_KD.items()
# }

# Atmospheric lapse rates Γ (K/m) by PG stability class.
# Determines temperature profile T(z) = T_surface - Γ·z.
# Dry adiabatic lapse rate: Γ_d = 0.0098 K/m (9.8 K/km).
# Superadiabatic (Γ > Γ_d): unstable — convective mixing.
# Subadiabatic (Γ < Γ_d): stable — suppressed vertical motion.
# Negative Γ: temperature inversion — very stable.
# Representative single values derived from the ΔT/Δz ranges published by
# the NOAA Air Resources Laboratory READY system:
#   A: > 19.0 K/km,  B: 17–19 K/km,  C: 15–17 K/km,  D: 5–15 K/km (≈9.8),
#   E: −15 to 5 K/km,  F: −40 to −15 K/km,  G: < −40 K/km.
# Reference: NOAA ARL READY (https://www.ready.noaa.gov);
#            Stull (1988), "An Introduction to Boundary Layer Meteorology", Ch. 5.
PG_LAPSE_RATES = {
    "A":  0.020,    # 20 K/km  — strong superadiabatic (representative of > 19 range)
    "B":  0.018,    # 18 K/km  — moderate superadiabatic (midpoint 17–19)
    "C":  0.016,    # 16 K/km  — slightly superadiabatic (midpoint 15–17)
    "D":  0.0098,   # 9.8 K/km — dry adiabatic / neutral (midpoint 5–15 ≈ 9.8)
    "E": -0.005,    # -5 K/km  — subadiabatic (midpoint −15 to 5)
    "F": -0.0275,   # -27.5 K/km — temperature inversion (midpoint −40 to −15)
    "G": -0.045,    # -45 K/km — strong temperature inversion (representative of < −40)
}

# Methane buoyancy: derived from density difference
# ρ_CH4 = 0.657 kg/m³, ρ_air = 1.225 kg/m³ at STP
# a_b = g · (ρ_air - ρ_CH4) / ρ_air ≈ 9.81 · 0.464 ≈ 4.55 m/s²
# This is the UNDILUTED buoyancy for pure methane; real plumes dilute
# rapidly via entrainment.  Following Briggs (1975), the effective
# buoyancy decays with volumetric dilution: a_eff = a_b · (σ₀/σ)³.
# With σ² = σ₀² + 2·K_D·t the puff rise saturates at a finite Δh,
# reproducing the Briggs plume-rise result without an explicit Δh formula.
# Reference: Briggs (1975), "Plume Rise Predictions";
#            Seinfeld & Pandis (2016) §18.4.
# Used only by the legacy grid-gas path (integrate_rho, use_grid_gas=True).
# Puffs use buoyancy_mode="rise" with an initial rise *velocity* w_rise0.
METHANE_BUOYANCY_ACCEL = 9.81 * (1.225 - 0.657) / 1.225  # ≈ 4.55 m/s²
T_SURFACE_DEFAULT = 293.15  # K (20°C reference surface temperature)

# Power-law shear exponents α_s by PG stability class.
# Wind shear profile: u(z) = U_ref·(z/z_ref)^α_s with z_ref = SHEAR_Z_REF
# (U_ref = base wind speed at 10 m).  Pre-LSM: s = ((k+0.5)/NZ)^α, i.e. z
# normalised by the 128 m domain top, so U_ref was the wind at 128 m.
# Unstable → well-mixed boundary layer → low shear exponent.
# Stable → strong wind gradient → high shear exponent.
# Range α_s ∈ {0.10, …, 0.35} as in wind_field_method.tex.
PG_SHEAR_EXPONENTS = {
    "A": 0.10,   # Very unstable — well-mixed, nearly uniform profile
    "B": 0.15,   # Unstable
    "C": 0.20,   # Slightly unstable
    "D": 0.25,   # Neutral (default)
    "E": 0.30,   # Slightly stable
    "F": 0.35,   # Stable — strong shear near ground
    "G": 0.40,   # Extremely stable — very strong shear
}
SHEAR_Z_REF = 10.0   # m, reference height of the shear profile
EDDY_HIST_N = 16384  # stored shared-eddy release velocities (≥ 3200 s at 0.2 s)

# ── Surface-layer turbulence (v1.1 two-scale LSM, surface_layer=True) ─────────
# Small scales: mechanical surface-layer turbulence, independent per puff,
#   σ_u, σ_v, σ_w = (2.4, 1.9, 1.25)·u*   (Panofsky & Dutton 1984),
#   σ_w(z) = 1.25 u*(1 − 3z/L)^{1/3} for L < 0 (Panofsky et al. 1977),
#   ε(z) = u*³/(κz)·φ_ε(z/L), φ_ε = (1 + 0.5|ζ|^{2/3})^{3/2} (ζ<0), 1 + 5ζ (ζ≥0)
#   (Kaimal & Finnigan 1994), T_i(z) = 2σ_i²/(C0 ε(z)) (Thomson 1987),
#   with C0 = 3.125 so that T_w = 0.5 z/σ_w in neutral conditions (Wilson,
#   Thurtell & Kidd 1981).  Vertical: Thomson's well-mixed drift
#   ½(1 + w²/σ_w²)∂σ_w²/∂z.  Large scales (unstable only): convective lateral
#   eddies, σ_c² = σ_v,P² − (1.9u*)² with σ_v,P = u*(12 + 0.5 z_i/|L|)^{1/3}
#   (Panofsky et al. 1977), T_c = 0.15 z_i/σ_v,P (Hanna 1982); coherent across
#   puffs through the shared-eddy release process.
# Met inputs: u*, L, z_i — from (class, U10, z0) via the Golder (1972) 1/L–z0
# relation and Monin–Obukhov similarity, or given (met=dict(ustar, inv_L, zi)).
KARMAN = 0.4
SL_SIGMA_MECH = (2.4, 1.9, 1.25)
SL_C0 = 3.125
SL_Z_MIN = 0.5           # m, floor of z in ε(z) and T(z)
GOLDER_AB = {"A": (-0.096, 0.029), "B": (-0.037, 0.029), "C": (-0.002, 0.018),
             "D": (0.0, 0.0), "E": (0.004, -0.018), "F": (0.035, -0.036)}


def golder_inv_L(pg_class, z0):
    """1/L (1/m) for a Pasquill class and roughness z0 (Golder 1972, fitted form
    1/L = a + b·log10 z0 as tabulated in Seinfeld & Pandis)."""
    a, b = GOLDER_AB[pg_class.upper()]
    return a + b * math.log10(z0)


def psi_m(zeta):
    """Businger–Dyer stability correction of the wind profile."""
    if zeta >= 0:
        return -5.0 * min(zeta, 1.0) - (5.0 * math.log(zeta) if zeta > 1 else 0.0)
    x = (1 - 16 * zeta) ** 0.25
    return 2 * math.log((1 + x) / 2) + math.log((1 + x * x) / 2) - 2 * math.atan(x) + math.pi / 2


def friction_velocity(U, z, z0, inv_L):
    """u* (m/s) from the wind speed U at height z (log law with ψ_m)."""
    den = math.log(z / z0) - psi_m(z * inv_L) + psi_m(z0 * inv_L)
    return max(KARMAN * U / max(den, 1e-3), 0.01)


def sl_phi_eps(zeta):
    if zeta < 0:
        return (1 + 0.5 * abs(zeta) ** (2.0 / 3.0)) ** 1.5
    return 1 + 5 * min(zeta, 10.0)


def sl_sigma_w(z, ustar, inv_L):
    z = max(z, SL_Z_MIN)
    return SL_SIGMA_MECH[2] * ustar * ((1 - 3 * z * inv_L) ** (1.0 / 3.0) if inv_L < 0 else 1.0)


def sl_eps(z, ustar, inv_L):
    z = max(z, SL_Z_MIN)
    return ustar ** 3 / (KARMAN * z) * sl_phi_eps(z * inv_L)


@wp.func
def sl_phi_eps_w(zeta: float):
    if zeta < 0.0:
        return wp.pow(1.0 + 0.5 * wp.pow(wp.abs(zeta), 2.0 / 3.0), 1.5)
    return 1.0 + 5.0 * wp.min(zeta, 10.0)


@wp.func
def sl_sigma_w2_w(z: float, ustar: float, inv_L: float):
    if inv_L < 0.0:
        return 1.5625 * ustar * ustar * wp.pow(1.0 - 3.0 * z * inv_L, 2.0 / 3.0)
    return 1.5625 * ustar * ustar


@wp.func
def sl_dsigma_w2_dz_w(z: float, ustar: float, inv_L: float):
    if inv_L < 0.0:
        return -3.125 * ustar * ustar * inv_L * wp.pow(1.0 - 3.0 * z * inv_L, -1.0 / 3.0)
    return 0.0

# Wind presets (stochastic resolved-wind amplitude, see HybridFluidPuffSimulation).
#   calibrated: resolved fluctuations small, mean-direction meander ≲ 3° at the
#               env wind (1.5 m/s); puff turbulence comes from the LSM, which is
#               calibrated to Briggs.  Default.
#   meandering: pre-LSM amplitudes (coarse OU field σ_stat ≈ 2.7 m/s, mean-wind
#               meander ≈ 14° at 3 m/s).  Harder RL; adds to the LSM, so the
#               plume is wider than the class calibration.
WIND_PRESETS = {
    "calibrated": {"turb_scale": 0.1},
    "meandering": {"turb_scale": 1.0},
}

# ── Grid dimensions (single source of truth for all files) ────────────────────
NX, NY, NZ = 64, 32, 32          # cells
DX = 4.0                          # m/cell → domain 256×128×128 m

WIND_NX = wp.constant(NX)         # compile-time copies for Warp kernels
WIND_NY = wp.constant(NY)
WIND_NZ = wp.constant(NZ)

_MAX_SPHERES = 10
_MAX_CUBOIDS = 10
MAX_SPHERES = wp.constant(_MAX_SPHERES)
MAX_CUBOIDS = wp.constant(_MAX_CUBOIDS)

# Coarse turbulence grid — one-eighth resolution per axis
_TURB_NX, _TURB_NY, _TURB_NZ = NX // 8, NY // 8, NZ // 8
TURB_NX = wp.constant(_TURB_NX)
TURB_NY = wp.constant(_TURB_NY)
TURB_NZ = wp.constant(_TURB_NZ)

# ── SDF ──────────────────────────────────────────────────────────────────────

@wp.func
def sdf_sphere(p: wp.vec3, center: wp.vec3, radius: float):
    return wp.length(p - center) - radius

@wp.func
def normal_sphere(p: wp.vec3, center: wp.vec3):
    d = p - center
    if wp.length(d) < 1.0e-6:      # degenerate: point at the centre → push upward
        return wp.vec3(0.0, 0.0, 1.0)
    return wp.normalize(d)

@wp.func
def sdf_cuboid(p: wp.vec3, center: wp.vec3, half_extents: wp.vec3):
    d = wp.vec3(
        wp.abs(p[0]-center[0])-half_extents[0],
        wp.abs(p[1]-center[1])-half_extents[1],
        wp.abs(p[2]-center[2])-half_extents[2],
    )
    outside = wp.length(wp.vec3(wp.max(d[0],0.0),wp.max(d[1],0.0),wp.max(d[2],0.0)))
    inside  = wp.min(wp.max(d[0],wp.max(d[1],d[2])),0.0)
    return outside + inside

@wp.func
def normal_cuboid(p: wp.vec3, center: wp.vec3, half_extents: wp.vec3):
    d = wp.vec3(
        wp.abs(p[0]-center[0])-half_extents[0],
        wp.abs(p[1]-center[1])-half_extents[1],
        wp.abs(p[2]-center[2])-half_extents[2],
    )
    max_d = wp.max(d[0],wp.max(d[1],d[2]))
    n = wp.vec3(0.0,0.0,0.0)
    if max_d == d[0]:   n[0] = wp.sign(p[0]-center[0])
    elif max_d == d[1]: n[1] = wp.sign(p[1]-center[1])
    else:               n[2] = wp.sign(p[2]-center[2])
    return n

# ── Line-of-sight tests (exact segment–primitive intersection) ──────────────
# Used to occlude puffs whose centre is not visible from the sample point
# (same approximation as GADEN's filament-centre line-of-sight test).

@wp.func
def seg_hits_sphere(p0: wp.vec3, p1: wp.vec3, c: wp.vec3, r: float):
    d = p1 - p0
    t = wp.clamp(wp.dot(c - p0, d) / wp.max(wp.dot(d, d), 1.0e-9), 0.0, 1.0)
    return wp.length(p0 + t * d - c) < r

@wp.func
def seg_hits_box(p0: wp.vec3, p1: wp.vec3, c: wp.vec3, h: wp.vec3):
    # Slab test on the segment parameter t ∈ [0, 1]
    d = p1 - p0
    tmin = float(0.0)
    tmax = float(1.0)
    hit = int(1)
    for a in range(3):
        if wp.abs(d[a]) < 1.0e-9:
            if wp.abs(p0[a] - c[a]) > h[a]:
                hit = 0
        else:
            inv = 1.0 / d[a]
            t1 = (c[a] - h[a] - p0[a]) * inv
            t2 = (c[a] + h[a] - p0[a]) * inv
            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
    if tmin > tmax:
        hit = 0
    return hit == 1

@wp.func
def los_blocked(
    p0: wp.vec3, p1: wp.vec3,
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
):
    blocked = int(0)
    for s in range(n_spheres):
        if seg_hits_sphere(p0, p1, sphere_centers[s], sphere_radii[s]):
            blocked = 1
    for ci in range(n_cuboids):
        if seg_hits_box(p0, p1, cuboid_centers[ci], cuboid_extents[ci]):
            blocked = 1
    return blocked == 1

@wp.func
def inside_obstacle(
    p: wp.vec3,
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
):
    inside = int(0)
    for s in range(n_spheres):
        if sdf_sphere(p, sphere_centers[s], sphere_radii[s]) < 0.0:
            inside = 1
    for ci in range(n_cuboids):
        if sdf_cuboid(p, cuboid_centers[ci], cuboid_extents[ci]) < 0.0:
            inside = 1
    return inside == 1

# ── Obstacle mask computation ────────────────────────────────────────────────
# Computes a per-cell flag: 1 = inside obstacle (solid), 0 = fluid
# Uses the same SDF primitives the puff system already uses.


@wp.kernel
def compute_obstacle_mask(
    mask: wp.array3d(dtype=int),
    dx: float,
    n_spheres: int,
    sphere_centers: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    n_cuboids: int,
    cuboid_centers: wp.array(dtype=wp.vec3),
    cuboid_extents: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    px = (float(i) + 0.5) * dx
    py = (float(j) + 0.5) * dx
    pz = (float(k) + 0.5) * dx
    p = wp.vec3(px, py, pz)

    # Declare as dynamic float so Warp allows mutation inside loops
    min_sdf = float(1.0e10)

    for s in range(n_spheres):
        d = sdf_sphere(p, sphere_centers[s], sphere_radii[s])
        if d < min_sdf:
            min_sdf = d

    for ci in range(n_cuboids):
        d = sdf_cuboid(p, cuboid_centers[ci], cuboid_extents[ci])
        if d < min_sdf:
            min_sdf = d

    if min_sdf <= 0.0:
        mask[i, j, k] = 1
    else:
        mask[i, j, k] = 0
# ── Enforce no-slip: zero velocity inside obstacles ─────────────────────────

@wp.kernel
def enforce_velocity_obstacle(
    u: wp.array3d(dtype=wp.vec3),
    mask: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    if mask[i, j, k] == 1:
        u[i, j, k] = wp.vec3(0.0, 0.0, 0.0)


# ── Zero density inside obstacles ───────────────────────────────────────────

@wp.kernel
def enforce_density_obstacle(
    rho: wp.array3d(dtype=float),
    mask: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    if mask[i, j, k] == 1:
        rho[i, j, k] = 0.0


# ── Velocity + scalar interpolation ──────────────────────────────────────────

# Grid convention: cell (i, j, k) stores the value at its centre
# ((i+0.5)·dx, (j+0.5)·dx, (k+0.5)·dx), matching the obstacle mask and the
# shear profile. World position p maps to grid coordinate p/dx - 0.5
# (world_to_grid). Samplers clamp the coordinate to [0, N-1] *before*
# splitting into cell + fraction, so points outside the grid take the edge
# value (clamp-to-edge). Pre-fix code sampled at p/dx (a 2 m shift) and
# clamped the cell index after computing the fraction.

@wp.func
def world_to_grid(p: wp.vec3, dx: float):
    return wp.vec3(p[0]/dx - 0.5, p[1]/dx - 0.5, p[2]/dx - 0.5)

@wp.func
def sample_vel_3d(u: wp.array3d(dtype=wp.vec3), x: float, y: float, z: float):
    x=wp.clamp(x,0.0,float(WIND_NX-1)); y=wp.clamp(y,0.0,float(WIND_NY-1)); z=wp.clamp(z,0.0,float(WIND_NZ-1))
    lx=wp.min(int(wp.floor(x)),WIND_NX-2); ly=wp.min(int(wp.floor(y)),WIND_NY-2); lz=wp.min(int(wp.floor(z)),WIND_NZ-2)
    tx=x-float(lx); ty=y-float(ly); tz=z-float(lz)
    c000=u[lx,ly,lz];     c100=u[lx+1,ly,lz]
    c010=u[lx,ly+1,lz];   c110=u[lx+1,ly+1,lz]
    c001=u[lx,ly,lz+1];   c101=u[lx+1,ly,lz+1]
    c011=u[lx,ly+1,lz+1]; c111=u[lx+1,ly+1,lz+1]
    c00=wp.lerp(c000,c100,tx); c01=wp.lerp(c001,c101,tx)
    c10=wp.lerp(c010,c110,tx); c11=wp.lerp(c011,c111,tx)
    c0=wp.lerp(c00,c10,ty);    c1=wp.lerp(c01,c11,ty)
    return wp.lerp(c0,c1,tz)

@wp.func
def lookup_rho_3d(rho: wp.array3d(dtype=float), x: int, y: int, z: int):
    x=wp.clamp(x,0,WIND_NX-1); y=wp.clamp(y,0,WIND_NY-1); z=wp.clamp(z,0,WIND_NZ-1)
    return rho[x,y,z]

@wp.func
def sample_rho_3d(rho: wp.array3d(dtype=float), x: float, y: float, z: float):
    x=wp.clamp(x,0.0,float(WIND_NX-1)); y=wp.clamp(y,0.0,float(WIND_NY-1)); z=wp.clamp(z,0.0,float(WIND_NZ-1))
    lx=wp.min(int(wp.floor(x)),WIND_NX-2); ly=wp.min(int(wp.floor(y)),WIND_NY-2); lz=wp.min(int(wp.floor(z)),WIND_NZ-2)
    tx=x-float(lx); ty=y-float(ly); tz=z-float(lz)
    c000=lookup_rho_3d(rho,lx,ly,lz);     c100=lookup_rho_3d(rho,lx+1,ly,lz)
    c010=lookup_rho_3d(rho,lx,ly+1,lz);   c110=lookup_rho_3d(rho,lx+1,ly+1,lz)
    c001=lookup_rho_3d(rho,lx,ly,lz+1);   c101=lookup_rho_3d(rho,lx+1,ly,lz+1)
    c011=lookup_rho_3d(rho,lx,ly+1,lz+1); c111=lookup_rho_3d(rho,lx+1,ly+1,lz+1)
    c00=wp.lerp(c000,c100,tx); c01=wp.lerp(c001,c101,tx)
    c10=wp.lerp(c010,c110,tx); c11=wp.lerp(c011,c111,tx)
    c0=wp.lerp(c00,c10,ty);    c1=wp.lerp(c01,c11,ty)
    return wp.lerp(c0,c1,tz)

# ── Unified advect: velocity + scalar ─────────────────────────────────────────

@wp.kernel
def advect_vel_and_rho(
    u0: wp.array3d(dtype=wp.vec3), u1: wp.array3d(dtype=wp.vec3),
    rho0: wp.array3d(dtype=float), rho1: wp.array3d(dtype=float),
    dt: float,
):
    i,j,k = wp.tid()
    u = u0[i,j,k]
    p = wp.vec3(float(i),float(j),float(k)) - u*(dt/DX)
    u1[i,j,k]   = sample_vel_3d(u0, p[0],p[1],p[2])
    rho1[i,j,k] = sample_rho_3d(rho0, p[0],p[1],p[2])

# ── Isotropic Laplacian diffusion on rho ─────────────────────────────────────

@wp.kernel
def diffuse_rho(
    rho_in:  wp.array3d(dtype=float),
    rho_out: wp.array3d(dtype=float),
    mask:    wp.array3d(dtype=int),
    D_iso:   float,
    dt:      float,
    dx:      float,
):
    i,j,k = wp.tid()
    # Inside obstacle: density stays zero
    if mask[i, j, k] == 1:
        rho_out[i, j, k] = 0.0
        return
    c = rho_in[i,j,k]
    # Neighbors: if neighbor is obstacle or boundary, use Neumann (no-flux) = copy c
    r_xm = rho_in[i-1,j,k] if (i > 0         and mask[i-1,j,k] == 0) else c
    r_xp = rho_in[i+1,j,k] if (i < WIND_NX-1 and mask[i+1,j,k] == 0) else c
    r_ym = rho_in[i,j-1,k] if (j > 0         and mask[i,j-1,k] == 0) else c
    r_yp = rho_in[i,j+1,k] if (j < WIND_NY-1 and mask[i,j+1,k] == 0) else c
    r_zm = rho_in[i,j,k-1] if (k > 0         and mask[i,j,k-1] == 0) else c
    r_zp = rho_in[i,j,k+1] if (k < WIND_NZ-1 and mask[i,j,k+1] == 0) else c

    lap = (r_xp + r_xm + r_yp + r_ym + r_zp + r_zm - 6.0*c) / (dx*dx)
    rho_out[i,j,k] = c + D_iso * lap * dt

# ── Buoyancy + decay (temperature-profile-based) ────────────────────────────
# Buoyancy acceleration derived from methane-air density difference,
# scaled by local concentration. The environmental temperature profile
# T_env(z) = T_surface - Γ·z modulates the buoyancy via the stability.
# Reference: Stull (1988), Ch. 5; Seinfeld & Pandis (2016), Ch. 18.

@wp.kernel
def integrate_rho(
    u: wp.array3d(dtype=wp.vec3), rho: wp.array3d(dtype=float),
    dt: float, buoy_accel: float, decay_rate: float,
    lapse_rate: float, t_surface: float, dx: float,
    sigma_0: float, K_d_z: float, sim_time: float,
):
    i,j,k = wp.tid()
    r = rho[i,j,k]
    if r < 1.0e-9: return
    # Height-dependent buoyancy via temperature profile
    z = (float(k) + 0.5) * dx
    t_env = t_surface - lapse_rate * z
    # Briggs (1975) dilution-limited buoyancy: a_eff = a_b · (σ₀/σ)³
    # σ² = σ₀² + 2·K·t  →  dilution = (σ₀²/σ²)^(3/2)   (legacy: K = K_rel)
    # For the grid density we use a global age estimate (sim_time) since
    # individual cell ages are not tracked.  This makes the plume buoyancy
    # decay rapidly after the first few seconds, producing a finite rise.
    sig2 = sigma_0 * sigma_0 + 2.0 * K_d_z * wp.max(sim_time, 0.01)
    sig2_0 = sigma_0 * sigma_0
    dilution = wp.pow(sig2_0 / sig2, 1.5)  # (σ₀/σ)³
    a_b = buoy_accel * dilution * (293.15 / wp.max(t_env, 200.0))
    u[i,j,k] = u[i,j,k] + wp.vec3(0.0, 0.0, a_b * dt)
    rho[i,j,k] = r * (1.0 - decay_rate * dt)

# ── Fixed Emitter ────────────────────────────────────────────────────────────

@wp.kernel
def emit_density_fixed(
    rho:      wp.array3d(dtype=float),
    u:        wp.array3d(dtype=wp.vec3),
    src_gx:   float, src_gy: float, src_gz: float,
    emit_radius:   float,
    emit_strength: float,
    inject_dir: wp.vec3,
):
    i,j,k = wp.tid()
    dx_=float(i)-src_gx; dy_=float(j)-src_gy; dz_=float(k)-src_gz
    dist2=dx_*dx_+dy_*dy_+dz_*dz_; r2=emit_radius*emit_radius
    if dist2 < r2*9.0:
        w = wp.exp(-dist2/(2.0*r2))
        rho[i,j,k] = rho[i,j,k] + emit_strength*w
        u[i,j,k] = u[i,j,k] + inject_dir*w*0.5

# ── Pressure / divergence (obstacle-aware) ───────────────────────────────────

@wp.kernel
def wind_divergence_3d(
    u: wp.array3d(dtype=wp.vec3),
    div: wp.array3d(dtype=float),
    mask: wp.array3d(dtype=int),
):
    i,j,k = wp.tid()
    # Ground (k=0): solid wall — zero divergence
    if k == 0:
        div[i,j,k] = 0.0
        return
    # Inside obstacle: zero divergence (no fluid here)
    if mask[i,j,k] == 1:
        div[i,j,k] = 0.0
        return
    # Fluid cell: compute divergence with obstacle-aware neighbors
    # If a neighbor is obstacle or boundary, use current cell velocity (no-penetration)
    ux_p = u[i+1,j,k][0] if (i < WIND_NX-1 and mask[i+1,j,k] == 0) else u[i,j,k][0]
    ux_m = u[i-1,j,k][0] if (i > 0         and mask[i-1,j,k] == 0) else u[i,j,k][0]
    uy_p = u[i,j+1,k][1] if (j < WIND_NY-1 and mask[i,j+1,k] == 0) else u[i,j,k][1]
    uy_m = u[i,j-1,k][1] if (j > 0         and mask[i,j-1,k] == 0) else u[i,j,k][1]
    uz_p = u[i,j,k+1][2] if (k < WIND_NZ-1 and mask[i,j,k+1] == 0) else u[i,j,k][2]
    uz_m = u[i,j,k-1][2] if                    (mask[i,j,k-1] == 0) else u[i,j,k][2]
    div[i,j,k] = (ux_p - ux_m + uy_p - uy_m + uz_p - uz_m) * 0.5


# Red-Black SOR pressure solver (Young 1971).
# Cells are partitioned by parity (i+j+k) % 2.  Red cells (parity=0) are
# updated first using black neighbours, then black cells using updated red
# neighbours.  Each half-sweep is fully GPU-parallel.
# SOR update: p_new = p_old + ω·(p_GS - p_old), with ω = 1.7.

@wp.kernel
def pressure_sor_rb(
    p: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    mask: wp.array3d(dtype=int),
    omega: float,
    color: int,
):
    i,j,k = wp.tid()
    # Only update cells matching the current color
    if (i + j + k) % 2 != color:
        return
    if mask[i,j,k] == 1:
        p[i,j,k] = 0.0
        return

    p_sum = float(0.0)
    n_fluid = float(0.0)

    # x-minus
    if i > 0:
        if mask[i-1,j,k] == 0:
            p_sum += p[i-1,j,k]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        n_fluid += 1.0

    # x-plus
    if i < WIND_NX-1:
        if mask[i+1,j,k] == 0:
            p_sum += p[i+1,j,k]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        n_fluid += 1.0

    # y-minus
    if j > 0:
        if mask[i,j-1,k] == 0:
            p_sum += p[i,j-1,k]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        n_fluid += 1.0

    # y-plus
    if j < WIND_NY-1:
        if mask[i,j+1,k] == 0:
            p_sum += p[i,j+1,k]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        n_fluid += 1.0

    # z-minus
    if k > 0:
        if mask[i,j,k-1] == 0:
            p_sum += p[i,j,k-1]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        p_sum += p[i,j,0]
        n_fluid += 1.0

    # z-plus
    if k < WIND_NZ-1:
        if mask[i,j,k+1] == 0:
            p_sum += p[i,j,k+1]
        else:
            p_sum += p[i,j,k]
        n_fluid += 1.0
    else:
        n_fluid += 1.0

    p_gs = (p_sum - div[i,j,k]) / n_fluid  # Gauss-Seidel value
    old = p[i,j,k]
    p[i,j,k] = old + omega * (p_gs - old)  # SOR (no residual tracking needed)

@wp.kernel
def pressure_apply_3d(
    p: wp.array3d(dtype=float),
    u: wp.array3d(dtype=wp.vec3),
    mask: wp.array3d(dtype=int),
):
    i,j,k = wp.tid()
    # Inside obstacle: velocity stays zero (already enforced, but be safe)
    if mask[i,j,k] == 1:
        u[i,j,k] = wp.vec3(0.0, 0.0, 0.0)
        return
    # Ground (k=0): solid wall — zero out vertical velocity, skip correction
    if k==0:
        vel = u[i,j,k]
        u[i,j,k] = wp.vec3(vel[0], vel[1], 0.0)
        return
    # Pressure gradient with obstacle-aware Neumann BCs:
    # If neighbor is obstacle, use current cell's pressure (∂p/∂n = 0 → gradient = 0 in that direction)
    p_c = p[i,j,k]
    p_xp = p[i+1,j,k] if (i < WIND_NX-1 and mask[i+1,j,k] == 0) else p_c
    p_xm = p[i-1,j,k] if (i > 0         and mask[i-1,j,k] == 0) else p_c
    p_yp = p[i,j+1,k] if (j < WIND_NY-1 and mask[i,j+1,k] == 0) else p_c
    p_ym = p[i,j-1,k] if (j > 0         and mask[i,j-1,k] == 0) else p_c
    p_zp = p[i,j,k+1] if (k < WIND_NZ-1 and mask[i,j,k+1] == 0) else p_c
    p_zm = p[i,j,k-1] if                    (mask[i,j,k-1] == 0) else p_c  # k>0 guaranteed
    # Open domain boundaries: Dirichlet p=0
    if i == WIND_NX-1 and mask[i,j,k] == 0:
        p_xp = 0.0
    if i == 0 and mask[i,j,k] == 0:
        p_xm = 0.0
    if j == WIND_NY-1 and mask[i,j,k] == 0:
        p_yp = 0.0
    if j == 0 and mask[i,j,k] == 0:
        p_ym = 0.0
    if k == WIND_NZ-1 and mask[i,j,k] == 0:
        p_zp = 0.0

    grad_p = wp.vec3(
        (p_xp - p_xm)*0.5,
        (p_yp - p_ym)*0.5,
        (p_zp - p_zm)*0.5,
    )
    u[i,j,k] = u[i,j,k] - grad_p

@wp.func
def sample_turb_3d(turb: wp.array3d(dtype=wp.vec3), x: float, y: float, z: float):
    """Trilinear interpolation on the coarse turbulence grid (clamp-to-edge)."""
    x = wp.clamp(x, 0.0, float(TURB_NX - 1)); y = wp.clamp(y, 0.0, float(TURB_NY - 1)); z = wp.clamp(z, 0.0, float(TURB_NZ - 1))
    lx = wp.min(int(wp.floor(x)), TURB_NX - 2); ly = wp.min(int(wp.floor(y)), TURB_NY - 2); lz = wp.min(int(wp.floor(z)), TURB_NZ - 2)
    tx = x - float(lx); ty = y - float(ly); tz = z - float(lz)
    c000 = turb[lx, ly, lz];       c100 = turb[lx+1, ly, lz]
    c010 = turb[lx, ly+1, lz];     c110 = turb[lx+1, ly+1, lz]
    c001 = turb[lx, ly, lz+1];     c101 = turb[lx+1, ly, lz+1]
    c011 = turb[lx, ly+1, lz+1];   c111 = turb[lx+1, ly+1, lz+1]
    c00 = wp.lerp(c000, c100, tx);  c01 = wp.lerp(c001, c101, tx)
    c10 = wp.lerp(c010, c110, tx);  c11 = wp.lerp(c011, c111, tx)
    c0 = wp.lerp(c00, c10, ty);     c1 = wp.lerp(c01, c11, ty)
    return wp.lerp(c0, c1, tz)

@wp.kernel
def apply_global_wind(
    u: wp.array3d(dtype=wp.vec3),
    wind_x:float, wind_y:float, wind_z:float,
    alpha:float, dt:float, shear_exp:float,
    turb_field: wp.array3d(dtype=wp.vec3),
    mask: wp.array3d(dtype=int),
    z_ref:float, dx:float,
    drag_rate:float, n_drag:int, drag_comp:int,
):
    """Newtonian relaxation (nudging) toward target wind profile.

    Instead of accumulating wind as a force (u += w·dt, unbounded),
    we relax toward the target: u += α·(u_target − u)·dt.
    This is bounded by construction — |u| ≤ |u_target|/α in steady state.
    Reference: Stauffer & Seaman (1990), Mon. Wea. Rev.
    """
    i,j,k = wp.tid()
    if mask[i,j,k] == 1:
        return
    # Power law u(z) = U_ref·(z/z_ref)^α at the cell centre.
    # (Pre-LSM: z_frac = (k+0.5)/NZ, shear = max(z_frac, 0.02)^α — z
    # normalised by the domain top.)
    z=(float(k)+0.5)*dx
    shear=wp.pow(wp.max(z,0.5)/z_ref,shear_exp)
    # Drag compensation: relaxation (rate n = αΔt per step) followed by the
    # ground-drag factor d_k gives a steady state u/u_tgt = d·n/(1 − d(1 − n))
    # (0.79 at the 10 m cell).  Dividing the target by this factor in the drag
    # layers makes the mean resolved profile follow the power law, so the
    # realised 10 m wind equals the requested base speed; drag still damps
    # deviations from the target near the ground.
    if drag_comp != 0 and k < n_drag:
        nn=wp.min(alpha*dt,1.0)
        d=wp.exp(-drag_rate*dt*(1.0-float(k)/float(n_drag)))
        shear=shear*(1.0-d*(1.0-nn))/(d*nn)
    # Map fine grid coords to coarse turbulence grid coords
    tx = (float(i) + 0.5) / float(WIND_NX) * float(TURB_NX - 1)
    ty = (float(j) + 0.5) / float(WIND_NY) * float(TURB_NY - 1)
    tz = (float(k) + 0.5) / float(WIND_NZ) * float(TURB_NZ - 1)
    perturb = sample_turb_3d(turb_field, tx, ty, tz)
    # Target wind at this height: mean wind + spatial turbulence, shaped by shear
    u_tgt_x = (wind_x + perturb[0]) * shear
    u_tgt_y = (wind_y + perturb[1]) * shear
    u_tgt_z = (wind_z + perturb[2]) * shear * 0.3
    vel = u[i,j,k]
    # Relaxation: u_new = u + α·(u_target − u)·dt
    nudge = alpha * dt
    # Clamp nudge factor to [0, 1] to ensure stability
    nudge = wp.min(nudge, 1.0)
    u[i,j,k] = wp.vec3(
        vel[0] + nudge * (u_tgt_x - vel[0]),
        vel[1] + nudge * (u_tgt_y - vel[1]),
        vel[2] + nudge * (u_tgt_z - vel[2]),
    )

@wp.kernel
def apply_ground_drag(u: wp.array3d(dtype=wp.vec3), drag_rate:float, dt:float, n_layers:int):
    """Linear ground drag in the lowest n_layers cells, scaled by dt:
        u ← u · exp(−drag_rate · dt · (1 − k/n_layers)).
    (Pre-fix: u ← u·max(1 − 0.08·(1 − k/n), 0) per frame, i.e. frame-rate
    dependent; drag_rate = 2.4 /s reproduces it at 30 fps.)"""
    i,j,k = wp.tid()
    if k<n_layers:
        frac=float(k)/float(n_layers)
        u[i,j,k]=u[i,j,k]*wp.exp(-drag_rate*dt*(1.0-frac))

@wp.kernel
def reflect_puffs_at_boundary(
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_vel: wp.array(dtype=wp.vec3),
    domain_max_x:float, domain_max_y:float, domain_max_z:float,
    counters: wp.array(dtype=int),
    puff_mass: wp.array(dtype=float), mass_counters: wp.array(dtype=float),
    closed: int, puff_vs: wp.array(dtype=wp.vec3),
):
    i=wp.tid()
    if puff_active[i]==0: return
    p=puff_pos[i]; px=p[0]; py=p[1]; pz=p[2]
    if closed != 0:
        # Closed box (tests only): reflect position and turbulent velocity at
        # every face, as at the ground.
        v=puff_vel[i]; vx=v[0]; vy=v[1]; vz=v[2]
        if px<0.0:
            px=-px; vx=-vx
        if px>domain_max_x:
            px=2.0*domain_max_x-px; vx=-vx
        if py<0.0:
            py=-py; vy=-vy
        if py>domain_max_y:
            py=2.0*domain_max_y-py; vy=-vy
        if pz<0.0:
            pz=-pz; vz=-vz
        if pz>domain_max_z:
            pz=2.0*domain_max_z-pz; vz=-vz
        puff_pos[i]=wp.vec3(px,py,pz); puff_vel[i]=wp.vec3(vx,vy,vz)
        s=puff_vs[i]; sx=s[0]; sy=s[1]; sz=s[2]
        if p[0]<0.0 or p[0]>domain_max_x:
            sx=-sx
        if p[1]<0.0 or p[1]>domain_max_y:
            sy=-sy
        if p[2]<0.0 or p[2]>domain_max_z:
            sz=-sz
        puff_vs[i]=wp.vec3(sx,sy,sz)
        return
    # Open faces (x, y, z_top): deactivate puffs that leave the domain
    if px<0.0 or px>domain_max_x or py<0.0 or py>domain_max_y or pz>domain_max_z:
        puff_active[i]=0
        wp.atomic_add(counters, COUNTER_EXITED, 1)
        wp.atomic_add(mass_counters, COUNTER_EXITED, puff_mass[i])
        return
    # Ground (z=0): solid wall — reflect position and the vertical
    # turbulent velocity (perfect reflection keeps the LSM well mixed)
    if pz<0.0:
        pz=-pz
        v=puff_vel[i]
        puff_vel[i]=wp.vec3(v[0], v[1], -v[2])
        s=puff_vs[i]
        puff_vs[i]=wp.vec3(s[0], s[1], -s[2])
    puff_pos[i]=wp.vec3(px, py, pz)

# ── Lagrangian puffs ─────────────────────────────────────────────────────────

# Indices into HybridFluidPuffSimulation._counters
COUNTER_OVERWRITTEN = wp.constant(0)
COUNTER_EXITED = wp.constant(1)

@wp.kernel
def emit_puff_kernel(
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float),
    puff_vel: wp.array(dtype=wp.vec3),
    idx: int, pos: wp.vec3, emit_time: float, mass: float,
    counters: wp.array(dtype=int),
    vel: wp.vec3,
    mass_counters: wp.array(dtype=float),
    puff_vs: wp.array(dtype=wp.vec3), vs: wp.vec3,
):
    # A still-active puff in this ring-buffer slot means max_puffs is too
    # small and its mass is lost: count it.
    if puff_active[idx] == 1:
        wp.atomic_add(counters, COUNTER_OVERWRITTEN, 1)
        wp.atomic_add(mass_counters, COUNTER_OVERWRITTEN, puff_mass[idx])
    puff_pos[idx] = pos
    puff_active[idx] = 1
    puff_emit_time[idx] = emit_time
    puff_mass[idx] = mass
    # Turbulent velocity at release (host: σ·ζ, ζ ~ N(0,1), shared eddies)
    puff_vel[idx] = vel
    # Small-scale (surface-layer) velocity, independent per puff
    puff_vs[idx] = vs

BUOY_NONE = wp.constant(0)
BUOY_RISE = wp.constant(1)

SUBGRID_Z_FLOOR = 0.1   # m, lowest height of the sub-grid wind profile
SUBGRID_Z_FLOOR_W = wp.constant(SUBGRID_Z_FLOOR)


@wp.func
def subgrid_scale(v: wp.vec3, z: float, dx: float, shear_exp: float, on: int):
    """Sub-grid surface-layer wind below the lowest cell centre z1 = dx/2.
    The resolved field is constant below z1 (clamped interpolation), so puffs
    near the ground moved at the z1 wind.  Extend the power-law profile the
    wind is relaxed to: u_h(z) = u_h(z1)·(max(z, z_floor)/z1)^α for z < z1."""
    z1 = 0.5 * dx
    if on == 0 or z >= z1:
        return v
    f = wp.pow(wp.max(z, SUBGRID_Z_FLOOR_W) / z1, shear_exp)
    return wp.vec3(v[0] * f, v[1] * f, v[2])


@wp.kernel
def advect_puffs_rk2(
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), wind_field: wp.array3d(dtype=wp.vec3),
    dt:float, dx:float, current_time:float,
    puff_vel: wp.array(dtype=wp.vec3), ou_decay: wp.vec3, ou_noise: wp.vec3,
    n_spheres:int, sphere_centers:wp.array(dtype=wp.vec3), sphere_radii:wp.array(dtype=float),
    n_cuboids:int, cuboid_centers:wp.array(dtype=wp.vec3), cuboid_extents:wp.array(dtype=wp.vec3),
    lapse_rate:float, t_surface:float, seed:int,
    buoy_mode:int, w_rise0:float, sigma_0:float,
    K_rel_x:float, K_rel_y:float, K_rel_z:float,
    subgrid:int, shear_exp:float,
    puff_vs: wp.array(dtype=wp.vec3), sl_on:int, ustar:float, inv_L:float,
    sl_su:float, sl_sv:float, sl_c0:float, sl_zmin:float,
):
    i=wp.tid()
    if puff_active[i]==0: return
    pos=puff_pos[i]; g=world_to_grid(pos,dx); gx=g[0]; gy=g[1]; gz=g[2]
    age=current_time-puff_emit_time[i]
    if age<0.0: age=0.0

    # Buoyant rise as a velocity (m/s), added to the sampled wind:
    #   w_b = w_rise0 · (σ₀/σ)³ · (T_ref/T_env),  (σ₀/σ)³ = σ₀³/(σx·σy·σz)
    # with σ from the puff growth law σ_α² = σ₀² + 2·K_rel_α·age.  The rise
    # decays with volumetric dilution, so the height gain is finite
    # (Briggs-like plateau, Δh ≈ ∫w_b dt).  w_rise0 is an empirical initial
    # rise speed for buoyant gases (methane); "none" for neutral/dilute gases.
    # (Pre-fix: the 4.55 m/s² methane *acceleration* was added as a velocity.)
    buoy=float(0.0)
    if buoy_mode==BUOY_RISE:
        s02=sigma_0*sigma_0
        vol=wp.sqrt((s02+2.0*K_rel_x*age)*(s02+2.0*K_rel_y*age)*(s02+2.0*K_rel_z*age))
        dilution=s02*sigma_0/vol   # (σ₀/σ)³
        t_env=t_surface-lapse_rate*pos[2]
        buoy=w_rise0*dilution*(293.15/wp.max(t_env,200.0))

    # Turbulent velocity: exact OU update per axis (Thomson 1987, homogeneous
    # Gaussian turbulence).  ou_decay = e^{−dt/T_L}, ou_noise = σ·√(1 − e^{−2dt/T_L}).
    # Noise: Warp's seeded PRNG, stream = (per-step seed, puff index).
    state=wp.rand_init(seed,i)
    v=puff_vel[i]
    v=wp.vec3(v[0]*ou_decay[0]+ou_noise[0]*wp.randn(state),
              v[1]*ou_decay[1]+ou_noise[1]*wp.randn(state),
              v[2]*ou_decay[2]+ou_noise[2]*wp.randn(state))

    # Small-scale surface-layer turbulence (surface_layer=True): exact OU update
    # with the local T_i(z) = 2σ_i²/(C0 ε(z)) plus Thomson's (1987) well-mixed
    # drift for the height-dependent σ_w (Gaussian, 1-D inhomogeneous).
    vs=puff_vs[i]
    if sl_on!=0:
        zz=wp.max(pos[2],sl_zmin)
        eps=ustar*ustar*ustar/(0.4*zz)*sl_phi_eps_w(zz*inv_L)
        sw2=sl_sigma_w2_w(zz,ustar,inv_L)
        dsw2=float(0.0)
        if pos[2]>sl_zmin:
            dsw2=sl_dsigma_w2_dz_w(zz,ustar,inv_L)
        du=wp.exp(-dt*sl_c0*eps/(2.0*sl_su*sl_su))
        dv=wp.exp(-dt*sl_c0*eps/(2.0*sl_sv*sl_sv))
        dw=wp.exp(-dt*sl_c0*eps/(2.0*sw2))
        ws=vs[2]
        vs=wp.vec3(vs[0]*du+sl_su*wp.sqrt(1.0-du*du)*wp.randn(state),
                   vs[1]*dv+sl_sv*wp.sqrt(1.0-dv*dv)*wp.randn(state),
                   ws*dw+wp.sqrt(sw2*(1.0-dw*dw))*wp.randn(state)+0.5*(1.0+ws*ws/sw2)*dsw2*dt)
    # RK2 advection on resolved wind + turbulent velocity + buoyant rise
    vb=wp.vec3(v[0]+vs[0],v[1]+vs[1],v[2]+vs[2]+buoy)
    vel1=subgrid_scale(sample_vel_3d(wind_field,gx,gy,gz),pos[2],dx,shear_exp,subgrid)+vb; k1=vel1*dt
    pm=pos+0.5*k1; gm=world_to_grid(pm,dx)
    vel2=subgrid_scale(sample_vel_3d(wind_field,gm[0],gm[1],gm[2]),pm[2],dx,shear_exp,subgrid)+vb
    new_pos=pos+vel2*dt

    # Obstacles — push puffs 2 m from the surface (half a cell) so they
    # reach a zone with nonzero wind velocity for advection.
    margin = 2.0
    # Obstacle bounce: also flip the turbulent velocity's normal component
    # when it points into the surface.  Two passes over all obstacles handle
    # most overlaps; a puff still inside one (e.g. in a crevice narrower than
    # 2·margin between overlapping obstacles) is lifted until it is clear —
    # obstacles stand on the ground, so moving up always terminates.
    for npass in range(2):
        for s in range(n_spheres):
            c=sphere_centers[s]; r=sphere_radii[s]; dist=sdf_sphere(new_pos,c,r)
            if dist<margin:
                n=normal_sphere(new_pos,c); new_pos=new_pos+n*(margin-dist)
                vn=wp.dot(v,n)
                if vn<0.0: v=v-2.0*vn*n
                vsn=wp.dot(vs,n)
                if vsn<0.0: vs=vs-2.0*vsn*n
        for ci in range(n_cuboids):
            c=cuboid_centers[ci]; e=cuboid_extents[ci]; dist=sdf_cuboid(new_pos,c,e)
            if dist<margin:
                n=normal_cuboid(new_pos,c,e); new_pos=new_pos+n*(margin-dist)
                vn=wp.dot(v,n)
                if vn<0.0: v=v-2.0*vn*n
                vsn=wp.dot(vs,n)
                if vsn<0.0: vs=vs-2.0*vsn*n
    for k in range(64):
        phi=float(1.0e10)
        for s in range(n_spheres):
            phi=wp.min(phi, sdf_sphere(new_pos,sphere_centers[s],sphere_radii[s]))
        for ci in range(n_cuboids):
            phi=wp.min(phi, sdf_cuboid(new_pos,cuboid_centers[ci],cuboid_extents[ci]))
        if phi>=0.0:
            break
        new_pos=new_pos+wp.vec3(0.0,0.0,margin-phi)
        v=wp.vec3(v[0],v[1],wp.abs(v[2]))
        vs=wp.vec3(vs[0],vs[1],wp.abs(vs[2]))
    puff_pos[i]=new_pos
    puff_vel[i]=v
    puff_vs[i]=vs

@wp.kernel
def sample_wind_point(u: wp.array3d(dtype=wp.vec3), p: wp.vec3, dx: float, out: wp.array(dtype=wp.vec3)):
    g = world_to_grid(p, dx)
    out[0] = sample_vel_3d(u, g[0], g[1], g[2])

# ── Concentration kernels ────────────────────────────────────────────────────

@wp.func
def puff_sum_at(
    p: wp.vec3,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float),
    current_time: float, sigma_0: float,
    K_rel_x: float, K_rel_y: float, K_rel_z: float, num_puffs: int,
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
    use_los: int,
):
    """Sum of Gaussian puffs (with ground image) at point p.

    Puff growth uses the relative-dispersion share of K:
        σ_α² = σ₀² + 2·K_rel_α·age  (relative dispersion only).
    Occlusion (use_los): points inside an obstacle read 0; a puff whose
    centre is not visible from p (segment centre→p hits an obstacle) is
    skipped entirely, image term included. Tested from the puff centre only.
    """
    c = float(0.0)
    if use_los != 0:
        if inside_obstacle(p, n_spheres,sphere_centers,sphere_radii,n_cuboids,cuboid_centers,cuboid_extents):
            return c
    for i in range(num_puffs):
        c = c + puff_contrib(p, i, puff_pos, puff_active, puff_emit_time, puff_mass, current_time, sigma_0,
                             K_rel_x, K_rel_y, K_rel_z, n_spheres, sphere_centers, sphere_radii,
                             n_cuboids, cuboid_centers, cuboid_extents, use_los)
    return c


@wp.func
def puff_contrib(
    p: wp.vec3, i: int,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float),
    current_time: float, sigma_0: float,
    K_rel_x: float, K_rel_y: float, K_rel_z: float,
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
    use_los: int,
):
    """Concentration at p from puff i (with ground image); 0 if inactive, beyond
    4σ on an axis, or occluded."""
    if puff_active[i] == 0:
        return float(0.0)
    age = current_time - puff_emit_time[i]
    if age < 1e-6:
        return float(0.0)
    s02 = sigma_0 * sigma_0
    sig2_x = s02 + 2.0*K_rel_x*age
    sig2_y = s02 + 2.0*K_rel_y*age
    sig2_z = s02 + 2.0*K_rel_z*age
    sig_x = wp.sqrt(sig2_x); sig_y = wp.sqrt(sig2_y); sig_z = wp.sqrt(sig2_z)
    q = puff_pos[i]
    diff = p - q
    # Early exit on horizontal distance only
    if wp.abs(diff[0]) > 4.0*sig_x or wp.abs(diff[1]) > 4.0*sig_y:
        return float(0.0)
    # Ground reflection (image source method):
    #   real puff at z_i → dz_real = z - z_i;  image at -z_i → dz_image = z + z_i
    dz_real = diff[2]
    dz_image = p[2] + q[2]
    if wp.abs(dz_real) > 4.0*sig_z and wp.abs(dz_image) > 4.0*sig_z:
        return float(0.0)
    if use_los != 0:
        if los_blocked(q, p, n_spheres,sphere_centers,sphere_radii,n_cuboids,cuboid_centers,cuboid_extents):
            return float(0.0)
    r2_xy = diff[0]*diff[0]/sig2_x + diff[1]*diff[1]/sig2_y
    gauss_xy = wp.exp(-0.5*r2_xy)
    gauss_z_real = wp.exp(-0.5*dz_real*dz_real/sig2_z)
    gauss_z_image = wp.exp(-0.5*dz_image*dz_image/sig2_z)
    norm = puff_mass[i] / (15.74960995*sig_x*sig_y*sig_z)  # (2π)^{3/2} σxσyσz
    return norm*gauss_xy*(gauss_z_real + gauss_z_image)


@wp.kernel
def query_sensor_hashed(
    grid: wp.uint64, radius: float,
    sensor_pos: wp.array(dtype=wp.vec3), dx: float,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float), current_time: float,
    sigma_0: float, K_rel_x: float, K_rel_y: float, K_rel_z: float,
    concentration: wp.array(dtype=float),
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
    use_los: int,
):
    """query_sensor_hybrid (puff part) over the puffs within `radius` of the sensor
    only (hash grid).  `radius` bounds every puff's 4σ box, so the sum is the same
    as the brute-force loop.  The image term needs no extra search: the image is
    never closer to a point above ground than the puff itself."""
    s = wp.tid(); p = sensor_pos[s]
    if p[2] < 0.0 or p[0] < 0.0 or p[1] < 0.0 or p[0] > float(WIND_NX)*dx or p[1] > float(WIND_NY)*dx or p[2] > float(WIND_NZ)*dx:
        concentration[s] = 0.0
        return
    c = float(0.0)
    if use_los != 0:
        if inside_obstacle(p, n_spheres,sphere_centers,sphere_radii,n_cuboids,cuboid_centers,cuboid_extents):
            concentration[s] = 0.0
            return
    query = wp.hash_grid_query(grid, p, radius)
    i = int(0)
    while wp.hash_grid_query_next(query, i):
        c = c + puff_contrib(p, i, puff_pos, puff_active, puff_emit_time, puff_mass, current_time, sigma_0,
                             K_rel_x, K_rel_y, K_rel_z, n_spheres, sphere_centers, sphere_radii,
                             n_cuboids, cuboid_centers, cuboid_extents, use_los)
    concentration[s] = c


@wp.kernel
def query_sensor_hybrid(
    sensor_pos: wp.array(dtype=wp.vec3),
    rho_grid: wp.array3d(dtype=float), rho_scale:float, dx:float,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float), current_time:float,
    sigma_0:float, K_rel_x:float, K_rel_y:float, K_rel_z:float,
    num_puffs:int, concentration: wp.array(dtype=float),
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
    use_los: int,
):
    s=wp.tid(); p=sensor_pos[s]
    # Points below ground or outside the domain read 0 (no gas is simulated
    # there; the image term would otherwise mirror the plume below z = 0).
    if p[2] < 0.0 or p[0] < 0.0 or p[1] < 0.0 or p[0] > float(WIND_NX)*dx or p[1] > float(WIND_NY)*dx or p[2] > float(WIND_NZ)*dx:
        concentration[s]=0.0
        return
    c_grid=float(0.0)
    if rho_scale > 0.0:   # legacy grid gas only
        g=world_to_grid(p,dx)
        c_grid=sample_rho_3d(rho_grid,g[0],g[1],g[2])*rho_scale
    c_puff=puff_sum_at(p,puff_pos,puff_active,puff_emit_time,puff_mass,current_time,
                       sigma_0,K_rel_x,K_rel_y,K_rel_z,num_puffs,
                       n_spheres,sphere_centers,sphere_radii,n_cuboids,cuboid_centers,cuboid_extents,use_los)
    concentration[s]=c_grid+c_puff

@wp.kernel
def compute_concentration_grid_kernel(
    conc_grid: wp.array3d(dtype=float), rho_grid: wp.array3d(dtype=float), rho_scale:float,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), puff_mass: wp.array(dtype=float), current_time:float,
    sigma_0:float, K_rel_x:float, K_rel_y:float, K_rel_z:float, dx:float, num_puffs:int,
    n_spheres: int, sphere_centers: wp.array(dtype=wp.vec3), sphere_radii: wp.array(dtype=float),
    n_cuboids: int, cuboid_centers: wp.array(dtype=wp.vec3), cuboid_extents: wp.array(dtype=wp.vec3),
    use_los: int,
):
    i,j,k=wp.tid()
    pos=wp.vec3((float(i)+0.5)*dx,(float(j)+0.5)*dx,(float(k)+0.5)*dx)
    c_puff=puff_sum_at(pos,puff_pos,puff_active,puff_emit_time,puff_mass,current_time,
                       sigma_0,K_rel_x,K_rel_y,K_rel_z,num_puffs,
                       n_spheres,sphere_centers,sphere_radii,n_cuboids,cuboid_centers,cuboid_extents,use_los)
    conc_grid[i,j,k]=rho_grid[i,j,k]*rho_scale+c_puff


# ── Atmospheric wind model (vector-valued OU + gusts + turbulence) ────────────
# Convention: (u, v, w) = (x-component, y-component, vertical)
# Each component follows an independent OU process: dU_i = -κ_i(U_i - U_i0)dt + σ_i dW_i
# Reference: Thomson (1987) for Lagrangian stochastic models.

class WindVectorModel:
    """Wind model using vector-valued OU processes for (u, v, w) components.

    Stabilized version: Gusts and turbulence are scaled down to allow for 
    a predictable, conical plume during RL training.
    """
    def __init__(self, center_angle=45.0, angle_range=20.0, base_speed=1.5,
                 speed_variance=0.5, ou_theta=0.02, ou_sigma=0.15,  # <-- Reduced noise (was 0.8)
                 gust_interval=15.0, gust_duration=2.0,
                 gust_strength_range=(0.5, 1.5), vertical_component=0.1, # <-- Weakened gusts
                 shear_exponent=0.25, seed=42,
                 enable_turbulence=True, enable_gusts=True, steady=False,
                 turb_scale=1.0):
        """
        enable_turbulence : False zeroes the three OU turbulence layers.
        enable_gusts      : False disables the gust model.
        steady            : True removes all randomness (mean-wind OU, turbulence
                            layers, gusts): the wind is exactly the mean vector.
        turb_scale        : multiplies the mean-wind OU σ and the turbulence-layer σs.
        """
        self.center_rad = math.radians(center_angle)
        # Mean wind vector from direction and speed
        self.u_mean = base_speed * math.cos(self.center_rad)
        self.v_mean = base_speed * math.sin(self.center_rad)
        self.w_mean = 0.0  # zero mean vertical wind

        self.base_speed = base_speed
        self.speed_variance = speed_variance
        self.shear_exponent = shear_exponent
        self.angle_range_rad = math.radians(angle_range)
        self.rng = np.random.default_rng(seed)
        self.steady = bool(steady)
        self.enable_turbulence = bool(enable_turbulence) and not self.steady
        self.enable_gusts = bool(enable_gusts) and not self.steady

        # OU parameters per component: [u, v, w]
        # Tighter reversion and lower noise for a stable plume
        self._kappa = np.array([ou_theta, ou_theta, 0.05])
        self._sigma = np.array([ou_sigma, ou_sigma, 0.05]) * turb_scale
        if self.steady:
            self._sigma = np.zeros(3)

        # Wind state (u, v, w) — warm-started from stationary distribution
        self._wind_mean = np.array([self.u_mean, self.v_mean, self.w_mean])
        _stat_std = self._sigma / np.sqrt(2.0 * self._kappa)
        self._wind = self._wind_mean + _stat_std * self.rng.standard_normal(3)

        # Multi-scale turbulence: scaled down to avoid tearing the plume apart
        self._turb_kappas = [0.05, 0.15, 0.40]
        self._turb_sigmas = [0.20 * turb_scale, 0.15 * turb_scale, 0.10 * turb_scale] # <-- Was 0.80, 0.50, 0.30
        if not self.enable_turbulence:
            self._turb_sigmas = [0.0, 0.0, 0.0]
        self._turb_states = np.zeros((3, 3))  
        for _layer_idx in range(3):
            _layer_std = self._turb_sigmas[_layer_idx] / np.sqrt(2.0 * self._turb_kappas[_layer_idx])
            self._turb_states[_layer_idx] = _layer_std * self.rng.standard_normal(3)
            self._turb_states[_layer_idx, 2] *= 0.1  

        # Gust model 
        self.gust_interval = gust_interval if self.enable_gusts else math.inf
        self.gust_duration = gust_duration
        self.gust_strength_range = gust_strength_range
        self._gust_timer = 0.0
        self._gust_active = False
        self._gust_elapsed = 0.0
        self._gust_vec = np.zeros(3)  
        # Gust onsets are a Poisson process: the quiet time before the next
        # gust is drawn once, Exp(mean = gust_interval).  (Pre-fix drew a new
        # Exp sample every frame and compared it to the timer, so gusts fired
        # ~1 s apart instead of ~gust_interval.)
        self._gust_wait = self.rng.exponential(gust_interval) if self.enable_gusts else math.inf
        # Gusts only add along-wind speed, so their time mean is non-zero:
        #   duty·E[strength]·E[envelope]·E[cos(direction offset)]
        #   = D/(I + D) · mean(range) · 0.8 · sin(r)/r   (≈ 0.09 m/s for the defaults).
        # It is subtracted in get_wind() so gusts are zero-mean fluctuations and
        # the time-mean wind equals base_speed (A5).
        if self.enable_gusts:
            duty = gust_duration / (gust_interval + gust_duration)
            r = self.angle_range_rad
            along = math.sin(r) / r if r > 0 else 1.0
            g = duty * 0.5 * (gust_strength_range[0] + gust_strength_range[1]) * 0.8 * along
            self._gust_mean = np.array([g * math.cos(self.center_rad), g * math.sin(self.center_rad), 0.0])
        else:
            self._gust_mean = np.zeros(3)

    def step(self, dt):
        # OU update for each wind component
        for i in range(3):
            k = self._kappa[i]
            decay = math.exp(-k * dt)
            ns = self._sigma[i] * math.sqrt((1.0 - math.exp(-2.0 * k * dt)) / (2.0 * k))
            self._wind[i] = self._wind_mean[i] + (self._wind[i] - self._wind_mean[i]) * decay + ns * self.rng.standard_normal()

        # Clamp horizontal speed to reasonable range
        h_speed = math.sqrt(self._wind[0]**2 + self._wind[1]**2)
        max_speed = self.base_speed + self.speed_variance
        if h_speed > max_speed and h_speed > 0:
            scale = max_speed / h_speed
            self._wind[0] *= scale
            self._wind[1] *= scale
        if h_speed < 0.3 and h_speed > 0:
            scale = 0.3 / h_speed
            self._wind[0] *= scale
            self._wind[1] *= scale

        # Multi-scale turbulence OU per component
        for layer in range(3):
            th = self._turb_kappas[layer]
            sig = self._turb_sigmas[layer]
            d = math.exp(-th * dt)
            ns = sig * math.sqrt((1.0 - math.exp(-2.0 * th * dt)) / (2.0 * th))
            self._turb_states[layer] *= d
            self._turb_states[layer] += ns * self.rng.standard_normal(3)
            self._turb_states[layer, 2] *= 0.1

        # Gust model
        if not self.enable_gusts:
            return
        self._gust_timer += dt
        if not self._gust_active:
            if self._gust_timer > self._gust_wait:
                self._gust_active = True
                self._gust_elapsed = 0.0
                self._gust_timer = 0.0
                
                # FIXED: Gusts now blow generally in the direction of the mean wind (± angle_range)
                # rather than random 360-degree chaos.
                gust_angle = self.rng.uniform(
                    self.center_rad - self.angle_range_rad, 
                    self.center_rad + self.angle_range_rad
                )
                gust_speed = self.rng.uniform(*self.gust_strength_range)
                self._gust_vec = np.array([
                    gust_speed * math.cos(gust_angle),
                    gust_speed * math.sin(gust_angle),
                    0.0,
                ])
        else:
            self._gust_elapsed += dt
            if self._gust_elapsed > self.gust_duration:
                self._gust_active = False
                self._gust_timer = 0.0
                self._gust_wait = self.rng.exponential(self.gust_interval)

    def get_wind(self, t):
        turb = np.sum(self._turb_states, axis=0)
        eff = self._wind + turb - self._gust_mean

        if self._gust_active:
            frac = self._gust_elapsed / self.gust_duration
            env = frac / 0.15 if frac < 0.15 else (1.0 - frac) / 0.25 if frac > 0.75 else 1.0
            eff = eff + self._gust_vec * env

        return float(eff[0]), float(eff[1]), float(eff[2]), self.shear_exponent

    def set_direction(self, center_angle):
        """Rotate the mean wind to a new direction (degrees), keeping its speed.
        The current OU state, gust vector and gust mean rotate with it."""
        new = math.radians(center_angle)
        rot = new - self.center_rad
        c, s_ = math.cos(rot), math.sin(rot)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        self.center_rad = new
        self.u_mean = self.base_speed * math.cos(new)
        self.v_mean = self.base_speed * math.sin(new)
        self._wind_mean = np.array([self.u_mean, self.v_mean, self.w_mean])
        self._wind = R @ self._wind
        self._gust_vec = R @ self._gust_vec
        self._gust_mean = R @ self._gust_mean

    def get_angle_degrees(self):
        """Derive wind direction from current u,v components (backward compat)."""
        return math.degrees(math.atan2(self._wind[1], self._wind[0]))
# Backward compatibility alias
WindDirectionModel = WindVectorModel


# ── Units ────────────────────────────────────────────────────────────────────
R_GAS = 8.314462618          # J/(mol·K)
M_CH4 = 0.01604              # kg/mol


def kgm3_to_ppm(c, T=293.15, P=101325.0, M=M_CH4):
    """Mass concentration (kg/m³) → volume mixing ratio (ppm): c·R·T/(M·P)·1e6."""
    return np.asarray(c, dtype=np.float64) * R_GAS * T / (M * P) * 1e6


def ppm_to_kgm3(ppm, T=293.15, P=101325.0, M=M_CH4):
    """Volume mixing ratio (ppm) → mass concentration (kg/m³)."""
    return np.asarray(ppm, dtype=np.float64) * 1e-6 * M * P / (R_GAS * T)


# ── Main simulation class ────────────────────────────────────────────────────

class HybridFluidPuffSimulation:
    # Puff dispersion: OU turbulent velocity per puff (PG_LSM, set by
    # set_stability_class; class D by default) + relative puff growth K_rel.

    # Grid scalar diffusion (turbulent eddy diffusivity, NOT molecular D_mol)
    D_ISO = 15.0       # m²/s — turbulent mixing on the Eulerian grid (reduced from 50)

    RHO_DECAY_RATE = 0.0008   # 1/s — reduced so grid density persists downstream
    RHO_SCALE      = 1e-3

    N_DRAG_LAYERS = 3          # ground-drag layers (k < 3, z < 12 m)

    # Temperature-profile-based buoyancy (Stull 1988)
    LAPSE_RATE = PG_LAPSE_RATES["D"]  # K/m, set by stability class
    T_SURFACE  = T_SURFACE_DEFAULT    # K, surface temperature

    def __init__(self, wind_center_angle=45.0, wind_angle_range=40.0,
                 wind_base_speed=4.5, wind_seed=42, max_puffs=1024,
                 seed=None,
                 use_grid_gas=False,
                 K_rel=0.05,
                 buoyancy_mode="none",
                 w_rise0=0.0,
                 ground_drag_rate=2.4,
                 release_jitter=0.0,
                 enable_turbulence=True,
                 enable_gusts=True,
                 steady_wind=False,
                 shear_exponent=None,
                 use_los=True,
                 legacy_emission=False,
                 turb_scale=None,
                 wind_preset="calibrated",
                 stability_class="D",
                 shared_eddies=True,
                 drag_compensation=True,
                 class_sigma0=None,
                 sigma_min=0.05,
                 fps=30,
                 closed_domain=False,
                 lsm_table=None,
                 hp_gamma=HP_GAMMA,
                 subgrid_wind=False,
                 surface_layer=False,
                 z0=0.03,
                 zi=1000.0,
                 met=None,
                 sl_c0=SL_C0,
                 meander_frac=0.0,
                 meander_TE=60.0):
        """
        seed              : puff RNG seed (defaults to wind_seed).
        use_grid_gas      : legacy Eulerian gas field + source momentum jet.
                            Kept only to reproduce pre-fix figures.
        K_rel             : m²/s, puff growth σ² = σ₀² + 2·K_rel·age (relative
                            dispersion only; absolute dispersion is the LSM).
                            Replaces f_rel (pre-LSM share of K for growth).
        buoyancy_mode     : "none" (neutral / dilute gas) or "rise".
        w_rise0           : initial rise speed (m/s) for "rise" mode.
        ground_drag_rate  : 1/s. Old value was 0.08 per frame at 30 fps (~2.4/s).
        release_jitter    : m, std of release position noise (old hidden value 1.5).
        enable_turbulence : coarse OU field + three OU turbulence layers.
        enable_gusts      : gust model.
        steady_wind       : no wind randomness at all (validation).
        shear_exponent    : overrides the PG shear exponent (V1 uses 0).
        use_los           : line-of-sight occlusion of puffs by obstacles.
        legacy_emission   : pre-fix emission (two puffs per interval). Legacy only.
        turb_scale        : multiplies every stochastic wind amplitude (coarse OU
                            field σ, OU turbulence layers, mean-wind OU σ).
                            None → from wind_preset.
        wind_preset       : "calibrated" (default, turb_scale 0.1) or "meandering"
                            (turb_scale 1.0, pre-LSM amplitudes; harder RL).
        stability_class   : PG class A–F (LSM intensities, T_L, shear, lapse rate).
        shared_eddies     : correlate successive puffs' release velocities
                            (corr e^{−τ_e/T_E}, T_E = T_L·i/γ) so the plume meanders
                            coherently.  False → independent N(0, σ²) draws.
        hp_gamma          : γ in the Hay–Pasquill ratio T_L/T_E = γ/i.
        fps               : simulation steps per second (Δt = 1/fps); default 30.
        closed_domain     : reflect puffs at all faces instead of removing them
                            (well-mixed test only).
        sigma_min         : m/s, floor on the LSM velocity σ (calm wind; σ = a·U
                            otherwise vanishes).  Affects only a·U < sigma_min.
        class_sigma0      : optional {class: σ₀ (m)} overriding sigma_0 for those
                            classes (A7).  Requires a T_L table fitted for those σ₀
                            (lsm_table; validation/fit_tl.py --class-sigma0).
        lsm_table         : optional {class: {"a_y", "a_z", "TL_y", "TL_z"}} overriding
                            entries of PG_LSM.
        surface_layer     : v1.1 two-scale LSM (see SL_* constants): small-scale
                            mechanical surface-layer turbulence per puff (u*, L,
                            ε(z), T(z), Thomson drift) + convective coherent
                            lateral eddies when unstable.  Replaces the
                            Briggs-fitted homogeneous LSM (PG_LSM / lsm_table).
        z0                : m, roughness length (surface_layer met from class).
        zi                : m, mixed-layer depth (convective eddies, unstable).
        met               : optional dict(ustar=…, inv_L=…, zi=…) overriding the
                            class-derived surface-layer met (measured inputs).
        sl_c0             : Lagrangian Kolmogorov constant C0 of T = 2σ²/(C0 ε).
        meander_frac      : surface_layer, neutral/stable only: fraction f of the
                            mechanical lateral variance (1.9u*)² carried by a coherent
                            low-frequency meander (shared eddies, Eulerian time scale
                            meander_TE); the small-scale σ_u, σ_v are reduced so the
                            total variance is unchanged.  0 = off.
        meander_TE        : s, Eulerian time scale of that meander.
        subgrid_wind      : puffs below the lowest cell centre (z < dx/2 = 2 m) are
                            advected with the power-law profile extended below
                            the grid, u_h(z) = u_h(dx/2)·(max(z, 0.1)/(dx/2))^α,
                            instead of the dx/2 wind (clamped interpolation).
        drag_compensation : scale the target in the drag layers so the mean
                            resolved wind follows the power law (realised 10 m
                            wind = wind_base_speed).  False: pre-A5 behaviour
                            (10 m wind ≈ 21% below target).
        """
        self.seed = int(wind_seed if seed is None else seed)
        self.subgrid_wind = bool(subgrid_wind)
        self.surface_layer = bool(surface_layer)
        self.z0 = float(z0); self.zi = float(zi); self.sl_c0 = float(sl_c0)
        if not (self.z0 > 0 and self.zi > 0 and self.sl_c0 > 0):
            raise ValueError("z0, zi and sl_c0 must be > 0")
        self.met_override = dict(met) if met else None
        self.meander_frac = float(meander_frac); self.meander_TE = float(meander_TE)
        if not (0.0 <= self.meander_frac < 1.0 and self.meander_TE > 0):
            raise ValueError("meander_frac must be in [0, 1) and meander_TE > 0")
        self._vs_rng = np.random.default_rng([self.seed, 11])
        self.use_grid_gas = bool(use_grid_gas)
        self.k_rel = float(K_rel)
        if self.k_rel < 0.0:
            raise ValueError("K_rel must be >= 0")
        self.shared_eddies = bool(shared_eddies)
        self.drag_compensation = bool(drag_compensation)
        self.sigma_min = float(sigma_min)
        if self.sigma_min < 0:
            raise ValueError("sigma_min must be >= 0")
        self.class_sigma0 = dict(class_sigma0) if class_sigma0 else {}
        self.lsm_table = {k.upper(): dict(v) for k, v in (lsm_table or {}).items()}
        for v in self.class_sigma0.values():
            if not v > 0:
                raise ValueError("class_sigma0 values must be > 0")
        self.hp_gamma = float(hp_gamma)
        self._release_rng = np.random.default_rng([self.seed, 7])
        self._vs_rng = np.random.default_rng([self.seed, 11])
        self._release_zeta = None
        self._eddy_hist_t = np.full(EDDY_HIST_N, -np.inf); self._eddy_hist_v = np.zeros((EDDY_HIST_N, 3))
        self._eddy_hist_i = 0
        if buoyancy_mode not in ("none", "rise"):
            raise ValueError("buoyancy_mode must be 'none' or 'rise'")
        self.buoyancy_mode = buoyancy_mode
        self.w_rise0 = float(w_rise0)
        if buoyancy_mode == "rise" and self.k_rel == 0.0:
            import warnings
            warnings.warn("buoyancy_mode='rise' with K_rel=0: puffs never dilute, so the "
                          "rise never decays (no plateau).")
        self.ground_drag_rate = float(ground_drag_rate)
        self.release_jitter = float(release_jitter)
        self.enable_turbulence = bool(enable_turbulence) and not steady_wind
        self.enable_gusts = bool(enable_gusts) and not steady_wind
        self.steady_wind = bool(steady_wind)
        self.shear_exponent_override = None if shear_exponent is None else float(shear_exponent)
        self.use_los = bool(use_los)
        self.legacy_emission = bool(legacy_emission)
        if wind_preset not in WIND_PRESETS:
            raise ValueError(f"wind_preset must be one of {list(WIND_PRESETS)}")
        self.wind_preset = wind_preset
        self.turb_scale = float(WIND_PRESETS[wind_preset]["turb_scale"] if turb_scale is None else turb_scale)
        self.wind_angle_range = wind_angle_range
        if not wind_base_speed >= 0:
            raise ValueError(f"wind_base_speed must be >= 0 (got {wind_base_speed})")
        self.wind_base_speed = wind_base_speed

        self.closed_domain = bool(closed_domain)
        if not fps > 0:
            raise ValueError("fps must be > 0")
        self.fps=float(fps); self.frame_dt=1.0/self.fps; self.sim_dt=self.frame_dt; self.sim_time=0.0
        self.dx=DX; self.wind_shape=(NX, NY, NZ)

        self.u0=wp.zeros(self.wind_shape,dtype=wp.vec3); self.u1=wp.zeros(self.wind_shape,dtype=wp.vec3)
        self.p0=wp.zeros(self.wind_shape,dtype=float)
        self.div=wp.zeros(self.wind_shape,dtype=float)
        self.rho0=wp.zeros(self.wind_shape,dtype=float); self.rho1=wp.zeros(self.wind_shape,dtype=float)
        self.rho_tmp=wp.zeros(self.wind_shape,dtype=float)
        self.conc_grid=wp.zeros(self.wind_shape,dtype=float)

        # Obstacle mask: 1 = solid, 0 = fluid
        self.obstacle_mask = wp.zeros(self.wind_shape, dtype=int)

        self.max_puffs=max_puffs
        self.puff_pos=wp.zeros(self.max_puffs,dtype=wp.vec3)
        self.puff_active=wp.zeros(self.max_puffs,dtype=int)
        self.puff_emit_time=wp.zeros(self.max_puffs,dtype=float)
        self.puff_mass=wp.zeros(self.max_puffs,dtype=float)   # kg per puff, set at release
        self.puff_vel=wp.zeros(self.max_puffs,dtype=wp.vec3)  # LSM turbulent velocity (m/s)
        self.puff_vs=wp.zeros(self.max_puffs,dtype=wp.vec3)   # small-scale surface-layer velocity
        self.next_puff_idx=0; self.emit_interval=0.05
        self._max_emit_interval=0.0; self._min_inject_emit=None; self._hash=None
        self.use_hash_query=True
        self._next_emit_time=0.0   # emission schedule: t_n = n·emit_interval
        self.n_emitted=0
        self._counters=wp.zeros(2,dtype=int)   # [overwritten, exited]
        self._mass_counters=wp.zeros(2,dtype=float)   # kg [overwritten, exited]
        self.mass_emitted=0.0                  # kg released (host, float64)
        self._overflow_warned=False
        self._step_counter=0   # puff-noise stream index (reset with the sim)

        self._spheres_host=[]; self._cuboids_host=[]
        self._source_pos=np.array([NX//4*DX, NY//4*DX, 2*DX],dtype=np.float32)

        self.Q = 20.0           # kg/s — methane emission rate (overridden by env)
        self.sigma_0 = 1.0      # m — initial puff standard deviation (was 2.0; 2 m
                                #     exceeds Briggs D σ_z(25 m) = 1.47 m, see PG_LSM)

        self.wind_model=self._make_wind_model(wind_center_angle, wind_seed)

        self._rng = np.random.default_rng(wind_seed)

        self.n_spheres=0; self.n_cuboids=0
        self.sphere_centers=wp.zeros(_MAX_SPHERES,dtype=wp.vec3); self.sphere_radii=wp.zeros(_MAX_SPHERES,dtype=float)
        self.cuboid_centers=wp.zeros(_MAX_CUBOIDS,dtype=wp.vec3); self.cuboid_extents=wp.zeros(_MAX_CUBOIDS,dtype=wp.vec3)
        self._domain_max=np.array([NX*DX, NY*DX, NZ*DX],dtype=np.float32)

        # Spatially-correlated wind perturbation field (coarse 8x4x4 grid)
        # Warm-started from OU stationary distribution: std = σ/√(2κ)
        # (Gardiner 2009, §4.5) — eliminates zero-turbulence cold start.
        self._turb_shape = (_TURB_NX, _TURB_NY, _TURB_NZ)
        self._turb_ou_theta = 0.1
        self._turb_ou_sigma = 1.2 * self.turb_scale if self.enable_turbulence else 0.0
        _turb_stat_std = self._turb_ou_sigma / math.sqrt(2.0 * self._turb_ou_theta)
        self._turb_np = (_turb_stat_std * self._rng.standard_normal(
            (*self._turb_shape, 3))).astype(np.float32)
        self.turb_field = wp.array(self._turb_np, dtype=wp.vec3)

        # Pressure solve: Red-Black SOR iterations (see divergence_residual)
        self.sor_iters = 25
        self._div_scratch = wp.zeros(self.wind_shape, dtype=float)

        self._U_src = None
        self._wind_probe = wp.zeros(1, dtype=wp.vec3)
        self._next_U_update = 0.0
        self.set_stability_class(stability_class)

    def divergence_residual(self, u_ref=None):
        """Normalised divergence of the current (projected) wind field.

        Returns mean |∇·u| over fluid cells divided by U_ref/dx, i.e.
        mean |div_cell| / U_ref with div_cell the kernel's central-difference
        divergence in m/s.  Also stores the same norm of the pre-projection
        divergence of the last step in self.last_div_pre.  Diagnostics only
        (forces a device sync).
        """
        if u_ref is None:
            u_ref = max(float(self.wind_base_speed), 1e-6)
        wp.launch(wind_divergence_3d, dim=self.wind_shape,
                  inputs=[self.u0, self._div_scratch, self.obstacle_mask])
        fluid = self.obstacle_mask.numpy() == 0
        fluid[:, :, 0] = False   # ground row: divergence defined as 0
        post = float(np.abs(self._div_scratch.numpy()[fluid]).mean()) / u_ref
        self.last_div_pre = float(np.abs(self.div.numpy()[fluid]).mean()) / u_ref
        return post

    def _make_wind_model(self, center_angle, seed):
        wm = WindVectorModel(
            center_angle=center_angle,
            angle_range=self.wind_angle_range,
            base_speed=self.wind_base_speed,
            seed=seed,
            enable_turbulence=self.enable_turbulence,
            enable_gusts=self.enable_gusts,
            steady=self.steady_wind,
            turb_scale=self.turb_scale,
        )
        if self.shear_exponent_override is not None:
            wm.shear_exponent = self.shear_exponent_override
        return wm

    def set_wind_direction(self, center_angle):
        """Rotate the mean wind direction (degrees) mid-run."""
        self.wind_model.set_direction(center_angle)

    def reset(self, wind_center_angle, wind_seed, seed=None):
        """Reset all state in place (reuses GPU allocations and kernels)."""
        for a in (self.u0, self.u1, self.p0, self.div, self.rho0, self.rho1,
                  self.rho_tmp, self.conc_grid):
            a.zero_()
        self.puff_pos.zero_(); self.puff_active.zero_(); self.puff_emit_time.zero_()
        self.puff_mass.zero_(); self.puff_vel.zero_(); self.puff_vs.zero_(); self._counters.zero_()
        self._mass_counters.zero_(); self.mass_emitted = 0.0; self._overflow_warned = False
        self.next_puff_idx = 0
        self._max_emit_interval = 0.0; self._min_inject_emit = None
        self._next_emit_time = 0.0
        self.n_emitted = 0
        self._step_counter = 0
        self._U_src = None
        self._next_U_update = 0.0
        self.sim_time = 0.0
        if seed is not None:
            self.seed = int(seed)
        self._release_rng = np.random.default_rng([self.seed, 7])
        self._vs_rng = np.random.default_rng([self.seed, 11])
        self._release_zeta = None
        self._eddy_hist_t = np.full(EDDY_HIST_N, -np.inf); self._eddy_hist_v = np.zeros((EDDY_HIST_N, 3))
        self._eddy_hist_i = 0
        self._rng = np.random.default_rng(wind_seed)
        # Spatial turbulence field — warm-start from OU stationary distribution
        _turb_stat_std = self._turb_ou_sigma / math.sqrt(2.0 * self._turb_ou_theta)
        self._turb_np = (_turb_stat_std * self._rng.standard_normal(
            (*self._turb_shape, 3))).astype(np.float32)
        wp.copy(self.turb_field, wp.array(self._turb_np, dtype=wp.vec3))
        shear = self.wind_model.shear_exponent
        self.wind_model = self._make_wind_model(wind_center_angle, wind_seed)
        self.wind_model.shear_exponent = shear

    @property
    def K_rel(self):
        """Puff-growth diffusivity (m²/s) per axis (isotropic K_rel)."""
        return (self.k_rel, self.k_rel, self.k_rel)

    def puff_sigma(self, ages):
        """Per-axis puff σ (m) for an array of ages: σ² = σ₀² + 2·K_rel·age."""
        ages = np.asarray(ages, dtype=np.float64)
        return np.stack([np.sqrt(self.sigma_0**2 + 2.0*k*ages) for k in self.K_rel], axis=-1)

    def _U_target_at_source(self):
        alpha = self.wind_model.shear_exponent
        z = max(float(self.source_pos[2]), 1.0)
        return max(float(self.wind_base_speed) * (z / SHEAR_Z_REF) ** alpha, 0.1)

    @property
    def U_src(self):
        """Mean horizontal wind speed (m/s) at the source, used to scale the LSM
        intensities and T_L: a running mean (τ = U_SRC_TAU) of the resolved
        wind sampled at the source once per second, initialised with the
        target profile U_ref·(z_s/z_ref)^α.  The resolved wind is used because
        ground drag makes it ~20% slower than the target at 10 m."""
        if self._U_src is None:
            self._U_src = self._U_target_at_source()
        return max(self._U_src, 0.1)

    U_SRC_TAU = 30.0       # s, running-mean time constant for U_src
    U_SRC_EVERY = 1.0      # s, sampling interval

    def _update_U_src(self):
        wp.launch(sample_wind_point, dim=1, inputs=[
            self.u0, wp.vec3(*[float(v) for v in self.source_pos]), float(self.dx), self._wind_probe])
        u = self._wind_probe.numpy()[0]
        speed = float(math.hypot(u[0], u[1]))
        U = self.U_src
        self._U_src = U + (self.U_SRC_EVERY / self.U_SRC_TAU) * (speed - U)

    def _update_met(self):
        """Surface-layer met (u*, 1/L, z_i) for surface_layer=True."""
        if self.met_override:
            m = self.met_override
            self.inv_L = float(m.get("inv_L", 0.0)); self.zi_eff = float(m.get("zi", self.zi))
            self.ustar = float(m["ustar"]) if "ustar" in m else friction_velocity(
                max(float(self.wind_base_speed), 0.3), SHEAR_Z_REF, self.z0, self.inv_L)
        else:
            self.inv_L = golder_inv_L(self.stability_class, self.z0)
            self.zi_eff = self.zi
            self.ustar = friction_velocity(max(float(self.wind_base_speed), 0.3), SHEAR_Z_REF, self.z0, self.inv_L)

    def _meander_sigma(self):
        """σ of the coherent neutral/stable meander (0 if off or unstable)."""
        if not self.surface_layer or self.meander_frac <= 0 or self.inv_L < 0:
            return 0.0
        return math.sqrt(self.meander_frac) * SL_SIGMA_MECH[1] * self.ustar

    def sl_sigma(self, z):
        """Small-scale σ_u, σ_v, σ_w (m/s) at height z (surface_layer); the
        meander share (if any) is removed from the horizontal components."""
        su, sv, _ = SL_SIGMA_MECH
        sm2 = self._meander_sigma() ** 2
        su_ = math.sqrt(max((su * self.ustar) ** 2 - sm2, 0.0))
        sv_ = math.sqrt(max((sv * self.ustar) ** 2 - sm2, 0.0))
        return (su_, sv_, sl_sigma_w(z, self.ustar, self.inv_L))

    def sl_T(self, z):
        """Small-scale Lagrangian time scales T_i = 2σ_i²/(C0 ε(z)) (s)."""
        e = sl_eps(z, self.ustar, self.inv_L)
        return tuple(2 * s_ * s_ / (self.sl_c0 * e) for s_ in self.sl_sigma(z))

    def _convective(self):
        """(σ_c, T_c): coherent large-scale lateral eddies — convective (unstable,
        Panofsky/Hanna) or, with meander_frac > 0, the neutral/stable meander
        (T_L = β·meander_TE, β = γ/i, the Hay–Pasquill ratio used for T_E)."""
        if not self.surface_layer:
            return 0.0, 1e6
        if self.inv_L >= 0:
            sm = self._meander_sigma()
            if sm <= 0:
                return 0.0, 1e6
            beta = max(self.hp_gamma / max(sm / self.U_src, 1e-6), 1.0)
            return sm, beta * self.meander_TE
        svp = self.ustar * (12.0 + 0.5 * self.zi_eff * abs(self.inv_L)) ** (1.0 / 3.0)
        sm = SL_SIGMA_MECH[1] * self.ustar
        return math.sqrt(max(svp * svp - sm * sm, 0.0)), 0.15 * self.zi_eff / svp

    @property
    def lsm_sigma(self):
        """LSM velocity standard deviations (σ_u, σ_v, σ_w) in m/s."""
        if self.surface_layer:
            sc, _ = self._convective()
            return (sc, sc, 0.0)
        U = self.U_src
        m = self.sigma_min
        return (max(self.a_y * U, m), max(self.a_y * U, m), max(self.a_z * U, m))

    @property
    def lsm_TL(self):
        """Lagrangian time scales (s) per axis, T_L,fit·LSM_U_FIT/U."""
        if self.surface_layer:
            _, tc = self._convective()
            return (tc, tc, 1e6)
        f = LSM_U_FIT / self.U_src
        return (self.TL_y * f, self.TL_y * f, self.TL_z * f)

    @property
    def lsm_TE(self):
        """Eulerian time scales (s) at the source, T_E = T_L/β, β = γ/i ≥ 1."""
        TL = self.lsm_TL
        if self.surface_layer:
            i = max(self.lsm_sigma[0] / self.U_src, 1e-6)
            b = max(self.hp_gamma / i, 1.0)
            return (TL[0] / b, TL[1] / b, TL[2])
        beta = [max(self.hp_gamma / a, 1.0) for a in (self.a_y, self.a_y, self.a_z)]
        return tuple(T / b for T, b in zip(TL, beta))

    def _ou_coeffs(self, dt):
        sig, TL = self.lsm_sigma, self.lsm_TL
        decay = [math.exp(-dt / T) for T in TL]
        noise = [s_ * math.sqrt(1.0 - d * d) for s_, d in zip(sig, decay)]
        return wp.vec3(*decay), wp.vec3(*noise)

    def set_stability_class(self, pg_class: str = "D"):
        """Set LSM intensities and T_L, lapse rate and shear exponent (A–F)."""
        pg_class = pg_class.upper()
        if pg_class not in PG_LSM:
            raise ValueError(f"Unknown PG class '{pg_class}'. Use A–F.")
        self.stability_class = pg_class
        c = dict(PG_LSM[pg_class])
        c.update(self.lsm_table.get(pg_class, {}))
        self.a_y, self.a_z = c["a_y"], c["a_z"]
        self.TL_y, self.TL_z = c["TL_y"], c["TL_z"]
        if pg_class in self.class_sigma0:
            self.sigma_0 = float(self.class_sigma0[pg_class])
        self._update_met()
        # Lapse rate (used only by buoyancy_mode="rise")
        self.LAPSE_RATE = PG_LAPSE_RATES[pg_class]
        # Shear exponent for power-law vertical wind profile
        self.wind_model.shear_exponent = (PG_SHEAR_EXPONENTS[pg_class]
                                          if self.shear_exponent_override is None
                                          else self.shear_exponent_override)

    def _run_pressure_sor(self):
        """Red-Black SOR pressure solve with fixed iterations (no CPU-GPU syncs)."""
        omega = 1.7
        fixed_iters = self.sor_iters   # see divergence_residual() / V8
        shape = self.wind_shape

        for _ in range(fixed_iters):
            # Red sweep (parity 0)
            wp.launch(pressure_sor_rb, dim=shape,
                      inputs=[self.p0, self.div, self.obstacle_mask, omega, 0])
            # Black sweep (parity 1)
            wp.launch(pressure_sor_rb, dim=shape,
                      inputs=[self.p0, self.div, self.obstacle_mask, omega, 1])
            
    @property
    def Q(self):
        """Emission rate (kg/s)."""
        return self._Q

    @Q.setter
    def Q(self, q):
        q = float(q)
        if not q >= 0:
            raise ValueError(f"Q must be >= 0 kg/s (got {q})")
        self._Q = q

    @property
    def sigma_0(self):
        """Initial puff standard deviation (m)."""
        return self._sigma_0

    @sigma_0.setter
    def sigma_0(self, s):
        s = float(s)
        if not s > 0:
            raise ValueError(f"sigma_0 must be > 0 m (got {s})")
        self._sigma_0 = s

    @property
    def source_pos(self):
        return self._source_pos

    @source_pos.setter
    def source_pos(self, p):
        """Source position (m).  Rejected with ValueError if it lies inside an
        obstacle; coordinates outside the domain are also rejected."""
        p = np.asarray(p, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError(f"source position {p} is not finite")
        dmax = np.array([NX*DX, NY*DX, NZ*DX])
        if np.any(p < 0) or np.any(p > dmax):
            raise ValueError(f"source position {p} is outside the domain [0, {dmax.tolist()}]")
        d = self.obstacle_sdf(p)
        if d < 0:
            raise ValueError(f"source position {p} is inside an obstacle (signed distance {d:.2f} m); "
                             "move the source or the obstacle")
        self._source_pos = p

    def obstacle_sdf(self, p):
        """Signed distance (m) from point p to the nearest obstacle (+inf if none)."""
        p = np.asarray(p, float)
        d = np.inf
        for s in self._spheres_host:
            d = min(d, float(np.linalg.norm(p - np.asarray(s.center, float)) - float(s.radius)))
        for b in self._cuboids_host:
            q = np.abs(p - np.asarray(b.center, float)) - np.asarray(b.half_extents, float)
            d = min(d, float(np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)))
        return d

    def set_obstacles(self, spheres, cuboids):
        dmax = np.array([NX*DX, NY*DX, NZ*DX])
        if len(spheres) > _MAX_SPHERES or len(cuboids) > _MAX_CUBOIDS:
            raise ValueError(f"at most {_MAX_SPHERES} spheres and {_MAX_CUBOIDS} cuboids")
        for o in list(spheres) + list(cuboids):
            c = np.asarray(o.center, float)
            ext = np.full(3, float(o.radius)) if hasattr(o, "radius") else np.asarray(o.half_extents, float)
            if not (np.all(np.isfinite(c)) and np.all(ext > 0)):
                raise ValueError(f"invalid obstacle {o}")
            if np.any(c + ext < 0) or np.any(c - ext > dmax):
                raise ValueError(f"obstacle at {c.tolist()} lies entirely outside the domain [0, {dmax.tolist()}]")
        self._spheres_host = list(spheres)[:_MAX_SPHERES]
        self._cuboids_host = list(cuboids)[:_MAX_CUBOIDS]
        self.n_spheres=min(len(spheres),_MAX_SPHERES); self.n_cuboids=min(len(cuboids),_MAX_CUBOIDS)
        sc=np.zeros((_MAX_SPHERES,3),dtype=np.float32); sr=np.zeros(_MAX_SPHERES,dtype=np.float32)
        for i in range(self.n_spheres): sc[i]=spheres[i].center; sr[i]=spheres[i].radius
        cc=np.zeros((_MAX_CUBOIDS,3),dtype=np.float32); ce=np.zeros((_MAX_CUBOIDS,3),dtype=np.float32)
        for i in range(self.n_cuboids): cc[i]=cuboids[i].center; ce[i]=cuboids[i].half_extents
        wp.copy(self.sphere_centers,wp.array(sc,dtype=wp.vec3)); wp.copy(self.sphere_radii,wp.array(sr,dtype=float))
        wp.copy(self.cuboid_centers,wp.array(cc,dtype=wp.vec3)); wp.copy(self.cuboid_extents,wp.array(ce,dtype=wp.vec3))

        # Recompute obstacle mask on the grid
        wp.launch(compute_obstacle_mask, dim=self.wind_shape, inputs=[
            self.obstacle_mask, float(self.dx),
            self.n_spheres, self.sphere_centers, self.sphere_radii,
            self.n_cuboids, self.cuboid_centers, self.cuboid_extents,
        ])

        # No graph invalidation needed — SOR is not captured in CUDA graph

    def _step_spatial_turbulence(self, dt):
        """Evolve the coarse wind perturbation field with OU dynamics."""
        th = self._turb_ou_theta
        decay = math.exp(-th * dt)
        noise_std = self._turb_ou_sigma * math.sqrt((1.0 - math.exp(-2.0 * th * dt)) / (2.0 * th))
        self._turb_np *= decay
        self._turb_np += noise_std * self._rng.standard_normal(self._turb_np.shape).astype(np.float32)
        wp.copy(self.turb_field, wp.array(self._turb_np.reshape(*self._turb_shape, 3), dtype=wp.vec3))

    @property
    def n_overwritten(self):
        """Puffs overwritten while still active (ring buffer too small → mass lost)."""
        return int(self._counters.numpy()[0])

    @property
    def mass_overwritten(self):
        """kg of puffs overwritten while active (lost)."""
        return float(self._mass_counters.numpy()[0])

    @property
    def mass_exited(self):
        """kg of puffs removed at the open faces."""
        return float(self._mass_counters.numpy()[1])

    @property
    def mass_in_domain(self):
        act = self.puff_active.numpy() == 1
        return float(self.puff_mass.numpy()[act].astype(np.float64).sum())

    def _check_overflow(self):
        """Warn once if the ring buffer overwrote active puffs, naming the
        max_puffs (env N_PUFFS) needed: 1.25 × oldest active age / emit_interval."""
        if self._overflow_warned:
            return
        n = self.n_overwritten
        if n == 0:
            return
        act = self.puff_active.numpy() == 1
        age = float((self.sim_time - self.puff_emit_time.numpy()[act]).max()) if act.any() else 0.0
        need = int(math.ceil(1.25 * max(age, self.max_puffs * self.emit_interval) / self.emit_interval))
        import warnings
        warnings.warn(f"puff ring buffer overflow: {n} active puffs overwritten ({self.mass_overwritten:.3g} kg lost); "
                      f"oldest active puff is {age:.0f} s old. Use max_puffs (env N_PUFFS) >= {need}.",
                      RuntimeWarning, stacklevel=2)
        self._overflow_warned = True

    @property
    def n_exited(self):
        """Puffs removed at the open domain faces."""
        return int(self._counters.numpy()[1])

    def emit_puff(self):
        """Release puffs on the fixed schedule t_n = n·emit_interval.

        Mass per puff = Q · emit_interval, so the release rate is exactly Q.
        (Pre-fix: two puffs per interval → 2Q, and float drift in
        `sim_time - last_emit < emit_interval` stretched 0.2 s to 7 frames.)
        """
        n_per = 2 if self.legacy_emission else 1
        if self.sim_time + 1e-9 >= self._next_emit_time:
            self._max_emit_interval = max(self._max_emit_interval, float(self.emit_interval))
        while self.sim_time + 1e-9 >= self._next_emit_time:
            mass = float(self.Q * self.emit_interval)
            for _ in range(n_per):
                idx = self.next_puff_idx % self.max_puffs
                pos = self.source_pos.astype(np.float64)
                if self.release_jitter > 0.0:
                    pos = pos + self._rng.standard_normal(3) * self.release_jitter
                wp.launch(emit_puff_kernel, dim=1, inputs=[
                    self.puff_pos, self.puff_active, self.puff_emit_time, self.puff_mass,
                    self.puff_vel,
                    idx, wp.vec3(*[float(v) for v in pos]), float(self.sim_time), mass,
                    self._counters, self._release_velocity(), self._mass_counters,
                    self.puff_vs, self._release_small_scale(pos[2])])
                self.next_puff_idx += 1
                self.n_emitted += 1
                self.mass_emitted += mass
            self._next_emit_time += self.emit_interval

    def _release_small_scale(self, z):
        """Small-scale velocity of a new puff: N(0, σ_i(z)²), independent per puff."""
        if not self.surface_layer:
            return wp.vec3(0.0, 0.0, 0.0)
        v = np.array(self.sl_sigma(z)) * self._vs_rng.standard_normal(3)
        return wp.vec3(*[float(c) for c in v])

    def _release_velocity(self):
        """Turbulent velocity of a new puff: σ·ζ, ζ ~ N(0, 1) per axis.

        With shared_eddies, ζ is an OU process in release time:
        ζ_n = ρ ζ_{n−1} + √(1 − ρ²) ξ,  ρ = e^{−τ_e/T_E} (per axis).
        """
        xi = self._release_rng.standard_normal(3)
        if not self.shared_eddies or self._release_zeta is None:
            self._release_zeta = xi
        else:
            rho = np.exp(-self.emit_interval / np.array(self.lsm_TE))
            self._release_zeta = rho * self._release_zeta + np.sqrt(1.0 - rho * rho) * xi
        v = np.array(self.lsm_sigma) * self._release_zeta
        k = self._eddy_hist_i % EDDY_HIST_N
        self._eddy_hist_t[k] = self.sim_time; self._eddy_hist_v[k] = v; self._eddy_hist_i += 1
        return wp.vec3(*[float(c) for c in v])

    def small_scale_ou(self, state, z, dt, rng):
        """One exact OU step of the local small-scale turbulence seen by a fixed or
        slowly moving sensor at height z (surface_layer): σ_i(z), time scale T_i(z).
        Returns the new (3,) state; zeros when surface_layer is off."""
        if not self.surface_layer:
            return np.zeros(3)
        sig = np.array(self.sl_sigma(z)); T = np.array(self.sl_T(z))
        a = np.exp(-dt / T)
        return a * np.asarray(state, float) + sig * np.sqrt(1.0 - a * a) * rng.standard_normal(3)

    def coherent_velocity(self, points):
        """Large-scale turbulent velocity (m/s) that moves the plume, at points (M, 3).

        With shared eddies, puffs leave the source with the velocity σ·ζ(t) of an
        OU process with the Eulerian time scale T_E (Hay–Pasquill): the eddy at
        the source.  A point downwind sees that eddy after the advection time
        (Taylor's frozen turbulence): v(x, t) = σ·ζ(t − max(x_along, 0)/U_src),
        x_along the downwind distance from the source along the mean wind.  An
        anemometer reading = resolved wind + coherent_velocity, so wind-direction
        changes at the sensor precede and predict the plume's meander.  Zero
        without shared eddies (puff velocities are then independent)."""
        P = np.atleast_2d(np.asarray(points, np.float64))
        out = np.zeros((len(P), 3))
        if not self.shared_eddies or self._eddy_hist_i == 0:
            return out
        wm = self.wind_model
        d = np.array([wm.u_mean, wm.v_mean]); nd = np.linalg.norm(d)
        d = d / nd if nd > 0 else np.array([1.0, 0.0])
        along = (P[:, :2] - np.asarray(self.source_pos, float)[:2]) @ d
        t_q = self.sim_time - np.maximum(along, 0.0) / self.U_src
        n = min(self._eddy_hist_i, EDDY_HIST_N)
        order = np.argsort(self._eddy_hist_t[:n])
        ts, vs = self._eddy_hist_t[:n][order], self._eddy_hist_v[:n][order]
        idx = np.clip(np.searchsorted(ts, t_q, side="right") - 1, 0, n - 1)
        return vs[idx]

    def _fixed_inject_dir(self):
        return wp.vec3(0.0, 0.0, 2.0)

    def step_wind(self):
        dt=self.sim_dt; shape=self.wind_shape

        self.wind_model.step(dt)
        wx,wy,wz,shear_exp=self.wind_model.get_wind(self.sim_time)
        self._step_spatial_turbulence(dt)

        # Newtonian relaxation coefficient α (1/s).
        # α ≈ 3.0 → τ_relax = 1/α ≈ 0.33 s: fast enough to track
        # the OU wind model, slow enough for fluid features to develop.
        _nudge_alpha = 3.0
        wp.launch(apply_global_wind, dim=shape,
                  inputs=[self.u0,float(wx),float(wy),float(wz),
                          _nudge_alpha,dt,float(shear_exp),self.turb_field,
                          self.obstacle_mask,float(SHEAR_Z_REF),float(self.dx),
                          float(self.ground_drag_rate),self.N_DRAG_LAYERS,1 if self.drag_compensation else 0])
        wp.launch(apply_ground_drag, dim=shape, inputs=[self.u0,float(self.ground_drag_rate),float(dt),self.N_DRAG_LAYERS])

        # Zero velocity inside obstacles before advection
        wp.launch(enforce_velocity_obstacle, dim=shape,
                  inputs=[self.u0, self.obstacle_mask])

        wp.launch(advect_vel_and_rho,dim=shape,
                  inputs=[self.u0,self.u1,self.rho0,self.rho1,dt])
        self.u0,self.u1=self.u1,self.u0; self.rho0,self.rho1=self.rho1,self.rho0

        # Zero velocity inside obstacles after advection (semi-Lagrangian can smear)
        wp.launch(enforce_velocity_obstacle, dim=shape,
                  inputs=[self.u0, self.obstacle_mask])

        # Pressure solve (obstacle-aware divergence + Red-Black SOR)
        wp.launch(wind_divergence_3d, dim=shape,
                  inputs=[self.u0, self.div, self.obstacle_mask])
        self.p0.zero_()

        # Red-Black SOR with adaptive convergence
        self._run_pressure_sor()

        wp.launch(pressure_apply_3d, dim=shape,
                  inputs=[self.p0, self.u0, self.obstacle_mask])

        # Final velocity cleanup inside obstacles
        wp.launch(enforce_velocity_obstacle, dim=shape,
                  inputs=[self.u0, self.obstacle_mask])

        # Legacy Eulerian gas field: buoyancy on the wind, grid diffusion and the
        # source momentum jet (emit_density_fixed adds (0,0,2)*w*0.5 m/s to the
        # wind every substep). Off by default: the sensor reads puffs only.
        if not self.use_grid_gas:
            return

        wp.launch(integrate_rho,dim=shape,
                  inputs=[self.u0,self.rho0,dt,
                          float(METHANE_BUOYANCY_ACCEL),self.RHO_DECAY_RATE,
                          float(self.LAPSE_RATE),float(self.T_SURFACE),float(self.dx),
                          float(self.sigma_0),float(self.k_rel),float(self.sim_time)])

        # Diffusion (obstacle-aware Neumann no-flux at surfaces)
        wp.launch(diffuse_rho, dim=shape,
                  inputs=[self.rho0, self.rho_tmp, self.obstacle_mask,
                          float(self.D_ISO), dt, float(self.dx)])
        self.rho0, self.rho_tmp = self.rho_tmp, self.rho0

        wp.launch(enforce_density_obstacle, dim=shape,
                  inputs=[self.rho0, self.obstacle_mask])

        src_gx=float(self.source_pos[0]/self.dx-0.5)
        src_gy=float(self.source_pos[1]/self.dx-0.5)
        src_gz=float(self.source_pos[2]/self.dx-0.5)
        emit_strength=float(self.Q*dt*0.08)
        inject_dir=self._fixed_inject_dir()
        wp.launch(emit_density_fixed, dim=shape,
                  inputs=[self.rho0,self.u0,src_gx,src_gy,src_gz,
                          2.0,emit_strength,inject_dir])

    def step_puffs(self):
        n=min(self.next_puff_idx,self.max_puffs)
        if n==0: return
        wp.launch(advect_puffs_rk2,dim=n,inputs=[
            self.puff_pos,self.puff_active,self.puff_emit_time,
            self.u0,self.sim_dt,self.dx,self.sim_time,
            self.puff_vel,*self._ou_coeffs(self.sim_dt),
            self.n_spheres,self.sphere_centers,self.sphere_radii,
            self.n_cuboids,self.cuboid_centers,self.cuboid_extents,
            float(self.LAPSE_RATE),float(self.T_SURFACE),
            int((self.seed*1000003+self._step_counter)%2147483647),
            1 if self.buoyancy_mode=="rise" else 0, float(self.w_rise0), float(self.sigma_0),
            *[float(v) for v in self.K_rel],
            1 if self.subgrid_wind else 0, float(self.wind_model.shear_exponent),
            self.puff_vs, 1 if self.surface_layer else 0,
            float(getattr(self, "ustar", 0.0)), float(getattr(self, "inv_L", 0.0)),
            *[float(v) for v in (self.sl_sigma(1.0)[:2] if self.surface_layer else (0.0, 0.0))],
            float(self.sl_c0), float(SL_Z_MIN),
        ])
        self._step_counter+=1
        wp.launch(reflect_puffs_at_boundary,dim=n,inputs=[
            self.puff_pos,self.puff_active,self.puff_vel,
            float(self._domain_max[0]),float(self._domain_max[1]),float(self._domain_max[2]),
            self._counters, self.puff_mass, self._mass_counters,
            1 if self.closed_domain else 0, self.puff_vs,
        ])

    def _obstacle_args(self):
        return (self.n_spheres, self.sphere_centers, self.sphere_radii,
                self.n_cuboids, self.cuboid_centers, self.cuboid_extents,
                1 if self.use_los else 0)

    def _rho_scale(self):
        """Grid-gas contribution to sensed concentration (0 unless legacy mode)."""
        return float(self.RHO_SCALE) if self.use_grid_gas else 0.0

    def compute_conc_grid(self):
        n=min(self.next_puff_idx,self.max_puffs)
        wp.launch(compute_concentration_grid_kernel,dim=self.wind_shape,inputs=[
            self.conc_grid,self.rho0,self._rho_scale(),
            self.puff_pos,self.puff_active,self.puff_emit_time,self.puff_mass,
            self.sim_time,float(self.sigma_0),*[float(v) for v in self.K_rel],self.dx,n,
            *self._obstacle_args(),
        ])

    def _influence_radius(self):
        """Upper bound (m) of any active puff's 4σ box half-diagonal.  A live emitted
        puff is at most max_puffs·emit_interval old (the ring buffer overwrites it
        after that); injected puffs are tracked by their oldest emit time."""
        age = self.max_puffs * self._max_emit_interval
        if self._min_inject_emit is not None:
            age = max(age, self.sim_time - self._min_inject_emit)
        sig = math.sqrt(self.sigma_0 ** 2 + 2.0 * max(self.K_rel) * max(age, 0.0))
        return 4.0 * math.sqrt(3.0) * sig * 1.0001

    def query_concentration(self, sensor_positions_np):
        """Concentration (kg/m³) at each sensor position.  Returns 0 below
        ground, outside the domain and (with use_los) inside obstacles.
        Uses a hash grid over puff centres (use_hash_query, default) unless the
        legacy grid gas is on; same sum as the brute-force loop."""
        if self.use_hash_query and not self.use_grid_gas:
            return self._query_hashed(sensor_positions_np)
        M=sensor_positions_np.shape[0]; sensors=wp.array(sensor_positions_np,dtype=wp.vec3)
        conc=wp.zeros(M,dtype=float); n=min(self.next_puff_idx,self.max_puffs)
        wp.launch(query_sensor_hybrid,dim=M,inputs=[
            sensors,self.rho0,self._rho_scale(),self.dx,
            self.puff_pos,self.puff_active,self.puff_emit_time,self.puff_mass,
            self.sim_time,float(self.sigma_0),*[float(v) for v in self.K_rel],n,conc,
            *self._obstacle_args(),
        ])
        return conc.numpy()

    def _query_hashed(self, sensor_positions_np):
        M = sensor_positions_np.shape[0]
        sensors = wp.array(np.asarray(sensor_positions_np, np.float32), dtype=wp.vec3)
        conc = wp.zeros(M, dtype=float)
        R = self._influence_radius()
        if self._hash is None:
            self._hash = wp.HashGrid(NX, NY, NZ)
        self._hash.build(points=self.puff_pos, radius=R)
        wp.launch(query_sensor_hashed, dim=M, inputs=[
            self._hash.id, float(R), sensors, self.dx,
            self.puff_pos, self.puff_active, self.puff_emit_time, self.puff_mass,
            self.sim_time, float(self.sigma_0), *[float(v) for v in self.K_rel], conc,
            *self._obstacle_args(),
        ])
        return conc.numpy()

    def step(self):
        self.emit_puff(); self.step_wind()
        if self.sim_time + 1e-9 >= self._next_U_update:
            self._update_U_src(); self._next_U_update += self.U_SRC_EVERY
            self._check_overflow()
        self.step_puffs(); self.sim_time+=self.sim_dt

    def inject_puffs(self, positions, ages=None, masses=1.0, velocities=None):
        """Replace all puffs with the given set (tests/validation only).

        positions: (N, 3) m; ages: (N,) s (default 1 s); masses: kg, scalar or (N,);
        velocities: (N, 3) LSM turbulent velocities (default: drawn from
        N(0, lsm_sigma²) with a numpy RNG seeded by self.seed).
        N must be ≤ max_puffs. Puff σ follows the growth law for its age.
        """
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        n = len(positions)
        if n > self.max_puffs:
            raise ValueError(f"{n} puffs > max_puffs={self.max_puffs}")
        ages = np.ones(n) if ages is None else np.broadcast_to(np.asarray(ages, float), (n,))
        masses = np.broadcast_to(np.asarray(masses, float), (n,))
        pos = np.zeros((self.max_puffs, 3), np.float32); pos[:n] = positions
        act = np.zeros(self.max_puffs, np.int32); act[:n] = 1
        emt = np.zeros(self.max_puffs, np.float32); emt[:n] = self.sim_time - ages
        self._min_inject_emit = float(emt[:n].min()) if n else None
        mas = np.zeros(self.max_puffs, np.float32); mas[:n] = masses
        if velocities is None:
            velocities = np.random.default_rng(self.seed).standard_normal((n, 3)) * np.array(self.lsm_sigma)
        vel = np.zeros((self.max_puffs, 3), np.float32); vel[:n] = velocities
        wp.copy(self.puff_vel, wp.array(vel, dtype=wp.vec3))
        self.puff_vs.zero_()
        wp.copy(self.puff_pos, wp.array(pos, dtype=wp.vec3))
        wp.copy(self.puff_active, wp.array(act, dtype=int))
        wp.copy(self.puff_emit_time, wp.array(emt, dtype=float))
        wp.copy(self.puff_mass, wp.array(mas, dtype=float))
        self.next_puff_idx = n

    def get_puff_data(self):
        pos=self.puff_pos.numpy(); active=self.puff_active.numpy(); emit_t=self.puff_emit_time.numpy()
        n=min(self.next_puff_idx,self.max_puffs); mask=active[:n]==1
        return pos[:n][mask], self.sim_time-emit_t[:n][mask]

    def get_wind_slice_xy(self, z_idx=None):
        if z_idx is None: z_idx=1
        u_np=self.u0.numpy(); s=u_np[:,:,z_idx,:]
        return np.sqrt(s[:,:,0]**2+s[:,:,1]**2+s[:,:,2]**2), s[:,:,0], s[:,:,1]

    def get_conc_slice_xy(self, z_idx=None):
        if z_idx is None: z_idx=1
        self.compute_conc_grid(); return self.conc_grid.numpy()[:,:,z_idx]

    def get_rho_slice_xy(self, z_idx=None):
        if z_idx is None: z_idx=1
        return self.rho0.numpy()[:,:,z_idx]

    def get_obstacle_mask_slice_xy(self, z_idx=None):
        """Return a 2D slice of the obstacle mask for visualization."""
        if z_idx is None: z_idx = 1
        return self.obstacle_mask.numpy()[:, :, z_idx]

    # ── Column-integrated projections (Mod 1) ──────────────────────────
    def get_conc_column_xy(self):
        """Column-integrated concentration projected onto XY plane (kg/m²)."""
        self.compute_conc_grid()
        return np.sum(self.conc_grid.numpy(), axis=2) * self.dx

    def get_wind_mean_xy(self):
        """Vertically-averaged wind vector on XY plane.
        Returns: (speed, ux, uy) — speed magnitude and horizontal components.
        """
        u_np = self.u0.numpy()
        u_mean = np.mean(u_np, axis=2)  # shape (NX, NY, 3)
        speed = np.sqrt(u_mean[:, :, 0]**2 + u_mean[:, :, 1]**2 + u_mean[:, :, 2]**2)
        return speed, u_mean[:, :, 0], u_mean[:, :, 1]

    def get_obstacle_mask_any_z(self):
        """Return 2D mask: 1 if any z-level has an obstacle at (x,y)."""
        return np.any(self.obstacle_mask.numpy() == 1, axis=2).astype(int)


# ── Standalone visualisation (run with --mode 3d) ─────────────────────────

def run_3d_puff_visualization(center_angle, angle_range, base_speed):
    import matplotlib.pyplot as plt

    sim = HybridFluidPuffSimulation(
        wind_center_angle=center_angle,
        wind_angle_range=angle_range,
        wind_base_speed=base_speed,
    )

    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    xmax = NX * DX
    ymax = NY * DX
    zmax = NZ * DX

    plt.ion()
    plt.show()

    for frame in range(600):
        sim.step()

        if frame % 3 != 0:
            continue

        ax3d.cla()
        positions, ages = sim.get_puff_data()

        if len(positions) > 0:
            sigmas = sim.puff_sigma(ages).max(axis=1)
            sizes = sigmas * 8.0
            max_age = max(ages.max(), 1.0)
            colors = plt.cm.viridis(1.0 - ages / max_age)
            colors[:, 3] = np.clip(1.0 - ages / max_age, 0.1, 0.8)

            ax3d.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                s=sizes, c=colors, depthshade=True, edgecolors='none'
            )

        ax3d.scatter(*sim.source_pos, c='lime', s=200, marker='*',
                     edgecolors='black', linewidths=1, zorder=10)

        ax3d.set_xlim(0, xmax)
        ax3d.set_ylim(0, ymax)
        ax3d.set_zlim(0, zmax)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')

        wind_deg = sim.wind_model.get_angle_degrees()
        ax3d.set_title(
            f'3D Puff Positions  t={sim.sim_time:.1f}s\n'
            f'Active: {len(positions)}  Wind: {wind_deg:.0f}°'
        )

        ax2d.cla()
        # Column-integrated concentration (kg/m²) — projects all z-levels
        conc_col = sim.get_conc_column_xy()

        # Overlay obstacle mask (any z-level)
        obs_mask = sim.get_obstacle_mask_any_z()

        ax2d.imshow(
            conc_col.T, origin='lower', cmap='viridis',
            extent=[0, xmax, 0, ymax], aspect='auto',
            vmin=0, vmax=max(conc_col.max(), 1e-6),
            interpolation='bilinear',
        )

        # Draw obstacles as gray overlay
        if obs_mask.any():
            obs_rgba = np.zeros((*obs_mask.T.shape, 4), dtype=np.float32)
            obs_rgba[obs_mask.T == 1] = [0.4, 0.4, 0.4, 0.8]
            ax2d.imshow(obs_rgba, origin='lower',
                        extent=[0, xmax, 0, ymax], aspect='auto')

        # Vertically-averaged wind vectors
        speed, ux, uy = sim.get_wind_mean_xy()
        skip = 2
        xs = np.arange(0, NX, skip) * DX + DX * 0.5
        ys = np.arange(0, NY, skip) * DX + DX * 0.5
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        ax2d.quiver(X, Y, ux[::skip, ::skip], uy[::skip, ::skip],
                    color='cyan', alpha=0.5, scale=150, width=0.003)

        ax2d.plot(sim.source_pos[0], sim.source_pos[1], 'g*',
                  markersize=15, markeredgecolor='white')

        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.set_title(
            f'Column-integrated concentration (kg/m$^2$) t={sim.sim_time:.1f}s  '
            f'wind={wind_deg:.0f}°'
        )

        fig.suptitle(
            '3D Stable Fluids + Hybrid Gaussian-Puff + Grid',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.show()


def run_diagnostics(center_angle, angle_range, base_speed):
    import matplotlib.pyplot as plt

    sim = HybridFluidPuffSimulation(
        wind_center_angle=center_angle,
        wind_angle_range=angle_range,
        wind_base_speed=base_speed,
    )

    sensor_pos = np.array([[
        sim.source_pos[0] + 40.0,
        sim.source_pos[1] + 40.0,
        sim.source_pos[2],
    ]], dtype=np.float32)

    times, conc_hist, div_hist = [], [], []
    puff_hist, speed_hist, angle_hist = [], [], []

    print("Running diagnostics (400 steps)...")
    for frame in range(400):
        sim.step()
        times.append(sim.sim_time)
        conc_hist.append(sim.query_concentration(sensor_pos)[0])
        div_hist.append(np.sqrt(np.mean(sim.div.numpy()**2)))
        puff_hist.append(np.sum(sim.puff_active.numpy()))

        u_np = sim.u0.numpy()
        sx = int(np.clip(sim.source_pos[0] / DX, 0, NX - 1))
        sy = int(np.clip(sim.source_pos[1] / DX, 0, NY - 1))
        sz = int(np.clip(sim.source_pos[2] / DX, 0, NZ - 1))
        speed_hist.append(np.linalg.norm(u_np[sx, sy, sz]))
        angle_hist.append(sim.wind_model.get_angle_degrees())

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].plot(times, conc_hist, 'r-', lw=1.5)
    axes[0, 0].set_title('Concentration at Sensor')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('c (kg/m³)')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(times, div_hist, 'b-', lw=1.5)
    axes[0, 1].set_title('RMS Divergence')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(times, puff_hist, 'g-', lw=1.5)
    axes[0, 2].set_title('Active Puffs')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(times, speed_hist, 'm-', lw=1.5)
    axes[1, 0].set_title('Wind Speed at Source')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('|u| (m/s)')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(times, angle_hist, 'k-', lw=1.5)
    axes[1, 1].axhline(y=math.degrees(sim.wind_model.center_rad),
                        color='r', ls='--', alpha=0.5, label='center')
    axes[1, 1].axhline(y=math.degrees(sim.wind_model.angle_min),
                        color='b', ls=':', alpha=0.5, label='min')
    axes[1, 1].axhline(y=math.degrees(sim.wind_model.angle_max),
                        color='b', ls=':', alpha=0.5, label='max')
    axes[1, 1].set_title('Wind Direction (degrees)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    angles_rad = np.radians(angle_hist)
    axes[1, 2].remove()
    ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
    ax_polar.hist(angles_rad, bins=36, density=True, alpha=0.7,
                  color='steelblue')
    ax_polar.set_title('Wind Rose', pad=15)

    fig.suptitle('Atmospheric Wind Diagnostics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagnostics_wind.png', dpi=150)
    print("Saved diagnostics_wind.png")
    plt.show()


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3D Stable Fluids + Hybrid Gaussian-Puff "
                    "(Realistic Atmospheric Wind)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default="3d",
                        choices=["3d", "diagnostics", "test"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--wind-center", type=float, default=45.0,
                        help="Center wind direction (degrees, e.g., 45 for NE)")
    parser.add_argument("--wind-range", type=float, default=40.0,
                        help="Wind angular range (degrees meander)")
    parser.add_argument("--wind-speed", type=float, default=4.5,
                        help="Base wind speed (m/s)")
    parser.add_argument("--wind-seed", type=int, default=42,
                        help="RNG seed for wind model and puff emission")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        if args.headless or args.mode == "test":
            sim = HybridFluidPuffSimulation(
                wind_center_angle=args.wind_center,
                wind_angle_range=args.wind_range,
                wind_base_speed=args.wind_speed,
                wind_seed=args.wind_seed,
            )
            print(f"Running {args.num_frames} frames, "
                  f"wind: {args.wind_center}° ± "
                  f"{args.wind_range / 2}°  seed={args.wind_seed}")
            for i in range(args.num_frames):
                sim.step()
                if (i + 1) % 50 == 0:
                    positions, ages = sim.get_puff_data()
                    deg = sim.wind_model.get_angle_degrees()
                    if len(ages) > 0:
                        spread_y = float(np.std(positions[:, 1]))
                        spread_x = float(np.std(positions[:, 0]))
                        print(
                            f"  Frame {i+1}: t={sim.sim_time:.1f}s, "
                            f"puffs={len(positions)}, "
                            f"spread_x={spread_x:.1f}m, spread_y={spread_y:.1f}m, "
                            f"rho_max={sim.rho0.numpy().max():.4f}, "
                            f"wind={deg:.0f}°"
                        )
                    else:
                        print(f"  Frame {i+1}: t={sim.sim_time:.1f}s, no puffs")
            print("Done.")
        elif args.mode == "3d":
            run_3d_puff_visualization(args.wind_center, args.wind_range, args.wind_speed)
        elif args.mode == "diagnostics":
            run_diagnostics(args.wind_center, args.wind_range, args.wind_speed)