
"""
stable_fluid_3d.py — 3-D Stable Fluids + Hybrid Gaussian-Puff concentration model.

Target gas: Methane (CH₄) — M=16.04 g/mol, ρ=0.657 kg/m³ at STP.
Concentration units: kg/m³.  Emission rate Q: kg/s.

Physics:
  - Jos Stam stable fluids (advect, pressure-project, diffuse) on 64×32×32 grid
  - Obstacle-aware velocity/pressure/density (SDF-based Neumann BCs)
  - Lagrangian Gaussian puffs with RK2 advection + Langevin noise
  - Ground reflection via method of images at z=0
  - Pasquill-Gifford stability classes A–F for K_D + Langevin coefficients
  - Temperature-profile-based methane buoyancy (Stull 1988)
"""

import warp as wp
import math
import numpy as np

# ── Methane (CH₄) physical constants ─────────────────────────────────────────
# M = 16.04 g/mol, ρ_STP = 0.657 kg/m³, D_mol = 2.2e-5 m²/s (negligible vs K_D)
# Buoyancy: w_b ≈ 0.5–1.5 m/s (entrainment-limited dilute plume)
# Q in kg/s; each puff carries Q · emit_interval (kg)
# Concentration in kg/m³; to ppm: c × (R·T)/(M·P) × 1e6

# Pasquill-Gifford turbulent diffusivities K_D (m²/s) for σ² = σ₀² + 2·K_D·t.
# Derived from Briggs (1973) rural σ_y(x), σ_z(x) power-law formulas
# at reference distance x=200 m, mean wind U=3 m/s (travel time t≈67 s):
#   K_D = σ²(x) / (2·t)
# Cross-wind (Y) and vertical (Z) from Briggs curves; along-wind (X) ≈ 0.3·K_D_Y
# (along-wind spreading dominated by shear, not turbulent diffusion).
# References: Gifford (1961), Briggs (1973), Seinfeld & Pandis (2016) Ch. 18.
PASQUILL_GIFFORD_KD = {
    "A": {"K_D_X": 4.3,  "K_D_Y": 14.2, "K_D_Z": 11.9,  "desc": "Very unstable"},
    "B": {"K_D_X": 3.0,  "K_D_Y":  8.5, "K_D_Z":  5.5,  "desc": "Unstable"},
    "C": {"K_D_X": 2.2,  "K_D_Y":  5.5, "K_D_Z":  2.8,  "desc": "Slightly unstable"},
    "D": {"K_D_X": 1.5,  "K_D_Y":  3.8, "K_D_Z":  1.3,  "desc": "Neutral (default)"},
    "E": {"K_D_X": 1.0,  "K_D_Y":  2.2, "K_D_Z":  0.6,  "desc": "Slightly stable"},
    "F": {"K_D_X": 0.5,  "K_D_Y":  1.1, "K_D_Z":  0.15, "desc": "Stable"},
    "G": {"K_D_X": 0.25, "K_D_Y":  0.5, "K_D_Z":  0.07, "desc": "Extremely stable"},
}

# Langevin sub-grid noise coefficients ℓ (m/√s): ℓ_i = √(2·K_D_i).
# This follows from the Langevin equation connection to diffusion:
#   σ² = ℓ²·t = 2·K_D·t  →  ℓ = √(2·K_D)
# Reference: Thomson (1987).
import math as _math
PASQUILL_GIFFORD_LANGEVIN = {
    cls: {
        "L_X": round(_math.sqrt(2.0 * kd["K_D_X"]), 2),
        "L_Y": round(_math.sqrt(2.0 * kd["K_D_Y"]), 2),
        "L_Z": round(_math.sqrt(2.0 * kd["K_D_Z"]), 2),
    }
    for cls, kd in PASQUILL_GIFFORD_KD.items()
}

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
METHANE_BUOYANCY_ACCEL = 9.81 * (1.225 - 0.657) / 1.225  # ≈ 4.55 m/s²
T_SURFACE_DEFAULT = 293.15  # K (20°C reference surface temperature)

# Power-law shear exponents α_s by PG stability class.
# Wind shear profile: s(k) = (z/z_ref)^α_s
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
    return wp.normalize(p - center)

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

@wp.func
def sample_vel_3d(u: wp.array3d(dtype=wp.vec3), x: float, y: float, z: float):
    lx=int(wp.floor(x)); ly=int(wp.floor(y)); lz=int(wp.floor(z))
    tx=x-float(lx); ty=y-float(ly); tz=z-float(lz)
    lx=wp.clamp(lx,0,WIND_NX-2); ly=wp.clamp(ly,0,WIND_NY-2); lz=wp.clamp(lz,0,WIND_NZ-2)
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
    lx=int(wp.floor(x)); ly=int(wp.floor(y)); lz=int(wp.floor(z))
    tx=x-float(lx); ty=y-float(ly); tz=z-float(lz)
    lx=wp.clamp(lx,0,WIND_NX-2); ly=wp.clamp(ly,0,WIND_NY-2); lz=wp.clamp(lz,0,WIND_NZ-2)
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
    # σ² = σ₀² + 2·K_D_Z·t  →  dilution = (σ₀²/σ²)^(3/2)
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
    """Trilinear interpolation on the coarse turbulence grid."""
    lx = int(wp.floor(x)); ly = int(wp.floor(y)); lz = int(wp.floor(z))
    tx = x - float(lx); ty = y - float(ly); tz = z - float(lz)
    lx = wp.clamp(lx, 0, TURB_NX - 2); ly = wp.clamp(ly, 0, TURB_NY - 2); lz = wp.clamp(lz, 0, TURB_NZ - 2)
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
    z_frac=(float(k)+0.5)/float(WIND_NZ)
    shear=wp.pow(wp.max(z_frac,0.02),shear_exp)
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
def apply_ground_drag(u: wp.array3d(dtype=wp.vec3), drag:float, n_layers:int):
    i,j,k = wp.tid()
    if k<n_layers:
        frac=float(k)/float(n_layers)
        u[i,j,k]=u[i,j,k]*wp.max(1.0-drag*(1.0-frac),0.0)

@wp.kernel
def reflect_puffs_at_boundary(
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    domain_max_x:float, domain_max_y:float, domain_max_z:float,
):
    i=wp.tid()
    if puff_active[i]==0: return
    p=puff_pos[i]; px=p[0]; py=p[1]; pz=p[2]
    # Open faces (x, y, z_top): deactivate puffs that leave the domain
    if px<0.0 or px>domain_max_x or py<0.0 or py>domain_max_y or pz>domain_max_z:
        puff_active[i]=0
        return
    # Ground (z=0): solid wall — reflect back
    if pz<0.0:
        pz=-pz
    puff_pos[i]=wp.vec3(px, py, pz)

# ── Lagrangian puffs ─────────────────────────────────────────────────────────

@wp.kernel
def advect_puffs_rk2(
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), wind_field: wp.array3d(dtype=wp.vec3),
    dt:float, dx:float, buoy_accel:float, current_time:float,
    langevin_x:float, langevin_y:float, langevin_z:float,
    n_spheres:int, sphere_centers:wp.array(dtype=wp.vec3), sphere_radii:wp.array(dtype=float),
    n_cuboids:int, cuboid_centers:wp.array(dtype=wp.vec3), cuboid_extents:wp.array(dtype=wp.vec3),
    lapse_rate:float, t_surface:float,
):
    i=wp.tid()
    if puff_active[i]==0: return
    pos=puff_pos[i]; gx=pos[0]/dx; gy=pos[1]/dx; gz=pos[2]/dx
    age=current_time-puff_emit_time[i]
    if age<0.0: age=0.0
    # Briggs (1975) dilution-limited buoyancy: a_eff = a_b · (σ₀/σ)³
    # σ² = σ₀² + 2·K_D_Z·age  →  dilution ratio = (σ₀²/σ²)^(3/2)
    # This replaces the ad-hoc exp(-0.2·age) with a physically motivated
    # decay that produces a finite plume rise consistent with Briggs.
    t_env = t_surface - lapse_rate * pos[2]
    sig2_0 = 2.0 * 2.0  # σ₀² (using σ₀ = 2 m, matched to HybridFluidPuffSimulation.sigma_0)
    sig2 = sig2_0 + 2.0 * langevin_z * langevin_z * 0.5 * wp.max(age, 0.01)  # σ² = σ₀² + 2·K_D_Z·t, K_D_Z = L_Z²/2
    dilution = wp.pow(sig2_0 / sig2, 1.5)  # (σ₀/σ)³
    buoy = buoy_accel * dilution * (293.15 / wp.max(t_env, 200.0))

    # RK2 advection
    vel1=sample_vel_3d(wind_field,gx,gy,gz)
    vel1=wp.vec3(vel1[0],vel1[1],vel1[2]+buoy); k1=vel1*dt
    mid_gx=(pos[0]+0.5*k1[0])/dx; mid_gy=(pos[1]+0.5*k1[1])/dx; mid_gz=(pos[2]+0.5*k1[2])/dx
    vel2=sample_vel_3d(wind_field,mid_gx,mid_gy,mid_gz)
    vel2=wp.vec3(vel2[0],vel2[1],vel2[2]+buoy)
    new_pos=pos+vel2*dt

    # Langevin noise
    t_int=int(current_time*100.0); seed=i*2654435761+t_int*1234567891
    h0=float((seed^(seed>>16))&0x7FFFFFFF)/2147483647.0+1e-9
    h1=float(((seed*1664525)^(seed>>13))&0x7FFFFFFF)/2147483647.0+1e-9
    h2=float(((seed*22695477)^(seed>>11))&0x7FFFFFFF)/2147483647.0+1e-9
    h3=float(((seed*6364136)^(seed>>17))&0x7FFFFFFF)/2147483647.0+1e-9
    h4=float(((seed*214013)^(seed>>14))&0x7FFFFFFF)/2147483647.0+1e-9
    h5=float(((seed*1140671)^(seed>>9))&0x7FFFFFFF)/2147483647.0+1e-9
    nx_r=wp.sqrt(-2.0*wp.log(h0))*wp.cos(6.28318530*h1)
    ny_r=wp.sqrt(-2.0*wp.log(h2))*wp.cos(6.28318530*h3)
    nz_r=wp.sqrt(-2.0*wp.log(h4))*wp.cos(6.28318530*h5)
    sqrt_dt=wp.sqrt(dt)
    new_pos=new_pos+wp.vec3(nx_r*langevin_x*sqrt_dt,
                            ny_r*langevin_y*sqrt_dt,
                            nz_r*langevin_z*sqrt_dt)

    # Obstacles — push puffs 2 m from the surface (half a cell) so they
    # reach a zone with nonzero wind velocity for advection.
    margin = 2.0
    for s in range(n_spheres):
        c=sphere_centers[s]; r=sphere_radii[s]; dist=sdf_sphere(new_pos,c,r)
        if dist<margin: n=normal_sphere(new_pos,c); new_pos=new_pos+n*(margin-dist)
    for ci in range(n_cuboids):
        c=cuboid_centers[ci]; e=cuboid_extents[ci]; dist=sdf_cuboid(new_pos,c,e)
        if dist<margin: n=normal_cuboid(new_pos,c,e); new_pos=new_pos+n*(margin-dist)
    puff_pos[i]=new_pos

# ── Concentration kernels ────────────────────────────────────────────────────

@wp.kernel
def query_sensor_hybrid(
    sensor_pos: wp.array(dtype=wp.vec3),
    rho_grid: wp.array3d(dtype=float), rho_scale:float, dx:float,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), current_time:float,
    Q_dt:float, sigma_0:float, K_d_x:float, K_d_y:float, K_d_z:float,
    num_puffs:int, concentration: wp.array(dtype=float),
):
    s=wp.tid(); p=sensor_pos[s]
    gx=p[0]/dx; gy=p[1]/dx; gz=p[2]/dx
    c_grid=sample_rho_3d(rho_grid,gx,gy,gz)*rho_scale
    c_puff=float(0.0)
    for i in range(num_puffs):
        if puff_active[i]==0: continue
        age=current_time-puff_emit_time[i]
        if age<1e-6: continue
        sig2_x=sigma_0*sigma_0+2.0*K_d_x*age
        sig2_y=sigma_0*sigma_0+2.0*K_d_y*age
        sig2_z=sigma_0*sigma_0+2.0*K_d_z*age
        sig_x=wp.sqrt(sig2_x); sig_y=wp.sqrt(sig2_y); sig_z=wp.sqrt(sig2_z)
        diff=p-puff_pos[i]
        # Early exit on horizontal distance only
        if wp.abs(diff[0])>4.0*sig_x or wp.abs(diff[1])>4.0*sig_y: continue
        # Ground reflection (image source method):
        #   Real source at z_i  → dz_real  = z - z_i
        #   Image source at -z_i → dz_image = z + z_i
        dz_real=diff[2]
        dz_image=p[2]+puff_pos[i][2]
        if wp.abs(dz_real)>4.0*sig_z and wp.abs(dz_image)>4.0*sig_z: continue
        r2_xy=diff[0]*diff[0]/sig2_x+diff[1]*diff[1]/sig2_y
        gauss_xy=wp.exp(-0.5*r2_xy)
        gauss_z_real=wp.exp(-0.5*dz_real*dz_real/sig2_z)
        gauss_z_image=wp.exp(-0.5*dz_image*dz_image/sig2_z)
        norm=Q_dt/(15.74960995*sig_x*sig_y*sig_z)
        c_puff=c_puff+norm*gauss_xy*(gauss_z_real+gauss_z_image)
    concentration[s]=c_grid+c_puff

@wp.kernel
def compute_concentration_grid_kernel(
    conc_grid: wp.array3d(dtype=float), rho_grid: wp.array3d(dtype=float), rho_scale:float,
    puff_pos: wp.array(dtype=wp.vec3), puff_active: wp.array(dtype=int),
    puff_emit_time: wp.array(dtype=float), current_time:float,
    Q_dt:float, sigma_0:float, K_d_x:float, K_d_y:float, K_d_z:float, dx:float, num_puffs:int,
):
    i,j,k=wp.tid()
    px=(float(i)+0.5)*dx; py=(float(j)+0.5)*dx; pz=(float(k)+0.5)*dx
    pos=wp.vec3(px,py,pz); total_c=rho_grid[i,j,k]*rho_scale
    for n in range(num_puffs):
        if puff_active[n]==0: continue
        age=current_time-puff_emit_time[n]
        if age<1e-6: continue
        sig2_x=sigma_0*sigma_0+2.0*K_d_x*age
        sig2_y=sigma_0*sigma_0+2.0*K_d_y*age
        sig2_z=sigma_0*sigma_0+2.0*K_d_z*age
        sig_x=wp.sqrt(sig2_x); sig_y=wp.sqrt(sig2_y); sig_z=wp.sqrt(sig2_z)
        diff=pos-puff_pos[n]
        if wp.abs(diff[0])>4.0*sig_x or wp.abs(diff[1])>4.0*sig_y: continue
        dz_real=diff[2]
        dz_image=pos[2]+puff_pos[n][2]
        if wp.abs(dz_real)>4.0*sig_z and wp.abs(dz_image)>4.0*sig_z: continue
        r2_xy=diff[0]*diff[0]/sig2_x+diff[1]*diff[1]/sig2_y
        gauss_xy=wp.exp(-0.5*r2_xy)
        gauss_z_real=wp.exp(-0.5*dz_real*dz_real/sig2_z)
        gauss_z_image=wp.exp(-0.5*dz_image*dz_image/sig2_z)
        norm=Q_dt/(15.74960995*sig_x*sig_y*sig_z)
        total_c=total_c+norm*gauss_xy*(gauss_z_real+gauss_z_image)
    conc_grid[i,j,k]=total_c


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
                 shear_exponent=0.25, seed=42):
        
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

        # OU parameters per component: [u, v, w]
        # Tighter reversion and lower noise for a stable plume
        self._kappa = np.array([ou_theta, ou_theta, 0.05])
        self._sigma = np.array([ou_sigma, ou_sigma, 0.05])

        # Wind state (u, v, w) — warm-started from stationary distribution
        self._wind_mean = np.array([self.u_mean, self.v_mean, self.w_mean])
        _stat_std = self._sigma / np.sqrt(2.0 * self._kappa)
        self._wind = self._wind_mean + _stat_std * self.rng.standard_normal(3)

        # Multi-scale turbulence: scaled down to avoid tearing the plume apart
        self._turb_kappas = [0.05, 0.15, 0.40]
        self._turb_sigmas = [0.20, 0.15, 0.10] # <-- Was 0.80, 0.50, 0.30
        self._turb_states = np.zeros((3, 3))  
        for _layer_idx in range(3):
            _layer_std = self._turb_sigmas[_layer_idx] / np.sqrt(2.0 * self._turb_kappas[_layer_idx])
            self._turb_states[_layer_idx] = _layer_std * self.rng.standard_normal(3)
            self._turb_states[_layer_idx, 2] *= 0.1  

        # Gust model 
        self.gust_interval = gust_interval
        self.gust_duration = gust_duration
        self.gust_strength_range = gust_strength_range
        self._gust_timer = 0.0
        self._gust_active = False
        self._gust_elapsed = 0.0
        self._gust_vec = np.zeros(3)  

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
        self._gust_timer += dt
        if not self._gust_active:
            if self._gust_timer > self.rng.exponential(self.gust_interval):
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

    def get_wind(self, t):
        turb = np.sum(self._turb_states, axis=0)
        eff = self._wind + turb

        if self._gust_active:
            frac = self._gust_elapsed / self.gust_duration
            env = frac / 0.15 if frac < 0.15 else (1.0 - frac) / 0.25 if frac > 0.75 else 1.0
            eff = eff + self._gust_vec * env

        return float(eff[0]), float(eff[1]), float(eff[2]), self.shear_exponent

    def get_angle_degrees(self):
        """Derive wind direction from current u,v components (backward compat)."""
        return math.degrees(math.atan2(self._wind[1], self._wind[0]))
# Backward compatibility alias
WindDirectionModel = WindVectorModel


# ── Main simulation class ────────────────────────────────────────────────────

class HybridFluidPuffSimulation:
    # Pasquill-Gifford Class D (neutral) — default atmospheric stability.
    # Change via set_stability_class() or override directly.
    K_D_X = 1.5    # along-wind turbulent diffusivity (m²/s)
    K_D_Y = 3.8    # cross-wind turbulent diffusivity (m²/s)
    K_D_Z = 1.3    # vertical turbulent diffusivity (m²/s)

    LANGEVIN_X = 1.73  # along-wind Langevin noise coefficient (m/√s) = √(2·K_D_X)
    LANGEVIN_Y = 2.76  # cross-wind Langevin noise coefficient (m/√s) = √(2·K_D_Y)
    LANGEVIN_Z = 1.61  # vertical Langevin noise coefficient (m/√s) = √(2·K_D_Z)

    # Grid scalar diffusion (turbulent eddy diffusivity, NOT molecular D_mol)
    D_ISO = 15.0       # m²/s — turbulent mixing on the Eulerian grid (reduced from 50)

    RHO_DECAY_RATE = 0.0008   # 1/s — reduced so grid density persists downstream
    RHO_SCALE      = 1e-3

    # Temperature-profile-based buoyancy (Stull 1988)
    LAPSE_RATE = PG_LAPSE_RATES["D"]  # K/m, set by stability class
    T_SURFACE  = T_SURFACE_DEFAULT    # K, surface temperature

    def __init__(self, wind_center_angle=45.0, wind_angle_range=40.0,
                 wind_base_speed=4.5, wind_seed=42, max_puffs=1024):
        self.fps=30; self.frame_dt=1.0/self.fps; self.sim_dt=self.frame_dt; self.sim_time=0.0
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
        self.next_puff_idx=0; self.emit_interval=0.05; self.last_emit_time=-999.0

        self.source_pos=np.array([NX//4*DX, NY//4*DX, 2*DX],dtype=np.float32)

        self.Q = 20.0           # kg/s — methane emission rate (overridden by env)
        self.sigma_0 = 2.0      # m — initial puff standard deviation
        self.K_d = max(self.K_D_X, self.K_D_Y, self.K_D_Z)  # isotropic proxy for viz

        self.wind_model=WindVectorModel(
            center_angle=wind_center_angle,
            angle_range=wind_angle_range,
            base_speed=wind_base_speed,
            seed=wind_seed,
        )

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
        self._turb_ou_sigma = 1.2
        _turb_stat_std = self._turb_ou_sigma / math.sqrt(2.0 * self._turb_ou_theta)
        self._turb_np = (_turb_stat_std * self._rng.standard_normal(
            (*self._turb_shape, 3))).astype(np.float32)
        self.turb_field = wp.array(self._turb_np, dtype=wp.vec3)

        # SOR pressure solver residual buffer
        self._sor_residual = wp.zeros(1, dtype=float)

    def set_stability_class(self, pg_class: str = "D"):
        """Set K_D, Langevin, lapse rate, and shear exponent from PG class (A–G)."""
        pg_class = pg_class.upper()
        if pg_class not in PASQUILL_GIFFORD_KD:
            raise ValueError(f"Unknown PG class '{pg_class}'. Use A–G.")
        kd = PASQUILL_GIFFORD_KD[pg_class]
        lg = PASQUILL_GIFFORD_LANGEVIN[pg_class]
        self.K_D_X = kd["K_D_X"]
        self.K_D_Y = kd["K_D_Y"]
        self.K_D_Z = kd["K_D_Z"]
        self.LANGEVIN_X = lg["L_X"]
        self.LANGEVIN_Y = lg["L_Y"]
        self.LANGEVIN_Z = lg["L_Z"]
        # Lapse rate for temperature-based buoyancy (NOAA READY ranges)
        self.LAPSE_RATE = PG_LAPSE_RATES[pg_class]
        # Shear exponent for power-law vertical wind profile
        self.wind_model.shear_exponent = PG_SHEAR_EXPONENTS[pg_class]
        # Legacy scalar K_d — max of anisotropic values (used by viz for puff sizing)
        self.K_d = max(self.K_D_X, self.K_D_Y, self.K_D_Z)

    def _run_pressure_sor(self):
        """Red-Black SOR pressure solve with fixed iterations (no CPU-GPU syncs)."""
        omega = 1.7
        # 25 iterations is a sweet spot for speed vs. stability in RL.
        fixed_iters = 25 
        shape = self.wind_shape

        for _ in range(fixed_iters):
            # Red sweep (parity 0)
            wp.launch(pressure_sor_rb, dim=shape,
                      inputs=[self.p0, self.div, self.obstacle_mask, omega, 0])
            # Black sweep (parity 1)
            wp.launch(pressure_sor_rb, dim=shape,
                      inputs=[self.p0, self.div, self.obstacle_mask, omega, 1])
            
    def set_obstacles(self, spheres, cuboids):
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

    def emit_puff(self):
        if self.sim_time-self.last_emit_time<self.emit_interval: return
        pos_np=self.puff_pos.numpy(); act_np=self.puff_active.numpy(); emt_np=self.puff_emit_time.numpy()
        for _ in range(2):
            idx=self.next_puff_idx%self.max_puffs
            pos_np[idx]=self.source_pos + self._rng.standard_normal(3).astype(np.float32)*1.5
            act_np[idx]=1; emt_np[idx]=self.sim_time; self.next_puff_idx+=1
        wp.copy(self.puff_pos,wp.array(pos_np,dtype=wp.vec3))
        wp.copy(self.puff_active,wp.array(act_np,dtype=int))
        wp.copy(self.puff_emit_time,wp.array(emt_np,dtype=float))
        self.last_emit_time=self.sim_time

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
                          self.obstacle_mask])
        wp.launch(apply_ground_drag, dim=shape, inputs=[self.u0,0.08,3])

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

        wp.launch(integrate_rho,dim=shape,
                  inputs=[self.u0,self.rho0,dt,
                          float(METHANE_BUOYANCY_ACCEL),self.RHO_DECAY_RATE,
                          float(self.LAPSE_RATE),float(self.T_SURFACE),float(self.dx),
                          float(self.sigma_0),float(self.K_D_Z),float(self.sim_time)])

        # Diffusion (obstacle-aware Neumann no-flux at surfaces)
        wp.launch(diffuse_rho, dim=shape,
                  inputs=[self.rho0, self.rho_tmp, self.obstacle_mask,
                          float(self.D_ISO), dt, float(self.dx)])
        self.rho0, self.rho_tmp = self.rho_tmp, self.rho0

        wp.launch(enforce_density_obstacle, dim=shape,
                  inputs=[self.rho0, self.obstacle_mask])

        src_gx=float(self.source_pos[0]/self.dx)
        src_gy=float(self.source_pos[1]/self.dx)
        src_gz=float(self.source_pos[2]/self.dx)
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
            self.u0,self.sim_dt,self.dx,float(METHANE_BUOYANCY_ACCEL),self.sim_time,
            float(self.LANGEVIN_X),float(self.LANGEVIN_Y),float(self.LANGEVIN_Z),
            self.n_spheres,self.sphere_centers,self.sphere_radii,
            self.n_cuboids,self.cuboid_centers,self.cuboid_extents,
            float(self.LAPSE_RATE),float(self.T_SURFACE),
        ])
        wp.launch(reflect_puffs_at_boundary,dim=n,inputs=[
            self.puff_pos,self.puff_active,
            float(self._domain_max[0]),float(self._domain_max[1]),float(self._domain_max[2])
        ])

    def compute_conc_grid(self):
        n=min(self.next_puff_idx,self.max_puffs); Q_dt=self.Q*self.emit_interval
        wp.launch(compute_concentration_grid_kernel,dim=self.wind_shape,inputs=[
            self.conc_grid,self.rho0,float(self.RHO_SCALE),
            self.puff_pos,self.puff_active,self.puff_emit_time,
            self.sim_time,Q_dt,self.sigma_0,self.K_D_X,self.K_D_Y,self.K_D_Z,self.dx,n
        ])

    def query_concentration(self, sensor_positions_np):
        M=sensor_positions_np.shape[0]; sensors=wp.array(sensor_positions_np,dtype=wp.vec3)
        conc=wp.zeros(M,dtype=float); n=min(self.next_puff_idx,self.max_puffs); Q_dt=self.Q*self.emit_interval
        wp.launch(query_sensor_hybrid,dim=M,inputs=[
            sensors,self.rho0,float(self.RHO_SCALE),self.dx,
            self.puff_pos,self.puff_active,self.puff_emit_time,
            self.sim_time,Q_dt,self.sigma_0,self.K_D_X,self.K_D_Y,self.K_D_Z,n,conc
        ])
        return conc.numpy()

    def step(self):
        self.emit_puff(); self.step_wind(); self.step_puffs(); self.sim_time+=self.sim_dt

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
            max_kd = max(sim.K_D_X, sim.K_D_Y, sim.K_D_Z)
            sigmas = np.sqrt(sim.sigma_0**2 + 2.0 * max_kd * ages)
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