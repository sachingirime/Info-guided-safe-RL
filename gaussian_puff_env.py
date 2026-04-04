"""
gaussian_puff_env.py — Gymnasium RL environment for drone-based gas source localisation.

Wraps a GPU-accelerated 3D Stable Fluids + Gaussian Puff simulation.
State:  s = [p_norm(3), w_norm(3), c_soft(1), c_detect(1), ζ⁻¹(1)] = 9D,
        all channels in [0, 1]; concentration channels use soft log1p/sigmoid
        (no hard gate — sub-threshold traces are visible to the network)
Action: velocity command in R³, clipped to [-V_MAX, V_MAX]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List

import warp as wp

from stable_fluid_3d import (
    HybridFluidPuffSimulation,
    NX, NY, NZ, DX as FLUID_DX,
    PASQUILL_GIFFORD_KD,
    PASQUILL_GIFFORD_LANGEVIN,
)


# ── Obstacle primitives ──────────────────────────────────────────────────────
@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float

    def h(self, p: np.ndarray) -> float:
        # Linear distance from the surface
        return float(np.linalg.norm(p - self.center) - self.radius)

    def grad_h(self, p: np.ndarray) -> np.ndarray:
        # Gradient of linear distance is the unit vector pointing outward
        diff = p - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return diff / dist

    def distance(self, p: np.ndarray) -> float:
        return self.h(p)
    
@dataclass
class CuboidObstacle:
    center: np.ndarray
    half_extents: np.ndarray

    def _signed_axis_gaps(self, p: np.ndarray) -> np.ndarray:
        return np.abs(p - self.center) - self.half_extents

    def h(self, p: np.ndarray) -> float:
        return float(np.max(self._signed_axis_gaps(p)))

    def grad_h(self, p: np.ndarray) -> np.ndarray:
        gaps = self._signed_axis_gaps(p)
        idx = int(np.argmax(gaps))
        g = np.zeros(3)
        g[idx] = np.sign(p[idx] - self.center[idx])
        return g

    def distance(self, p: np.ndarray) -> float:
        gaps = self._signed_axis_gaps(p)
        if np.all(gaps < 0):
            return float(np.max(gaps))
        return float(np.linalg.norm(np.maximum(gaps, 0.0)))


# ── Environment ──────────────────────────────────────────────────────────────

class GaussianPuffEnv(gym.Env):
    """
    3D Gaussian Puff Environment backed by GPU Stable Fluids simulation
    with full 360-degree atmospheric wind model.

    Observation (9D, all normalised to [0, 1]):
        0-2  p_norm      position / domain_max
        3-5  w_norm       (w / WIND_MAX + 1) / 2  — normalised wind vector
        6    c_norm       soft log1p concentration (no hard gate; sub-threshold
                          traces visible; ~0.10 at noise, ~0.20 at 3σ floor)
        7    c_detected   soft sigmoid detection confidence (0.5 at noise floor;
                          continuous gradient replaces binary 0/1 flag)
        8    ζ_inv        EMGR observability metric (injected by training loop)

    Phase behaviour (soft transitions via c_detected sigmoid):
        c_detected ≈ 0 → exploration.  Wind vector is the primary cue;
            wind-biased Lévy flights dominate action selection.
        c_detected ≈ 0.5 → plume edge.  Blended Lévy + SAC actions;
            faint concentration gradient provides directional hints.
        c_detected ≈ 1 → exploitation.  SAC policy dominates;
            concentration gradient + EMGR info gain drive convergence.

    Action: u_k = velocity command in R^3, clipped to [-V_MAX, V_MAX]
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Domain — matched to the fluid grid
    #   Grid: 64 x 32 x 32 cells, each 4m -> 256 x 128 x 128 meters
    # ------------------------------------------------------------------
    DOMAIN_MIN = np.array([0.0, 0.0, 0.0])
    DOMAIN_MAX = np.array([
        NX * FLUID_DX,    # 256.0
        NY * FLUID_DX,    # 128.0
        NZ * FLUID_DX,    # 128.0
    ])

    # ------------------------------------------------------------------
    # Height zones
    # ------------------------------------------------------------------
 
    SOURCE_Z_MIN = 0.0
    SOURCE_Z_MAX = 12.0 # (Was 8.0). Lets it float up a bit, but stays within easy reach of the drone.

    DRONE_Z_MIN  = 4.0
    DRONE_Z_MAX  = 60.0

    SPAWN_Z_MIN  = 4.0
    SPAWN_Z_MAX  = 12.0

    # ------------------------------------------------------------------
    # Source XY spawn window
    # ------------------------------------------------------------------
    #SOURCE_XY_MIN = np.array([88.0, 44.0])    # centered: domain_center ± 40m (x), ± 20m (y)
    #SOURCE_XY_MAX = np.array([168.0, 84.0])

    # Expanded slightly, but keeping a safe 15m padding from the walls so 
    # the source doesn't hide in the absolute corners where wind gets weird.
    SOURCE_XY_MIN = np.array([15.0, 15.0])    
    SOURCE_XY_MAX = np.array([241.0, 113.0])

    # ------------------------------------------------------------------
    # Puff parameters (methane CH₄, Q in kg/s)
    # ------------------------------------------------------------------
    Q_INIT      = 5.0       # kg/s — methane mass emission rate
    Q_GROWTH    = 0.0       # kg/s² — emission rate increase over time (0 = constant)
    PUFF_PERIOD = 0.2       # s — interval between puff emissions
    N_PUFFS     = 1024      # max Lagrangian puff particles

    SIGMA_0 = 2.0           # m — initial puff standard deviation

    # ------------------------------------------------------------------
    # Mobile source — exact OU
    # ------------------------------------------------------------------

    SRC_THETA = 0.03  # (Was 0.05). A gentle rubber band. It can wander, but won't sprint away.
    SRC_SIGMA = 0.4   # (Was 0.3). Slightly faster than your baseline, but smooth. Avoid 1.0!

    # ------------------------------------------------------------------
    # Sensor model (methane TDLAS)
    # ------------------------------------------------------------------
    SIGMA_E    = 1e-4       # kg/m³
    BIAS       = 1e-4       # kg/m³
    SENSOR_TAU = 1.0        # s — sensor response time constant

    # ------------------------------------------------------------------
    # Concentration normalization (signal-aware)
    #   Subtracts sensor bias, gates on 3σ noise floor, then log-maps
    #   the signal-above-noise to [0, 1].  Below the noise floor → 0.
    #   c_detected flag derived from c_norm > 0.
    # ------------------------------------------------------------------
    LOG_C_EPS        = 1e-10          # guard against log(0)
    CONC_NOISE_FLOOR = 3e-4           # 3 × SIGMA_E — detection threshold (kg/m³)
    LOG_C_SIG_MIN    = np.log(3e-4)   # = -8.11  (detection edge)
    LOG_C_SIG_MAX    = np.log(5e-3)   # = -2.30  (near-source peak)

    # Soft log1p concentration normalisation (replaces hard-gated log mapping)
    #   c_soft = log(1 + c_signal/CONC_LOG_SCALE) / log(1 + C_SIG_CEIL/CONC_LOG_SCALE)
    #   Provides gradient at ALL levels, including sub-threshold traces.
    #   With CONC_LOG_SCALE = 1e-4 (≈ sensor σ_e):
    #     noise-only  (σ_e ≈ 1e-4)  → c_soft ≈ 0.10
    #     noise floor (3σ  = 3e-4)   → c_soft ≈ 0.20
    #     moderate    (1e-2)          → c_soft ≈ 0.67
    #     near-source (1e-1)          → c_soft = 1.00
    CONC_LOG_SCALE     = 1e-4         # log1p sensitivity scale (≈ sensor σ_e)
    CONC_SIG_CEIL      = 5e-3         # kg/m³ — normalisation ceiling
    DETECT_TEMPERATURE = 1e-4         # kg/m³ — sigmoid width for soft detection

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    DT    = 0.1
    T_MAX = 2000            # steps — 200s at DT=0.1
    V_MAX = 10.0            # m/s — 1m per step, covers diagonal in ~313 steps
    D_MAX = 0.05

    # ------------------------------------------------------------------
    # Wind model defaults (passed to fluid simulation)
    # ------------------------------------------------------------------
    WIND_ANGLE_RANGE = 20.0     # degrees — meander band width
    WIND_BASE_SPEED  = 1.5      # m/s — mean horizontal wind speed

    # ------------------------------------------------------------------
    # Wind normalization — symmetric clamp then shift to [0, 1]
    #   w_norm_i = clip(w_i / WIND_MAX, -1, 1) * 0.5 + 0.5
    # ------------------------------------------------------------------
    WIND_MAX = 5.0  # m/s — symmetric clamp for each component

    # ------------------------------------------------------------------
    # Reward / convergence
    # ------------------------------------------------------------------
    LAMBDA_CTRL   = 0.01
    CONVERGE_DIST = 0.0

    # ------------------------------------------------------------------
    # Fixed obstacles
    # ------------------------------------------------------------------
    FIXED_OBSTACLES = [
        SphereObstacle(
            center=np.array([46.0, 92.0, 21.3]),
            radius=12.8,
        ),
        SphereObstacle(
            center=np.array([158.7, 35.8, 19.2]),
            radius=11.5,
        ),
        CuboidObstacle(
            center=np.array([46.0, 32.0, 34.1]),
            half_extents=np.array([12.8, 5.1, 34.1]),
        ),
        CuboidObstacle(
            center=np.array([158.7, 96.0, 25.6]),
            half_extents=np.array([10.2, 6.4, 25.6]),
        ),
    ]

    N_RAND_SPHERES  = 2
    N_RAND_CUBOIDS  = 1

    RAND_SPHERE_R_MIN = 5.0
    RAND_SPHERE_R_MAX = 12.0

    RAND_CUBOID_HE_MIN = np.array([5.0, 5.0, 10.0])
    RAND_CUBOID_HE_MAX = np.array([13.0, 13.0, 30.0])

    OBSTACLE_SPAWN_MARGIN = 8.0

    # ------------------------------------------------------------------

    def __init__(self, seed: int = None, source_mobile: bool = False,
                 stability_class: str = "D", info_map_size: int = 0,
                 source_xy_min=None, source_xy_max=None, source_z_max=None):
        super().__init__()

        self.source_mobile = source_mobile
        self.stability_class = stability_class
        self._init_seed = seed  # stored for reproducible fluid sim seeding
        self._info_map_size = info_map_size

        if source_xy_min is not None:
            self.SOURCE_XY_MIN = np.asarray(source_xy_min, dtype=np.float64)
        if source_xy_max is not None:
            self.SOURCE_XY_MAX = np.asarray(source_xy_max, dtype=np.float64)
        if source_z_max is not None:
            self.SOURCE_Z_MAX = float(source_z_max)

        self.action_space = spaces.Box(
            low=-self.V_MAX, high=self.V_MAX,
            shape=(3,), dtype=np.float32,
        )

        # Observation: 10D base + info_map_size EMGR information map channels
        obs_dim = 10 + info_map_size
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self._theta_hat : np.ndarray = None # To hold the MLE estimate
        self._info_map = np.zeros(info_map_size, dtype=np.float32)

        self.p          : np.ndarray = None
        self.v          : np.ndarray = None
        self.w          : np.ndarray = None
        self.theta      : np.ndarray = None
        self.t          : float      = 0.0
        self.step_count : int        = 0
        self.obstacles  : List       = []

        # EMGR observability index (set externally by training loop)
        self._zeta_inv_norm : float = 0.0

        self._c_true    : float = 0.0
        self._y_sensor  : float = 0.0   # cached per step
        self._y_lagged  : float = 0.0   # first-order lag state
        self._sensor_alpha = np.exp(-self.DT / self.SENSOR_TAU)

        self._rng = np.random.default_rng(seed)

        wind_seed = int(self._rng.integers(0, 2**31)) if seed is not None else 42
        random_wind_angle = float(self._rng.uniform(0.0, 360.0))

        # GPU fluid simulation (created once, reset in-place per episode)
        self.fluid_sim = HybridFluidPuffSimulation(
            wind_center_angle=random_wind_angle,
            wind_angle_range=self.WIND_ANGLE_RANGE,
            wind_base_speed=self.WIND_BASE_SPEED,
            max_puffs=self.N_PUFFS,
            wind_seed=wind_seed,
        )

        self._configure_fluid_sim()
        self._precompute_ou_coeffs()

    # ------------------------------------------------------------------

    def close(self):
        """Release GPU resources."""
        wp.synchronize()
        super().close()

    def _configure_fluid_sim(self):
        """Push env-level parameters to the fluid simulation."""
        self.fluid_sim.Q             = self.Q_INIT
        self.fluid_sim.sigma_0       = self.SIGMA_0
        self.fluid_sim.emit_interval = self.PUFF_PERIOD
        self.fluid_sim.max_puffs     = self.N_PUFFS
        self.fluid_sim.set_stability_class(self.stability_class)

    # ------------------------------------------------------------------

    def _reset_fluid_sim_inplace(self, wind_angle: float, wind_seed: int):
        """Reset simulation state in-place (reuses GPU allocations + kernels)."""
        sim = self.fluid_sim

        # Zero all field arrays
        sim.u0.zero_()
        sim.u1.zero_()
        sim.p0.zero_()
        sim.div.zero_()
        sim.rho0.zero_()
        sim.rho1.zero_()
        sim.rho_tmp.zero_()
        sim.conc_grid.zero_()

        # Zero puff pool
        sim.puff_pos     = wp.zeros(sim.max_puffs, dtype=wp.vec3)
        sim.puff_active  = wp.zeros(sim.max_puffs, dtype=int)
        sim.puff_emit_time = wp.zeros(sim.max_puffs, dtype=float)
        sim.next_puff_idx   = 0
        sim.last_emit_time  = -999.0

        # Reset time
        sim.sim_time = 0.0

        # Reset spatial turbulence field — warm-start from OU stationary dist
        import math as _m
        _turb_stat_std = sim._turb_ou_sigma / _m.sqrt(2.0 * sim._turb_ou_theta)
        sim._turb_np = (_turb_stat_std * sim._rng.standard_normal(
            (*sim._turb_shape, 3))).astype(np.float32)
        wp.copy(sim.turb_field, wp.array(sim._turb_np, dtype=wp.vec3))
        sim._rng = np.random.default_rng(wind_seed)

        # Re-initialise wind model with new angle and seed
        from stable_fluid_3d import WindVectorModel
        sim.wind_model = WindVectorModel(
            center_angle=wind_angle,
            angle_range=self.WIND_ANGLE_RANGE,
            base_speed=self.WIND_BASE_SPEED,
            seed=wind_seed,
        )

    # ------------------------------------------------------------------

    def _precompute_ou_coeffs(self):
        dt = self.DT
        self._src_decay = np.exp(-self.SRC_THETA * dt)
        self._src_noise_std = self.SRC_SIGMA * np.sqrt(
            (1.0 - np.exp(-2.0 * self.SRC_THETA * dt))
            / (2.0 * self.SRC_THETA)
        )

    # ── Wind field queries (trilinear interpolation on u0 grid) ───────────

    def _sample_wind_at(self, world_pos: np.ndarray) -> np.ndarray:
        u_np = self.fluid_sim.u0.numpy()
        dx   = self.fluid_sim.dx

        gx = world_pos[0] / dx
        gy = world_pos[1] / dx
        gz = world_pos[2] / dx

        nx, ny, nz = u_np.shape[:3]

        lx = int(np.floor(gx))
        ly = int(np.floor(gy))
        lz = int(np.floor(gz))
        tx = gx - lx; ty = gy - ly; tz = gz - lz

        lx = np.clip(lx, 0, nx - 2)
        ly = np.clip(ly, 0, ny - 2)
        lz = np.clip(lz, 0, nz - 2)

        c000 = u_np[lx,   ly,   lz]
        c100 = u_np[lx+1, ly,   lz]
        c010 = u_np[lx,   ly+1, lz]
        c110 = u_np[lx+1, ly+1, lz]
        c001 = u_np[lx,   ly,   lz+1]
        c101 = u_np[lx+1, ly,   lz+1]
        c011 = u_np[lx,   ly+1, lz+1]
        c111 = u_np[lx+1, ly+1, lz+1]

        c00 = c000*(1-tx) + c100*tx
        c01 = c001*(1-tx) + c101*tx
        c10 = c010*(1-tx) + c110*tx
        c11 = c011*(1-tx) + c111*tx
        c0  = c00*(1-ty)  + c10*ty
        c1  = c01*(1-ty)  + c11*ty
        return (c0*(1-tz) + c1*tz).astype(np.float64)

    # ── Obstacle generation ───────────────────────────────────────────────

    def _sample_obstacles(self) -> List:
        obs_list = list(self.FIXED_OBSTACLES)
        placed_centers = [o.center.copy() for o in obs_list]

        rx_min = max(self.SOURCE_XY_MIN[0] - 20.0, self.DOMAIN_MIN[0] + 20.0)
        rx_max = min(self.SOURCE_XY_MAX[0] + 20.0, self.DOMAIN_MAX[0] - 20.0)
        ry_min = max(self.SOURCE_XY_MIN[1] - 20.0, self.DOMAIN_MIN[1] + 20.0)
        ry_max = min(self.SOURCE_XY_MAX[1] + 20.0, self.DOMAIN_MAX[1] - 20.0)

        def _clear(cand, r):
            for c in placed_centers:
                if np.linalg.norm(cand - c) < (r + self.OBSTACLE_SPAWN_MARGIN + 12.0):
                    return False
            if np.linalg.norm(cand[:2] - self.theta[:2]) < 25.0:
                return False
            return True

        for _ in range(self.N_RAND_SPHERES):
            for _attempt in range(300):
                r  = self._rng.uniform(self.RAND_SPHERE_R_MIN, self.RAND_SPHERE_R_MAX)
                cx = self._rng.uniform(rx_min, rx_max)
                cy = self._rng.uniform(ry_min, ry_max)
                cand = np.array([cx, cy, r])
                if _clear(cand, r):
                    obs_list.append(SphereObstacle(center=cand, radius=r))
                    placed_centers.append(cand)
                    break

        for _ in range(self.N_RAND_CUBOIDS):
            for _attempt in range(300):
                he = self._rng.uniform(self.RAND_CUBOID_HE_MIN, self.RAND_CUBOID_HE_MAX)
                cx = self._rng.uniform(rx_min + he[0], rx_max - he[0])
                cy = self._rng.uniform(ry_min + he[1], ry_max - he[1])
                cand = np.array([cx, cy, he[2]])
                if _clear(cand, np.max(he)):
                    obs_list.append(CuboidObstacle(center=cand, half_extents=he))
                    placed_centers.append(cand)
                    break

        return obs_list

    # ── Collision helpers ─────────────────────────────────────────────────

    def _in_collision(self, p: np.ndarray) -> bool:
        return any(obs.distance(p) < -0.25 for obs in self.obstacles)

    def nearest_obstacle_distance(self, p: np.ndarray) -> float:
        if not self.obstacles:
            return float("inf")
        return min(obs.distance(p) for obs in self.obstacles)

    def get_cbf_data(self, p: np.ndarray):
        return [(obs.h(p), obs.grad_h(p)) for obs in self.obstacles]

    # ── Gymnasium interface ───────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Derive wind seed from env rng for reproducibility
        wind_seed = int(self._rng.integers(0, 2**31))
        random_wind_angle = float(self._rng.uniform(0.0, 360.0))

        # Reset fluid sim in-place (reuses GPU allocations)
        self._reset_fluid_sim_inplace(random_wind_angle, wind_seed)
        self._configure_fluid_sim()

        # Source position
        self.theta = np.array([
            self._rng.uniform(self.SOURCE_XY_MIN[0], self.SOURCE_XY_MAX[0]),
            self._rng.uniform(self.SOURCE_XY_MIN[1], self.SOURCE_XY_MAX[1]),
            self._rng.uniform(self.SOURCE_Z_MIN, self.SOURCE_Z_MAX),
        ])
        self._src_mean = self.theta.copy()

        # Set source position before warmup so puffs emit from the right place
        self.fluid_sim.source_pos = self.theta.astype(np.float32)

        # Warm up
        for _ in range(10):
            self.fluid_sim.step()

        # Build obstacles
        self.obstacles = self._sample_obstacles()
        spheres = [o for o in self.obstacles if isinstance(o, SphereObstacle)]
        cuboids = [o for o in self.obstacles if isinstance(o, CuboidObstacle)]
        self.fluid_sim.set_obstacles(spheres, cuboids)

        # Drone spawn
        for _ in range(2000):
            xy = self._rng.uniform(
                self.DOMAIN_MIN[:2] + 5.0,
                self.DOMAIN_MAX[:2] - 5.0,
            )
            z  = self._rng.uniform(self.SPAWN_Z_MIN, self.SPAWN_Z_MAX)
            p  = np.array([xy[0], xy[1], z])

            src_xy_dist = np.linalg.norm(p[:2] - self.theta[:2])
            in_cluster  = (
                self.SOURCE_XY_MIN[0] - 12 <= p[0] <= self.SOURCE_XY_MAX[0] + 12
                and self.SOURCE_XY_MIN[1] - 12 <= p[1] <= self.SOURCE_XY_MAX[1] + 12
            )
            if (src_xy_dist > 30.0
                    and not in_cluster
                    and not self._in_collision(p)
                    and self.nearest_obstacle_distance(p) > 5.0):
                self.p = p
                break
        else:
            # Fallback position (verified >60m from all fixed obstacles)
            self.p = np.array([230.0, 115.0, self.SPAWN_Z_MIN + 5.0])

        self.v = np.zeros(3, dtype=np.float64)
        self.w = self._sample_wind_at(self.p)
        self.t = 0.0
        self.step_count = 0
        self._zeta_inv_norm = 0.0    # reset EMGR metric each episode
        self._info_map[:] = 0.0      # reset information map

        self._c_true = self.concentration(self.p)
        # Initialise sensor lag to equilibrium
        self._y_lagged = self._c_true + self.BIAS
        self._y_sensor = self._get_sensor_reading()

        return self._get_obs().astype(np.float32), self._get_info()

    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        u = np.clip(np.array(action, dtype=np.float64), -self.V_MAX, self.V_MAX)

        # Process noise
        d  = self._rng.standard_normal(3)
        dn = np.linalg.norm(d)
        d  = (d / dn * self._rng.uniform(0.0, self.D_MAX)
              if dn > 1e-12 else np.zeros(3))

        new_p = np.clip(self.p + self.DT * (u + d), self.DOMAIN_MIN, self.DOMAIN_MAX)
        new_p[2] = np.clip(new_p[2], self.DRONE_Z_MIN, self.DRONE_Z_MAX)
        self.p = new_p
        self.v = u.copy()

        # Update emission rate
        self.fluid_sim.Q = self.Q_INIT + (self.Q_GROWTH * self.t)

        # Mobile source
        if self.source_mobile:
            self._step_source()
            self.fluid_sim.source_pos = self.theta.astype(np.float32)

        # Advance fluid sim
        n_substeps = max(1, int(round(self.DT / self.fluid_sim.frame_dt)))
        for _ in range(n_substeps):
            self.fluid_sim.step()

        self.t          += self.DT
        self.step_count += 1

        self.w       = self._sample_wind_at(self.p)
        self._c_true = self.concentration(self.p)
        self._y_sensor = self._get_sensor_reading()

        # Placeholder reward (replace with EMGR-based reward later)
        reward = float(self._y_sensor) - self.LAMBDA_CTRL * float(np.dot(u, u))

        dist_to_source = float(np.linalg.norm(self.p - self.theta))
        in_collision   = self._in_collision(self.p)
        terminated     = dist_to_source < self.CONVERGE_DIST
        truncated      = self.step_count >= self.T_MAX or in_collision

        info = self._get_info()
        info["dist_to_source"] = dist_to_source
        info["c_true"]         = self._c_true
        info["y_noisy"]        = self._y_sensor
        info["c_norm"]         = self._normalize_concentration(self._y_sensor)
        info["c_detected"]     = self._soft_detection(self._y_sensor)
        info["in_collision"]   = in_collision
        info["min_obs_dist"]   = self.nearest_obstacle_distance(self.p)

        return self._get_obs().astype(np.float32), reward, terminated, truncated, info

    # ── Source mobility (OU process) ─────────────────────────────────────

    def _step_source(self):
        noise     = self._rng.standard_normal(3)
        new_theta = (self._src_mean
                     + (self.theta - self._src_mean) * self._src_decay
                     + self._src_noise_std * noise)
        new_theta[0] = np.clip(new_theta[0], self.SOURCE_XY_MIN[0], self.SOURCE_XY_MAX[0])
        new_theta[1] = np.clip(new_theta[1], self.SOURCE_XY_MIN[1], self.SOURCE_XY_MAX[1])
        new_theta[2] = np.clip(new_theta[2], self.SOURCE_Z_MIN, self.SOURCE_Z_MAX)
        self.theta   = new_theta

    # ── Concentration query (delegates to GPU) ───────────────────────────

    def concentration(self, p: np.ndarray) -> float:
        sensor_pos = p.astype(np.float32).reshape(1, 3)
        c = self.fluid_sim.query_concentration(sensor_pos)
        return float(c[0])

    def concentration_at(self, p: np.ndarray, theta: np.ndarray) -> float:
        """Stub: currently ignores theta. Will be needed for EMGR sensitivity."""
        return self.concentration(p)

    # ── Sensor model (first-order lag + noise) ──────────────────────────

    def _get_sensor_reading(self) -> float:
        """TDLAS sensor: additive noise + first-order exponential lag."""
        noise = self._rng.normal(0.0, self.SIGMA_E)
        y_raw = float(np.maximum(self._c_true + self.BIAS + noise, 0.0))
        # First-order lag: exponential moving average with time constant SENSOR_TAU
        alpha = self._sensor_alpha
        self._y_lagged = alpha * self._y_lagged + (1.0 - alpha) * y_raw
        return self._y_lagged

    # ── Observation / info ────────────────────────────────────────────────

    # ── Normalisation helpers ──────────────────────────────────────────────
    def _normalize_concentration(self, c_raw: float) -> float:
        """
        Soft log1p concentration normalisation.
        c_soft = log1p(c_signal / scale) / log1p(ceil / scale)
        Provides a smooth gradient at ALL levels, ensuring sub-threshold traces
        are visible to the network without flatlining to zero.
        """
        # Strip bias, but allow values below the noise floor to survive
        c_signal = max(c_raw - self.BIAS, 0.0) 
        
        # log1p safely handles zero (log(1+0) = 0) and naturally squashes massive spikes
        numerator = np.log1p(c_signal / self.CONC_LOG_SCALE)
        denominator = np.log1p(self.CONC_SIG_CEIL / self.CONC_LOG_SCALE)
        
        c_norm = numerator / denominator
        
        return float(np.clip(c_norm, 0.0, 1.0))

    def _soft_detection(self, c_raw: float) -> float:
        """
        Smooth sigmoid transition centered around the noise floor.
        Replaces binary 0/1 detection flags to prevent policy shock.
        """
        c_signal = max(c_raw - self.BIAS, 0.0)
        
        # Sigmoid centered at CONC_NOISE_FLOOR.
        # Temperature controls how steep the transition from 'exploring' to 'tracking' is.
        x = (c_signal - self.CONC_NOISE_FLOOR) / self.DETECT_TEMPERATURE
        
        # Prevent overflow warnings in exp
        x = np.clip(x, -20.0, 20.0) 
        
        return float(1.0 / (1.0 + np.exp(-x)))
    
    def _normalize_wind(self, w: np.ndarray) -> np.ndarray:
        """Normalise 3D wind vector to [0, 1]^3.

        Each component: w_norm_i = clip(w_i / WIND_MAX, -1, 1) * 0.5 + 0.5

        The network can derive speed (‖w‖) and direction (atan2) if needed.
        Values near 0.5 = calm; values near 0 or 1 = strong.
        """
        return np.clip(w / self.WIND_MAX, -1.0, 1.0) * 0.5 + 0.5

    def set_zeta_inv(self, zeta_inv_norm: float):
        """Inject normalised EMGR observability index from training loop.

        Parameters
        ----------
        zeta_inv_norm : float in [0, 1]
            Normalised ζ⁻¹(W_k); 0 = fully unobservable, 1 = well-conditioned.
        """
        self._zeta_inv_norm = float(np.clip(zeta_inv_norm, 0.0, 1.0))

    def set_theta_hat(self, theta_hat: np.ndarray):
        """Inject MLE estimate from training loop to calculate distance."""
        self._theta_hat = theta_hat.copy()

    def set_info_map(self, values: np.ndarray):
        """Inject normalised EMGR information map from training loop.

        Parameters
        ----------
        values : ndarray of shape (info_map_size,), each in [0, 1]
            Observability at each grid point: 1 = well-observed, 0 = uncertain.
        """
        self._info_map = np.clip(values, 0.0, 1.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector: 10D base + info map channels."""
        # Position → [0, 1]
        p_norm = self.p / self.DOMAIN_MAX

        # Wind → [0, 1]^3
        w_norm = self._normalize_wind(self.w)

        # Concentration → soft log1p [0, 1] + soft sigmoid detection [0, 1]
        c_norm = self._normalize_concentration(self._y_sensor)
        c_detected = self._soft_detection(self._y_sensor)
        
        # MLE Distance → [0, 1]
        if self._theta_hat is None:
            dist_mle_norm = 1.0  # Fallback before MLE is initialized
        else:
            dist_mle = np.linalg.norm(self.p - self._theta_hat)
            max_dist = np.linalg.norm(self.DOMAIN_MAX)
            dist_mle_norm = float(np.clip(dist_mle / max_dist, 0.0, 1.0))

        base = np.array([
            p_norm[0], p_norm[1], p_norm[2],
            w_norm[0], w_norm[1], w_norm[2],
            c_norm, c_detected,
            self._zeta_inv_norm,
            dist_mle_norm,
        ], dtype=np.float32)
        return np.concatenate([base, self._info_map])
    
    def _get_info(self) -> dict:
        positions, ages = self.fluid_sim.get_puff_data()
        return {
            "source_location":    self.theta.copy(),
            "source_mobile":      self.source_mobile,
            "sensor_position":    self.p.copy(),
            "velocity":           self.v.copy(),
            "wind_local":         self.w.copy(),
            "wind_direction_deg": self.fluid_sim.wind_model.get_angle_degrees(),
            "time":               self.t,
            "step_count":         self.step_count,
            "n_active_puffs":     len(positions),
            "c_true":             self._c_true,
            "n_obstacles":        len(self.obstacles),
            "min_obs_dist":       self.nearest_obstacle_distance(self.p),
            "fluid_sim_time":     self.fluid_sim.sim_time,
            "rho_max":            float(self.fluid_sim.rho0.numpy().max()),
        }

    @property
    def wind_field(self):
        """Expose the full 3D wind field as numpy for visualization."""
        return self.fluid_sim.u0.numpy()
