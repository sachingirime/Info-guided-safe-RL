"""
gaden_client.py — Python side of bench/gaden_driver (persistent GADEN process).

    g = GadenClient(env_bin, params)      # params: dict, written as params.txt
    g.set_wind(u)                         # u: (nx, ny, nz, 3) array, our [i, j, k] layout
    g.step(n); c_ppm = g.query(points)    # points: (M, 3) m
    g.close()

Unit/parameter mapping between our puff model and GADEN filaments:
    σ₀ [m]      → filamentInitialSigma = 100·σ₀ [cm]
    K_rel [m²/s] → filamentGrowthGamma = 2·K_rel·1e4 [cm²/s]   (σ² = σ₀² + γ t)
    puff mass m = Q/n_fil [kg] → filamentPPMcenter_initial (GADEN mole bookkeeping)
    GADEN ppm   → kg/m³ via the same T, P and molar mass M (M cancels in C/Q).
"""

import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "external", "driver_build", "gaden_driver")

R_GADEN = 82.057338          # cm³·atm/(mol·K), GasTypes.hpp
M_REF = 0.016043             # kg/mol; any value works, it cancels in C/Q
T_REF, P_REF = 298.0, 1.0    # K, atm (GADEN defaults)
GAS_ETHANOL, GAS_METHANE = 0, 1


def n_air_cm3(T=T_REF, P=P_REF):
    """Moles of air per cm³ (GADEN's numMolesAllGasesIncm3)."""
    return P / (R_GADEN * T)


def ppm_center_for_mass(mass_kg, sigma0_cm, T=T_REF, P=P_REF, M=M_REF):
    """filamentPPMcenter_initial that makes one filament carry mass_kg."""
    moles = mass_kg / M
    return moles / (n_air_cm3(T, P) * (2 * np.pi) ** 1.5 * sigma0_cm ** 3) * 1e6


def ppm_to_kgm3(ppm, T=T_REF, P=P_REF, M=M_REF):
    return np.asarray(ppm, np.float64) * 1e-6 * n_air_cm3(T, P) * 1e6 * M


def write_env(path, dims, cells_ijk, cell_size, min_coord=(0.0, 0.0, 0.0)):
    """cells_ijk: uint8 array (nx, ny, nz) of GADEN CellState values."""
    nx, ny, nz = dims
    with open(path, "wb") as f:
        np.array([nx, ny, nz], np.int32).tofile(f)
        np.array([*min_coord, cell_size], np.float32).tofile(f)
        np.ascontiguousarray(np.transpose(cells_ijk, (2, 1, 0)), np.uint8).tofile(f)


def env_cells(obstacle_mask_ijk, outlets=("x+", "y-", "y+", "z+")):
    """GADEN cell states from our obstacle mask: obstacles → 1, the listed
    boundary faces (one cell thick) → 2 Outlet, everything else Free."""
    m = np.asarray(obstacle_mask_ijk)
    cells = np.where(m != 0, 1, 0).astype(np.uint8)
    sl = {"x-": (0, slice(None), slice(None)), "x+": (-1, slice(None), slice(None)),
          "y-": (slice(None), 0, slice(None)), "y+": (slice(None), -1, slice(None)),
          "z+": (slice(None), slice(None), -1)}
    for face in outlets:
        c = cells[sl[face]]
        cells[sl[face]] = np.where(c == 1, 1, 2)
    return cells


def gaden_params(source, Q, n_fil_sec, sigma0_m, K_rel, noise_std=0.0, dt=0.1,
                 gas_type=GAS_ETHANOL, seed=0, expected_iters=20000):
    """GADEN parameters equivalent to our puff release (Q kg/s, σ₀, K_rel)."""
    s0_cm = 100.0 * sigma0_m
    return {
        "source_x": source[0], "source_y": source[1], "source_z": source[2],
        "gas_type": gas_type, "deltaTime": dt, "temperature": T_REF, "pressure": P_REF,
        "filamentPPMcenter_initial": ppm_center_for_mass(Q / n_fil_sec, s0_cm),
        "filamentInitialSigma": s0_cm, "filamentGrowthGamma": 2.0 * K_rel * 1e4,
        "filamentNoise_std": noise_std, "numFilaments_sec": n_fil_sec,
        "expectedNumIterations": expected_iters, "seed": seed,
    }


class GadenClient:
    def __init__(self, env_bin, params, threads=None, workdir=None):
        self.workdir = workdir or tempfile.mkdtemp(prefix="gaden_")
        os.makedirs(self.workdir, exist_ok=True)
        self.params = dict(params)
        ppath = os.path.join(self.workdir, "params.txt")
        with open(ppath, "w") as f:
            for k, v in self.params.items():
                f.write(f"{k} {float(v):.9g}\n")
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        self.p = subprocess.Popen([DRIVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        r = self._cmd(f"INIT {env_bin} {ppath}")
        self.num_cells = int(r.split()[1])
        with open(env_bin, "rb") as f:
            self.dims = tuple(np.fromfile(f, np.int32, 3))

    def _cmd(self, line, payload=None):
        self.p.stdin.write((line + "\n").encode())
        if payload is not None:
            self.p.stdin.write(payload)
        self.p.stdin.flush()
        r = self.p.stdout.readline().decode().strip()
        if not r.startswith("OK"):
            raise RuntimeError(f"gaden_driver: {line!r} -> {r!r}")
        return r

    def set_wind(self, u_ijk):
        """u_ijk: (nx, ny, nz, 3) wind in our [i, j, k] layout (m/s)."""
        u = np.ascontiguousarray(np.transpose(np.asarray(u_ijk, np.float32), (2, 1, 0, 3)))
        assert u.size == 3 * self.num_cells
        self._cmd("WIND", u.tobytes())

    def step(self, n=1):
        r = self._cmd(f"STEP {int(n)}").split()
        return float(r[1]), int(r[2])

    def query(self, pts):
        pts = np.ascontiguousarray(np.asarray(pts, np.float32).reshape(-1, 3))
        self._cmd(f"QUERY {len(pts)}", pts.tobytes())
        return np.frombuffer(self.p.stdout.read(4 * len(pts)), np.float32).copy()

    def query_kgm3(self, pts):
        return ppm_to_kgm3(self.query(pts))

    def filaments(self):
        n = int(self._cmd("FIL").split()[1])
        return np.frombuffer(self.p.stdout.read(16 * n), np.float32).reshape(n, 4).copy()

    def close(self):
        if self.p.poll() is None:
            try:
                self._cmd("QUIT")
            except Exception:
                pass
            self.p.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
