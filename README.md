# Warp Gas Plume Gym Environment

A GPU-accelerated [Gymnasium](https://gymnasium.farama.org/) environment for **drone-based gas source localization** in 3D turbulent wind fields.

Built on [NVIDIA Warp](https://developer.nvidia.com/warp), the environment couples an Eulerian Stable Fluids wind solver with a Lagrangian Gaussian Puff dispersion model to produce physically realistic methane plume dynamics at interactive rates.

## Features

- **GPU-accelerated 3D fluid simulation** via NVIDIA Warp (64x32x32 grid, 256x128x128 m domain)
- **Lagrangian Gaussian Puff model** with up to 1024 puffs, RK2 advection, and Langevin noise
- **Pasquill-Gifford stability classes** (A-F) for turbulent diffusivity and atmospheric conditions
- **Realistic methane sensor model** with noise floor, detection threshold, and soft sigmoid gating
- **Mobile gas source** with Ornstein-Uhlenbeck drift
- **Obstacle support** (sphere and cuboid primitives with SDF-based collision)
- **Standard Gymnasium API** (`reset`, `step`, `observation_space`, `action_space`)

See a simulation at: https://youtu.be/etPAydxxQGs

## Observation Space (10D)

| Index | Name | Description |
|-------|------|-------------|
| 0-2 | `p_norm` | Drone position, normalized to [0, 1] |
| 3-5 | `w_norm` | Wind vector at drone, normalized to [0, 1] |
| 6 | `c_norm` | Soft log1p concentration |
| 7 | `c_detected` | Sigmoid detection confidence |
| 8 | `zeta_inv` | Observability metric (injected externally) |
| 9 | `c_raw_norm` | Raw concentration (clipped & normalized) |

## Action Space

Continuous velocity command in R^3, clipped to [-V_MAX, V_MAX] (default 5 m/s).

## Installation

```bash
pip install numpy gymnasium matplotlib warp-lang
```

Requires a CUDA-capable GPU for the Warp fluid solver.

## Quick Start

```python
from gaussian_puff_env import GaussianPuffEnv

env = GaussianPuffEnv()
obs, info = env.reset(seed=42)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Testing

```bash
python test_env.py
```

This runs dimension checks, a random-action episode, and generates a concentration field visualization.

## Environment Details

### Wind Field
The 3D wind field is computed using Jos Stam's Stable Fluids method on a 64x32x32 grid with obstacle-aware velocity/pressure/density using SDF-based Neumann boundary conditions. Full 360-degree atmospheric wind with sinusoidal direction variation.

### Gas Dispersion
Methane (CH4) dispersion uses a hybrid approach:
- **Eulerian**: Stable Fluids for the background wind field
- **Lagrangian**: Gaussian puffs emitted at regular intervals, advected by the wind field with RK2 integration and Pasquill-Gifford turbulent diffusion
- Ground reflection via method of images at z=0
- Temperature-profile-based buoyancy (Stull 1988)

### Sensor Model
TDLAS-style methane sensor with:
- Noise floor: 1e-6 kg/m^3
- Detection threshold: 5e-6 kg/m^3
- Gaussian measurement noise (sigma = 5e-7 kg/m^3)

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
