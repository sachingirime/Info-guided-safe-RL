import numpy as np
import matplotlib.pyplot as plt
from gaussian_puff_env import GaussianPuffEnv

def test_concentration_field(env):
    """
    Verify the puff model produces a sensible 2D concentration
    slice at z = source_z.  Should show a peak near the source
    (shifted by wind advection).
    """
    env.reset(seed=0)

    # fix wind and source for a clean visual
    env.theta = np.array([25.0, 25.0, 10.0])
    env.w     = np.array([ 0.5,  0.3,  0.0])
    env.t     = 20.0   # let puffs drift a little

    # pre-fill puff history at fixed period
    env._puff_times = [
        env.t - env.PUFF_PERIOD * i for i in range(env.N_PUFFS)
    ]

    # --- 2D slice at z = source_z ---
    xs = np.linspace(0, 50, 80)
    ys = np.linspace(0, 50, 80)
    C  = np.zeros((80, 80))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y, env.theta[2]])
            C[j, i] = env.concentration(p)

    plt.figure(figsize=(7, 6))
    plt.contourf(xs, ys, C, levels=30, cmap="hot_r")
    plt.colorbar(label="Concentration [kg/m³]")
    plt.scatter(*env.theta[:2], s=200, c="cyan",
                marker="*", zorder=5, label="Source")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Gaussian Puff — z={env.theta[2]:.1f}m slice\n"
              f"wind={env.w}, t={env.t}s")
    plt.legend()
    plt.tight_layout()
    plt.savefig("puff_concentration_slice.png", dpi=150)
   # plt.show()
    # replace plt.show() with this in test_concentration_field()
    plt.savefig("puff_concentration_slice.png", dpi=150)
    plt.close()   # non-blocking, works headless too
    print("✓ Saved puff_concentration_slice.png")
    print("Concentration at source:", env.concentration(env.theta))
    print("Concentration at origin:", env.concentration(np.zeros(3)))


def test_random_episode(env, n_steps=50):
    """
    Run a short random-action episode and print step-by-step info.
    Verifies reset/step/obs dimensions and concentration values.
    """
    obs, info = env.reset(seed=42)
    print(f"\nSource location : {info['source_location']}")
    print(f"Start position  : {info['sensor_position']}")
    print(f"Obs shape       : {obs.shape}  (expect (10,))")
    print(f"Action space    : {env.action_space}")
    print("-" * 55)
    print(f"{'Step':>4}  {'dist':>8}  {'conc':>12}  {'reward':>8}")
    print("-" * 55)

    total_reward = 0.0
    for k in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        p = obs[:3]
        c = obs[9]
        dist = info["dist_to_source"]

        if k % 10 == 0 or terminated:
            print(f"{k:>4}  {dist:>8.3f}  {c:>12.6e}  {reward:>8.4f}")

        if terminated:
            print(f"\n  ✓ Converged at step {k}!")
            break
        if truncated:
            print(f"\n  ✗ Truncated at step {k}.")
            break

    print(f"\nTotal reward: {total_reward:.4f}")


def test_obs_dimensions(env):
    """Assert observation and action shapes are correct."""
    obs, _ = env.reset(seed=1)
    assert obs.shape == (10,), f"Bad obs shape: {obs.shape}"
    assert env.action_space.shape == (3,), "Bad action shape"
    action = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(action)
    assert obs2.shape == (10,), f"Bad obs shape after step: {obs2.shape}"
    assert isinstance(r, float), "Reward must be float"
    print("✓ Dimension checks passed")


if __name__ == "__main__":
    env = GaussianPuffEnv()

    print("=== 1. Dimension checks ===")
    test_obs_dimensions(env)

    print("\n=== 2. Random episode ===")
    test_random_episode(env, n_steps=50)

    print("\n=== 3. Concentration field slice ===")
    test_concentration_field(env)