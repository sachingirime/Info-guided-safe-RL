"""
realtime_viz.py — Paper-quality RL episode visualisation.

3D view (PyVista):  Puff particles (spheres sized by σ_puff), drone with full
                    trail, obstacles, wind vane.
2D panels (matplotlib): Concentration (column-integrated, puff-only) and wind,
                    as two figures with colorbars.

Paper figures (headless, offscreen):
    python realtime_viz.py --headless          # writes figures/sim_3d_render.png,
                                               # sim_2d_conc.{png,pdf}, sim_2d_wind.{png,pdf}
Capture time is env time; reset() already advanced WARMUP_S (120 s), so the
default --capture 60 is 180 s of sim time.
"""

import os
import sys
import numpy as np

HEADLESS = "--headless" in sys.argv
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")
import pyvista as pv
import matplotlib.pyplot as plt
from gaussian_puff_env import GaussianPuffEnv, SphereObstacle, CuboidObstacle


# ── Built-in exploration policies ────────────────────────────────────────────

def random_policy(obs, env):
    return env.action_space.sample()


def gradient_policy(obs, env):
    """Finite-difference concentration ascent."""
    if env._c_true < env.SIGMA_E:
        return env.action_space.sample()
    grad = np.zeros(3)
    eps = 2.0
    for i, e_i in enumerate(np.eye(3)):
        cp = env.concentration(env.p + eps * e_i)
        cm = env.concentration(env.p - eps * e_i)
        grad[i] = (cp - cm) / (2.0 * eps)
    norm = np.linalg.norm(grad)
    if norm > 1e-15:
        return (env.V_MAX * grad / norm).astype(np.float32)
    return env.action_space.sample()


def spiral_policy(obs, env, _state={"t": 0.0}):
    _state["t"] += env.DT
    t = _state["t"]
    return np.array([
        env.V_MAX * 0.7 * np.cos(0.2 * t),
        env.V_MAX * 0.7 * np.sin(0.2 * t),
        env.V_MAX * 0.1 * np.sin(0.05 * t),
    ], dtype=np.float32)


POLICIES = {"random": random_policy, "gradient": gradient_policy, "spiral": spiral_policy}


# ── Colour palette — clean, paper-friendly on white background ───────────────

COLOURS = {
    "bg":           "white",
    "domain_wire":  "#d0d0d0",
    "ground_wire":  "#e8e8e8",
    "sphere_fixed": "#7eaec1",     # steel blue
    "sphere_rand":  "#b5c9a8",     # sage green
    "cuboid_fixed": "#c9a87e",     # warm sand
    "cuboid_rand":  "#c1a0d0",     # soft lavender
    "obs_edge":     "#555555",
    "drone_body":   "#1a1a1a",
    "drone_x":      "#cc0000",
    "drone_y":      "#00aa00",
    "drone_z":      "#0044cc",
    "trail":        "#2166ac",     # solid blue
    "source":       "#22cc44",     # vivid green
    "source_edge":  "#111111",
    "wind_arrow":   "#00b8d4",     # teal
    "puff_young":   "#ffe066",     # warm yellow core
}


def viridis_rgba(age_norm):
    """Map normalised puff age [0=young, 1=old] to RGBA via viridis."""
    intensity = 1.0 - age_norm
    colors = plt.cm.viridis(intensity)
    colors[:, 3] = np.clip(intensity, 0.1, 0.8)
    return (colors * 255).astype(np.uint8)


# ── 2D panels (module level so pre-fix code can be plotted identically) ─────

def _draw_obstacles(ax, env):
    for obs in env.obstacles:
        if isinstance(obs, SphereObstacle):
            ax.add_patch(plt.Circle(obs.center[:2], obs.radius, fill=False, color="white",
                                    linewidth=1, linestyle="--", alpha=0.6))
        elif isinstance(obs, CuboidObstacle):
            c, h = obs.center, obs.half_extents
            ax.add_patch(plt.Rectangle((c[0]-h[0], c[1]-h[1]), 2*h[0], 2*h[1], fill=False,
                                       color="white", linewidth=1, linestyle="--", alpha=0.6))


def _wind_grid(env, skip=4):
    sim = env.fluid_sim
    speed, ux, uy = sim.get_wind_mean_xy()
    nx, ny = speed.shape
    xs = np.arange(0, nx, skip) * sim.dx + sim.dx * 0.5
    ys = np.arange(0, ny, skip) * sim.dx + sim.dx * 0.5
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return speed, X, Y, ux[::skip, ::skip], uy[::skip, ::skip], nx * sim.dx, ny * sim.dx


def draw_conc_panel(fig, env, trail=None, title=None):
    """Column-integrated concentration (kg/m²) from sim.get_conc_column_xy().
    With the fixed code this is puff-only (no grid gas)."""
    sim = env.fluid_sim
    conc_col = sim.get_conc_column_xy()
    speed, X, Y, U, V, xmax, ymax = _wind_grid(env)
    fig.clf()
    ax = fig.add_subplot(111)
    im = ax.imshow(conc_col.T, origin="lower", cmap="viridis", extent=[0, xmax, 0, ymax],
                   aspect="equal", vmin=0, vmax=max(conc_col.max(), 1e-6), interpolation="bilinear")
    fig.colorbar(im, ax=ax, label=r"kg/m$^2$", shrink=0.62)
    ax.quiver(X, Y, U, V, color="cyan", alpha=0.6, scale=40, width=0.003)
    ax.plot(*env.theta[:2], "g*", markersize=15, markeredgecolor="white")
    ax.plot(*env.p[:2], "wo", markersize=8, markeredgecolor="cyan", markeredgewidth=2)
    if trail is not None and len(trail) > 1:
        t = np.array(trail)[:, :2]
        ax.plot(t[:, 0], t[:, 1], "-", color=COLOURS["trail"], linewidth=1.0, alpha=0.6)
    _draw_obstacles(ax, env)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title or f"Column-integrated concentration  t={sim.sim_time:.0f} s",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()


def draw_wind_panel(fig, env, title=None):
    """Vertically averaged wind speed with quiver."""
    speed, X, Y, U, V, xmax, ymax = _wind_grid(env)
    fig.clf()
    ax = fig.add_subplot(111)
    im = ax.imshow(speed.T, origin="lower", cmap="viridis", extent=[0, xmax, 0, ymax], aspect="equal",
                   vmin=0, vmax=max(speed.max(), 1e-3), interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="m/s", shrink=0.62)
    ax.quiver(X, Y, U, V, color="white", alpha=0.7, scale=40, width=0.003)
    ax.plot(*env.theta[:2], "g*", markersize=15, markeredgecolor="white")
    ax.plot(*env.p[:2], "wo", markersize=8, markeredgecolor="cyan", markeredgewidth=2)
    _draw_obstacles(ax, env)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title or f"Mean wind speed  |u|_max={speed.max():.1f} m/s", fontsize=11, fontweight="bold")
    fig.tight_layout()


def save_2d(fig_conc, fig_wind, out_dir="figures", suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        if fig_conc is not None:
            fig_conc.savefig(f"{out_dir}/sim_2d_conc{suffix}.{ext}", dpi=300, bbox_inches="tight")
        if fig_wind is not None:
            fig_wind.savefig(f"{out_dir}/sim_2d_wind{suffix}.{ext}", dpi=300, bbox_inches="tight")


# ── Visualiser ───────────────────────────────────────────────────────────────

class RealTimeVizPV:
    def __init__(
        self,
        env: GaussianPuffEnv,
        mode: str = "episode",
        policy: str = "gradient",
        agent=None,
        n_steps: int = 1000,
        update_field: int = 5,
        seed: int = 42,
        show_wind_vectors: bool = True,
        show_slice_panel: bool = True,
        capture_time: float = 60.0,
        headless: bool = False,
    ):
        self.env = env
        self.mode = mode
        self.policy_name = policy
        self.n_steps = n_steps
        self.update_field = update_field
        self.show_wind_vectors = show_wind_vectors
        self.show_slice_panel = show_slice_panel
        self.capture_time = capture_time   # env time (s); +WARMUP_S of sim time before t=0
        self.headless = headless
        self._captured = False

        if agent is not None:
            self._policy = lambda obs: np.array(agent(obs), dtype=np.float32)
        else:
            pol = POLICIES.get(policy, random_policy)
            self._policy = lambda obs: pol(obs, env)

        self.obs, _ = env.reset(seed=seed)
        self._step_idx = 0
        self._done = False
        self.positions = [env.p.copy()]
        self.dists = [float(np.linalg.norm(env.p - env.theta))]

        # 3D plotter
        self.plotter = pv.Plotter(window_size=(1200, 900), off_screen=headless)
        self.plotter.set_background(COLOURS["bg"])
        self.plotter.add_axes(
            color="black", line_width=2,
            x_color=COLOURS["drone_x"],
            y_color=COLOURS["drone_y"],
            z_color=COLOURS["drone_z"],
        )
        self._build_static_scene()
        self.plotter.add_text(
            "", position="upper_left",
            font_size=9, color="black",
            font="courier", name="info_text",
        )

        # 2D matplotlib panels (Split into two figures)
        self._fig_conc = None
        self._fig_wind = None
        if self.show_slice_panel:
            self._fig_conc = plt.figure(figsize=(7, 5))
            self._fig_wind = plt.figure(figsize=(7, 5))
            if not headless:
                self._fig_conc.canvas.manager.set_window_title("Concentration")
                self._fig_wind.canvas.manager.set_window_title("Wind Field")
                plt.ion()
                plt.show(block=False)

    # ── Static geometry (drawn once) ───────────────────────────────────

    def _create_star_mesh(self, center, size=4.0):
        mesh = pv.Sphere(radius=size * 0.35, center=center)
        for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            offset = np.array(d) * (size * 0.45)
            cone = pv.Cone(center=center+offset, direction=d,
                           height=size*0.9, radius=size*0.35)
            mesh = mesh + cone
        return mesh

    def _build_static_scene(self):
        e = self.env
        bounds = [e.DOMAIN_MIN[0], e.DOMAIN_MAX[0],
                  e.DOMAIN_MIN[1], e.DOMAIN_MAX[1],
                  e.DOMAIN_MIN[2], e.DOMAIN_MAX[2]]
        self.plotter.add_mesh(
            pv.Box(bounds), style="wireframe",
            color=COLOURS["domain_wire"], line_width=1, opacity=0.4)

        cx = (e.DOMAIN_MIN[0] + e.DOMAIN_MAX[0]) / 2
        cy = (e.DOMAIN_MIN[1] + e.DOMAIN_MAX[1]) / 2
        ground = pv.Plane(
            center=[cx, cy, e.DOMAIN_MIN[2]], direction=(0, 0, 1),
            i_size=e.DOMAIN_MAX[0] - e.DOMAIN_MIN[0],
            j_size=e.DOMAIN_MAX[1] - e.DOMAIN_MIN[1],
            i_resolution=16, j_resolution=8,
        )
        self.plotter.add_mesh(
            ground, style="wireframe",
            color=COLOURS["ground_wire"], line_width=0.5, opacity=0.15)

        n_fixed = len(e.FIXED_OBSTACLES)
        for idx, obs in enumerate(e.obstacles):
            is_fixed = idx < n_fixed
            if isinstance(obs, SphereObstacle):
                colour = COLOURS["sphere_fixed"] if is_fixed else COLOURS["sphere_rand"]
                mesh = pv.Sphere(radius=obs.radius, center=obs.center,
                                 theta_resolution=32, phi_resolution=32)
                self.plotter.add_mesh(
                    mesh, color=colour, opacity=0.35, smooth_shading=True)
                self.plotter.add_mesh(
                    mesh, style="wireframe",
                    color=COLOURS["obs_edge"], line_width=0.4, opacity=0.15)

            elif isinstance(obs, CuboidObstacle):
                colour = COLOURS["cuboid_fixed"] if is_fixed else COLOURS["cuboid_rand"]
                c, h = obs.center, obs.half_extents
                bnds = [c[0]-h[0], c[0]+h[0],
                        c[1]-h[1], c[1]+h[1],
                        c[2]-h[2], c[2]+h[2]]
                mesh = pv.Cube(bounds=bnds)
                self.plotter.add_mesh(mesh, color=colour, opacity=0.35)
                self.plotter.add_mesh(
                    mesh, style="wireframe",
                    color=COLOURS["obs_edge"], line_width=0.6, opacity=0.2)

    def _update_wind_vane(self):
        self.plotter.remove_actor("wind_arrows")
        wx, wy, wz, _ = self.env.fluid_sim.wind_model.get_wind(self.env.t)
        w = np.array([wx, wy, wz])
        speed = np.linalg.norm(w)
        if speed < 0.1:
            return

        anchor = np.array([
            self.env.DOMAIN_MAX[0] / 2,
            self.env.DOMAIN_MAX[1] - 20,
            self.env.DOMAIN_MAX[2] - 10,
        ])
        direction = w / speed
        length = min(speed * 4, 25)
        arr = pv.Arrow(
            start=anchor - direction * (length / 2),
            direction=direction, scale=length,
            shaft_radius=0.08, tip_radius=0.25,
        )
        self.plotter.add_mesh(
            arr, color=COLOURS["wind_arrow"], opacity=0.55,
            name="wind_arrows", smooth_shading=True, show_scalar_bar=False,
        )

    def _update_scene(self):
        env = self.env
        p, v = env.p, env.v
        v_norm = np.linalg.norm(v)

        positions, ages = env.fluid_sim.get_puff_data()
        self.plotter.remove_actor("puffs")
        self.plotter.remove_actor("puffs_core")

        if len(positions) > 0:
            max_age = max(ages.max(), 1.0)
            age_norm = ages / max_age
            rgba = viridis_rgba(age_norm)

            # Puff spheres sized by σ_puff = √(σ₀² + 2·K_rel·age)
            cloud = pv.PolyData(positions)
            cloud["rgba"] = rgba
            cloud["sigma"] = env.fluid_sim.puff_sigma(ages)[:, 0]
            spheres = cloud.glyph(geom=pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8),
                                  scale="sigma", orient=False)
            self.plotter.add_mesh(
                spheres, scalars="rgba", rgba=True, smooth_shading=True,
                name="puffs", show_scalar_bar=False,
            )

            young = age_norm < 0.15
            if young.any():
                self.plotter.add_mesh(
                    pv.PolyData(positions[young]),
                    color=COLOURS["puff_young"],
                    render_points_as_spheres=True,
                    point_size=6, opacity=0.9,
                    name="puffs_core", show_scalar_bar=False,
                )

        star = self._create_star_mesh(env.theta, size=4.0)
        self.plotter.add_mesh(
            star, color=COLOURS["source"], opacity=1.0,
            smooth_shading=True, specular=0.5, name="source",
        )
        self.plotter.add_mesh(
            star, style="wireframe", color=COLOURS["source_edge"],
            line_width=0.8, opacity=0.35, name="source_edge",
        )

        self.plotter.add_mesh(
            pv.Sphere(radius=1.5, center=p),
            color=COLOURS["drone_body"], opacity=1.0,
            smooth_shading=True, specular=0.5, name="drone_core",
        )
        for axis, colour, label in [
            ((1,0,0), COLOURS["drone_x"], "drone_x"),
            ((0,1,0), COLOURS["drone_y"], "drone_y"),
            ((0,0,1), COLOURS["drone_z"], "drone_z"),
        ]:
            arr = pv.Arrow(start=p, direction=axis, scale=4.5,
                           shaft_radius=0.06, tip_radius=0.18)
            self.plotter.add_mesh(arr, color=colour, name=label, opacity=0.85)

        if v_norm > 0.1:
            arr_v = pv.Arrow(start=p, direction=v / v_norm,
                             scale=min(v_norm * 1.8, 5.0),
                             shaft_radius=0.04, tip_radius=0.12)
            self.plotter.add_mesh(arr_v, color="#888888", name="drone_v", opacity=0.5)
        else:
            self.plotter.remove_actor("drone_v")

        if len(self.positions) > 1:
            pts = np.array(self.positions)
            n = len(pts)
            lines = np.column_stack(
                (np.full(n - 1, 2), np.arange(n - 1), np.arange(1, n))
            ).flatten()
            trail = pv.PolyData(pts, lines=lines)
            self.plotter.add_mesh(
                trail, color=COLOURS["trail"],
                line_width=2.5, opacity=0.85,
                name="trail", show_scalar_bar=False,
            )

        if self.show_wind_vectors and self._step_idx % (self.update_field * 2) == 0:
            self._update_wind_vane()

        n_puffs = len(positions) if len(positions) > 0 else 0
        status = ("CONVERGED" if self._done and self.dists[-1] < env.CONVERGE_DIST
                  else "TRUNCATED" if self._done else "RUNNING")
        
        if "info_text" in self.plotter.actors:
            self.plotter.add_text(
                f"t={env.t:.1f}s  step={self._step_idx}  [{status}]\n"
                f"Drone: [{p[0]:.0f},{p[1]:.0f},{p[2]:.0f}]  v={v_norm:.1f}m/s\n"
                f"Source: [{env.theta[0]:.0f},{env.theta[1]:.0f},{env.theta[2]:.0f}]  "
                f"d={self.dists[-1]:.1f}m\n"
                f"Conc: {env._c_true:.2e}  Puffs: {n_puffs}",
                position="upper_left", font_size=9, color="black",
                font="courier", name="info_text",
            )

    # ── 2D matplotlib slice panels ──────────────────────────────────────

    def _update_slice_panel(self):
        if self._fig_conc is None or self._fig_wind is None:
            return
        draw_conc_panel(self._fig_conc, self.env, self.positions)
        draw_wind_panel(self._fig_wind, self.env)
        if not self.headless:
            self._fig_conc.canvas.draw_idle(); self._fig_wind.canvas.draw_idle()
            self._fig_conc.canvas.flush_events(); self._fig_wind.canvas.flush_events()


    # ── Main loop ──────────────────────────────────────────────────────

    def _capture(self):
        """Save figures/sim_3d_render.png and the 2D panels (PNG + PDF)."""
        os.makedirs("figures", exist_ok=True)
        print(f"\nCapturing paper figures at env t={self.env.t:.1f} s "
              f"(sim t={self.env.fluid_sim.sim_time:.1f} s)...")
        if "info_text" in self.plotter.actors:
            self.plotter.remove_actor("info_text")
        self.plotter.screenshot("figures/sim_3d_render.png", window_size=(1920, 1080),
                                transparent_background=False)
        if self.show_slice_panel:
            self._update_slice_panel()
            save_2d(self._fig_conc, self._fig_wind)
        print("Saved figures/sim_3d_render.png, sim_2d_conc.{png,pdf}, sim_2d_wind.{png,pdf}")
        self._captured = True

    def run(self):
        e = self.env
        print(f"\n  Policy : {self.policy_name}")
        print(f"  Domain : {e.DOMAIN_MAX} m")
        print(f"  Grid   : {e.fluid_sim.wind_shape}")
        print(f"  PG     : {e.stability_class}\n")

        if not self.headless:
            self.plotter.show(interactive_update=True)

        while not self._done and self._step_idx < self.n_steps:
            if not self.headless and (getattr(self.plotter, "iren", None) is None
                                      or getattr(self.plotter, "_closed", True)):
                break

            action = self._policy(self.obs)
            self.obs, _, terminated, truncated, info = self.env.step(action)
            self._step_idx += 1

            self.positions.append(self.env.p.copy())
            self.dists.append(info["dist_to_source"])

            if terminated or truncated:
                self._done = True

            if not self.headless:
                self._update_scene()
                self.plotter.update()
                if self.show_slice_panel and self._step_idx % 10 == 0:
                    self._update_slice_panel()

            if not self._captured and self.env.t >= self.capture_time:
                if self.headless:
                    self._update_scene()
                self._capture()
                if self.headless:
                    break

        if self.headless:
            self.plotter.close()
            return
        if not getattr(self.plotter, "_closed", True):
            self.plotter.show(interactive_update=False)
        if self.show_slice_panel:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL episode visualisation")
    parser.add_argument("--mode",      type=str, default="episode")
    parser.add_argument("--policy",    type=str, default="gradient")
    parser.add_argument("--steps",     type=int, default=1000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--mobile",    action="store_true", default=False)
    parser.add_argument("--stability", type=str, default="D", help="PG class A–F")
    parser.add_argument("--no-wind",   action="store_true", default=False)
    parser.add_argument("--no-slice",  action="store_true", default=False)
    parser.add_argument("--capture",   type=float, default=60.0,
                        help="Env time (s) to capture figures; reset() already ran WARMUP_S (120 s), "
                             "so 60 → 180 s of sim time")
    parser.add_argument("--headless",  action="store_true", default=False,
                        help="Offscreen render, save figures at --capture and exit")
    args = parser.parse_args()

    env = GaussianPuffEnv(source_mobile=args.mobile, stability_class=args.stability)
    viz = RealTimeVizPV(
        env=env,
        mode=args.mode,
        policy=args.policy,
        n_steps=args.steps,
        seed=args.seed,
        show_wind_vectors=not args.no_wind,
        show_slice_panel=not args.no_slice,
        capture_time=args.capture,
        headless=args.headless,
    )
    viz.run()
