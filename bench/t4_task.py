"""
t4_task.py — Tier 4: does the simulator change source-seeking conclusions?

One paired run per episode: our simulator (as shipped) and GADEN worlds driven
by our time-varying wind in lockstep.  Agents are passive sensors, so every
agent variant runs in every world on the same plume realisation.

Scenario (env-like, open field): wind direction ~ U[0, 360), U10 = 1.5 m/s,
class D; source uniform in x 15–241, y 15–113, z 2–12 m; agent flies at the
source height (2-D search) at 2 m/s, starting ≥ 30 m (horizontal) from the
source; 120 s plume warm-up, 300 s episode; success = within 2 m.

Agents
  cast-surge(θ, T_s): surge upwind while C > θ and for T_s after the last
      detection, else cast crosswind with half-periods 2, 4, 8, … s
      (Kowadlo & Russell 2008 review; moth-inspired).
  infotaxis(θ): Vergassola et al. (2007) 2-D encounter-rate model on a 4 m
      grid of candidate sources; binary hit = C > θ within each 1 s decision.

Worlds: "ours", "G-default" (GADEN example filament params), "G-tuned" (GADEN
example params with filamentNoise_std from Tier 2).

    python bench/t4_task.py --episodes 0-119 --gaden-tuned-noise 0.7
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import k0e as k0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import gaden_client as gc                # noqa: E402
import paired as pr                      # noqa: E402
import stable_fluid_3d as sf             # noqa: E402
import t2_run                            # noqa: E402

OUT = os.path.join(HERE, "data", "t4")
Q_ENV = 5.0            # kg/s (env Q_INIT)
WARMUP, T_EP, DT = 120.0, 300.0, 0.1
SPEED, R_SUCCESS = 2.0, 2.0
DOM = np.array([256.0, 128.0])
CS = [(th, ts) for th in (1e-5, 1e-4, 1e-3) for ts in (2.0, 5.0)]
INF = [1e-5, 1e-4, 1e-3]


def scenario(ep):
    rng = np.random.default_rng([ep, 2026])
    angle = float(rng.uniform(0, 360)); wseed = int(rng.integers(1, 2 ** 31))
    src = np.array([rng.uniform(15, 241), rng.uniform(15, 113), rng.uniform(2, 12)])
    while True:
        p0 = np.array([rng.uniform(4, 252), rng.uniform(4, 124)])
        if np.hypot(*(p0 - src[:2])) >= 30.0:
            break
    return angle, wseed, src, p0


class CastSurge:
    def __init__(self, th, ts):
        self.th, self.ts = th, ts
        self.t_last = -1e9; self.cast_dir = 1.0; self.cast_T = 2.0; self.cast_t = 0.0

    def act(self, t, c, wind_xy, p):
        w = wind_xy / (np.linalg.norm(wind_xy) + 1e-9)
        if c > self.th:
            self.t_last = t; self.cast_T = 2.0; self.cast_t = 0.0
        if t - self.t_last <= self.ts:
            return -w                                    # surge upwind
        self.cast_t += DT
        if self.cast_t >= self.cast_T:
            self.cast_t = 0.0; self.cast_dir *= -1; self.cast_T = min(self.cast_T * 2, 32.0)
        return self.cast_dir * np.array([-w[1], w[0]])   # crosswind cast


class Infotaxis:
    """Vergassola et al. 2007, 2-D: R(r|r0) = R0/ln(λ/a) · exp(V·(r0−r)·ŵ_up/(2D)) · K0(|r−r0|/λ)."""
    D, TAU, A, R0, H = 2.0, 300.0, 0.5, 1.0, 4.0     # m²/s, s, m (sensor size), 1/s, grid m

    def __init__(self, th):
        self.th = th
        xs = np.arange(2.0, 256.0, self.H); ys = np.arange(2.0, 128.0, self.H)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.cells = np.stack([X.ravel(), Y.ravel()], 1)
        self.logp = np.full(len(self.cells), -np.log(len(self.cells)))
        self.hit = False; self.t_dec = 0.0; self.move = np.zeros(2); self.wsum = np.zeros(2)

    def rate(self, r, wind_xy):
        V = np.linalg.norm(wind_xy) + 1e-9; up = -wind_xy / V
        lam = np.sqrt(self.D * self.TAU / (1 + V * V * self.TAU / (4 * self.D)))
        a = min(self.A, lam / 2)                     # model requires λ > a
        d = np.maximum(np.linalg.norm(self.cells - r, axis=1), a)
        along = (self.cells - r) @ up
        return self.R0 / np.log(lam / a) * np.exp(np.clip(V * along / (2 * self.D) - d / lam, -50, 50)) \
            * k0(d / lam)                            # k0e(x) = K0(x)·e^x, paired with −d/λ above

    def _update(self, r, hit, wind_xy):
        R = self.rate(r, wind_xy)
        lik = (1 - np.exp(-R)) if hit else np.exp(-R)
        lp = self.logp + np.log(np.maximum(lik, 1e-300))
        lp[np.linalg.norm(self.cells - r, axis=1) < R_SUCCESS] = -np.inf   # not found here (episode continues)
        lp -= np.logaddexp.reduce(lp[np.isfinite(lp)])
        self.logp = lp

    def _entropy(self, lp):
        p = np.exp(lp[np.isfinite(lp)]); p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def act(self, t, c, wind_xy, p):
        self.hit |= c > self.th; self.wsum += wind_xy
        self.t_dec += DT
        if self.t_dec + 1e-9 < 1.0:
            return self.move
        w = self.wsum / max(round(self.t_dec / DT), 1)
        self._update(p, self.hit, w)
        self.hit = False; self.t_dec = 0.0; self.wsum[:] = 0
        S = self._entropy(self.logp); P = np.exp(self.logp); best, bestv = None, None
        for mv in (np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])):
            r2 = np.clip(p + mv * SPEED * 1.0, 1, DOM - 1)
            near = np.linalg.norm(self.cells - r2, axis=1) < self.H / 2
            p_found = float(P[near].sum())
            R = self.rate(r2, w); ph = float((P * (1 - np.exp(-R))).sum())
            l1 = self.logp + np.log(np.maximum(1 - np.exp(-R), 1e-300)); l1 -= np.logaddexp.reduce(l1)
            l0 = self.logp - R; l0 -= np.logaddexp.reduce(l0)
            ev = p_found * (-S) + (1 - p_found) * (ph * (self._entropy(l1) - S) + (1 - ph) * (self._entropy(l0) - S))
            if bestv is None or ev < bestv:
                best, bestv = mv, ev
        self.move = best.astype(float)
        return self.move


def make_agents():
    ag = {f"cs_th{th:g}_ts{ts:g}": CastSurge(th, ts) for th, ts in CS}
    ag.update({f"info_th{th:g}": Infotaxis(th) for th in INF})
    return ag


V11 = dict(surface_layer=True, subgrid_wind=True, K_rel=0.01)   # GaussianPuffEnv.SIM_DEFAULTS (v1.1)


def run_episode(ep, tuned_noise, gaden_threads=1, version="v1.0"):
    angle, wseed, src, p0 = scenario(ep)
    v11 = version == "v1.1"
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=angle, wind_angle_range=20.0, wind_base_speed=1.5,
                                       wind_seed=wseed, max_puffs=10240 if v11 else 2560, **(V11 if v11 else {}))
    sim.Q = Q_ENV; sim.sigma_0 = 0.3 if v11 else 1.0; sim.emit_interval = 0.05 if v11 else 0.2
    rng_an = np.random.default_rng([ep, 99])
    anemo_ss = None
    sim.set_stability_class("D"); sim.set_obstacles([], []); sim.source_pos = src.astype(np.float32)
    wd = os.path.join(OUT, f"work_{ep:03d}"); os.makedirs(wd, exist_ok=True)
    env_bin = pr.gaden_env(sim, os.path.join(wd, "env.bin"))
    gads = {"G-default": gc.GadenClient(env_bin, t2_run.gaden_example_params(tuple(src), 0.01), threads=gaden_threads,
                                        workdir=os.path.join(wd, "gd")),
            "G-tuned": gc.GadenClient(env_bin, t2_run.gaden_example_params(tuple(src), tuned_noise), threads=gaden_threads,
                                      workdir=os.path.join(wd, "gt"))}
    worlds = ["ours", *gads]
    agents = {w: make_agents() for w in worlds}
    names = list(agents["ours"])
    pos = {w: {n: p0.copy() for n in names} for w in worlds}
    rec = {w: {n: {"t_success": None, "t_first_det": None, "path": 0.0} for n in names} for w in worlds}
    fps = sim.fps; ratio = int(round(DT * fps))
    n_warm = int(round(WARMUP * fps)); n_ep = int(round(T_EP * fps))
    t0 = time.time()
    for f in range(n_warm + n_ep):
        if f % ratio == 0:
            u = sim.u0.numpy()
            for g in gads.values():
                g.set_wind(u); g.step(1)
            if f >= n_warm:
                t = (f - n_warm) / fps
                for w in worlds:
                    P = np.array([[*pos[w][n], src[2]] for n in names], np.float32)
                    if w == "ours":
                        c = sim.query_concentration(P)
                    else:
                        c = gads[w].query_kgm3(P) * (Q_ENV / pr.Q_TRACER)
                    wind = pr.trilinear(u, P, sim.dx)[:, :2]
                    if v11:   # v1.1 anemometer: + coherent eddies + local small-scale turbulence (shared by worlds)
                        coh = sim.coherent_velocity(P)[:, :2]      # depends on each agent's position
                        if w == worlds[0]:
                            if anemo_ss is None:
                                anemo_ss = np.array(sim.sl_sigma(float(src[2]))) * rng_an.standard_normal((len(names), 3))
                            anemo_ss = np.array([sim.small_scale_ou(a, float(src[2]), DT, rng_an) for a in anemo_ss])
                        wind = wind + coh + anemo_ss[:, :2]
                    for k, n in enumerate(names):
                        r = rec[w][n]
                        if r["t_success"] is not None:
                            continue
                        if c[k] > agents[w][n].th and r["t_first_det"] is None:
                            r["t_first_det"] = t
                        v = agents[w][n].act(t, float(c[k]), wind[k], pos[w][n])
                        nv = np.linalg.norm(v)
                        step = (v / nv * SPEED * DT) if nv > 0 else np.zeros(2)
                        newp = np.clip(pos[w][n] + step, 1.0, DOM - 1.0)
                        r["path"] += float(np.linalg.norm(newp - pos[w][n]))
                        pos[w][n] = newp
                        if np.hypot(*(newp - src[:2])) < R_SUCCESS:
                            r["t_success"] = t + DT
        sim.step()
    for g in gads.values():
        g.close()
    for w in worlds:
        for n in names:
            rec[w][n]["final_dist"] = float(np.hypot(*(pos[w][n] - src[:2])))
    return {"ep": ep, "angle": angle, "src": src.tolist(), "p0": p0.tolist(),
            "d0": float(np.hypot(*(p0 - src[:2]))), "results": rec, "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0-119")
    ap.add_argument("--gaden-tuned-noise", type=float, required=True)
    ap.add_argument("--version", default="v1.0", choices=["v1.0", "v1.1"])
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.episodes.split("-"))
    global OUT
    if a.version == "v1.1":
        OUT = OUT + "_v11"
    os.makedirs(OUT, exist_ok=True)
    for ep in range(lo, hi + 1):
        path = os.path.join(OUT, f"ep_{ep:03d}.json")
        if os.path.exists(path):
            continue
        r = run_episode(ep, a.gaden_tuned_noise, version=a.version)
        r["gaden_tuned_noise"] = a.gaden_tuned_noise; r["version"] = a.version
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(r, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        ok = {w: sum(v["t_success"] is not None for v in r["results"][w].values()) for w in r["results"]}
        print(f"ep {ep} d0 {r['d0']:.0f} m wall {r['wall_s']:.0f}s successes/agent-variants {ok}", flush=True)


if __name__ == "__main__":
    main()
