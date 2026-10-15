"""
a5_u10.py — A5: realised time-mean resolved wind at z = 10 m vs the requested base speed.

Over the 10 m cell layer (k = 2), excluding 20 m next to the lateral faces,
averaged 60–300 s (default preset, no obstacles):
  U10_realised    = |time- and plane-mean horizontal wind vector|  (transport speed; criterion)
  mean_speed_10m  = time- and plane-mean of |u_h|  (includes rectified fluctuations)

    python validation/a5_u10.py     # → validation/a5_u10.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import warp as wp; wp.config.quiet = True
import stable_fluid_3d as sf


def realised_u10(U, pg="D", comp=True, seed=1, spin=60.0, avg=240.0, **kw):
    sim = sf.HybridFluidPuffSimulation(wind_center_angle=0.0, wind_base_speed=U, wind_seed=seed, seed=seed,
                                       stability_class=pg, drag_compensation=comp, **kw)
    sim.source_pos = np.array([30.0, 64.0, 10.0], np.float32)
    sim.set_obstacles([], [])
    k10 = int(round(10.0 / sim.dx - 0.5))
    vals, vec, usrc = [], [], []
    for i in range(int((spin + avg) / sim.frame_dt)):
        sim.step()
        if i * sim.frame_dt >= spin and i % 15 == 0:
            u = sim.u0.numpy()[5:-5, 5:-5, k10, :2]
            vals.append(np.hypot(u[..., 0], u[..., 1]).mean()); vec.append(u.reshape(-1, 2).mean(0)); usrc.append(sim.U_src)
    return float(np.linalg.norm(np.mean(vec, 0))), float(np.mean(vals)), float(np.mean(usrc))


if __name__ == "__main__":
    # 1200 s averages and 3 seeds: the domain-wide OU mean wind wanders with a
    # 50 s time scale (σ ≈ 0.075 m/s), so 240 s averages scatter by ±3% at 1.5 m/s.
    rows = []
    for comp in (False, True):
        for U in (1.5, 3.0, 8.0):
            for sd in ((1,) if not comp else (1, 2, 3)):
                u10, spd, us = realised_u10(U, "D", comp, seed=sd, avg=1200.0)
                rows.append({"drag_compensation": comp, "U_requested": U, "seed": sd, "U10_realised": u10,
                             "ratio": u10 / U, "mean_speed_10m": spd, "U_src": us})
                print(f"comp={comp!s:5} U={U:4.1f} seed {sd}: |time-mean u_h| at 10 m = {u10:.3f} (×{u10/U:.3f}), "
                      f"mean |u_h| = {spd:.3f}, U_src = {us:.3f}", flush=True)
    summ = {}
    for U in (1.5, 3.0, 8.0):
        r = [x["ratio"] for x in rows if x["drag_compensation"] and x["U_requested"] == U]
        summ[str(U)] = {"mean_ratio": float(np.mean(r)), "min": float(min(r)), "max": float(max(r))}
    ok = all(abs(v["mean_ratio"] - 1) < 0.03 for v in summ.values())
    print("A5", "PASS" if ok else "FAIL", summ)
    json.dump({"rows": rows, "summary": summ, "pass": ok,
               "criterion": "|seed-mean realised U10 / requested − 1| < 3% (drag compensation + zero-mean gusts)"},
              open(os.path.join(HERE, "a5_u10.json"), "w"), indent=1)
