"""
fit_tl.py — Offline fit of the Lagrangian time scales T_L (per class and axis).

Model of the mean-plume spread at downwind distance x (travel time t = x/U):

    σ²(x) = σ₀² + 2·K_rel·t + 2·σ_v²·T_L²·(t/T_L − 1 + e^{−t/T_L})     (Taylor 1921)

with σ_v = a·U from the Briggs (1973) open-country near-field slopes
(a_y for σ_y, a_z for σ_z).  T_L is fitted (least squares on log σ over
x = 25…200 m, bounds [5, 2000] s) at the reference wind U_FIT = 3 m/s.  The
sim scales T_L ∝ 1/U (T_L = T_L,fit·U_FIT/U), which makes the Taylor term a
function of x only, like the Briggs curves (surface-layer scaling T_L ∝ z/u*).

    python validation/fit_tl.py [--sigma0 2.0] [--k-rel 0.2]

Writes validation/tl_fit.json and prints the table for stable_fluid_3d.PG_LSM.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

U_FIT = 3.0
X_FIT = np.linspace(25.0, 200.0, 36)
TL_BOUNDS = (5.0, 2000.0)

# Briggs (1973) open-country σ_y(x), σ_z(x) in m (x in m)
BRIGGS_Y = {
    "A": lambda x: 0.22 * x / np.sqrt(1 + 1e-4 * x),
    "B": lambda x: 0.16 * x / np.sqrt(1 + 1e-4 * x),
    "C": lambda x: 0.11 * x / np.sqrt(1 + 1e-4 * x),
    "D": lambda x: 0.08 * x / np.sqrt(1 + 1e-4 * x),
    "E": lambda x: 0.06 * x / np.sqrt(1 + 1e-4 * x),
    "F": lambda x: 0.04 * x / np.sqrt(1 + 1e-4 * x),
}
BRIGGS_Z = {
    "A": lambda x: 0.20 * x,
    "B": lambda x: 0.12 * x,
    "C": lambda x: 0.08 * x / np.sqrt(1 + 2e-4 * x),
    "D": lambda x: 0.06 * x / np.sqrt(1 + 1.5e-3 * x),
    "E": lambda x: 0.03 * x / (1 + 3e-4 * x),
    "F": lambda x: 0.016 * x / (1 + 3e-4 * x),
}
SLOPES = {"A": (0.22, 0.20), "B": (0.16, 0.12), "C": (0.11, 0.08),
          "D": (0.08, 0.06), "E": (0.06, 0.03), "F": (0.04, 0.016)}


def model_sigma(x, a, TL, sigma0, k_rel, U=U_FIT, TL_ref_U=U_FIT):
    TL = TL * TL_ref_U / U
    t = x / U
    sv = a * U
    return np.sqrt(sigma0 ** 2 + 2 * k_rel * t + 2 * sv ** 2 * TL ** 2 * (t / TL - 1 + np.exp(-t / TL)))


def fit(sigma0, k_rel, class_sigma0=None):
    out = {}
    for c in "ABCDEF":
        s0 = (class_sigma0 or {}).get(c, sigma0)
        row = {"a_y": SLOPES[c][0], "a_z": SLOPES[c][1], "sigma0": s0}
        for ax, ref, a in (("y", BRIGGS_Y[c], SLOPES[c][0]), ("z", BRIGGS_Z[c], SLOPES[c][1])):
            f = lambda lt: np.sum(np.log(model_sigma(X_FIT, a, np.exp(lt), s0, k_rel) / ref(X_FIT)) ** 2)
            r = minimize_scalar(f, bounds=np.log(TL_BOUNDS), method="bounded", options={"xatol": 1e-6})
            TL = float(np.exp(r.x))
            ratio = model_sigma(X_FIT, a, TL, s0, k_rel) / ref(X_FIT)
            row[f"TL_{ax}"] = round(TL, 1)
            row[f"fit_{ax}"] = {
                "rms_log_ratio": float(np.sqrt(np.mean(np.log(ratio) ** 2))),
                "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                "ratio_at_25m": float(ratio[0]), "ratio_at_50m": float(np.interp(50, X_FIT, ratio)),
                "ratio_at_200m": float(ratio[-1]),
                "at_bound": bool(TL <= TL_BOUNDS[0] * 1.001 or TL >= TL_BOUNDS[1] * 0.999),
            }
        out[c] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--k-rel", type=float, default=0.05)
    ap.add_argument("--class-sigma0", default="", help="per-class σ0, e.g. E=0.5,F=0.3")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tl_fit.json"))
    args = ap.parse_args()
    cs0 = {kv.split("=")[0].upper(): float(kv.split("=")[1]) for kv in args.class_sigma0.split(",") if kv}
    table = fit(args.sigma0, args.k_rel, cs0)
    res = {"U_fit": U_FIT, "x_fit_m": [25.0, 200.0], "TL_bounds_s": TL_BOUNDS,
           "sigma0": args.sigma0, "class_sigma0": cs0, "K_rel": args.k_rel, "classes": table,
           "model": "σ² = σ0² + 2 K_rel t + 2 σ_v² T_L² (t/T_L − 1 + exp(−t/T_L)), t = x/U, σ_v = a U"}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    for c, r in table.items():
        fy, fz = r["fit_y"], r["fit_z"]
        print(f"    \"{c}\": {{\"a_y\": {r['a_y']:<5}, \"a_z\": {r['a_z']:<5}, \"TL_y\": {r['TL_y']:7.1f}, "
              f"\"TL_z\": {r['TL_z']:7.1f}}},   # ratio y [{fy['ratio_min']:.2f}, {fy['ratio_max']:.2f}]"
              f"{' (bound)' if fy['at_bound'] else ''}  z [{fz['ratio_min']:.2f}, {fz['ratio_max']:.2f}]"
              f"{' (bound)' if fz['at_bound'] else ''}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
