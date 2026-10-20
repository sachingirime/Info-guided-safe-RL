"""
prairie_grass.py — Project Prairie Grass (1956) data: arcs, meteorology, stability.

Data: J. Irwin's digitisation (harmo.org/jsirwin/PrairieGrassDiscussion.html),
files in bench/data/prairie_grass/:
  PGARCS.txt           arc sampler concentrations (D6589 Table 2.1 format; polar:
                       angle [deg], arc radius [m]; concentration mg/m³ as in the data
                       volume, not yet corrected for SO2 evaporative loss)
  PGrassTABLE_5_4.TXT  evaporative-loss correction factors f (C_corr = C / f)
  PGrassTTUU.txt       20-min mean wind and temperature profiles (0.25–16 m), z0 = 0.006 m

Stability: Monin–Obukhov fit of the profiles (Businger–Dyer, Dyer 1974) for u*,
θ*, L; Pasquill class from 1/L and z0 with the Golder (1972) curves in the
fitted form 1/L = a + b·log10(z0) (coefficients as tabulated in Seinfeld &
Pandis, Atmospheric Chemistry and Physics; nearest-class assignment).
"""

import os
import re

import numpy as np
from scipy.optimize import brentq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prairie_grass")
Z0 = 0.006
KAPPA, G = 0.4, 9.81
STANDARD_EXCLUDE = (3, 4, 13, 14)        # anomalous low wind / roughness (Aarhus OML subset)
GOLDER_AB = {"A": (-0.096, 0.029), "B": (-0.037, 0.029), "C": (-0.002, 0.018),
             "D": (0.0, 0.0), "E": (0.004, -0.018), "F": (0.035, -0.036)}
S_RUNS = {70: "35S", 71: "48S"}


def read_arcs(path=None):
    """{exp: {"Q_gs", "hs", "zr", "date", "time", "arcs": {r: (deg, C_mgm3)}}}"""
    lines = open(path or os.path.join(DATA, "PGARCS.txt")).read().splitlines()[1:]
    runs, i = {}, 0
    while i < len(lines):
        h2 = lines[i].strip()
        if not h2:
            i += 1; continue
        m = re.match(r"\s*(\d+),\s*(\d+),'([^']*)','([^']*)',\s*([\d.]+)", lines[i])
        exp, trav, date, tim, dist = int(m[1]), int(m[2]), m[3].strip(), m[4].strip(), float(m[5])
        h3 = [float(v) for v in lines[i + 1].split(",")]
        nrec = int(h3[0]); Q, hs, zr, mult = h3[7], h3[8], h3[9], h3[10]
        rec = np.array([[float(v) for v in lines[i + 2 + k].split()] for k in range(nrec)])
        i += 2 + nrec
        deg, r, c = rec[:, 0], rec[:, 2], rec[:, 3] * mult
        run = runs.setdefault(exp, {"Q_gs": Q, "hs": hs, "zr": zr, "date": date, "time": tim, "arcs": {}})
        run["arcs"][int(round(r[0]))] = (deg, c)
    return runs


def read_corrections(path=None):
    """{exp: {arc: f}}; blanks → missing.  Run 12 at 200 m reads 0.09 (all others
    0.85–1.00): treated as a transcription error and left uncorrected."""
    out = {}
    for ln in open(path or os.path.join(DATA, "PGrassTABLE_5_4.TXT")).read().splitlines():
        if not re.match(r"^\s+\d+\s", ln) or "Exp" in ln:
            continue
        exp = int(ln[:8])
        vals = {}
        for k, arc in enumerate((50, 100, 200, 400, 800)):
            s = ln[8 + 9 * k: 8 + 9 * (k + 1)].strip()
            if s:
                vals[arc] = float(s)
        if exp == 12 and vals.get(200) == 0.09:
            vals.pop(200)
        out[exp] = vals
    return out


def read_profiles(path=None):
    """{exp: (z, T_C, U)} with missing (-999) removed."""
    txt = open(path or os.path.join(DATA, "PGrassTTUU.txt")).read()
    out = {}
    for blk in re.split(r"\n\s*Exp\s+", txt)[1:]:
        lines = blk.strip().splitlines()
        exp = int(lines[0].split()[0])
        rows = []
        for ln in lines[1:]:
            p = ln.split()
            if len(p) != 3:
                break
            rows.append([float(v) for v in p])
        a = np.array(rows)
        out[exp] = a
    return out


def _psi(zeta, kind):
    zeta = np.asarray(zeta, float)
    out = np.empty_like(zeta)
    st = zeta >= 0
    out[st] = -5.0 * np.minimum(zeta[st], 1.0) - np.where(zeta[st] > 1, 5.0 * np.log(np.maximum(zeta[st], 1.0)), 0.0)
    x = (1 - 16 * zeta[~st]) ** 0.25
    if kind == "m":
        out[~st] = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
    else:
        out[~st] = 2 * np.log((1 + x * x) / 2)
    return out


def mo_fit(prof, z_min=0.5, z_max=8.0):
    """Fit u*, θ*, L from one profile.  Returns dict or None."""
    z, T, U = prof[:, 0], prof[:, 1], prof[:, 2]
    ok_u = (U > -900) & (z >= z_min) & (z <= z_max)
    ok_t = (T > -900) & (z >= z_min) & (z <= z_max)
    if ok_u.sum() < 3 or ok_t.sum() < 3:
        return None
    zu, uu = z[ok_u], U[ok_u]
    zt, th = z[ok_t], T[ok_t] + 273.15 + 0.0098 * z[ok_t]   # potential temperature
    Tm = th.mean()

    def fit_given(eta):
        fu = (np.log(zu / Z0) - _psi(zu * eta, "m") + _psi(np.array([Z0 * eta]), "m")[0]) / KAPPA
        ustar = float((fu @ uu) / (fu @ fu))
        A = np.stack([np.ones_like(zt), (np.log(zt) - _psi(zt * eta, "h")) / KAPPA], 1)
        coef, *_ = np.linalg.lstsq(A, th, rcond=None)
        return ustar, float(coef[1]), fu, A, coef

    def g(eta):
        ustar, tstar, *_ = fit_given(eta)
        return eta - KAPPA * G * tstar / (ustar ** 2 * Tm)

    grid = np.concatenate([-np.logspace(1, -4, 60), [0.0], np.logspace(-4, 1, 60)])
    vals = np.array([g(e) for e in grid])
    roots = []
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.sign(va) != np.sign(vb):
            roots.append(brentq(g, a, b))
    if not roots:
        return None
    # several roots: keep the one with the smallest wind-profile residual
    best = None
    for eta in roots:
        ustar, tstar, fu, A, coef = fit_given(eta)
        res = float(np.sqrt(np.mean((fu * ustar - uu) ** 2)))
        if best is None or res < best[0]:
            best = (res, eta, ustar, tstar)
    res, eta, ustar, tstar = best
    U_at = lambda zz: float(ustar / KAPPA * (np.log(zz / Z0) - _psi(np.array([zz * eta]), "m")[0] + _psi(np.array([Z0 * eta]), "m")[0]))
    return {"ustar": ustar, "thetastar": tstar, "inv_L": float(eta), "L": float(1 / eta) if eta != 0 else np.inf,
            "U2": U_at(2.0), "U10": U_at(10.0), "U_fit_rms": res, "n_roots": len(roots)}


def pg_class(inv_L, z0=Z0):
    lz = np.log10(z0)
    ref = {c: a + b * lz for c, (a, b) in GOLDER_AB.items()}
    return min(ref, key=lambda c: abs(inv_L - ref[c]))


def arc_stats(deg, c, r):
    """Arc max, CWIC (∫C dy along the arc), σ_y (arc length), centroid angle.
    edge_max: maximum at the first/last sampler (plume not captured)."""
    c = np.maximum(np.asarray(c, float), 0.0)
    if c.max() <= 0:
        return None
    # angles relative to the peak sampler, wrapped to (-180, 180]: arcs may cross 0/360
    rel = (np.asarray(deg, float) - deg[int(np.argmax(c))] + 180.0) % 360.0 - 180.0
    o = np.argsort(rel); d = rel[o]; c = c[o]
    s = np.deg2rad(d) * r
    cw = float(np.trapezoid(c, s))
    if cw <= 0:
        return None
    sc = float(np.trapezoid(c * s, s) / cw)
    sy = float(np.sqrt(max(np.trapezoid(c * (s - sc) ** 2, s) / cw, 0.0)))
    k = int(np.argmax(c))
    return {"cmax": float(c[k]), "cwic": cw, "sigma_y": sy, "theta_c": float(sc / r),
            "edge_max": bool(k == 0 or k == len(c) - 1), "n": int(len(c))}


def load(arcs=(50, 100, 200), exclude=STANDARD_EXCLUDE):
    """Evaluation table: one entry per run with met, class and per-arc C/Q stats
    (C/Q in s/m³, CWIC/Q in s/m², corrected for evaporative loss)."""
    runs, corr, prof = read_arcs(), read_corrections(), read_profiles()
    table, dropped = {}, {}
    for exp, run in sorted(runs.items()):
        if exp in exclude:
            dropped[exp] = "standard exclusion"; continue
        if exp not in prof:
            dropped[exp] = "no profile"; continue
        met = mo_fit(prof[exp])
        if met is None:
            dropped[exp] = "MO fit failed"; continue
        Qgs = run["Q_gs"]
        entry = {"exp": exp, "name": S_RUNS.get(exp, str(exp)), "Q_gs": Qgs, "hs": run["hs"], "zr": run["zr"],
                 "date": run["date"], "time": run["time"], **met, "pg": pg_class(met["inv_L"]), "arcs": {}}
        for a in arcs:
            if a not in run["arcs"]:
                continue
            deg, c = run["arcs"][a]
            f = corr.get(exp, {}).get(a)
            cc = c / f if f else c
            st = arc_stats(deg, cc * 1e-3, float(a))      # mg/m³ → g/m³
            if st is None:
                continue
            st.update({"cmax_Q": st["cmax"] / Qgs, "cwic_Q": st["cwic"] / Qgs, "evap_f": f})
            entry["arcs"][a] = st
        if entry["arcs"]:
            table[exp] = entry
        else:
            dropped[exp] = "no arcs"
    return table, dropped


if __name__ == "__main__":
    t, d = load()
    print(f"{len(t)} runs, dropped {d}")
    for e, v in t.items():
        a50 = v["arcs"].get(50, {})
        print(f"{v['name']:>4} {v['pg']} U10 {v['U10']:5.2f} u* {v['ustar']:.3f} 1/L {v['inv_L']:+.4f} "
              f"Q {v['Q_gs']:5.1f} hs {v['hs']:.2f}  50m: Cmax/Q {a50.get('cmax_Q', np.nan):.2e} "
              f"CWIC/Q {a50.get('cwic_Q', np.nan):.3e} σy {a50.get('sigma_y', np.nan):5.1f} edge {a50.get('edge_max')}")
