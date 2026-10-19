"""
nag2024.py — loader for Nag & van Breugel (2024) outdoor odor-plume data
(Dryad doi:10.5061/dryad.2547d7wvr, CC0), pandas 'fixed'-format HDF5 read with
h5py (no PyTables needed).  Files in bench/data/nag2024/data/data/.
"""
import os

import h5py
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nag2024", "data", "data")
SETS = {"desert_low_wind": "NotWindyMASigned.h5", "desert_high_wind": "WindyMASigned.h5",
        "forest": "ForestMASigned.h5"}


def load(name, cols=None):
    """dict column → np.ndarray for one dataset (all blocks of the 'fdf' frame)."""
    with h5py.File(os.path.join(DATA, SETS.get(name, name)), "r") as f:
        g = f[list(f.keys())[0]]
        out = {}
        for b in range(int(g.attrs["nblocks"])):
            items = [s.decode() for s in g[f"block{b}_items"][()]]
            want = [i for i, s in enumerate(items) if cols is None or s in cols]
            if not want:
                continue
            vals = g[f"block{b}_values"]
            arr = vals[:, want] if len(want) < len(items) else vals[()]
            for k, i in enumerate(want):
                out[items[i]] = np.asarray(arr[:, k] if arr.ndim == 2 else arr)
        return out
