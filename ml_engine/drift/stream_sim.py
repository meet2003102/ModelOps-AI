"""Synthetic concept-drift streams for the controlled monitoring benchmark.

SEA concepts (Street & Kim, 2001): three features uniform in [0, 10];
label = 1 if x0 + x1 <= theta. Variants use theta = 8, 9, 7, 9.5.
Feature x2 is irrelevant noise. `noise` flips that share of labels.

Drift follows River's ConceptDriftStream definition: at sample t the new concept
is used with probability 1 / (1 + exp(-4 (t - position) / width)).
width = 1 gives sudden drift; a large width gives gradual drift.

All parameters are stored in the returned `config` so every run is reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEA_THETA = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}

SCENARIOS = {
    "stable":  {"n": 20000, "position": None,  "width": 1,    "from_variant": 0, "to_variant": 0},
    "sudden":  {"n": 20000, "position": 10000, "width": 1,    "from_variant": 0, "to_variant": 2},
    "gradual": {"n": 20000, "position": 10000, "width": 2000, "from_variant": 0, "to_variant": 2},
}


def sea_stream(n: int = 20000, position: int | None = 10000, width: int = 1,
               from_variant: int = 0, to_variant: int = 2, noise: float = 0.1,
               seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 3))
    t = np.arange(n)
    if position is None:
        p_new = np.zeros(n)
    else:
        z = np.clip(-4 * (t - position) / max(width, 1), -500, 500)
        p_new = 1 / (1 + np.exp(z))
    use_new = rng.random(n) < p_new
    theta = np.where(use_new, SEA_THETA[to_variant], SEA_THETA[from_variant])
    y = (X[:, 0] + X[:, 1] <= theta).astype(int)
    flip = rng.random(n) < noise
    y = np.where(flip, 1 - y, y)
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["y"] = y
    df["concept"] = use_new.astype(int)
    cfg = {"generator": "SEA", "n": n, "position": position, "width": width,
           "from_variant": from_variant, "to_variant": to_variant,
           "theta_from": SEA_THETA[from_variant], "theta_to": SEA_THETA[to_variant],
           "noise": noise, "seed": seed}
    return df, cfg


def scenario(name: str, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    df, cfg = sea_stream(seed=seed, **SCENARIOS[name])
    cfg["scenario"] = name
    return df, cfg
