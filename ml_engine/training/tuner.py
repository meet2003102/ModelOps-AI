"""Hyperparameter tuning: Optuna when installed, otherwise random search."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ml_engine.config import RANDOM_STATE

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False


def _prefixed(space: dict) -> dict:
    return {f"model__{k}": v for k, v in space.items()}


def _to_scipy(spec):
    kind = spec[0]
    if kind == "int":
        return stats.randint(spec[1], spec[2] + 1)
    if kind == "float":
        lo, hi, log = spec[1], spec[2], spec[3]
        return stats.loguniform(lo, hi) if log else stats.uniform(lo, hi - lo)
    return list(spec[1])


def tune(pipeline, space: dict, X, y, cv, scoring: str, n_trials: int = 20,
         timeout: int | None = 300) -> tuple[dict, float]:
    """Return (best params, best mean CV score). Score is 'higher is better'."""
    if not space:
        return {}, float(np.mean(cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)))
    space = _prefixed(space)

    if HAS_OPTUNA:
        def objective(trial):
            params = {}
            for name, spec in space.items():
                if spec[0] == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif spec[0] == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
                else:
                    params[name] = trial.suggest_categorical(name, spec[1])
            p = clone(pipeline).set_params(**params)
            return float(np.mean(cross_val_score(p, X, y, cv=cv, scoring=scoring)))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return study.best_params, float(study.best_value)

    search = RandomizedSearchCV(pipeline, {k: _to_scipy(v) for k, v in space.items()},
                                n_iter=n_trials, cv=cv, scoring=scoring,
                                random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)
