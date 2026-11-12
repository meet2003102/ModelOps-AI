"""Controlled concept-drift benchmark (prequential: predict first, then learn the label).

For each scenario x detector x {static, adaptive} it reports:
  detection rate, false alarms, detection delay, accuracy before / after drift,
  recovery time. 'adaptive' = champion-challenger retraining on a drift alarm.

Drift window: [position - width/2, position + width/2]. An alarm before the window
start counts as a false alarm; the first alarm inside or after it gives the delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_engine.drift.concept_monitor import make_detector
from ml_engine.drift.stream_sim import scenario

FEATURES = ["x0", "x1", "x2"]


def _fit(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


def _predict_one(m, x):
    return int(x @ m.coef_[0] + m.intercept_[0] > 0)


def _recovery(acc: pd.Series, start: int, end: int, pre: float, tol: float = 0.02,
              hold: int = 1000) -> int | None:
    """Samples from drift start until rolling accuracy is back within `tol` of the
    pre-drift level AND stays there for `hold` samples, counted from the worst point.
    0 = no real dip; None = never recovered."""
    window = acc.loc[start:]
    dip_zone = window.loc[:end + 1000]
    worst_t = int(dip_zone.idxmin())
    if dip_zone.min() >= pre - tol:
        return 0
    ok = (window.loc[worst_t:] >= pre - tol).to_numpy()
    idx = window.loc[worst_t:].index
    for i in range(len(ok) - hold):
        if ok[i] and ok[i:i + hold].all():
            return int(idx[i] - start)
    return None


def run_stream(df: pd.DataFrame, cfg: dict, detector: str | None = "adwin", adapt: bool = False,
               init: int = 1000, retrain_window: int = 1000, holdout: int = 200,
               min_gain: float = 0.01, acc_window: int = 500) -> dict:
    X, y = df[FEATURES].to_numpy(), df["y"].to_numpy()
    model = _fit(X[:init], y[:init])
    det = make_detector(detector) if detector else None
    correct = np.zeros(len(df))
    alarms, retrains = [], []
    pending = None          # time of the alarm whose retraining is waiting for new labels

    for t in range(init, len(df)):
        correct[t] = _predict_one(model, X[t]) == y[t]
        if det is None:
            continue
        # champion-challenger once `retrain_window` labels have arrived AFTER the alarm,
        # so the challenger learns the new concept, not a mix of old and new
        if adapt and pending is not None and t - pending >= retrain_window:
            hold = slice(t - holdout, t)
            train = slice(t - retrain_window, t - holdout)
            challenger = _fit(X[train], y[train])
            champ_acc = float((model.predict(X[hold]) == y[hold]).mean())
            chall_acc = float((challenger.predict(X[hold]) == y[hold]).mean())
            promoted = chall_acc >= champ_acc + min_gain
            if promoted:
                model = challenger
            retrains.append({"alarm_t": pending, "t": t, "champion_acc": round(champ_acc, 4),
                             "challenger_acc": round(chall_acc, 4), "promoted": promoted})
            pending = None
            det.reset()
        if det.update(1.0 - correct[t]):
            alarms.append(t)
            if adapt and pending is None:
                pending = t
            det.reset()

    acc = pd.Series(correct[init:]).rolling(acc_window, min_periods=50).mean()
    acc.index = range(init, len(df))
    pos, width = cfg["position"], cfg["width"]
    out = {"scenario": cfg.get("scenario"), "detector": det.name if det else "none",
           "adaptive": adapt, "alarms": alarms, "n_alarms": len(alarms), "retrains": retrains,
           "overall_accuracy": round(float(correct[init:].mean()), 4)}

    if pos is None:          # stable stream: every alarm is a false alarm
        out.update(false_alarms=len(alarms), detected=None, delay=None,
                   pre_drift_acc=out["overall_accuracy"], post_drift_acc=None, recovery=None)
    else:
        start, end = int(pos - width / 2), int(pos + width / 2)
        false = [a for a in alarms if a < start]
        hits = [a for a in alarms if a >= start]
        pre = float(correct[init:start].mean())
        post = float(correct[end:].mean())
        recovery = _recovery(acc, start, end, pre)
        out.update(false_alarms=len(false), detected=bool(hits),
                   delay=int(hits[0] - start) if hits else None,
                   pre_drift_acc=round(pre, 4), post_drift_acc=round(post, 4), recovery=recovery)
    out["accuracy_curve"] = {"t": acc.index[::100].tolist(),
                             "acc": acc.iloc[::100].round(4).fillna(0).tolist()}
    return out


def run_benchmark(detectors=("adwin", "fixed_window", None), scenarios=("stable", "sudden", "gradual"),
                  seeds=(42,), adaptive=(False, True)) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        for seed in seeds:
            df, cfg = scenario(sc, seed=seed)
            for d in detectors:
                for ad in adaptive:
                    if d is None and ad:
                        continue
                    r = run_stream(df, cfg, d, ad)
                    rows.append({k: v for k, v in r.items()
                                 if k not in ("alarms", "retrains", "accuracy_curve")} | {"seed": seed})
    return pd.DataFrame(rows)
