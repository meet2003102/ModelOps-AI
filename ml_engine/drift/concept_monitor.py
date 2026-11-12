"""Module 5B - concept drift: has the relationship between inputs and target changed?

Needs true labels. Each labelled prediction becomes an error value
(classification: 0 = correct, 1 = wrong; regression: |error| scaled to [0, 1]),
and a detector watches that error stream.

Detectors (same interface: update(err) -> True when drift is signalled):
  ADWINDetector        River's ADWIN (adaptive window; the main method)
  PageHinkleyDetector  classic sequential test; used only if River is missing
  FixedWindowDetector  simple baseline: error rate of the last N vs the reference error rate
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:
    from river import drift as river_drift
    HAS_RIVER = True
except ImportError:  # pragma: no cover
    HAS_RIVER = False


class ADWINDetector:
    name = "adwin"

    def __init__(self, delta: float = 0.002, grace: int = 30):
        if not HAS_RIVER:
            raise ImportError("river is not installed")
        self.delta, self.grace = delta, grace
        self.reset()

    def reset(self):
        self._d = river_drift.ADWIN(delta=self.delta)
        self.n = 0

    def update(self, err: float) -> bool:
        self.n += 1
        self._d.update(err)
        return bool(self._d.drift_detected) and self.n > self.grace


class PageHinkleyDetector:
    """Detects an increase in the mean error (fallback when River is unavailable)."""
    name = "page_hinkley"

    def __init__(self, delta: float = 0.005, threshold: float = 25.0, grace: int = 30):
        self.delta, self.threshold, self.grace = delta, threshold, grace
        self.reset()

    def reset(self):
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, err: float) -> bool:
        self.n += 1
        self.mean += (err - self.mean) / self.n
        self.cum += err - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return self.n > self.grace and (self.cum - self.min_cum) > self.threshold


class FixedWindowDetector:
    """Baseline: alarm when the error rate of the last `window` samples exceeds the
    reference error rate (first `window` samples after a reset) by `margin`."""
    name = "fixed_window"

    def __init__(self, window: int = 500, margin: float = 0.05):
        self.window, self.margin = window, margin
        self.reset()

    def reset(self):
        self.buf, self.ref, self.n = deque(maxlen=self.window), None, 0

    def update(self, err: float) -> bool:
        self.n += 1
        self.buf.append(err)
        if len(self.buf) < self.window:
            return False
        rate = float(np.mean(self.buf))
        if self.ref is None:
            self.ref = rate
            return False
        return rate > self.ref + self.margin


def make_detector(name: str = "adwin", **kw):
    if name == "adwin":
        return ADWINDetector(**kw) if HAS_RIVER else PageHinkleyDetector()
    if name == "page_hinkley":
        return PageHinkleyDetector(**kw)
    if name == "fixed_window":
        return FixedWindowDetector(**kw)
    raise ValueError(f"Unknown detector '{name}'")


def error_stream(y_true, y_pred, task: str, scale: float | None = None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if task == "classification":
        return (y_true != y_pred).astype(float)
    err = np.abs(y_true.astype(float) - y_pred.astype(float))
    scale = scale or (np.percentile(err, 95) or 1.0)
    return np.clip(err / scale, 0, 1)


def monitor_labelled_batch(y_true, y_pred, task: str, detector: str = "adwin",
                           window: int = 100) -> dict:
    """Run a detector over labelled production predictions (in arrival order)."""
    errs = error_stream(y_true, y_pred, task)
    det = make_detector(detector)
    alarms = []
    for i, e in enumerate(errs):
        if det.update(e):
            alarms.append(i)
            det.reset()          # start a fresh window after each alarm
    roll = np.convolve(errs, np.ones(window) / window, mode="valid") if len(errs) >= window else errs
    return {"detector": det.name, "n": int(len(errs)), "alarms": alarms[:50],
            "concept_drift": bool(alarms),
            "error_rate": round(float(errs.mean()), 4),
            "rolling_error": [round(float(v), 4) for v in roll[::max(1, len(roll) // 200)]]}
