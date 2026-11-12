"""Optional Evidently HTML data-drift report (visual only).

The platform's drift DECISIONS come from psi_ks.py; Evidently is used for its
interactive HTML report. Written for evidently 0.4.x (pinned in requirements).
Returns None if Evidently is missing or its API differs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def save_html_report(reference: pd.DataFrame, current: pd.DataFrame, path: str | Path) -> str | None:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception:
        return None
    try:
        warnings.filterwarnings("ignore", category=RuntimeWarning)   # Evidently's internal chi2
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(path))
        return str(path)
    except Exception:
        return None
