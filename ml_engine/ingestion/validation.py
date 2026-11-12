"""Module 1 - file and dataset validation."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ml_engine import config


class ValidationError(ValueError):
    """Raised when an uploaded file cannot be used at all."""


def validate_file(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported file type '{ext}'. Allowed: CSV, XLSX.")
    if size_bytes > config.MAX_FILE_MB * 1024 * 1024:
        raise ValidationError(f"File is larger than {config.MAX_FILE_MB} MB.")
    if size_bytes == 0:
        raise ValidationError("File is empty.")


def load_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """Read CSV/XLSX bytes into a DataFrame (tries common encodings)."""
    validate_file(filename, len(content))
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as e:
            raise ValidationError(f"Could not parse CSV: {e}") from e
    raise ValidationError("Could not decode file (tried utf-8 and latin-1).")


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Checks that decide whether the dataset can be used for training.

    Returns {"valid": bool, "errors": [...], "warnings": [...]}.
    """
    errors, warnings = [], []
    if df.shape[1] < 2:
        errors.append("Dataset needs at least 2 columns (features + target).")
    if len(df) < config.MIN_ROWS:
        errors.append(f"Dataset has {len(df)} rows; at least {config.MIN_ROWS} are needed.")
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        warnings.append(f"Completely empty columns will be dropped: {empty}")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        warnings.append(f"Unnamed columns (often a saved index): {unnamed}")
    if df.columns.duplicated().any():
        errors.append("Duplicate column names found.")
    # CSV/formula injection guard for values later exported to spreadsheets
    obj = df.select_dtypes(include="object")
    risky = [c for c in obj.columns
             if obj[c].astype(str).str.match(r"^[=+\-@]\w*\(").any()]
    if risky:
        warnings.append(f"Cells that look like spreadsheet formulas in: {risky}")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "n_rows": int(len(df)), "n_cols": int(df.shape[1])}
