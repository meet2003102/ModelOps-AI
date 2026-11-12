"""Backend settings. Override with environment variables (used by the tests)."""
import os
from pathlib import Path

from ml_engine import config as engine_config

ARTIFACTS_DIR = Path(os.getenv("MODELOPS_ARTIFACTS", engine_config.ARTIFACTS_DIR))
engine_config.ARTIFACTS_DIR = ARTIFACTS_DIR          # engine and API share one folder
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
DB_URL = os.getenv("MODELOPS_DB", f"sqlite:///{(ARTIFACTS_DIR / 'modelops.db').as_posix()}")
CORS_ORIGINS = os.getenv("MODELOPS_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
API_PREFIX = "/api/v1"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
