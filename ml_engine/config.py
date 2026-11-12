"""Global settings for the ML engine."""
from pathlib import Path

RANDOM_STATE = 42
MAX_FILE_MB = 50
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MIN_ROWS = 50
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

# Default metric used to pick the best model (user can override)
DEFAULT_METRIC = {"classification": "f1_macro", "regression": "rmse"}
