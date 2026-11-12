"""Point the backend at a temporary artifacts folder BEFORE it is imported."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="modelops_test_")
os.environ["MODELOPS_ARTIFACTS"] = _tmp
os.environ["MODELOPS_DB"] = f"sqlite:///{_tmp}/test.db".replace("\\", "/")
