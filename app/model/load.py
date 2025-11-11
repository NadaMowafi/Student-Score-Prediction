from functools import lru_cache
from joblib import load
import json, os

ARTIFACT_MODEL = os.getenv("MODEL_PATH", "artifacts/model.joblib")
ARTIFACT_COLUMNS = os.getenv("COLUMNS_PATH", "artifacts/columns.json")

@lru_cache(maxsize=1)
def get_model():
    try:
        return load(ARTIFACT_MODEL)
    except Exception:
        return None

@lru_cache(maxsize=1)
def get_feature_names():
    try:
        with open(ARTIFACT_COLUMNS, "r") as f:
            cols = json.load(f)
        return cols if isinstance(cols, list) else None
    except Exception:
        return None
