# import libraries

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ensure backward compatibility for simpleimputer across scikit-learn versions
try:
    from sklearn.impute import SimpleImputer

    if not hasattr(SimpleImputer, "_fill_dtype"):
        def _get_fill_dtype(self):
            if "_fill_dtype" in self.__dict__:
                return self.__dict__["_fill_dtype"]
            return getattr(self, "_fit_dtype", np.float64)

        def _set_fill_dtype(self, val):
            self.__dict__["_fill_dtype"] = val

        SimpleImputer._fill_dtype = property(_get_fill_dtype, _set_fill_dtype)
except Exception:
    pass

# project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "job_offer_acceptance_dataset.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

ENCODER_PATH = BASE_DIR / "models" / "encoders.pkl"

FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"

# cross version compatibility helper

def fix_sklearn_compatibility(obj):
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            fix_sklearn_compatibility(step)
    if hasattr(obj, "transformers_"):
        for _, trans, _ in obj.transformers_:
            fix_sklearn_compatibility(trans)
    if hasattr(obj, "transformers"):
        for item in obj.transformers:
            if len(item) >= 2:
                fix_sklearn_compatibility(item[1])
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            fix_sklearn_compatibility(step)
    if type(obj).__name__ == "SimpleImputer":
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = getattr(obj, "_fit_dtype", np.float64)
    return obj

# load dataset

def load_dataset():
    return pd.read_csv(DATASET_PATH)

# load best model

def load_model():
    model = joblib.load(MODEL_PATH)
    return fix_sklearn_compatibility(model)

# load scaler

def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    return fix_sklearn_compatibility(scaler)

# load encoders

def load_encoders():
    return joblib.load(ENCODER_PATH)

# load feature names

def load_feature_names():
    return joblib.load(FEATURE_PATH)

# load production pipeline

def load_pipeline():
    pipe = joblib.load(PIPELINE_PATH)
    return fix_sklearn_compatibility(pipe)

# load named model

def load_named_model(model_filename: str):
    path = BASE_DIR / "models" / model_filename
    if path.exists():
        model = joblib.load(path)
    else:
        model = joblib.load(MODEL_PATH)
    return fix_sklearn_compatibility(model)
