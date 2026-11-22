"""Environment and dependency smoke test."""

import importlib.metadata
import catboost
import dice_ml
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost


def test_core_imports():
    """Verify that core numerical, scientific, and ML libraries import properly."""
    assert np is not None
    assert pd is not None
    assert scipy is not None
    assert sklearn.__version__ is not None


def test_model_libraries():
    """Verify that XGBoost and CatBoost import and expose versions."""
    assert xgboost.__version__ is not None
    assert catboost.__version__ is not None


def test_interpretability_libraries():
    """Verify that DiCE and SHAP import cleanly without dependency/version conflicts."""
    assert dice_ml.Dice is not None
    assert dice_ml.Data is not None
    assert dice_ml.Model is not None
    assert shap.__version__ is not None
    assert importlib.metadata.version("dice-ml") is not None
