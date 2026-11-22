"""Tests for baseline tree models and interpretability artifacts."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")

CYTO_MODEL_PATH = MODELS_DIR / "cytotox_best_model.joblib"
SEL_MODEL_PATH = MODELS_DIR / "seltox_best_model.joblib"

CYTO_TRAIN_PATH = PROCESSED_DIR / "cytotox_train.csv"
CYTO_TEST_PATH = PROCESSED_DIR / "cytotox_test.csv"
SEL_TRAIN_PATH = PROCESSED_DIR / "seltox_train.csv"
SEL_TEST_PATH = PROCESSED_DIR / "seltox_test.csv"

CYTO_SHAP_PNG = FIGURES_DIR / "shap_cytotox_summary.png"
SEL_SHAP_PNG = FIGURES_DIR / "shap_seltox_summary.png"


def test_models_exist_and_load_cleanly():
    """1. Assert models/cytotox_best_model.joblib and models/seltox_best_model.joblib exist and load cleanly."""
    assert CYTO_MODEL_PATH.exists(), f"Missing model: {CYTO_MODEL_PATH}"
    assert SEL_MODEL_PATH.exists(), f"Missing model: {SEL_MODEL_PATH}"

    cyto_pipe = joblib.load(CYTO_MODEL_PATH)
    sel_pipe = joblib.load(SEL_MODEL_PATH)

    assert cyto_pipe is not None
    assert sel_pipe is not None
    assert hasattr(cyto_pipe, "predict")
    assert hasattr(cyto_pipe, "predict_proba")
    assert hasattr(sel_pipe, "predict")
    assert hasattr(sel_pipe, "predict_proba")


def test_zero_doi_overlap_between_train_and_test():
    """2. Assert zero doi overlap between {task}_train.csv and {task}_test.csv for both tasks."""
    for path in [CYTO_TRAIN_PATH, CYTO_TEST_PATH, SEL_TRAIN_PATH, SEL_TEST_PATH]:
        assert path.exists(), f"Missing split file: {path}"

    cyto_train = pd.read_csv(CYTO_TRAIN_PATH)
    cyto_test = pd.read_csv(CYTO_TEST_PATH)
    cyto_train_dois = set(cyto_train["doi"].dropna().unique())
    cyto_test_dois = set(cyto_test["doi"].dropna().unique())
    assert cyto_train_dois.isdisjoint(cyto_test_dois), "DOI overlap detected in Cytotox train/test split!"

    sel_train = pd.read_csv(SEL_TRAIN_PATH)
    sel_test = pd.read_csv(SEL_TEST_PATH)
    sel_train_dois = set(sel_train["doi"].dropna().unique())
    sel_test_dois = set(sel_test["doi"].dropna().unique())
    assert sel_train_dois.isdisjoint(sel_test_dois), "DOI overlap detected in SelTox train/test split!"


def test_pipelines_predict_proba_on_raw_test_dataframe():
    """3. Assert both loaded pipelines accept a raw 5-row DataFrame from {task}_test.csv and return valid predict_proba outputs with shape (5, 2) and probabilities summing to 1.0."""
    cyto_pipe = joblib.load(CYTO_MODEL_PATH)
    sel_pipe = joblib.load(SEL_MODEL_PATH)

    cyto_test = pd.read_csv(CYTO_TEST_PATH)
    sel_test = pd.read_csv(SEL_TEST_PATH)

    # Raw 5-row slices from holdout test sets
    cyto_sample = cyto_test.head(5)
    sel_sample = sel_test.head(5)

    # Cytotox inference validation
    cyto_probs = cyto_pipe.predict_proba(cyto_sample)
    assert cyto_probs.shape == (5, 2), f"Expected shape (5, 2), got {cyto_probs.shape}"
    assert np.all((cyto_probs >= 0.0) & (cyto_probs <= 1.0)), "Cytotox probabilities outside [0, 1]"
    np.testing.assert_allclose(cyto_probs.sum(axis=1), np.ones(5), atol=1e-5)

    # SelTox inference validation
    sel_probs = sel_pipe.predict_proba(sel_sample)
    assert sel_probs.shape == (5, 2), f"Expected shape (5, 2), got {sel_probs.shape}"
    assert np.all((sel_probs >= 0.0) & (sel_probs <= 1.0)), "SelTox probabilities outside [0, 1]"
    np.testing.assert_allclose(sel_probs.sum(axis=1), np.ones(5), atol=1e-5)


def test_shap_figures_exist():
    """4. Assert results/figures/shap_cytotox_summary.png and results/figures/shap_seltox_summary.png exist."""
    assert CYTO_SHAP_PNG.exists(), f"Missing SHAP figure: {CYTO_SHAP_PNG}"
    assert SEL_SHAP_PNG.exists(), f"Missing SHAP figure: {SEL_SHAP_PNG}"

    assert CYTO_SHAP_PNG.stat().st_size > 5000, f"{CYTO_SHAP_PNG} is unexpectedly small ({CYTO_SHAP_PNG.stat().st_size} bytes)"
    assert SEL_SHAP_PNG.stat().st_size > 5000, f"{SEL_SHAP_PNG} is unexpectedly small ({SEL_SHAP_PNG.stat().st_size} bytes)"
