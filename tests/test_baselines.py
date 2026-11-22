"""Validation tests for baseline modeling pipeline and artifacts."""

from pathlib import Path
import joblib
import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
TABLES_DIR = Path("results/tables")
REPORT_PATH = Path("docs/MODEL_BENCHMARK.md")

CYTO_TRAIN = PROCESSED_DIR / "cytotox_train.csv"
CYTO_TEST = PROCESSED_DIR / "cytotox_test.csv"
SEL_TRAIN = PROCESSED_DIR / "seltox_train.csv"
SEL_TEST = PROCESSED_DIR / "seltox_test.csv"

CYTO_MODEL = MODELS_DIR / "cytotox_best_model.joblib"
SEL_MODEL = MODELS_DIR / "seltox_best_model.joblib"
METRICS_CSV = TABLES_DIR / "baseline_metrics.csv"


def test_split_files_exist_and_leakage_free():
    """Verify train and holdout test files exist and have zero DOI leakage."""
    for path in [CYTO_TRAIN, CYTO_TEST, SEL_TRAIN, SEL_TEST]:
        assert path.exists(), f"Missing split file: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Split file is empty: {path}"

    cyto_tr = pd.read_csv(CYTO_TRAIN)
    cyto_te = pd.read_csv(CYTO_TEST)
    assert set(cyto_tr["doi"]).isdisjoint(set(cyto_te["doi"])), "DOI leakage detected in Cytotox train/test split!"

    sel_tr = pd.read_csv(SEL_TRAIN)
    sel_te = pd.read_csv(SEL_TEST)
    assert set(sel_tr["doi"]).isdisjoint(set(sel_te["doi"])), "DOI leakage detected in SelTox train/test split!"


def test_saved_models_load_and_predict():
    """Verify serialized model pipelines load and can infer on test samples."""
    assert CYTO_MODEL.exists(), f"Missing {CYTO_MODEL}"
    assert SEL_MODEL.exists(), f"Missing {SEL_MODEL}"

    cyto_pipe = joblib.load(CYTO_MODEL)
    sel_pipe = joblib.load(SEL_MODEL)

    cyto_te = pd.read_csv(CYTO_TEST).head(5)
    sel_te = pd.read_csv(SEL_TEST).head(5)

    cyto_preds = cyto_pipe.predict(cyto_te)
    cyto_probs = cyto_pipe.predict_proba(cyto_te)
    assert len(cyto_preds) == 5
    assert cyto_probs.shape == (5, 2)

    sel_preds = sel_pipe.predict(sel_te)
    sel_probs = sel_pipe.predict_proba(sel_te)
    assert len(sel_preds) == 5
    assert sel_probs.shape == (5, 2)


def test_baseline_metrics_and_report():
    """Verify metrics CSV and benchmark documentation report are generated."""
    assert METRICS_CSV.exists(), f"Missing {METRICS_CSV}"
    df = pd.read_csv(METRICS_CSV)
    assert len(df) >= 12  # 2 tasks * 3 models * 2 split types (CV + test)

    expected_cols = [
        "task",
        "model",
        "split_type",
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in metrics CSV"

    assert REPORT_PATH.exists(), f"Missing {REPORT_PATH}"
    assert REPORT_PATH.stat().st_size > 500, "Benchmark report is unexpectedly empty"
