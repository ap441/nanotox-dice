"""Validation tests for harmonized preprocessed datasets."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
CYTO_CLEAN_PATH = PROCESSED_DIR / "cytotox_clean.csv"
SELTOX_CLEAN_PATH = PROCESSED_DIR / "seltox_clean.csv"
REPORT_PATH = Path("docs/PREPROCESSING_REPORT.md")

CANONICAL_SHAPES = {"spherical", "rod", "nanotube", "cubic", "other", "unknown"}


@pytest.fixture(scope="module")
def cyto_df() -> pd.DataFrame:
    assert CYTO_CLEAN_PATH.exists(), f"Missing processed file: {CYTO_CLEAN_PATH}"
    df = pd.read_csv(CYTO_CLEAN_PATH)
    assert len(df) > 0, f"{CYTO_CLEAN_PATH} is empty"
    return df


@pytest.fixture(scope="module")
def sel_df() -> pd.DataFrame:
    assert SELTOX_CLEAN_PATH.exists(), f"Missing processed file: {SELTOX_CLEAN_PATH}"
    df = pd.read_csv(SELTOX_CLEAN_PATH)
    assert len(df) > 0, f"{SELTOX_CLEAN_PATH} is empty"
    return df


def test_files_exist_and_non_empty(cyto_df: pd.DataFrame, sel_df: pd.DataFrame):
    """1. Assert both files exist and are non-empty."""
    assert len(cyto_df) > 1000, f"Expected >1000 rows in Cytotox, found {len(cyto_df)}"
    assert len(sel_df) > 1000, f"Expected >1000 rows in SelTox, found {len(sel_df)}"


def test_zero_nulls_in_target_columns_and_doi(cyto_df: pd.DataFrame, sel_df: pd.DataFrame):
    """2. Assert zero nulls in target columns and doi."""
    # Cytotox target columns and study identifier
    assert cyto_df["is_biocompatible"].isna().sum() == 0, "Nulls found in Cytotox is_biocompatible"
    assert cyto_df["viability_%"].isna().sum() == 0, "Nulls found in Cytotox viability_%"
    assert cyto_df["doi"].isna().sum() == 0, "Nulls found in Cytotox doi"

    # SelTox target columns and study identifier
    assert sel_df["is_antibacterial"].isna().sum() == 0, "Nulls found in SelTox is_antibacterial"
    assert sel_df["mic_ug_ml"].isna().sum() == 0, "Nulls found in SelTox mic_ug_ml"
    assert sel_df["log10_mic"].isna().sum() == 0, "Nulls found in SelTox log10_mic"
    assert sel_df["doi"].isna().sum() == 0, "Nulls found in SelTox doi"


def test_target_value_bounds(cyto_df: pd.DataFrame, sel_df: pd.DataFrame):
    """3. Assert viability_% is strictly within [0, 120] and mic_ug_ml > 0."""
    assert (cyto_df["viability_%"] >= 0.0).all(), "viability_% contains negative values"
    assert (cyto_df["viability_%"] <= 120.0).all(), "viability_% exceeds clip limit of 120.0"

    assert (sel_df["mic_ug_ml"] > 0.0).all(), "mic_ug_ml contains zero or negative values"
    assert not np.isinf(sel_df["log10_mic"]).any(), "log10_mic contains infinite values"


def test_canonical_shapes_and_is_coated_values(cyto_df: pd.DataFrame, sel_df: pd.DataFrame):
    """4. Assert canonical shape categories and is_coated values ({0, 1}) match across both datasets."""
    cyto_shapes = set(cyto_df["shape"].unique())
    sel_shapes = set(sel_df["shape"].unique())

    assert cyto_shapes.issubset(CANONICAL_SHAPES), f"Cytotox has non-canonical shapes: {cyto_shapes - CANONICAL_SHAPES}"
    assert sel_shapes.issubset(CANONICAL_SHAPES), f"SelTox has non-canonical shapes: {sel_shapes - CANONICAL_SHAPES}"

    assert set(cyto_df["is_coated"].unique()) == {0, 1}, "Cytotox is_coated should be exactly {0, 1}"
    assert set(sel_df["is_coated"].unique()) == {0, 1}, "SelTox is_coated should be exactly {0, 1}"


def test_core_material_overlap_threshold(cyto_df: pd.DataFrame, sel_df: pd.DataFrame):
    """5. Assert core_material has at least 10 overlapping materials across the cleaned datasets."""
    cyto_materials = set(cyto_df["core_material"].dropna().str.strip())
    sel_materials = set(sel_df["core_material"].dropna().str.strip())
    overlap = cyto_materials.intersection(sel_materials)

    assert len(overlap) >= 10, f"Expected >= 10 overlapping materials, found {len(overlap)}: {overlap}"
