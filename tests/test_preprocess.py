"""Unit and integration tests for data preprocessing and harmonization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
CYTO_CLEAN_PATH = PROCESSED_DIR / "cytotox_clean.csv"
SELTOX_CLEAN_PATH = PROCESSED_DIR / "seltox_clean.csv"
REPORT_PATH = Path("docs/PREPROCESSING_REPORT.md")

CANONICAL_SHAPES = {"spherical", "rod", "nanotube", "cubic", "other", "unknown"}

CYTO_EXPECTED_COLS = [
    "core_material",
    "particle_size_nm",
    "zeta_potential_mv",
    "surface_coating",
    "is_coated",
    "shape",
    "cell_type",
    "concentration",
    "time_hr",
    "test_assay",
    "doi",
    "viability_%",
    "is_biocompatible",
    "is_shared_material",
]

SELTOX_EXPECTED_COLS = [
    "core_material",
    "particle_size_nm",
    "zeta_potential_mv",
    "is_coated",
    "shape",
    "bacteria",
    "mdr",
    "time_hr",
    "method",
    "doi",
    "mic_ug_ml",
    "log10_mic",
    "is_antibacterial",
    "is_shared_material",
]


@pytest.fixture(scope="module")
def cyto_df():
    assert CYTO_CLEAN_PATH.exists(), f"Missing {CYTO_CLEAN_PATH}"
    df = pd.read_csv(CYTO_CLEAN_PATH)
    assert len(df) > 0, "Cytotox clean dataset is empty"
    return df


@pytest.fixture(scope="module")
def sel_df():
    assert SELTOX_CLEAN_PATH.exists(), f"Missing {SELTOX_CLEAN_PATH}"
    df = pd.read_csv(SELTOX_CLEAN_PATH)
    assert len(df) > 0, "SelTox clean dataset is empty"
    return df


def test_cytotox_schema_and_columns(cyto_df):
    for col in CYTO_EXPECTED_COLS:
        assert col in cyto_df.columns, f"Column {col} missing from Cytotox clean"


def test_cytotox_target_bounds_and_filtering(cyto_df):
    # Null particle size and viability should have been dropped
    assert cyto_df["particle_size_nm"].isna().sum() == 0
    assert cyto_df["viability_%"].isna().sum() == 0

    # Viability clipped to [0, 120]
    assert cyto_df["viability_%"].min() >= 0.0
    assert cyto_df["viability_%"].max() <= 120.0

    # Binary target alignment
    expected_bio = (cyto_df["viability_%"] >= 70.0).astype(int)
    assert (cyto_df["is_biocompatible"] == expected_bio).all()
    assert set(cyto_df["is_biocompatible"].unique()).issubset({0, 1})


def test_cytotox_surface_coating_and_coating_flag(cyto_df):
    uncoated_mask = cyto_df["surface_coating"] == "Uncoated"
    assert (cyto_df.loc[uncoated_mask, "is_coated"] == 0).all()
    assert (cyto_df.loc[~uncoated_mask, "is_coated"] == 1).all()


def test_cytotox_shapes_canonical(cyto_df):
    shapes_found = set(cyto_df["shape"].unique())
    assert shapes_found.issubset(CANONICAL_SHAPES), f"Uncanonical shapes: {shapes_found - CANONICAL_SHAPES}"


def test_seltox_schema_and_columns(sel_df):
    for col in SELTOX_EXPECTED_COLS:
        assert col in sel_df.columns, f"Column {col} missing from SelTox clean"


def test_seltox_mic_targets(sel_df):
    # MIC must be strictly positive numeric values
    assert (sel_df["mic_ug_ml"] > 0).all()
    assert sel_df["mic_ug_ml"].isna().sum() == 0

    # log10_mic correctness
    expected_log10 = np.log10(sel_df["mic_ug_ml"])
    np.testing.assert_allclose(sel_df["log10_mic"], expected_log10, rtol=1e-5)

    # is_antibacterial threshold at 64.0 µg/mL
    expected_anti = (sel_df["mic_ug_ml"] <= 64.0).astype(int)
    assert (sel_df["is_antibacterial"] == expected_anti).all()
    assert set(sel_df["is_antibacterial"].unique()).issubset({0, 1})


def test_seltox_shapes_canonical(sel_df):
    shapes_found = set(sel_df["shape"].unique())
    assert shapes_found.issubset(CANONICAL_SHAPES), f"Uncanonical shapes: {shapes_found - CANONICAL_SHAPES}"


def test_shared_materials_flags(cyto_df, sel_df):
    assert cyto_df["is_shared_material"].dtype == bool
    assert sel_df["is_shared_material"].dtype == bool
    assert cyto_df["is_shared_material"].sum() > 0
    assert sel_df["is_shared_material"].sum() > 0


def test_report_generated():
    assert REPORT_PATH.exists()
    assert REPORT_PATH.stat().st_size > 500
