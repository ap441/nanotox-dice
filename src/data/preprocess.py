#!/usr/bin/env python3
"""
Data preprocessing and harmonization pipeline for Safe-by-Design Nanotoxicology.

Cleans and harmonizes:
  - data/raw/cytotox_raw.csv -> data/processed/cytotox_clean.csv
  - data/raw/seltox_raw.csv  -> data/processed/seltox_clean.csv

Generates audit summary at docs/PREPROCESSING_REPORT.md.
Calibrated against configs/experiment_config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# Standard canonical shape categories
CANONICAL_SHAPES = ["spherical", "rod", "nanotube", "cubic", "other", "unknown"]

# Cell line typography / hyphenation mapping
CELL_LINE_MAP: Dict[str, str] = {
    "SHSY5Y": "SH-SY5Y",
    "SH-SY5Y": "SH-SY5Y",
    "RAW264.7": "RAW 264.7",
    "RAW 264.7": "RAW 264.7",
    "Raw 264.7": "RAW 264.7",
    "Raw264.7": "RAW 264.7",
    "HaCat": "HaCaT",
    "HaCaT": "HaCaT",
    "Caco2": "Caco-2",
    "CACO-2": "Caco-2",
    "Caco-2": "Caco-2",
    "MDAMB231": "MDA-MB-231",
    "MDA-MB 231": "MDA-MB-231",
    "MDA-MB-231": "MDA-MB-231",
    "MCF7": "MCF-7",
    "MCF-7": "MCF-7",
    "HCT116": "HCT-116",
    "HCT 116": "HCT-116",
    "HCT-116": "HCT-116",
    "HEK293": "HEK-293",
    "HEK 293": "HEK-293",
    "HEK-293": "HEK-293",
    "HEK293T": "HEK-293T",
    "HEK 293T": "HEK-293T",
    "HEK-293T": "HEK-293T",
    "THP1": "THP-1",
    "THP 1": "THP-1",
    "THP-1": "THP-1",
    "NIH3T3": "NIH 3T3",
    "NIH-3T3": "NIH 3T3",
    "NIH 3T3": "NIH 3T3",
    "L929": "L-929",
    "L-929": "L-929",
    "MG63": "MG-63",
    "MG 63": "MG-63",
    "MG-63": "MG-63",
    "SKOV3": "SKOV-3",
    "SK-OV-3": "SKOV-3",
    "SKOV-3": "SKOV-3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and harmonize raw nanotoxicology datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
        help="Path to experiment config YAML (default: configs/experiment_config.yaml).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save cleaned CSVs (default: data/processed).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/PREPROCESSING_REPORT.md"),
        help="Path to output markdown report (default: docs/PREPROCESSING_REPORT.md).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_shape(val: Any) -> str:
    """Normalize morphological shape into canonical categories."""
    if pd.isna(val) or val is None:
        return "unknown"
    s = str(val).strip().lower()
    if s in ["", "nan", "none", "unknown", "unspecified"]:
        return "unknown"

    # Multi-morphology mixtures are mapped to 'other'
    if "/" in s or " and " in s:
        return "other"

    # Specific morphologies
    if any(k in s for k in ["nanotube", "tube", "tubular"]):
        return "nanotube"
    if any(k in s for k in ["nanorod", "rod", "cylind", "spindle"]):
        return "rod"
    if any(
        k in s
        for k in [
            "sphere",
            "spherical",
            "round",
            "spheroid",
            "quasi-spherical",
            "pseudo-spherical",
            "ellipsoid",
            "oval",
            "circular",
        ]
    ):
        return "spherical"
    if any(k in s for k in ["cube", "cubic", "cuboid", "rectang", "square", "tetragon"]):
        return "cubic"

    return "other"


def normalize_cell_line(val: Any) -> str:
    """Normalize cell line designations."""
    if pd.isna(val) or val is None:
        return "unknown"
    clean_val = str(val).strip()
    return CELL_LINE_MAP.get(clean_val, clean_val)


def process_cytotox(
    raw_df: pd.DataFrame,
    shared_materials: Set[str],
    viability_threshold: float = 70.0,
    viability_clip_max: float = 120.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and harmonize the Cytotox cohort."""
    stats: Dict[str, Any] = {
        "initial_rows": len(raw_df),
        "initial_cols": len(raw_df.columns),
    }

    df = raw_df.copy()

    # 1. Renaming and fallbacks
    df["core_material"] = df["material"].astype(str).str.strip()
    df["particle_size_nm"] = df["hydrodynamic_nm"].fillna(df["size_in_medium_nm"])
    df["zeta_potential_mv"] = df["potential_mv"].fillna(df["zeta_in_medium_mv"])

    # 2. Surface coating harmonization
    raw_coatings = df["coat_functional_group"].fillna("Uncoated").astype(str).str.strip()
    # Find top 10 non-uncoated ligands
    non_uncoated = raw_coatings[raw_coatings != "Uncoated"]
    top_10_ligands = non_uncoated.value_counts().head(10).index.tolist()
    stats["top_10_ligands"] = top_10_ligands

    def map_coating(c: str) -> str:
        if c == "Uncoated":
            return "Uncoated"
        if c in top_10_ligands:
            return c
        return "Other"

    df["surface_coating"] = raw_coatings.apply(map_coating)
    df["is_coated"] = (df["surface_coating"] != "Uncoated").astype(int)

    # 3. Shape normalization
    df["shape"] = df["shape"].apply(normalize_shape)

    # 4. Cell line normalization
    df["cell_type"] = df["cell_type"].apply(normalize_cell_line)

    # 5. Core contextual columns
    df["test_assay"] = df["test"].astype(str).str.strip()
    # Keep concentration and time_hr as-is (do not impute!)

    # 6. Filter null target / size
    null_viability = df["viability_%"].isna()
    null_size = df["particle_size_nm"].isna()
    drop_mask = null_viability | null_size

    stats["dropped_null_viability"] = int(null_viability.sum())
    stats["dropped_null_size"] = int(null_size.sum())
    stats["total_dropped"] = int(drop_mask.sum())

    df = df[~drop_mask].copy()

    # 7. Viability clipping & binary target
    clipped_high = (df["viability_%"] > viability_clip_max).sum()
    clipped_low = (df["viability_%"] < 0.0).sum()
    stats["clipped_high_count"] = int(clipped_high)
    stats["clipped_low_count"] = int(clipped_low)

    df["viability_%"] = df["viability_%"].clip(lower=0.0, upper=viability_clip_max)
    df["is_biocompatible"] = (df["viability_%"] >= viability_threshold).astype(int)

    # 8. Cross-study flag
    df["is_shared_material"] = df["core_material"].isin(shared_materials)

    # Select final columns
    final_cols = [
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
    df = df[final_cols].copy()
    stats["final_rows"] = len(df)
    stats["final_cols"] = len(df.columns)
    stats["biocompatible_count"] = int(df["is_biocompatible"].sum())
    stats["biocompatible_pct"] = (df["is_biocompatible"].sum() / len(df)) * 100

    return df, stats


def process_seltox(
    raw_df: pd.DataFrame,
    shared_materials: Set[str],
    mic_active_threshold: float = 64.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and harmonize the SelTox cohort."""
    stats: Dict[str, Any] = {
        "initial_rows": len(raw_df),
        "initial_cols": len(raw_df.columns),
    }

    df = raw_df.copy()

    # 1. Filter to MIC/MBC rows with non-null mic_np_µg_ml
    valid_method = df["method"].astype(str).str.upper().isin(["MIC", "MBC"])
    has_mic = df["mic_np_µg_ml"].notna()
    method_filtered = df[valid_method & has_mic].copy()

    stats["filtered_method_mic_mbc_rows"] = len(method_filtered)
    stats["dropped_non_mic_mbc_or_null"] = len(df) - len(method_filtered)

    # 2. Parse mic_np_µg_ml
    unparseable_list: List[str] = []
    zero_or_neg_count = 0
    clean_mics: List[float | None] = []

    for raw_val in method_filtered["mic_np_µg_ml"]:
        val_str = str(raw_val).strip()
        try:
            val_f = float(val_str)
            if val_f <= 0:
                zero_or_neg_count += 1
                clean_mics.append(None)
            else:
                clean_mics.append(val_f)
        except ValueError:
            unparseable_list.append(val_str)
            clean_mics.append(None)

    stats["unparseable_mic_strings"] = unparseable_list
    stats["unparseable_mic_count"] = len(unparseable_list)
    stats["zero_or_neg_mic_count"] = zero_or_neg_count

    method_filtered["mic_ug_ml"] = clean_mics
    valid_mic_df = method_filtered[method_filtered["mic_ug_ml"].notna()].copy()
    valid_mic_df["mic_ug_ml"] = valid_mic_df["mic_ug_ml"].astype(float)

    stats["retained_after_mic_parsing"] = len(valid_mic_df)

    # 3. Renaming and fallbacks
    valid_mic_df["core_material"] = valid_mic_df["np"].astype(str).str.strip()
    valid_mic_df["particle_size_nm"] = valid_mic_df["np_size_avg_nm"].fillna(
        valid_mic_df["hydrodynamic_diameter_nm"]
    )
    valid_mic_df["zeta_potential_mv"] = valid_mic_df["zeta_potential_mV"]
    valid_mic_df["is_coated"] = valid_mic_df["coating"].astype(int)
    valid_mic_df["shape"] = valid_mic_df["shape"].apply(normalize_shape)
    valid_mic_df["time_hr"] = valid_mic_df["time_set_hours"]
    valid_mic_df["bacteria"] = valid_mic_df["bacteria"].astype(str).str.strip()
    valid_mic_df["mdr"] = valid_mic_df["mdr"].astype(str).str.strip()
    valid_mic_df["method"] = valid_mic_df["method"].astype(str).str.strip()

    # 4. Target creation
    valid_mic_df["log10_mic"] = np.log10(valid_mic_df["mic_ug_ml"])
    valid_mic_df["is_antibacterial"] = (
        valid_mic_df["mic_ug_ml"] <= mic_active_threshold
    ).astype(int)

    # 5. Cross-study flag
    valid_mic_df["is_shared_material"] = valid_mic_df["core_material"].isin(shared_materials)

    final_cols = [
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
    final_df = valid_mic_df[final_cols].copy()
    stats["final_rows"] = len(final_df)
    stats["final_cols"] = len(final_df.columns)
    stats["antibacterial_count"] = int(final_df["is_antibacterial"].sum())
    stats["antibacterial_pct"] = (final_df["is_antibacterial"].sum() / len(final_df)) * 100

    return final_df, stats


def generate_preprocessing_report(
    cyto_df: pd.DataFrame,
    cyto_stats: Dict[str, Any],
    sel_df: pd.DataFrame,
    sel_stats: Dict[str, Any],
    report_path: Path,
) -> None:
    """Generate comprehensive Markdown preprocessing report."""
    lines: List[str] = [
        "# Nanotoxicology Preprocessing & Cohort Harmonization Report\n",
        "**Pipeline Component:** `src/data/preprocess.py`  ",
        "**Configuration:** `configs/experiment_config.yaml`\n",
        "---\n",
        "## 1. Attrition Summary\n",
        "| Dataset | Initial Rows | Dropped / Filtered | Final Clean Rows | Retention % | Primary Target |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Cytotox (Mammalian)** | {cyto_stats['initial_rows']:,} | {cyto_stats['total_dropped']:,} | {cyto_stats['final_rows']:,} | {(cyto_stats['final_rows']/cyto_stats['initial_rows'])*100:.2f}% | `viability_%`, `is_biocompatible` |",
        f"| **SelTox (Bacterial)** | {sel_stats['initial_rows']:,} | {sel_stats['initial_rows'] - sel_stats['final_rows']:,} | {sel_stats['final_rows']:,} | {(sel_stats['final_rows']/sel_stats['initial_rows'])*100:.2f}% | `mic_ug_ml`, `log10_mic`, `is_antibacterial` |",
        "\n",
        "### Attrition Breakdown Details\n",
        f"- **Cytotox Filtering**:\n"
        f"  - Dropped null `viability_%`: {cyto_stats['dropped_null_viability']} rows\n"
        f"  - Dropped null `particle_size_nm`: {cyto_stats['dropped_null_size']} rows\n"
        f"  - Net rows dropped: {cyto_stats['total_dropped']} rows\n",
        f"- **SelTox Filtering**:\n"
        f"  - Filtered out non-MIC/MBC or null assay rows: {sel_stats['dropped_non_mic_mbc_or_null']} rows\n"
        f"  - Filtered out non-numeric/bounded MIC strings ({sel_stats['unparseable_mic_count']}): {sel_stats['unparseable_mic_strings']}\n"
        f"  - Filtered out non-positive ($\le 0$) MIC values: {sel_stats['zero_or_neg_mic_count']} rows\n"
        f"  - Net valid assay rows retained: {sel_stats['final_rows']:,} rows\n",
        "---\n",
        "## 2. Target Distributions & Class Balance\n",
        "### Cytotox: Host Cell Safety",
        f"- **Biocompatible (`viability_% >= 70.0%`)**: {cyto_stats['biocompatible_count']:,} ({cyto_stats['biocompatible_pct']:.2f}%)",
        f"- **Cytotoxic (`viability_% < 70.0%`)**: {cyto_stats['final_rows'] - cyto_stats['biocompatible_count']:,} ({100 - cyto_stats['biocompatible_pct']:.2f}%)",
        f"- **Viability Clipping**: Clipped {cyto_stats['clipped_high_count']} rows with viability $> 120.0\\%$ down to 120.0% (assay noise/hormesis). 0 rows $< 0.0\\%$.\n",
        "### SelTox: Antimicrobial Efficacy",
        f"- **Antibacterial Active (`MIC <= 64.0 µg/mL`)**: {sel_stats['antibacterial_count']:,} ({sel_stats['antibacterial_pct']:.2f}%)",
        f"- **Antibacterial Inactive / Resistant (`MIC > 64.0 µg/mL`)**: {sel_stats['final_rows'] - sel_stats['antibacterial_count']:,} ({100 - sel_stats['antibacterial_pct']:.2f}%)",
        f"- **log10(MIC) Quartiles**: Min={sel_df['log10_mic'].min():.2f}, 25%={sel_df['log10_mic'].quantile(0.25):.2f}, Median={sel_df['log10_mic'].median():.2f}, 75%={sel_df['log10_mic'].quantile(0.75):.2f}, Max={sel_df['log10_mic'].max():.2f}\n",
        "---\n",
        "## 3. Feature Missingness in Cleaned Datasets\n",
        "> [!NOTE]\n"
        "> Missing numeric features (such as `zeta_potential_mv` and `concentration`) are intentionally **not imputed** globally here. Imputation is deferred to cross-validated model pipelines to avoid train-test data leakage.\n",
        "### Cytotox Clean Features Missingness\n",
        "| Column | Non-Null Count | Null Count | Null % | Data Role |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for col in cyto_df.columns:
        null_c = int(cyto_df[col].isna().sum())
        non_null_c = len(cyto_df) - null_c
        null_pct = (null_c / len(cyto_df)) * 100
        role = "Target" if "viability" in col or "biocompatible" in col else (
            "Flag" if "shared" in col or "coated" in col else "Feature"
        )
        lines.append(f"| `{col}` | {non_null_c:,} | {null_c:,} | {null_pct:.2f}% | {role} |")

    lines.append("\n### SelTox Clean Features Missingness\n")
    lines.append("| Column | Non-Null Count | Null Count | Null % | Data Role |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for col in sel_df.columns:
        null_c = int(sel_df[col].isna().sum())
        non_null_c = len(sel_df) - null_c
        null_pct = (null_c / len(sel_df)) * 100
        role = "Target" if "mic" in col or "antibacterial" in col else (
            "Flag" if "shared" in col or "coated" in col else "Feature"
        )
        lines.append(f"| `{col}` | {non_null_c:,} | {null_c:,} | {null_pct:.2f}% | {role} |")

    lines.append("\n---\n")
    lines.append("## 4. Shared Core Materials Coverage\n")
    cyto_shared = int(cyto_df["is_shared_material"].sum())
    sel_shared = int(sel_df["is_shared_material"].sum())
    lines.append(
        f"- **Cytotox Records with Shared Material**: {cyto_shared:,} / {len(cyto_df):,} ({(cyto_shared / len(cyto_df)) * 100:.2f}%)\n"
        f"- **SelTox Records with Shared Material**: {sel_shared:,} / {len(sel_df):,} ({(sel_shared / len(sel_df)) * 100:.2f}%)\n"
    )

    lines.append("### Shape Distribution Across Both Datasets\n")
    lines.append("| Canonical Shape | Cytotox Count (Pct) | SelTox Count (Pct) |")
    lines.append("| :--- | :--- | :--- |")
    for sh in CANONICAL_SHAPES:
        c_cnt = int((cyto_df["shape"] == sh).sum())
        s_cnt = int((sel_df["shape"] == sh).sum())
        c_p = (c_cnt / len(cyto_df)) * 100
        s_p = (s_cnt / len(sel_df)) * 100
        lines.append(f"| `{sh}` | {c_cnt:,} ({c_p:.1f}%) | {s_cnt:,} ({s_p:.1f}%) |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Preprocessing summary generated at {report_path.resolve()}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    shared_materials = set(config.get("materials", {}).get("shared_materials", []))
    target_cfg = config.get("target_thresholds", {})
    viability_min = target_cfg.get("viability_biocompatible_min", 70.0)
    viability_clip_max = target_cfg.get("viability_clip_max", 120.0)
    mic_max = target_cfg.get("mic_active_max_ug_ml", 64.0)

    print("=" * 75)
    print("Nanotox Data Preprocessing & Harmonization Pipeline")
    print(f"Config: {args.config.resolve()}")
    print(f"Raw Dir: {args.raw_dir.resolve()}")
    print(f"Processed Dir: {args.processed_dir.resolve()}")
    print(f"Shared Materials Count: {len(shared_materials)}")
    print("=" * 75)

    # 1. Process Cytotox
    cyto_raw_path = args.raw_dir / "cytotox_raw.csv"
    if not cyto_raw_path.exists():
        print(f"[ERROR] Missing {cyto_raw_path}", file=sys.stderr)
        return 1

    print(f"\n[PROCESSING] Cleaning Cytotox cohort from {cyto_raw_path.name}...")
    cyto_raw = pd.read_csv(cyto_raw_path, low_memory=False)
    cyto_clean, cyto_stats = process_cytotox(
        cyto_raw,
        shared_materials=shared_materials,
        viability_threshold=viability_min,
        viability_clip_max=viability_clip_max,
    )

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    cyto_out_path = args.processed_dir / "cytotox_clean.csv"
    cyto_clean.to_csv(cyto_out_path, index=False)
    print(
        f"[SAVED] {cyto_out_path.name}: {len(cyto_clean):,} rows, {len(cyto_clean.columns)} columns "
        f"({cyto_stats['biocompatible_pct']:.1f}% biocompatible)."
    )

    # 2. Process SelTox
    sel_raw_path = args.raw_dir / "seltox_raw.csv"
    if not sel_raw_path.exists():
        print(f"[ERROR] Missing {sel_raw_path}", file=sys.stderr)
        return 1

    print(f"\n[PROCESSING] Cleaning SelTox cohort from {sel_raw_path.name}...")
    sel_raw = pd.read_csv(sel_raw_path, low_memory=False)
    sel_clean, sel_stats = process_seltox(
        sel_raw,
        shared_materials=shared_materials,
        mic_active_threshold=mic_max,
    )

    sel_out_path = args.processed_dir / "seltox_clean.csv"
    sel_clean.to_csv(sel_out_path, index=False)
    print(
        f"[SAVED] {sel_out_path.name}: {len(sel_clean):,} rows, {len(sel_clean.columns)} columns "
        f"({sel_stats['antibacterial_pct']:.1f}% antibacterial active)."
    )

    # 3. Generate Report
    generate_preprocessing_report(
        cyto_clean, cyto_stats, sel_clean, sel_stats, args.report_path
    )

    print("\n[SUCCESS] Preprocessing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
