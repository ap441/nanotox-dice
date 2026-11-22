#!/usr/bin/env python3
"""
Data profiling script for raw nanotoxicology datasets.

Inspects all CSV files in data/raw/ and generates a comprehensive schema audit
and cross-dataset alignment report at docs/DATA_AUDIT.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile raw CSV datasets and generate schema audit report."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("docs/DATA_AUDIT.md"),
        help="Output markdown file path (default: docs/DATA_AUDIT.md).",
    )
    return parser.parse_args()


def format_number(val: Any) -> str:
    """Format floating point and numeric numbers cleanly."""
    if val is None or pd.isna(val):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        if abs(val) < 0.001 and val != 0:
            return f"{val:.4e}"
        return f"{val:.3f}".rstrip("0").rstrip(".") if "." in f"{val:.3f}" else f"{val:.3f}"
    return str(val)


def profile_single_dataset(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate Markdown profiling sections for a single dataset."""
    lines: List[str] = []
    total_rows, total_cols = df.shape

    lines.append(f"## Dataset Audit: `{dataset_name}`\n")
    lines.append(f"- **Total Rows:** {total_rows:,}")
    lines.append(f"- **Total Columns:** {total_cols:,}")
    lines.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB\n")

    # 1. Full Column Schema
    lines.append("### 1. Schema & Data Types\n")
    lines.append("| # | Column Name | Data Type | Non-Null Count | Null Count | Null % |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for idx, col in enumerate(df.columns, 1):
        null_count = int(df[col].isna().sum())
        non_null_count = total_rows - null_count
        null_pct = (null_count / total_rows) * 100
        lines.append(
            f"| {idx} | `{col}` | `{df[col].dtype}` | {non_null_count:,} | {null_count:,} | {null_pct:.2f}% |"
        )
    lines.append("\n")

    # 2. Missing Value Rankings (Sorted Descending)
    lines.append("### 2. Missing Value Distribution (Sorted Descending)\n")
    missing_series = df.isna().sum()
    missing_pct_series = (missing_series / total_rows) * 100
    missing_df = (
        pd.DataFrame({"missing_count": missing_series, "missing_pct": missing_pct_series})
        .sort_values(by="missing_count", ascending=False)
    )

    lines.append("| Column Name | Missing Count | Missing % | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for col, row in missing_df.iterrows():
        status = "Complete (0%)" if row["missing_count"] == 0 else (
            "Critical (>50%)" if row["missing_pct"] > 50 else (
                "Moderate (20-50%)" if row["missing_pct"] >= 20 else "Low (<20%)"
            )
        )
        lines.append(f"| `{col}` | {int(row['missing_count']):,} | {row['missing_pct']:.2f}% | {status} |")
    lines.append("\n")

    # 3. Numeric Summary Statistics (min, 5th, median, 95th, max)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines.append("### 3. Numeric Features Summary Statistics\n")
    if not num_cols:
        lines.append("*No numeric columns detected.*\n")
    else:
        lines.append("| Numeric Column | Count | Min | 5th Percentile | Median | 95th Percentile | Max | Mean | Std |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for col in num_cols:
            clean_s = df[col].dropna()
            if len(clean_s) == 0:
                lines.append(f"| `{col}` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            c_min = clean_s.min()
            p05 = np.percentile(clean_s, 5)
            c_med = clean_s.median()
            p95 = np.percentile(clean_s, 95)
            c_max = clean_s.max()
            c_mean = clean_s.mean()
            c_std = clean_s.std()
            lines.append(
                f"| `{col}` | {len(clean_s):,} | {format_number(c_min)} | {format_number(p05)} | "
                f"{format_number(c_med)} | {format_number(p95)} | {format_number(c_max)} | "
                f"{format_number(c_mean)} | {format_number(c_std)} |"
            )
        lines.append("\n")

    # 4. Categorical & String Features Distribution
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    lines.append("### 4. Categorical & String Features Cardinality & Top Categories\n")
    if not cat_cols:
        lines.append("*No string/categorical columns detected.*\n")
    else:
        lines.append("| Column Name | Unique Count | Top 10 Most Frequent Categories (Count, %) |")
        lines.append("| :--- | :--- | :--- |")
        for col in cat_cols:
            val_counts = df[col].dropna().value_counts()
            n_unique = int(df[col].nunique(dropna=True))
            top10 = val_counts.head(10)
            top_strs: List[str] = []
            for val, cnt in top10.items():
                pct = (cnt / total_rows) * 100
                top_strs.append(f"`{str(val)[:30]}`: {cnt:,} ({pct:.1f}%)")
            top_repr = "<br>".join(top_strs) if top_strs else "*All NaN*"
            lines.append(f"| `{col}` | {n_unique:,} | {top_repr} |")
        lines.append("\n")

    return "\n".join(lines)


def generate_cross_dataset_alignment(df_cyto: pd.DataFrame, df_sel: pd.DataFrame) -> str:
    """Generate in-depth cross-dataset alignment analysis."""
    lines: List[str] = []
    lines.append("## Cross-Dataset Alignment Analysis\n")
    lines.append(
        "This section maps and contrasts the schema and semantic domains of "
        "`cytotox_raw.csv` (mammalian cytotoxicity) and `seltox_raw.csv` (antimicrobial efficacy).\n"
    )

    # Feature Mapping Table
    lines.append("### 1. Core Feature & Target Mapping\n")
    lines.append(
        "| Canonical Role | Concept | Cytotox Column(s) | SelTox Column(s) | Alignment Status & Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| **Descriptor** | Core Material | `material` | `np` | **High alignment:** Both record chemical formulas (e.g., $Ag$, $ZnO$, $CuO$). |\n"
        "| **Descriptor** | Particle Size ($nm$) | `core_nm`, `hydrodynamic_nm`, `size_in_medium_nm` | `np_size_avg_nm`, `np_size_min_nm`, `np_size_max_nm`, `hydrodynamic_diameter_nm` | **Harmonizable:** Average TEM core size aligns with `core_nm` and `np_size_avg_nm`; DLS size aligns with `hydrodynamic_nm` and `hydrodynamic_diameter_nm`. |\n"
        "| **Descriptor** | Zeta Potential ($mV$) | `potential_mv`, `zeta_in_medium_mv`, `surface_charge` | `zeta_potential_mV` | **Harmonizable:** Continuous surface potential recorded in both; medium-specific measurements available in Cytotox. |\n"
        "| **Descriptor** | Surface Coating | `coat_functional_group` | `coating` | **Divergence:** Cytotox contains specific chemical ligand names (Citrate, PEG, PVP); SelTox records binary indicator ($0$ = uncoated, $1$ = coated). |\n"
        "| **Descriptor** | Shape / Morphology | `shape` | `shape` | **Harmonizable:** Both contain morphology strings (e.g., Sphere/spherical, Rod/rod-shaped) requiring standard casing and normalization. |\n"
        "| **Target** | Mammalian Viability | `viability_%` | *None* | **Cytotox-specific:** Primary objective target for host cell safety (continuous %). |\n"
        "| **Target** | Antimicrobial Efficacy | *None* | `mic_np_µg_ml`, `zoi_np_mm` | **SelTox-specific:** Primary objective target for bacterial suppression (MIC in $\\mu g/mL$). |\n"
        "| **Context** | Biological Strain | `cell_type`, `cell_source`, `cell_tissue` | `bacteria`, `strain`, `mdr` | **Distinct domains:** Host cell line models vs. bacterial pathogen strains and MDR resistance classifications. |\n"
        "| **Context** | Exposure / Dose | `concentration`, `time_hr` | `concentration`, `time_set_hours` | **Aligned:** Standard exposure duration and applied concentration metrics. |\n"
    )
    lines.append("\n")

    # Core Materials Overlap Analysis
    cyto_mats = set(df_cyto["material"].dropna().astype(str).str.strip())
    sel_mats = set(df_sel["np"].dropna().astype(str).str.strip())
    shared_mats = sorted(list(cyto_mats.intersection(sel_mats)))
    cyto_only = sorted(list(cyto_mats - sel_mats))
    sel_only = sorted(list(sel_mats - cyto_mats))

    cyto_shared_count = int(df_cyto["material"].isin(shared_mats).sum())
    cyto_shared_pct = (cyto_shared_count / len(df_cyto)) * 100
    sel_shared_count = int(df_sel["np"].isin(shared_mats).sum())
    sel_shared_pct = (sel_shared_count / len(df_sel)) * 100

    lines.append("### 2. Core Material Overlap\n")
    lines.append(f"- **Shared Core Materials Count:** {len(shared_mats)}")
    lines.append(f"- **Cytotox Records Covered by Shared Materials:** {cyto_shared_count:,} / {len(df_cyto):,} ({cyto_shared_pct:.1f}%)")
    lines.append(f"- **SelTox Records Covered by Shared Materials:** {sel_shared_count:,} / {len(df_sel):,} ({sel_shared_pct:.1f}%)\n")

    lines.append("| Shared Material | Cytotox Record Count (Rank) | SelTox Record Count (Rank) | Dual-Target Feasibility |")
    lines.append("| :--- | :--- | :--- | :--- |")

    cyto_counts = df_cyto["material"].value_counts()
    sel_counts = df_sel["np"].value_counts()

    # Sort shared materials by combined record frequency
    shared_mats_sorted = sorted(
        shared_mats,
        key=lambda m: (cyto_counts.get(m, 0) + sel_counts.get(m, 0)),
        reverse=True,
    )

    for m in shared_mats_sorted:
        c_cnt = cyto_counts.get(m, 0)
        s_cnt = sel_counts.get(m, 0)
        feasibility = "High Priority (Abundant data)" if (c_cnt >= 100 and s_cnt >= 50) else (
            "Viable (Moderate data)" if (c_cnt >= 20 and s_cnt >= 10) else "Sparse"
        )
        lines.append(f"| **{m}** | {c_cnt:,} | {s_cnt:,} | {feasibility} |")
    lines.append("\n")

    lines.append(f"- **Materials unique to Cytotox ({len(cyto_only)}):** {', '.join(f'`{m}`' for m in cyto_only)}")
    lines.append(f"- **Materials unique to SelTox ({len(sel_only)}):** {', '.join(f'`{m}`' for m in sel_only)}\n")

    # Surface Coating Analysis & Divergence
    lines.append("### 3. Surface Coating Representation & Divergence\n")
    cyto_coat_counts = df_cyto["coat_functional_group"].value_counts(dropna=False)
    sel_coat_counts = df_sel["coating"].value_counts(dropna=False)

    lines.append(
        "A critical finding from this audit is the differing schema representations of nanoparticle surface coating:\n"
    )
    lines.append(
        f"1. **`cytotox_raw.csv` (`coat_functional_group`)**:\n"
        f"   - **{df_cyto['coat_functional_group'].nunique(dropna=True)} distinct coating ligands** are cataloged.\n"
        f"   - **Missing / Uncoated:** {int(df_cyto['coat_functional_group'].isna().sum()):,} rows "
        f"({(df_cyto['coat_functional_group'].isna().sum() / len(df_cyto)) * 100:.1f}%) represent pristine or uncoated nanoparticles.\n"
        f"   - Top coatings include: `Citrate` ({cyto_coat_counts.get('Citrate', 0):,}), `PEG` ({cyto_coat_counts.get('PEG', 0):,}), "
        f"`PVP` ({cyto_coat_counts.get('PVP', 0):,}), `PEI` ({cyto_coat_counts.get('PEI', 0):,}), `Chitosan` ({cyto_coat_counts.get('Chitosan', 0):,}).\n"
    )
    lines.append(
        f"2. **`seltox_raw.csv` (`coating`)**:\n"
        f"   - Binary representation: `0` = uncoated ({sel_coat_counts.get(0, 0):,} records, {sel_coat_counts.get(0, 0)/len(df_sel)*100:.1f}%), "
        f"`1` = coated ({sel_coat_counts.get(1, 0):,} records, {sel_coat_counts.get(1, 0)/len(df_sel)*100:.1f}%).\n"
        f"   - Synthesis metadata columns (`solvent_for_extract`, `np_synthesis`, `method`) provide additional context regarding green/botanical extract coatings.\n"
    )
    lines.append(
        "> [!IMPORTANT]\n"
        "> **Harmonization Recommendation:** For unified Safe-by-Design modeling across both targets, the pipeline can:\n"
        "> 1. Maintain a high-level binary `is_coated` feature (1 if coated, 0 if pristine/uncoated) common to both datasets.\n"
        "> 2. For detailed DiCE counterfactual optimization on the mammalian safety model, leverage the granular `coat_functional_group` values (e.g., swapping cytotoxic CTAB/PEI to biocompatible PEG/Citrate).\n"
    )
    lines.append("\n")

    # Shape Alignment
    lines.append("### 4. Shape & Morphology Alignment\n")
    cyto_shapes = df_cyto["shape"].value_counts(dropna=False)
    sel_shapes = df_sel["shape"].value_counts(dropna=False)

    lines.append("| Morphology Group | Cytotox Terminology (Top Counts) | SelTox Terminology (Top Counts) | Standardization Target |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Spherical** | `Sphere` ({cyto_shapes.get('Sphere', 0):,}), `spherical` ({cyto_shapes.get('spherical', 0):,}) | `spherical` ({sel_shapes.get('spherical', 0):,}), `quasi-spherical` ({sel_shapes.get('quasi-spherical', 0):,}) | `spherical` |")
    lines.append(f"| **Rod / Cylindrical** | `Rod` ({cyto_shapes.get('Rod', 0):,}) | `rod-shaped` ({sel_shapes.get('rod-shaped', 0):,}), `nanorods` ({sel_shapes.get('nanorods', 0):,}) | `rod` |")
    lines.append(f"| **Other / Irregular** | `Nanotube` ({cyto_shapes.get('Nanotube', 0):,}), `Star` ({cyto_shapes.get('Star', 0):,}) | `cubic` ({sel_shapes.get('cubic', 0):,}), `hexagonal` ({sel_shapes.get('hexagonal', 0):,}) | Standardize to discrete categories or `other` |")
    lines.append("\n")

    return "\n".join(lines)


def run_profiling(input_dir: Path, output_file: Path) -> None:
    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {input_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[PROFILING] Found {len(csv_files)} CSV files in {input_dir}:")
    for f in csv_files:
        print(f"  - {f.name}")

    report_lines: List[str] = [
        "# Nanotoxicology Dual-Dataset Schema & Profiling Audit\n",
        f"**Audit Generated:** Auto-generated by `src/data/profile_datasets.py`  ",
        f"**Source Data Directory:** `{input_dir}`  ",
        f"**Inspected Files:** {', '.join(f'`{f.name}`' for f in csv_files)}\n",
        "---\n",
    ]

    datasets_dict: Dict[str, pd.DataFrame] = {}
    for csv_file in csv_files:
        print(f"[PROFILING] Reading and profiling {csv_file.name}...")
        df = pd.read_csv(csv_file, low_memory=False)
        datasets_dict[csv_file.name] = df
        dataset_report = profile_single_dataset(df, csv_file.name)
        report_lines.append(dataset_report)
        report_lines.append("\n---\n")

    # Cross-dataset alignment
    if "cytotox_raw.csv" in datasets_dict and "seltox_raw.csv" in datasets_dict:
        print("[PROFILING] Computing cross-dataset alignment between cytotox_raw.csv and seltox_raw.csv...")
        alignment_report = generate_cross_dataset_alignment(
            datasets_dict["cytotox_raw.csv"], datasets_dict["seltox_raw.csv"]
        )
        report_lines.append(alignment_report)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    full_content = "\n".join(report_lines)
    output_file.write_text(full_content, encoding="utf-8")

    print(f"[SUCCESS] Audit report written to {output_file.resolve()} ({len(full_content):,} chars).")


def main() -> int:
    args = parse_args()
    run_profiling(args.input_dir, args.output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
