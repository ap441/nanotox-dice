#!/usr/bin/env python3
"""
Download script for nanotoxicology datasets from Hugging Face.

Ingests:
  - ai-chem/Cytotox -> data/raw/cytotox_raw.csv
  - ai-chem/SelTox  -> data/raw/seltox_raw.csv

The script is idempotent: it skips downloading if target files exist unless --force is specified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from datasets import DatasetDict, load_dataset

DATASETS_CONFIG: Dict[str, Dict[str, Any]] = {
    "cytotox": {
        "hf_id": "ai-chem/Cytotox",
        "filename": "cytotox_raw.csv",
        "description": "Mammalian cytotoxicity assays for nanomaterials",
    },
    "seltox": {
        "hf_id": "ai-chem/SelTox",
        "filename": "seltox_raw.csv",
        "description": "Bacterial selectivity / antimicrobial efficacy assays",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and export raw nanotox datasets from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target output directory for raw CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing files.",
    )
    return parser.parse_args()


def download_and_save_dataset(
    key: str, config: Dict[str, Any], output_dir: Path, force: bool = False
) -> Path:
    target_path = output_dir / config["filename"]

    if target_path.exists() and not force:
        existing_df = pd.read_csv(target_path)
        print(
            f"[SKIPPED] {config['filename']} already exists at {target_path} "
            f"({len(existing_df):,} rows, {len(existing_df.columns)} columns). "
            f"Use --force to re-download."
        )
        return target_path

    print(f"[DOWNLOADING] Fetching '{config['hf_id']}' ({config['description']})...")
    ds = load_dataset(config["hf_id"])

    if isinstance(ds, DatasetDict):
        if "train" in ds:
            df = ds["train"].to_pandas()
        else:
            # If multiple splits exist, concatenate them
            dfs = [split_ds.to_pandas() for split_ds in ds.values()]
            df = pd.concat(dfs, ignore_index=True)
    else:
        df = ds.to_pandas()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)

    print(
        f"[SAVED] Successfully exported to {target_path} "
        f"({len(df):,} rows, {len(df.columns)} columns)."
    )
    print(f"        Columns ({len(df.columns)}): {list(df.columns[:10])}"
          f"{' ...' if len(df.columns) > 10 else ''}\n")
    return target_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Nanotox Safe-by-Design Data Ingestion Pipeline")
    print(f"Target Directory: {args.output_dir.resolve()}")
    print(f"Force Re-download: {args.force}")
    print("=" * 70)

    for key, config in DATASETS_CONFIG.items():
        try:
            download_and_save_dataset(key, config, args.output_dir, force=args.force)
        except Exception as exc:
            print(f"[ERROR] Failed to download {config['hf_id']}: {exc}", file=sys.stderr)
            return 1

    print("Data download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
