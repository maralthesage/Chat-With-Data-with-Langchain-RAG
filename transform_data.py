#!/usr/bin/env python3
"""
Data Transformation Script for CSV RAG Analyst
==============================================

This script transforms data from 4 separate country-specific CSV files
into the unified format required by the RAG system.

Input files:
- /rechnung/rechnung_F01.csv (DE - Germany)
- /rechnung/rechnung_F02.csv (FR - France)
- /rechnung/rechnung_F03.csv (AT - Austria)
- /rechnung/rechnung_F04.csv (CH - Switzerland)

Output: Combined CSV file with standardized columns for RAG analysis
"""

from pathlib import Path
import sys
from typing import Optional
import argparse

import pandas as pd

from rag_engine.schema import DATE_COLUMN, NUMERIC_COLUMNS, PRODUCT_ID_COLUMN, TARGET_COLUMNS


class RechnungDataTransformer:
    """Transforms and combines rechnung data from multiple country-specific files."""

    def __init__(self):
        self.country_mapping = {
            "F01": "DE",
            "F02": "FR",
            "F03": "AT",
            "F04": "CH",
        }

        self.target_columns = list(TARGET_COLUMNS)

    def load_file(self, file_path: str, country_code: str) -> Optional[pd.DataFrame]:
        """
        Load and preprocess a single CSV file.

        Args:
            file_path: Path to the CSV file
            country_code: Country code (DE, FR, AT, CH)

        Returns:
            Preprocessed DataFrame
        """
        try:
            print(f"Loading {file_path} for country {country_code}...")

            df = pd.read_csv(
                file_path,
                sep=";",
                encoding="cp850",
                na_values=["", "E"],
                keep_default_na=True,
                low_memory=False,
                on_bad_lines="skip",
            )

            print(f"  - Loaded {len(df)} rows")
            print(f"  - Columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def standardize_columns(self, df: pd.DataFrame, country_code: str) -> pd.DataFrame:
        """
        Keep only the columns used by the RAG and create missing ones if needed.
        """
        print(f"  - Standardizing columns for {country_code}...")

        existing_cols = set(df.columns)
        missing_cols = set(self.target_columns) - existing_cols

        if missing_cols:
            print(f"    Creating missing columns: {missing_cols}")
            for col in missing_cols:
                if col in NUMERIC_COLUMNS:
                    df[col] = 0.0
                else:
                    df[col] = ""

        return df[self.target_columns].copy()

    def apply_data_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same data transformations as in the original loader.py."""
        print("  - Applying data transformations...")

        df = df.copy()

        df["AUFTRAG_NR"] = df["AUFTRAG_NR"].astype(str).str.replace(".0", "", regex=False).str.strip()
        df["NUMMER"] = df["NUMMER"].astype(str).str.replace(".0", "", regex=False).str.strip()
        if PRODUCT_ID_COLUMN in df.columns:
            df[PRODUCT_ID_COLUMN] = (
                df[PRODUCT_ID_COLUMN]
                .fillna("")
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.strip()
            )

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.dropna(subset=[DATE_COLUMN])

        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        string_cols = [col for col in self.target_columns if col not in NUMERIC_COLUMNS + [DATE_COLUMN]]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    def transform_all_files(self, base_path: str, output_path: str) -> bool:
        """
        Transform all country files and combine them into a single output file.

        Args:
            base_path: Base directory containing the source files
            output_path: Path for the combined output file

        Returns:
            True if successful, False otherwise
        """
        combined_dfs = []

        for file_suffix, country_code in self.country_mapping.items():
            file_path = f"{base_path}/rechnung_{file_suffix}.csv"

            if not Path(file_path).exists():
                print(f"Warning: File {file_path} not found, skipping...")
                continue

            df = self.load_file(file_path, country_code)
            if df is None:
                continue

            df = self.standardize_columns(df, country_code)
            df = self.apply_data_transformations(df)

            combined_dfs.append(df)
            print(f"  - Processed {len(df)} rows for {country_code}")

        if not combined_dfs:
            print("Error: No files were successfully processed!")
            return False

        print("\nCombining all dataframes...")
        combined_df = pd.concat(combined_dfs, ignore_index=True)

        print(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        print(f"Columns: {list(combined_df.columns)}")

        print(f"\nSaving to {output_path}...")
        combined_df.to_csv(output_path, sep=";", index=False)

        print(f"✅ Successfully created combined file: {output_path}")
        return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Transform and combine rechnung CSV files for RAG analysis"
    )
    parser.add_argument(
        "--input-dir",
        default="/Volumes/MARAL/rechnung",
        help="Directory containing source CSV files (default: /Volumes/MARAL/rechnung)",
    )
    parser.add_argument(
        "--output",
        default="/Volumes/MARAL/rechnung/rechnung_gesamt_combined.csv",
        help="Output path for combined CSV file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files would be processed without actually processing them",
    )

    args = parser.parse_args()

    print("CSV RAG Analyst - Data Transformation Script")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print()

    if args.dry_run:
        print("DRY RUN - Checking input files...")
        transformer = RechnungDataTransformer()
        for file_suffix, country_code in transformer.country_mapping.items():
            file_path = f"{args.input_dir}/rechnung_{file_suffix}.csv"
            exists = "✅" if Path(file_path).exists() else "❌"
            print(f"{exists} {file_path} ({country_code})")
        return

    transformer = RechnungDataTransformer()
    success = transformer.transform_all_files(args.input_dir, args.output)

    if success:
        print("\n🎉 Transformation completed successfully!")
        print(f"You can now update config.py to use: {args.output}")
        sys.exit(0)

    print("\n❌ Transformation failed!")
    sys.exit(1)


if __name__ == "__main__":
    main()
