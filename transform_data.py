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

import polars as pl
from pathlib import Path
import sys
from typing import Dict, List, Optional
import argparse


class RechnungDataTransformer:
    """Transforms and combines rechnung data from multiple country-specific files."""
    
    def __init__(self):
        self.country_mapping = {
            'F01': 'DE',  # Germany
            'F02': 'FR',  # France
            'F03': 'AT',  # Austria  
            'F04': 'CH'   # Switzerland
        }
        
        # Expected output columns based on the RAG system requirements
        self.target_columns = [
            'NUMMER', 'AUFTRAG_NR', 'RECHNUNG', 'Land', 'SOURCE', 'DATUM',
            'Herkunft', 'Brutto_Umsatz', 'Netto_Umsatz', 'Produkt', 'MENGE',
            'Retouren', 'WG_NAME'
        ]
        
    def load_file(self, file_path: str, country_code: str) -> pl.DataFrame:
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
            
            # Load with polars for better performance
            df = pl.read_csv(
                file_path,
                separator=";",
                encoding="cp850",
                try_parse_dates=True,
                infer_schema_length=50000,  # Increased for better schema inference
                ignore_errors=True,  # Handle parsing errors gracefully
                null_values=["", "E"]  # Treat "E" as null value
            )
            
            print(f"  - Loaded {len(df)} rows")
            print(f"  - Columns: {df.columns}")
            
            # Add country information
            df = df.with_columns(pl.lit(country_code).alias("Land"))
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def standardize_columns(self, df: pl.DataFrame, country_code: str) -> pl.DataFrame:
        """
        Standardize column names and create missing columns.
        
        This method maps various column names from source files to the target schema
        and creates any missing columns with appropriate default values.
        """
        print(f"  - Standardizing columns for {country_code}...")
        
        # Column mapping based on the actual data structure we see
        column_mappings = {
            # Direct mappings from your actual data
            'MEDIACODE': 'SOURCE',
            'HERKUNFT': 'Herkunft', 
            'PREIS': 'Netto_Umsatz',
            'BEZEICHNG': 'Produkt',
            # Keep existing columns that match
            # 'NUMMER': 'NUMMER',  # already correct
            # 'AUFTRAG_NR': 'AUFTRAG_NR',  # already correct
            # 'RECHNUNG': 'RECHNUNG',  # already correct  
            # 'DATUM': 'DATUM',  # already correct
            # 'MENGE': 'MENGE',  # already correct
            # 'WG_NAME': 'WG_NAME',  # already correct
        }
        
        # Apply column mappings
        current_columns = df.columns
        rename_dict = {}
        
        for old_name in current_columns:
            if old_name in column_mappings:
                rename_dict[old_name] = column_mappings[old_name]
        
        if rename_dict:
            df = df.rename(rename_dict)
            print(f"    Renamed columns: {rename_dict}")
        
        # Handle special case for Retouren column - combine return information
        if "RETOUREGRD" in df.columns or "RETOUREART" in df.columns:
            if "RETOUREGRD" in df.columns and "RETOUREART" in df.columns:
                # Combine both return columns
                df = df.with_columns(
                    pl.concat_str([
                        pl.col("RETOUREGRD").cast(pl.Utf8).fill_null(""),
                        pl.lit(" "),
                        pl.col("RETOUREART").cast(pl.Utf8).fill_null("")
                    ]).str.strip_chars().alias("Retouren")
                )
            elif "RETOUREGRD" in df.columns:
                df = df.with_columns(pl.col("RETOUREGRD").cast(pl.Utf8).fill_null("").alias("Retouren"))
            elif "RETOUREART" in df.columns:
                df = df.with_columns(pl.col("RETOUREART").cast(pl.Utf8).fill_null("").alias("Retouren"))
        
        # Ensure all target columns exist
        existing_cols = set(df.columns)
        missing_cols = set(self.target_columns) - existing_cols
        
        if missing_cols:
            print(f"    Creating missing columns: {missing_cols}")
            for col in missing_cols:
                # Create appropriate default values based on column type
                if col in ['Netto_Umsatz', 'Brutto_Umsatz', 'MENGE']:
                    df = df.with_columns(pl.lit(0.0).alias(col))
                elif col == 'Land':
                    df = df.with_columns(pl.lit(country_code).alias(col))
                else:
                    df = df.with_columns(pl.lit("").alias(col))
        
        # Select only target columns
        df = df.select(self.target_columns)
        
        return df
    
    def apply_data_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the same data transformations as in the original loader.py
        """
        print("  - Applying data transformations...")
        
        # Convert and format AUFTRAG_NR (remove .0 and zero-pad to 9 digits)
        df = df.with_columns([
            pl.col("AUFTRAG_NR").cast(pl.Utf8).str.replace(".0", "").str.zfill(9),
            pl.col("NUMMER").cast(pl.Utf8).str.zfill(10)
        ])
        
        # Handle date parsing - check if it's already a date type or needs parsing
        if df["DATUM"].dtype == pl.Date:
            # Already a date, keep as is
            pass
        else:
            # Try to parse as date
            df = df.with_columns(
                pl.col("DATUM").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            )
        
        # Remove rows with invalid dates
        df = df.filter(pl.col("DATUM").is_not_null())
        
        # Create Brutto_Umsatz if it doesn't exist (calculate from Netto + MWST if available)
        if "Brutto_Umsatz" not in df.columns:
            if "MWST" in df.columns:
                # Calculate Brutto = Netto + MWST
                df = df.with_columns(
                    (pl.col("Netto_Umsatz").cast(pl.Float64, strict=False).fill_null(0.0) + 
                     pl.col("MWST").cast(pl.Float64, strict=False).fill_null(0.0)).alias("Brutto_Umsatz")
                )
            else:
                # Use Netto_Umsatz * 1.19 as approximation (19% VAT)
                df = df.with_columns(
                    (pl.col("Netto_Umsatz").cast(pl.Float64, strict=False).fill_null(0.0) * 1.19).alias("Brutto_Umsatz")
                )
        
        # Convert numeric columns and handle errors
        numeric_cols = ["Netto_Umsatz", "MENGE", "Brutto_Umsatz"]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0)
                )
        
        # Handle missing values for string columns
        string_cols = [col for col in self.target_columns if col not in numeric_cols + ["DATUM"]]
        for col in string_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).fill_null("")
                )
        
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
            
            # Load file
            df = self.load_file(file_path, country_code)
            if df is None:
                continue
                
            # Standardize columns
            df = self.standardize_columns(df, country_code)
            
            # Apply transformations
            df = self.apply_data_transformations(df)
            
            combined_dfs.append(df)
            print(f"  - Processed {len(df)} rows for {country_code}")
        
        if not combined_dfs:
            print("Error: No files were successfully processed!")
            return False
        
        # Combine all dataframes
        print("\nCombining all dataframes...")
        combined_df = pl.concat(combined_dfs, how="vertical")
        
        print(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        print(f"Columns: {combined_df.columns}")
        
        # Show country distribution
        country_counts = combined_df.group_by("Land").len().sort("Land")
        print("\nCountry distribution:")
        print(country_counts)
        
        # Save combined file
        print(f"\nSaving to {output_path}...")
        combined_df.write_csv(
            output_path,
            separator=";"
        )
        
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
        help="Directory containing source CSV files (default: /Volumes/MARAL/rechnung)"
    )
    parser.add_argument(
        "--output", 
        default="/Volumes/MARAL/rechnung/rechnung_gesamt_combined.csv",
        help="Output path for combined CSV file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files would be processed without actually processing them"
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
    
    # Create transformer and process files
    transformer = RechnungDataTransformer()
    success = transformer.transform_all_files(args.input_dir, args.output)
    
    if success:
        print(f"\n🎉 Transformation completed successfully!")
        print(f"You can now update config.py to use: {args.output}")
        sys.exit(0)
    else:
        print(f"\n❌ Transformation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
