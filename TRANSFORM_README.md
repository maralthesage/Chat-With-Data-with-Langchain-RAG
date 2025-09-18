# Data Transformation Script

This script (`transform_data.py`) transforms data from 4 separate country-specific CSV files into the unified format required by the CSV RAG Analyst system.

## Quick Start

### 1. Install Dependencies
```bash
# Activate your virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install the new dependency (Polars)
pip install polars==0.20.31

# Or install all requirements
pip install -r requirements.txt
```

### 2. Run the Transformation

**Check what files will be processed (dry run):**
```bash
python transform_data.py --dry-run
```

**Transform all files:**
```bash
python transform_data.py
```

**Custom paths:**
```bash
python transform_data.py --input-dir /path/to/your/files --output /path/to/output.csv
```

### 3. Update Config
After successful transformation, update `config.py`:
```python
rechnung_path = '/Volumes/MARAL/rechnung/rechnung_gesamt_combined.csv'
```

## Input Files Expected

- `/Volumes/MARAL/rechnung/rechnung_F01.csv` (Germany - DE)
- `/Volumes/MARAL/rechnung/rechnung_F02.csv` (France - FR)  
- `/Volumes/MARAL/rechnung/rechnung_F03.csv` (Austria - AT)
- `/Volumes/MARAL/rechnung/rechnung_F04.csv` (Switzerland - CH)

## Output Format

The script creates a combined CSV file with these standardized columns:
- `NUMMER` - Customer ID (zero-padded to 10 digits)
- `AUFTRAG_NR` - Order ID (zero-padded to 9 digits) 
- `RECHNUNG` - Invoice ID
- `Land` - Country code (DE/FR/AT/CH)
- `SOURCE` - Customer acquisition source
- `AUF_ANLAGE` - Order creation date
- `Herkunft` - Order channel (phone/email/internet)
- `Brutto_Umsatz` - Gross revenue
- `Netto_Umsatz` - Net revenue  
- `Produkt` - Product name/description
- `MENGE` - Quantity
- `Retouren` - Return status
- `WG_NAME` - Product category

## Performance

- Uses **Polars** instead of Pandas for ~3-5x faster processing
- Optimized for large datasets
- Memory-efficient streaming operations

## Customization

If your source files have different column names, edit the `column_mappings` dictionary in the `standardize_columns()` method.

## Troubleshooting

**File not found errors:**
- Check that all 4 source files exist
- Verify the file paths in the script
- Use `--dry-run` to check file availability

**Column mapping issues:**
- Inspect your source CSV files to see actual column names
- Update the `column_mappings` dictionary accordingly
- The script will show which columns were renamed

**Encoding issues:**
- The script uses `cp850` encoding (Windows-1252) 
- Change encoding in both `load_file()` and `write_csv()` if needed
