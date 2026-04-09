import pandas as pd

from .schema import DATE_COLUMN, NUMERIC_COLUMNS, PRODUCT_ID_COLUMN, TARGET_COLUMNS

LEGACY_COLUMN_ALIASES = {
    "PREIS": "Netto_Umsatz",
    "BEZEICHNG": "Produkt",
}

LEGACY_DERIVED_NUMERIC_COLUMNS = {
    "MWST": ("Brutto_Umsatz", "Netto_Umsatz"),
}


def load_and_prepare_csv(csv_path: str) -> pd.DataFrame:
    available_columns = pd.read_csv(
        csv_path,
        sep=";",
        encoding="cp850",
        nrows=0,
    ).columns.tolist()
    use_columns = [col for col in TARGET_COLUMNS if col in available_columns]
    legacy_source_columns = [
        source_column
        for target_column, source_column in LEGACY_COLUMN_ALIASES.items()
        if target_column not in use_columns and source_column in available_columns
    ]
    for target_column, source_columns in LEGACY_DERIVED_NUMERIC_COLUMNS.items():
        if target_column in use_columns:
            continue
        if all(source_column in available_columns for source_column in source_columns):
            legacy_source_columns.extend(source_columns)
    use_columns = list(dict.fromkeys(use_columns + legacy_source_columns))
    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="cp850",
        low_memory=False,
        ### Adapt this to your data and the columns it has
        usecols=use_columns,
    )

    for target_column, source_column in LEGACY_COLUMN_ALIASES.items():
        if target_column not in df.columns and source_column in df.columns:
            df[target_column] = df[source_column]

    for target_column, source_columns in LEGACY_DERIVED_NUMERIC_COLUMNS.items():
        if target_column in df.columns:
            continue
        left_column, right_column = source_columns
        if left_column in df.columns and right_column in df.columns:
            left = pd.to_numeric(df[left_column], errors="coerce").fillna(0)
            right = pd.to_numeric(df[right_column], errors="coerce").fillna(0)
            df[target_column] = (left - right).clip(lower=0)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            if col in NUMERIC_COLUMNS:
                df[col] = 0
            else:
                df[col] = ""

    if PRODUCT_ID_COLUMN not in df.columns:
        df[PRODUCT_ID_COLUMN] = ""
    df["AUFTRAG_NR"] = df["AUFTRAG_NR"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["NUMMER"] = df["NUMMER"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df[PRODUCT_ID_COLUMN] = df[PRODUCT_ID_COLUMN].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    string_cols = [col for col in TARGET_COLUMNS if col not in [DATE_COLUMN, *NUMERIC_COLUMNS]]
    for col in string_cols:
        df[col] = df[col].fillna("").astype(str)
    return df[TARGET_COLUMNS].copy()
