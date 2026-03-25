"""Data loading and renaming layer for Austrian banks DEA project."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


class DataLoader:
    """Load source dataset and standardize column names/types."""

    def __init__(self, config: dict):
        self.config = config

    def load_raw(self) -> pd.DataFrame:
        """Load raw Excel/CSV file from configured path with deterministic sheet handling."""
        raw_path = Path(self.config["paths"]["raw_data"])
        logging.info("Loading raw data from %s", raw_path)

        if raw_path.suffix.lower() in {".xls", ".xlsx"}:
            sheet = self.config["io"].get("dataset_sheet")
            if sheet is None:
                # Deterministic fallback: first worksheet in workbook.
                xls = pd.ExcelFile(raw_path)
                if not xls.sheet_names:
                    raise ValueError(f"Excel file has no sheets: {raw_path}")
                selected_sheet = xls.sheet_names[0]
                logging.warning("dataset_sheet is null, using first sheet: %s", selected_sheet)
                df = xls.parse(selected_sheet)
            else:
                df = pd.read_excel(raw_path, sheet_name=sheet)

            if isinstance(df, dict):
                # Additional safety in case a list/None was passed unexpectedly.
                first_key = sorted(df.keys())[0]
                logging.warning("read_excel returned dict; using first sheet key: %s", first_key)
                df = df[first_key]
        else:
            df = pd.read_csv(raw_path)
        return df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map source column names (possibly German) into canonical names."""
        mapping = self.config.get("column_mapping", {})
        out = df.rename(columns=mapping).copy()

        id_col = self.config["io"].get("id_column", "bank_id")
        if id_col not in out.columns:
            out[id_col] = [f"bank_{i:05d}" for i in range(1, len(out) + 1)]

        return out

    def map_bank_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize bank type names into target categories used in analysis."""
        bank_type_col = self.config["io"]["bank_type_column"]
        bank_type_map = self.config.get("bank_type_mapping", {})

        out = df.copy()
        if bank_type_col in out.columns:
            out[bank_type_col] = out[bank_type_col].astype(str).str.strip()
            out["bank_type_original"] = out[bank_type_col]
            out[bank_type_col] = out[bank_type_col].map(bank_type_map).fillna("other")
        return out
