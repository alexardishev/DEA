"""Data loading and renaming layer for Austrian banks DEA project."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd


class DataLoader:
    """Load source dataset and standardize column names/types."""

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _normalize_header(name: str) -> str:
        """Normalize header names to make mapping robust to spaces/umlauts/punctuation."""
        if name is None:
            return ""
        text = str(name).strip()
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text

    def load_raw(self) -> pd.DataFrame:
        """Load raw Excel/CSV file from configured path with deterministic sheet handling."""
        raw_path = Path(self.config["paths"]["raw_data"])
        logging.info("Loading raw data from %s", raw_path)

        if raw_path.suffix.lower() in {".xls", ".xlsx"}:
            sheet = self.config["io"].get("dataset_sheet")
            if sheet is None:
                xls = pd.ExcelFile(raw_path)
                if not xls.sheet_names:
                    raise ValueError(f"Excel file has no sheets: {raw_path}")
                selected_sheet = xls.sheet_names[0]
                logging.warning("dataset_sheet is null, using first sheet: %s", selected_sheet)
                df = xls.parse(selected_sheet)
            else:
                df = pd.read_excel(raw_path, sheet_name=sheet)

            if isinstance(df, dict):
                first_key = sorted(df.keys())[0]
                logging.warning("read_excel returned dict; using first sheet key: %s", first_key)
                df = df[first_key]
        else:
            df = pd.read_csv(raw_path)
        return df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map source column names (possibly German) into canonical names."""
        mapping = self.config.get("column_mapping", {})

        # 1) strip original column names to avoid trailing spaces mismatch.
        stripped_cols = {col: str(col).strip() for col in df.columns}
        out = df.rename(columns=stripped_cols).copy()

        # 2) build normalized mapping and rename robustly.
        normalized_mapping = {self._normalize_header(src): dst for src, dst in mapping.items()}
        rename_dict = {}
        for col in out.columns:
            norm_col = self._normalize_header(col)
            if norm_col in normalized_mapping:
                rename_dict[col] = normalized_mapping[norm_col]

        out = out.rename(columns=rename_dict)

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
