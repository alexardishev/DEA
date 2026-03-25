"""Validation and anomaly handling for DEA pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    cleaned_df: pd.DataFrame
    quality_report: pd.DataFrame
    excluded_df: pd.DataFrame


class DataValidator:
    """Apply configurable data quality rules without breaking pipeline."""

    def __init__(self, config: dict, filter_rules: dict):
        self.config = config
        self.filter_rules = filter_rules
        self.critical_cols = config["quality_rules"]["critical_nonpositive_columns"]

    def validate_required_columns(self, df: pd.DataFrame) -> None:
        required_cols = self.config.get("required_columns", [])
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run(self, df: pd.DataFrame) -> ValidationResult:
        self.validate_required_columns(df)

        out = df.copy()
        quality_rows: list[dict] = []

        if self.config["quality_rules"].get("drop_duplicates", True):
            out = out.drop_duplicates().copy()

        # Cast potential numeric columns defensively.
        for col in set(self.critical_cols + self.config.get("optional_columns", [])):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Row-wise issue detection.
        issue_cols = [c for c in self.critical_cols if c in out.columns]
        for idx, row in out.iterrows():
            reasons: list[str] = []

            missing_ratio = row[issue_cols].isna().mean() if issue_cols else 0.0
            if missing_ratio > self.filter_rules["critical_rules"]["max_missing_ratio_per_row"]:
                reasons.append(f"missing_ratio>{self.filter_rules['critical_rules']['max_missing_ratio_per_row']}")

            allow_zero = set(self.filter_rules["critical_rules"].get("allow_zero_for", []))
            for col in issue_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                if val < 0:
                    reasons.append(f"negative:{col}")
                elif val == 0 and col not in allow_zero:
                    reasons.append(f"zero:{col}")

            quality_rows.append(
                {
                    "row_index": idx,
                    "bank_name": row.get(self.config["io"]["bank_name_column"], np.nan),
                    "issue_count": len(reasons),
                    "issues": "|".join(reasons),
                    "is_problematic": len(reasons) > 0,
                }
            )

        quality_report = pd.DataFrame(quality_rows)
        problematic_indices = set(quality_report.loc[quality_report["is_problematic"], "row_index"].tolist())

        # Manual exclusions from config.
        manual_names = set(self.filter_rules.get("manual_exclusions", {}).get("bank_names", []))
        name_col = self.config["io"]["bank_name_column"]
        if name_col in out.columns and manual_names:
            problematic_indices |= set(out.index[out[name_col].isin(manual_names)].tolist())

        mode = self.filter_rules.get("mode", "exclude_critical")
        if mode == "exclude_critical":
            excluded_df = out.loc[sorted(problematic_indices)].copy()
            cleaned_df = out.drop(index=sorted(problematic_indices)).copy()
        else:
            excluded_df = out.iloc[0:0].copy()
            cleaned_df = out.copy()

        logging.info("Validation finished. total=%s cleaned=%s excluded=%s", len(out), len(cleaned_df), len(excluded_df))
        return ValidationResult(cleaned_df=cleaned_df, quality_report=quality_report, excluded_df=excluded_df)
