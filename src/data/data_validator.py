"""Validation and anomaly handling for DEA pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
    cleaned_df: pd.DataFrame
    quality_report: pd.DataFrame
    excluded_df: pd.DataFrame


class DataValidator:
    """Apply configurable data quality rules without crashing the pipeline."""

    def __init__(self, config: dict, filter_rules: dict):
        self.config = config
        self.filter_rules = filter_rules
        self.critical_cols = config["quality_rules"]["critical_nonpositive_columns"]
        self.id_col = config["io"]["id_column"]
        self.name_col = config["io"]["bank_name_column"]

    def validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.config.get("required_columns", []) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _outlier_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based outlier flags for numeric columns configured as critical."""
        flags = pd.DataFrame(False, index=df.index, columns=self.critical_cols)
        mult = self.filter_rules.get("outlier_rules", {}).get("multiplier", 3.0)

        for col in self.critical_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            low, high = q1 - mult * iqr, q3 + mult * iqr
            flags[col] = (series < low) | (series > high)

        return flags

    def run(self, df: pd.DataFrame) -> ValidationResult:
        self.validate_required_columns(df)
        out = df.copy()

        if self.config["quality_rules"].get("drop_duplicates", True):
            out = out.drop_duplicates().copy()

        for col in set(self.critical_cols + self.config.get("optional_columns", [])):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        outlier_flags = self._outlier_flags(out)
        records: list[dict] = []
        excluded_indices: set[int] = set()

        max_missing = self.filter_rules.get("missingness", {}).get("max_missing_ratio_per_row", 0.3)
        zero_input_action = self.filter_rules.get("zero_rules", {}).get("inputs", "exclude")
        zero_output_action = self.filter_rules.get("zero_rules", {}).get("outputs", "flag")
        allow_zero_outputs = set(self.filter_rules.get("zero_rules", {}).get("allow_zero_outputs", []))

        stage1_inputs = set(self.config["analysis"]["dea"]["stage1"]["inputs"])
        stage1_outputs = set(self.config["analysis"]["dea"]["stage1"]["outputs"])
        stage2_inputs = set(self.config["analysis"]["dea"]["stage2"]["inputs"])
        stage2_outputs = set(self.config["analysis"]["dea"]["stage2"]["outputs"])

        for idx, row in out.iterrows():
            bank_id = row.get(self.id_col)
            bank_name = row.get(self.name_col)
            row_excluded = False

            present_critical = [c for c in self.critical_cols if c in out.columns]
            missing_ratio = row[present_critical].isna().mean() if present_critical else 0.0
            if missing_ratio > max_missing:
                records.append(
                    {
                        "row_index": idx,
                        "bank_id": bank_id,
                        "bank_name": bank_name,
                        "issue_type": "missingness",
                        "issue_column": "multiple",
                        "issue_value": missing_ratio,
                        "stage": "all",
                        "action": "exclude",
                        "reason": f"missing_ratio>{max_missing}",
                    }
                )
                row_excluded = True

            for col in present_critical:
                val = row[col]
                if pd.isna(val):
                    continue

                if val < 0:
                    records.append(
                        {
                            "row_index": idx,
                            "bank_id": bank_id,
                            "bank_name": bank_name,
                            "issue_type": "negative",
                            "issue_column": col,
                            "issue_value": val,
                            "stage": "all",
                            "action": "exclude",
                            "reason": "negative numeric value",
                        }
                    )
                    row_excluded = True

                if val == 0:
                    involved_as_input = col in (stage1_inputs | stage2_inputs)
                    involved_as_output = col in (stage1_outputs | stage2_outputs)
                    if involved_as_input and zero_input_action == "exclude":
                        records.append(
                            {
                                "row_index": idx,
                                "bank_id": bank_id,
                                "bank_name": bank_name,
                                "issue_type": "zero_input",
                                "issue_column": col,
                                "issue_value": val,
                                "stage": "dea",
                                "action": "exclude",
                                "reason": "zero on DEA input",
                            }
                        )
                        row_excluded = True
                    elif involved_as_output and col not in allow_zero_outputs:
                        records.append(
                            {
                                "row_index": idx,
                                "bank_id": bank_id,
                                "bank_name": bank_name,
                                "issue_type": "zero_output",
                                "issue_column": col,
                                "issue_value": val,
                                "stage": "dea",
                                "action": zero_output_action,
                                "reason": "zero on DEA output",
                            }
                        )
                        if zero_output_action == "exclude":
                            row_excluded = True

                if col in outlier_flags.columns and bool(outlier_flags.loc[idx, col]):
                    outlier_action = self.filter_rules.get("outlier_rules", {}).get("action", "flag")
                    records.append(
                        {
                            "row_index": idx,
                            "bank_id": bank_id,
                            "bank_name": bank_name,
                            "issue_type": "outlier",
                            "issue_column": col,
                            "issue_value": val,
                            "stage": "all",
                            "action": outlier_action,
                            "reason": "IQR outlier",
                        }
                    )
                    if outlier_action == "exclude":
                        row_excluded = True

            if row_excluded:
                excluded_indices.add(idx)

        # Manual exclusions by name/id.
        manual_names = set(self.filter_rules.get("manual_exclusions", {}).get("bank_names", []))
        manual_ids = set(self.filter_rules.get("manual_exclusions", {}).get("bank_ids", []))
        for idx, row in out.iterrows():
            if row.get(self.name_col) in manual_names or row.get(self.id_col) in manual_ids:
                excluded_indices.add(idx)
                records.append(
                    {
                        "row_index": idx,
                        "bank_id": row.get(self.id_col),
                        "bank_name": row.get(self.name_col),
                        "issue_type": "manual_exclusion",
                        "issue_column": "n/a",
                        "issue_value": "n/a",
                        "stage": "all",
                        "action": "exclude",
                        "reason": "manual exclusion rule",
                    }
                )

        quality_report = pd.DataFrame(records)
        if quality_report.empty:
            quality_report = pd.DataFrame(
                columns=["row_index", "bank_id", "bank_name", "issue_type", "issue_column", "issue_value", "stage", "action", "reason"]
            )

        cleaned_df = out.drop(index=sorted(excluded_indices)).copy()
        excluded_df = out.loc[sorted(excluded_indices)].copy() if excluded_indices else out.iloc[0:0].copy()
        return ValidationResult(cleaned_df=cleaned_df, quality_report=quality_report, excluded_df=excluded_df)
