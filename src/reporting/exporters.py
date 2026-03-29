"""Tabular export utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table(df: pd.DataFrame, base_path: str, export_excel: bool = True, export_csv: bool = True) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    if export_csv:
        df.to_csv(base.with_suffix(".csv"), index=False)
    if export_excel:
        df.to_excel(base.with_suffix(".xlsx"), index=False)
