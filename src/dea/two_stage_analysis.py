"""Two-stage DEA workflow (funding efficiency -> income intermediation efficiency)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dea.dea_model import DEASolver


@dataclass
class TwoStageDEAOutput:
    stage1: pd.DataFrame
    stage2: pd.DataFrame
    merged: pd.DataFrame


def summarize_efficiency(df: pd.DataFrame, score_col: str = "efficiency") -> pd.DataFrame:
    """Compute descriptive distribution statistics for DEA scores."""
    grouped = df.groupby("bank_type")[score_col]
    summary = grouped.agg(["mean", "median", "std", "var", "min", "max", "count"]).reset_index()
    q = grouped.quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = ["bank_type", "q25", "q75"]
    return summary.merge(q, on="bank_type", how="left")


def run_two_stage_dea(df: pd.DataFrame, config: dict) -> TwoStageDEAOutput:
    """Run DEA stage 1 and stage 2 globally with same DEA setup from config."""
    dea_cfg = config["analysis"]["dea"]
    solver = DEASolver(
        returns_to_scale=dea_cfg["default_returns_to_scale"],
        orientation=dea_cfg["default_orientation"],
    )

    id_col = config["io"]["id_column"]
    name_col = config["io"]["bank_name_column"]

    stage1 = solver.solve(
        data=df,
        id_col=id_col,
        input_cols=["staff_expenses", "operating_expenses", "tangible_assets"],
        output_cols=["deposits"],
    ).scores.rename(columns={"efficiency": "stage1_efficiency"})

    stage2 = solver.solve(
        data=df,
        id_col=id_col,
        input_cols=["deposits"],
        output_cols=["interest_income", "fee_income", "customer_loans"],
    ).scores.rename(columns={"efficiency": "stage2_efficiency"})

    merged = (
        df[[id_col, name_col, "bank_type"]]
        .merge(stage1, on=id_col, how="left")
        .merge(stage2, on=id_col, how="left")
    )
    merged["combined_efficiency_mean"] = merged[["stage1_efficiency", "stage2_efficiency"]].mean(axis=1)

    return TwoStageDEAOutput(stage1=stage1, stage2=stage2, merged=merged)


def run_groupwise_dea(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute DEA separately per bank type (intra-group frontier)."""
    out = []
    for bank_type, grp in df.groupby("bank_type"):
        if len(grp) < 3:
            continue
        result = run_two_stage_dea(grp, config).merged
        result["dea_scope"] = "within_group"
        out.append(result)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_pooled_dea(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute DEA for all banks jointly (common frontier)."""
    pooled = run_two_stage_dea(df, config).merged
    pooled["dea_scope"] = "pooled"
    return pooled
