"""Two-stage DEA workflow with explicit sequential approximation notes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dea.dea_model import DEASolver


@dataclass
class TwoStageDEAOutput:
    stage1_scores: pd.DataFrame
    stage2_scores: pd.DataFrame
    combined_summary: pd.DataFrame
    problematic: pd.DataFrame


@dataclass
class DEAAnalysisOutput:
    within_scores: pd.DataFrame
    pooled_scores: pd.DataFrame
    within_summary: pd.DataFrame
    between_summary: pd.DataFrame
    problematic: pd.DataFrame
    sensitivity: pd.DataFrame


def _run_single_stage(
    df: pd.DataFrame,
    stage_name: str,
    id_col: str,
    stage_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver = DEASolver(
        returns_to_scale=stage_cfg["returns_to_scale"],
        orientation=stage_cfg["orientation"],
    )
    result = solver.solve(
        data=df,
        id_col=id_col,
        input_cols=stage_cfg["inputs"],
        output_cols=stage_cfg["outputs"],
    )
    scores = result.scores.rename(
        columns={
            "efficiency": f"{stage_name}_efficiency",
            "status": f"{stage_name}_status",
            "is_efficient": f"{stage_name}_is_efficient",
            "peer_count": f"{stage_name}_peer_count",
            "lambda_sum": f"{stage_name}_lambda_sum",
        }
    )
    scores["stage"] = stage_name

    problematic = result.problematic.copy()
    if not problematic.empty:
        problematic["stage"] = stage_name
    return scores, problematic


def run_two_stage_dea(df: pd.DataFrame, config: dict) -> TwoStageDEAOutput:
    """Run sequential two-stage DEA approximation and keep both stage scores separately."""
    dea_cfg = config["analysis"]["dea"]
    id_col = config["io"]["id_column"]
    name_col = config["io"]["bank_name_column"]

    stage1_scores, stage1_bad = _run_single_stage(df, "stage1", id_col, dea_cfg["stage1"])
    stage2_scores, stage2_bad = _run_single_stage(df, "stage2", id_col, dea_cfg["stage2"])

    keep_stage_cols = [
        id_col,
        "dmu_index",
        "raw_score",
        "returns_to_scale",
        "orientation",
        "stage1_efficiency",
        "stage1_status",
        "stage1_is_efficient",
        "stage1_peer_count",
        "stage1_lambda_sum",
    ]
    s1 = stage1_scores[keep_stage_cols].copy()

    s2 = stage2_scores[
        [
            id_col,
            "dmu_index",
            "raw_score",
            "returns_to_scale",
            "orientation",
            "stage2_efficiency",
            "stage2_status",
            "stage2_is_efficient",
            "stage2_peer_count",
            "stage2_lambda_sum",
        ]
    ].copy()

    bank_meta = df[[id_col, name_col, "bank_type"]].copy()
    s1 = bank_meta.merge(s1, on=id_col, how="left")
    s2 = bank_meta.merge(s2, on=id_col, how="left")

    combined = bank_meta.merge(s1.drop(columns=[name_col, "bank_type"]), on=id_col, how="left").merge(
        s2,
        on=id_col,
        how="left",
        suffixes=("_stage1", "_stage2"),
    )

    # This is not strict network DEA final efficiency. It is an approximation summary metric.
    combined["combined_summary_metric"] = combined[["stage1_efficiency", "stage2_efficiency"]].mean(axis=1)
    combined["combined_metric_note"] = "Sequential approximation metric (mean of stage scores), not strict network DEA efficiency"

    problematic = pd.concat([stage1_bad, stage2_bad], ignore_index=True) if (not stage1_bad.empty or not stage2_bad.empty) else pd.DataFrame()
    return TwoStageDEAOutput(stage1_scores=s1, stage2_scores=s2, combined_summary=combined, problematic=problematic)


def summarize_by_group(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    grouped = df.groupby("bank_type")[score_col]
    summary = grouped.agg(["mean", "median", "std", "var", "min", "max", "count"]).reset_index()
    q = grouped.quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = ["bank_type", "q25", "q75"]
    efficient_share = (df.assign(is_eff=(df[score_col] >= 0.999).astype(int)).groupby("bank_type")["is_eff"].mean().reset_index(name="efficient_share"))
    return summary.merge(q, on="bank_type", how="left").merge(efficient_share, on="bank_type", how="left")


def run_groupwise_dea(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bad = []
    for bank_type, grp in df.groupby("bank_type"):
        if len(grp) < 3:
            continue
        output = run_two_stage_dea(grp, config)
        part = output.combined_summary.copy()
        part["dea_scope"] = "within_group"
        part["rank_within_group_stage1"] = part["stage1_efficiency"].rank(method="dense", ascending=False)
        part["rank_within_group_stage2"] = part["stage2_efficiency"].rank(method="dense", ascending=False)
        rows.append(part)
        if not output.problematic.empty:
            temp = output.problematic.copy()
            temp["bank_type"] = bank_type
            temp["dea_scope"] = "within_group"
            bad.append(temp)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    problematic = pd.concat(bad, ignore_index=True) if bad else pd.DataFrame()
    return combined, problematic


def run_pooled_dea(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = run_two_stage_dea(df, config)
    pooled = out.combined_summary.copy()
    pooled["dea_scope"] = "pooled"
    pooled["rank_within_group_stage1"] = pooled.groupby("bank_type")["stage1_efficiency"].rank(method="dense", ascending=False)
    pooled["rank_within_group_stage2"] = pooled.groupby("bank_type")["stage2_efficiency"].rank(method="dense", ascending=False)

    stage1 = pooled[["bank_type", "stage1_efficiency"]].rename(columns={"stage1_efficiency": "score"})
    stage2 = pooled[["bank_type", "stage2_efficiency"]].rename(columns={"stage2_efficiency": "score"})
    between = (
        summarize_by_group(stage1.rename(columns={"score": "stage1_score"}), "stage1_score")
        .rename(columns={"mean": "stage1_mean", "median": "stage1_median", "efficient_share": "stage1_efficient_share"})
        .merge(
            summarize_by_group(stage2.rename(columns={"score": "stage2_score"}), "stage2_score").rename(
                columns={"mean": "stage2_mean", "median": "stage2_median", "efficient_share": "stage2_efficient_share"}
            )[["bank_type", "stage2_mean", "stage2_median", "stage2_efficient_share"]],
            on="bank_type",
            how="left",
        )
    )

    return pooled, between, out.stage1_scores, out.stage2_scores, out.problematic


def run_sensitivity_analysis(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    dea_cfg = config["analysis"]["dea"]
    sens_cfg = dea_cfg.get("sensitivity", {})
    if not sens_cfg.get("enabled", False):
        return pd.DataFrame()

    scenarios = sens_cfg.get("scenarios", [])
    rows = []
    base_stage1 = dea_cfg["stage1"].copy()
    base_stage2 = dea_cfg["stage2"].copy()

    for scenario in scenarios:
        cfg_copy = {**config}
        cfg_copy["analysis"] = {**config["analysis"]}
        cfg_copy["analysis"]["dea"] = {**dea_cfg}
        cfg_copy["analysis"]["dea"]["stage1"] = {**base_stage1, **scenario["stage1"]}
        cfg_copy["analysis"]["dea"]["stage2"] = {**base_stage2, **scenario["stage2"]}

        out = run_two_stage_dea(df, cfg_copy)
        temp = out.combined_summary[[config["io"]["id_column"], "stage1_efficiency", "stage2_efficiency", "combined_summary_metric"]].copy()
        temp["scenario"] = scenario["name"]
        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    agg = result.groupby("scenario")[["stage1_efficiency", "stage2_efficiency", "combined_summary_metric"]].mean().reset_index()
    return result.merge(agg, on="scenario", suffixes=("", "_scenario_mean"))


def run_dea_analysis(df: pd.DataFrame, config: dict) -> DEAAnalysisOutput:
    within_scores, within_bad = run_groupwise_dea(df, config)
    pooled_scores, between_summary, stage1_scores, stage2_scores, pooled_bad = run_pooled_dea(df, config)

    within_summary = summarize_by_group(within_scores, "stage1_efficiency") if not within_scores.empty else pd.DataFrame()
    sensitivity = run_sensitivity_analysis(df, config)

    problematic_list = [d for d in [within_bad, pooled_bad] if not d.empty]
    problematic = pd.concat(problematic_list, ignore_index=True) if problematic_list else pd.DataFrame()

    # Attach stage-level exports in attrs for orchestrator simplicity.
    pooled_scores.attrs["stage1_scores"] = stage1_scores
    pooled_scores.attrs["stage2_scores"] = stage2_scores

    return DEAAnalysisOutput(
        within_scores=within_scores,
        pooled_scores=pooled_scores,
        within_summary=within_summary,
        between_summary=between_summary,
        problematic=problematic,
        sensitivity=sensitivity,
    )
