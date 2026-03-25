"""End-to-end orchestration for Austrian banking DEA research pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.dea.two_stage_analysis import run_dea_analysis, summarize_by_group
from src.regression.models import run_regressions
from src.reporting.exporters import export_table
from src.reporting.text_reports import build_results_interpretation, write_methodology_templates, write_text
from src.utils.io_utils import configure_logging, ensure_parent_dir, load_yaml
from src.visualization.maps import plot_efficiency_map
from src.visualization.plots import (
    plot_bar_group_means,
    plot_box_efficiency,
    plot_density,
    plot_scatter_inputs_outputs,
    plot_violin_efficiency,
)


def run_data_stage(config: dict, filter_rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(config)
    raw = loader.load_raw()
    std = loader.rename_columns(raw)
    std = loader.map_bank_types(std)

    validator = DataValidator(config, filter_rules)
    result = validator.run(std)

    ensure_parent_dir(config["paths"]["processed_data"])
    result.cleaned_df.to_csv(config["paths"]["processed_data"], index=False)
    result.quality_report.to_csv(config["paths"]["quality_report"], index=False)
    result.excluded_df.to_csv(config["paths"]["excluded_observations"], index=False)
    return result.cleaned_df, result.quality_report


def run_dea_stage(df: pd.DataFrame, quality_report: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = run_dea_analysis(df, config)

    stage1_scores = output.pooled_scores.attrs.get("stage1_scores", pd.DataFrame())
    stage2_scores = output.pooled_scores.attrs.get("stage2_scores", pd.DataFrame())

    export_table(stage1_scores, "outputs/tables/dea_stage1_scores", True, True)
    export_table(stage2_scores, "outputs/tables/dea_stage2_scores", True, True)
    export_table(output.pooled_scores, "outputs/tables/dea_combined_summary", True, True)

    within_summary_stage1 = summarize_by_group(output.within_scores, "stage1_efficiency") if not output.within_scores.empty else pd.DataFrame()
    within_summary_stage2 = summarize_by_group(output.within_scores, "stage2_efficiency") if not output.within_scores.empty else pd.DataFrame()
    within_summary = within_summary_stage1.merge(
        within_summary_stage2[["bank_type", "mean", "median", "std", "var", "min", "max", "q25", "q75", "efficient_share"]].rename(
            columns={
                "mean": "stage2_mean",
                "median": "stage2_median",
                "std": "stage2_std",
                "var": "stage2_var",
                "min": "stage2_min",
                "max": "stage2_max",
                "q25": "stage2_q25",
                "q75": "stage2_q75",
                "efficient_share": "stage2_efficient_share",
            }
        ),
        on="bank_type",
        how="left",
    ) if not within_summary_stage1.empty else pd.DataFrame()

    if not within_summary.empty:
        within_summary = within_summary.rename(
            columns={
                "mean": "stage1_mean",
                "median": "stage1_median",
                "std": "stage1_std",
                "var": "stage1_var",
                "min": "stage1_min",
                "max": "stage1_max",
                "q25": "stage1_q25",
                "q75": "stage1_q75",
                "efficient_share": "stage1_efficient_share",
            }
        )
    export_table(within_summary, "outputs/tables/dea_within_group_summary", True, True)

    export_table(output.between_summary, "outputs/tables/dea_between_group_summary", True, True)

    problematic = pd.concat(
        [
            output.problematic if not output.problematic.empty else pd.DataFrame(),
            quality_report.assign(source="data_quality"),
        ],
        ignore_index=True,
        sort=False,
    )
    export_table(problematic, "outputs/tables/dea_problematic_observations", True, True)

    if not output.sensitivity.empty:
        export_table(output.sensitivity, "outputs/tables/dea_sensitivity_comparison", True, True)

    logging.info("DEA stage finished. pooled=%s problematic=%s", len(output.pooled_scores), len(problematic))
    return output.within_scores, output.pooled_scores


def run_regression_stage(pooled: pd.DataFrame, cleaned_df: pd.DataFrame, config: dict) -> None:
    id_col = config["io"]["id_column"]
    reg_df = pooled.merge(cleaned_df, on=id_col, how="left", suffixes=("", "_raw"))
    output = run_regressions(reg_df, config)

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/models/ols_summary.txt").write_text(output.ols_summary_text, encoding="utf-8")
    if output.fractional_glm_summary_text:
        Path("outputs/models/fractional_glm_summary.txt").write_text(output.fractional_glm_summary_text, encoding="utf-8")

    export_table(output.coefficients, "outputs/tables/regression_coefficients", True, True)
    export_table(output.diagnostics, "outputs/tables/regression_diagnostics_vif", True, True)


def run_visualization_stage(within: pd.DataFrame, pooled: pd.DataFrame, config: dict) -> None:
    if within.empty:
        return
    for stage_col in ["stage1_efficiency", "stage2_efficiency"]:
        suffix = "stage1" if "stage1" in stage_col else "stage2"
        plot_violin_efficiency(within, stage_col, config, f"violin_{suffix}_by_type")
        plot_box_efficiency(within, stage_col, config, f"box_{suffix}_by_type")
        plot_density(pooled, stage_col, config, f"density_{suffix}_pooled")
        plot_bar_group_means(within, stage_col, config, f"bar_mean_{suffix}_within_group")

    plot_scatter_inputs_outputs(pooled, config, "scatter_deposits_interest_income")
    map_note = plot_efficiency_map(pooled, "stage2_efficiency")
    write_text("outputs/text/map_note.txt", map_note)


def run_reporting_stage(within: pd.DataFrame, pooled: pd.DataFrame, config: dict) -> None:
    write_methodology_templates(config)
    interpretation = build_results_interpretation(within, pooled)
    write_text("outputs/text/dea_interpretation_draft.md", interpretation)

    summary_text = f"""# Pipeline summary

- Two-stage DEA completed as sequential approximation (not strict network DEA).
- Stage 1 setup: RTS={config['analysis']['dea']['stage1']['returns_to_scale']}, orientation={config['analysis']['dea']['stage1']['orientation']}.
- Stage 2 setup: RTS={config['analysis']['dea']['stage2']['returns_to_scale']}, orientation={config['analysis']['dea']['stage2']['orientation']}.
- Within-group rows: {len(within)}.
- Pooled rows: {len(pooled)}.
- Key DEA outputs: dea_stage1_scores, dea_stage2_scores, dea_combined_summary, dea_within_group_summary, dea_between_group_summary, dea_problematic_observations.
"""
    write_text("outputs/text/summary.md", summary_text)


def main(stage: str) -> None:
    configure_logging()
    config = load_yaml("config/pipeline_config.yaml")
    filter_rules = load_yaml("config/filter_rules.yaml")

    cleaned_df = None
    quality_report = pd.DataFrame()
    within = None
    pooled = None

    if stage in {"all", "data"}:
        cleaned_df, quality_report = run_data_stage(config, filter_rules)
    else:
        cleaned_df = pd.read_csv(config["paths"]["processed_data"])
        quality_report = pd.read_csv(config["paths"]["quality_report"]) if Path(config["paths"]["quality_report"]).exists() else pd.DataFrame()

    if stage in {"all", "dea"}:
        within, pooled = run_dea_stage(cleaned_df, quality_report, config)

    if stage in {"all", "regression"}:
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_combined_summary.csv")
        run_regression_stage(pooled, cleaned_df, config)

    if stage in {"all", "viz"}:
        if within is None:
            within = pd.read_csv("outputs/tables/dea_combined_summary.csv")
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_combined_summary.csv")
        run_visualization_stage(within, pooled, config)

    if stage in {"all", "report"}:
        if within is None:
            within = pd.read_csv("outputs/tables/dea_combined_summary.csv")
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_combined_summary.csv")
        run_reporting_stage(within, pooled, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Austrian banking DEA pipeline")
    parser.add_argument("--stage", default="all", choices=["all", "data", "dea", "regression", "viz", "report"])
    args = parser.parse_args()
    main(args.stage)
