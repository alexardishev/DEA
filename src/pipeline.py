"""End-to-end orchestration for Austrian banking DEA research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.dea.two_stage_analysis import run_groupwise_dea, run_pooled_dea, summarize_efficiency
from src.regression.models import run_regressions
from src.reporting.exporters import export_table
from src.reporting.text_reports import build_results_interpretation, write_methodology_templates, write_text
from src.utils.io_utils import configure_logging, ensure_parent_dir, load_yaml
from src.visualization.maps import plot_efficiency_map
from src.visualization.plots import plot_box_efficiency, plot_density, plot_scatter_inputs_outputs, plot_violin_efficiency


def run_data_stage(config: dict, filter_rules: dict) -> pd.DataFrame:
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
    return result.cleaned_df


def run_dea_stage(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = run_groupwise_dea(df, config)
    pooled = run_pooled_dea(df, config)

    export_table(within, "outputs/tables/dea_within_group", True, True)
    export_table(pooled, "outputs/tables/dea_pooled", True, True)

    if not within.empty:
        rank = within.sort_values(["bank_type", "stage2_efficiency"], ascending=[True, False]).copy()
        rank["rank_within_group"] = rank.groupby("bank_type")["stage2_efficiency"].rank(method="dense", ascending=False)
        export_table(rank, "outputs/tables/dea_within_group_ranking", True, True)

        summary = summarize_efficiency(within.rename(columns={"stage2_efficiency": "efficiency"}), "efficiency")
        export_table(summary, "outputs/tables/dea_distribution_summary", True, True)

    return within, pooled


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
    plot_violin_efficiency(within, "stage1_efficiency", config, "violin_stage1_by_type")
    plot_violin_efficiency(within, "stage2_efficiency", config, "violin_stage2_by_type")
    plot_box_efficiency(within, "stage2_efficiency", config, "box_stage2_by_type")
    plot_density(pooled, "stage2_efficiency", config, "density_stage2_pooled")
    plot_scatter_inputs_outputs(pooled, config, "scatter_deposits_interest_income")

    map_note = plot_efficiency_map(pooled, "stage2_efficiency")
    write_text("outputs/text/map_note.txt", map_note)


def run_reporting_stage(within: pd.DataFrame, pooled: pd.DataFrame, config: dict) -> None:
    write_methodology_templates()
    interpretation = build_results_interpretation(within, pooled)
    write_text("outputs/text/dea_interpretation_draft.md", interpretation)

    summary_text = f"""# Pipeline summary

- Two-stage DEA completed (stage 1 funding efficiency, stage 2 income/loan transformation).
- Default DEA setup: RTS={config['analysis']['dea']['default_returns_to_scale']}, orientation={config['analysis']['dea']['default_orientation']}.
- Within-group rows: {len(within)}.
- Pooled rows: {len(pooled)}.
- Key outputs are available in outputs/tables, outputs/figures, outputs/models, outputs/text.
"""
    write_text("outputs/text/summary.md", summary_text)


def main(stage: str) -> None:
    configure_logging()
    config = load_yaml("config/pipeline_config.yaml")
    filter_rules = load_yaml("config/filter_rules.yaml")

    cleaned_df = None
    within = None
    pooled = None

    if stage in {"all", "data"}:
        cleaned_df = run_data_stage(config, filter_rules)
    else:
        cleaned_df = pd.read_csv(config["paths"]["processed_data"])

    if stage in {"all", "dea"}:
        within, pooled = run_dea_stage(cleaned_df, config)

    if stage in {"all", "regression"}:
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_pooled.csv")
        run_regression_stage(pooled, cleaned_df, config)

    if stage in {"all", "viz"}:
        if within is None:
            within = pd.read_csv("outputs/tables/dea_within_group.csv")
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_pooled.csv")
        run_visualization_stage(within, pooled, config)

    if stage in {"all", "report"}:
        if within is None:
            within = pd.read_csv("outputs/tables/dea_within_group.csv")
        if pooled is None:
            pooled = pd.read_csv("outputs/tables/dea_pooled.csv")
        run_reporting_stage(within, pooled, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Austrian banking DEA pipeline")
    parser.add_argument("--stage", default="all", choices=["all", "data", "dea", "regression", "viz", "report"])
    args = parser.parse_args()
    main(args.stage)
