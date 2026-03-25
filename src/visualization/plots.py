"""Plot builders for DEA distributions and model diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.style import apply_academic_style


def _save_multi_format(fig, base_path: Path, formats: list[str]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), bbox_inches="tight")


def plot_violin_efficiency(df, score_col: str, cfg: dict, output_name: str) -> None:
    apply_academic_style()
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x=score_col, y="bank_type", inner="box", cut=0, ax=ax)
    ax.set_title(f"Distribution of {score_col} by bank type")
    ax.set_xlabel("Efficiency score")
    ax.set_ylabel("Bank type")
    _save_multi_format(fig, Path("outputs/figures") / output_name, cfg["visualization"]["figure_format"])
    plt.close(fig)


def plot_box_efficiency(df, score_col: str, cfg: dict, output_name: str) -> None:
    apply_academic_style()
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="bank_type", y=score_col, ax=ax)
    ax.set_title(f"{score_col}: boxplot by bank type")
    ax.set_xlabel("Bank type")
    ax.set_ylabel("Efficiency score")
    _save_multi_format(fig, Path("outputs/figures") / output_name, cfg["visualization"]["figure_format"])
    plt.close(fig)


def plot_density(df, score_col: str, cfg: dict, output_name: str) -> None:
    apply_academic_style()
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=score_col, hue="bank_type", fill=True, common_norm=False, alpha=0.25, ax=ax)
    ax.set_title(f"Density of {score_col} by bank type")
    _save_multi_format(fig, Path("outputs/figures") / output_name, cfg["visualization"]["figure_format"])
    plt.close(fig)


def plot_scatter_inputs_outputs(df, cfg: dict, output_name: str) -> None:
    apply_academic_style()
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="deposits", y="interest_income", hue="bank_type", alpha=0.7, ax=ax)
    ax.set_title("Deposits vs Interest income")
    _save_multi_format(fig, Path("outputs/figures") / output_name, cfg["visualization"]["figure_format"])
    plt.close(fig)
