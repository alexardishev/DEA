"""Centralized plotting style for publication-grade figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def apply_academic_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
