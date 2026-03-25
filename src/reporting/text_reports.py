"""Generate draft academic interpretations and methodology notes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build_results_interpretation(within_df: pd.DataFrame, pooled_df: pd.DataFrame) -> str:
    top = within_df.groupby("bank_type")["stage2_efficiency"].mean().sort_values(ascending=False)
    efficient_share = (
        pooled_df.assign(efficient=lambda d: (d["stage2_efficiency"] >= 0.999).astype(int))
        .groupby("bank_type")["efficient"]
        .mean()
        .sort_values(ascending=False)
    )

    lines = [
        "# Черновая интерпретация результатов DEA\n",
        "## Внутригрупповой анализ\n",
        "Средние значения stage2-эффективности по группам (внутригрупповой фронтир):",
        top.to_string(),
        "\n## Межгрупповой анализ\n",
        "Доли полностью эффективных банков (score≈1) на общей границе:",
        efficient_share.to_string(),
        "\nКраткий вывод: группы с более высоким средним и меньшим разбросом находятся ближе к эффективной границе.",
    ]
    return "\n".join(lines)


def write_methodology_templates() -> None:
    intro = """# Введение (черновик)\nИсследование оценивает эффективность банков Австрии с применением двухэтапного DEA и второй стадии регрессии."""
    methodology = """# Методология DEA (черновик)\nРеализованы модели CCR/BCC и input/output orientation. По умолчанию используется BCC-input как наиболее уместная постановка при гетерогенном масштабе банков."""
    assumptions = """# assumptions_and_method_notes\n- DEA-оценки ограничены интервалом [0,1], поэтому OLS используется с осторожностью.\n- В качестве расширения добавлена fractional logit (GLM Binomial).\n- При наличии сильной цензуры рекомендуется bootstrap DEA и truncated regression (future work)."""

    write_text("report/introduction.md", intro)
    write_text("report/methodology_dea.md", methodology)
    write_text("report/assumptions_and_method_notes.md", assumptions)
