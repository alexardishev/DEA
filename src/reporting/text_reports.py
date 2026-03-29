"""Generate draft academic interpretations and methodology notes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build_results_interpretation(within_df: pd.DataFrame, pooled_df: pd.DataFrame) -> str:
    if within_df.empty or pooled_df.empty:
        return "# Черновая интерпретация результатов DEA\nНедостаточно данных для генерации интерпретации."

    stage1_mean = within_df.groupby("bank_type")["stage1_efficiency"].mean().sort_values(ascending=False)
    stage2_mean = within_df.groupby("bank_type")["stage2_efficiency"].mean().sort_values(ascending=False)
    efficient_share = (
        pooled_df.assign(efficient=lambda d: (d["stage2_efficiency"] >= 0.999).astype(int))
        .groupby("bank_type")["efficient"]
        .mean()
        .sort_values(ascending=False)
    )

    lines = [
        "# Черновая интерпретация результатов DEA\n",
        "## Внутригрупповой анализ: Stage 1 (привлечение средств)\n",
        stage1_mean.to_string(),
        "\n## Внутригрупповой анализ: Stage 2 (трансформация в доходы и кредиты)\n",
        stage2_mean.to_string(),
        "\n## Межгрупповой анализ на общей границе\n",
        "Доли полностью эффективных банков (по stage2 score≈1):",
        efficient_share.to_string(),
        "\nВажно: combined_summary_metric — это агрегированная метрика-приближение, а не строгая network DEA эффективность.",
    ]
    return "\n".join(lines)


def write_methodology_templates(config: dict) -> None:
    stage1 = config["analysis"]["dea"]["stage1"]
    stage2 = config["analysis"]["dea"]["stage2"]
    intro = (
        "# Введение (черновик)\n"
        "Исследование оценивает эффективность банков Австрии с применением двухэтапного DEA и второй стадии регрессии."
    )

    methodology = f"""# Методология DEA (черновик)

## Что считается в проекте
В проекте реализована **sequential two-stage DEA approximation**.
- Stage 1: {stage1['inputs']} -> {stage1['outputs']}.
- Stage 2: {stage2['inputs']} -> {stage2['outputs']}.

Это **не строгая network DEA** с единой межэтапной оптимизацией; этапы решаются последовательно и отдельно,
после чего формируется вспомогательная агрегированная метрика `combined_summary_metric`.

## Параметры DEA
- Stage 1: RTS={stage1['returns_to_scale']}, orientation={stage1['orientation']}.
- Stage 2: RTS={stage2['returns_to_scale']}, orientation={stage2['orientation']}.

## Интерпретация
- Stage 1 score: эффективность привлечения фондирования.
- Stage 2 score: эффективность преобразования фондирования в доходы/кредиты.
- `combined_summary_metric`: только summary-индикатор для сравнений, не строгая итоговая DEA-оценка.
"""

    assumptions = """# assumptions_and_method_notes

## Методологические решения
1. Реализация использует radial DEA scores (CCR/BCC; input/output), solved via LP (PuLP).
2. Реализован sequential two-stage approximation, а не network DEA.
3. Проверяется статус LP-решения; не-Optimal наблюдения попадают в dea_problematic_observations.
4. По данным качества: нули в inputs и отрицательные значения могут исключаться по конфигу.

## Ограничения
- В текущей версии не рассчитываются slacks/targets; доступен только peer_count как упрощённый индикатор reference set.
- Для строгой второй стадии DEA-регрессии (Simar-Wilson) нужны дополнительные процедуры bootstrap/truncated regression.
"""

    write_text("report/introduction.md", intro)
    write_text("report/methodology_dea.md", methodology)

    assumptions_path = Path("report/assumptions_and_method_notes.md")
    if not assumptions_path.exists():
        write_text(str(assumptions_path), assumptions)
