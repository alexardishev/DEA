# Austrian Banking DEA Pipeline (Python-only)

Воспроизводимый пайплайн для анализа банков Австрии: подготовка данных, **двухэтапный DEA**, межгрупповые сравнения, регрессии второй стадии, визуализации и текстовые черновики для университетской работы.

## Ключевое методологическое уточнение

В проекте реализована **sequential two-stage DEA approximation**:
- Stage 1 и Stage 2 решаются отдельно;
- `combined_summary_metric` — это агрегированный summary-индикатор,
- это **не** строгая network DEA итоговая эффективность.

Подробно: `report/assumptions_and_method_notes.md`.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --stage all
```

## Структура

- `data_raw/` — исходные данные
- `data_processed/` — очищенные данные
- `config/` — конфиги
- `src/` — код пайплайна
- `outputs/` — таблицы, графики, логи, тексты
- `report/` — методологические и текстовые черновики
- `tests/` — тесты

## DEA-конфигурация по этапам

Файл `config/pipeline_config.yaml`:
- `analysis.dea.stage1.returns_to_scale`, `analysis.dea.stage1.orientation`
- `analysis.dea.stage2.returns_to_scale`, `analysis.dea.stage2.orientation`
- поддерживаются `bcc/ccr` и `input/output`.

## Обработка проблемных наблюдений

Файл `config/filter_rules.yaml`:
- правила нулей в inputs/outputs;
- правила отрицательных значений;
- IQR-outlier flag/exclude;
- manual exclusions.

Результаты записываются в:
- `outputs/logs/data_quality_report.csv`
- `outputs/logs/excluded_observations.csv`
- `outputs/tables/dea_problematic_observations.csv`

## DEA-выходы

- `outputs/tables/dea_stage1_scores.csv/xlsx`
- `outputs/tables/dea_stage2_scores.csv/xlsx`
- `outputs/tables/dea_combined_summary.csv/xlsx`
- `outputs/tables/dea_within_group_scores.csv/xlsx`
- `outputs/tables/dea_within_group_summary.csv/xlsx`
- `outputs/tables/dea_between_group_summary.csv/xlsx`
- `outputs/tables/dea_problematic_observations.csv/xlsx`
- `outputs/tables/dea_sensitivity_comparison.csv/xlsx` (если включено)

## Визуализации

Строятся отдельно для Stage 1 и Stage 2:
- violin, box, density, bar (group means),
- scatter.

## Поэтапный запуск

```bash
python -m src.pipeline --stage data
python -m src.pipeline --stage dea
python -m src.pipeline --stage regression
python -m src.pipeline --stage viz
python -m src.pipeline --stage report
```

## Ограничения текущей DEA-реализации

- рассчитываются radial efficiency scores;
- доступен упрощённый peer-индикатор (`peer_count`);
- slacks/targets не реализованы и это явно фиксируется в методологических заметках.
