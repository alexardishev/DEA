# Austrian Banking DEA Pipeline (Python-only)

Полностью воспроизводимый пайплайн для двухэтапного DEA-анализа банков Австрии, межгрупповых сравнений, второй стадии регрессий и формирования академичных визуализаций/текстов.

## 1) Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --stage all
```

## 2) Структура проекта

- `data_raw/` — исходные данные (без правок)
- `data_processed/` — очищенные данные
- `config/` — конфиги параметров и правил фильтрации
- `src/` — основная логика пайплайна
  - `src/data/` — загрузка/валидация/трансформация
  - `src/dea/` — DEA (CCR/BCC, input/output, LP на PuLP)
  - `src/regression/` — OLS + fractional GLM, диагностики
  - `src/visualization/` — графики уровня статьи
  - `src/reporting/` — экспорт таблиц и текстовых интерпретаций
- `outputs/` — артефакты (таблицы, графики, карты, модели, логи, тексты)
- `report/` — методологические черновики
- `tests/` — минимальные тесты

## 3) Двухэтапная DEA-логика

### Stage 1: эффективность привлечения средств
- Inputs: `staff_expenses`, `operating_expenses`, `tangible_assets`
- Output: `deposits`

### Stage 2: эффективность трансформации привлечённых средств
- Input: `deposits`
- Outputs: `interest_income`, `fee_income` (`Provisionserträge`), `customer_loans`

## 4) Внутригрупповой и межгрупповой анализ

- Внутригрупповой: DEA отдельно для `cooperative`, `commercial`, `savings`
- Межгрупповой: DEA на объединённой выборке (общая граница эффективности)
- Экспортируются распределения, ранги, доля эффективных банков, описательная статистика.

## 5) Модели DEA и переключение

Файл `config/pipeline_config.yaml`:
- `default_returns_to_scale: bcc|ccr`
- `default_orientation: input|output`
- `run_sensitivity: true` и блок `sensitivity_models`

По умолчанию выбрана `BCC + input` (гетерогенный масштаб банков и управляемость input-факторов).

## 6) Регрессии второй стадии

- Базово: OLS (HC3)
- Дополнительно: fractional GLM (binomial link) для bounded score
- Dummy-переменные по типу банка добавляются автоматически
- Диагностика: VIF, таблица коэффициентов и значимостей

## 7) Обработка проблемных наблюдений

`config/filter_rules.yaml` управляет режимом:
- `flag_only`
- `exclude_critical` (по умолчанию)
- `winsorize_and_flag`

Логи и артефакты:
- `outputs/logs/data_quality_report.csv`
- `outputs/logs/excluded_observations.csv`

## 8) Какие поля ожидаются во входном датасете

Обязательно (после переименования):
- `bank_name`, `bank_type`
- `staff_expenses`, `operating_expenses`, `tangible_assets`, `deposits`
- `interest_income`, `fee_income`, `customer_loans`

Опционально:
- `year`, `total_assets`, `equity`, `R1`, `R2`, `R3`, `latitude`, `longitude`

## 9) Замена датасета без переписывания кода

1. Положить новый файл в `data_raw/`.
2. Обновить `paths.raw_data` в `config/pipeline_config.yaml`.
3. При новых названиях колонок обновить `column_mapping`.
4. При новых названиях типов банков обновить `bank_type_mapping`.
5. Запустить `python src/pipeline.py --stage all`.

## 10) Поэтапный запуск

```bash
python src/pipeline.py --stage data
python src/pipeline.py --stage dea
python src/pipeline.py --stage regression
python src/pipeline.py --stage viz
python src/pipeline.py --stage report
```

## 11) Где смотреть результаты

- Таблицы: `outputs/tables/`
- Графики: `outputs/figures/`
- Карты: `outputs/maps/`
- Тексты/интерпретации: `outputs/text/`
- Логи: `outputs/logs/`

## 12) Методологические примечания

См. `report/assumptions_and_method_notes.md`.
