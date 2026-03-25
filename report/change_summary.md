# Change Summary (vs previous version)

1. Усилена двухэтапная DEA-логика: stage 1 и stage 2 сохраняются отдельно, агрегат переименован в `combined_summary_metric` и явно помечен как approximation.
2. DEA-параметры разделены по этапам: отдельные RTS/orientation для stage1 и stage2 в конфиге.
3. Добавлена проверка LP-статуса для каждой DMU с отдельной таблицей проблемных наблюдений.
4. Усилена обработка проблемных данных: правила нулей в inputs/outputs, outlier-правила, ручные исключения, причины в quality report.
5. Реально реализован sensitivity analysis по сценариям из конфига с экспортом `dea_sensitivity_comparison`.
6. Расширены DEA-выходы: отдельные stage-таблицы, within/between summaries, problematic observations.
7. Визуализации DEA теперь строятся отдельно для stage 1 и stage 2.
8. Расширены тесты DEA и data validation (RTS-ограничения, статусы, two-stage outputs, нулевые inputs).
9. Обновлена документация и добавлена методологическая записка `report/assumptions_and_method_notes.md`.
