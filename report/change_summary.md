# Change Summary (final acceptance fixes)

1. Исправлена загрузка Excel при `dataset_sheet: null`: теперь используется детерминированный fallback на первый лист и добавлена защитная обработка случая, когда `read_excel` вернул словарь листов.
2. Обновлён официальный запуск в README на стабильный entrypoint `python -m src.pipeline --stage ...`.
3. Исправлены standalone-режимы `viz` и `report`: теперь они читают `dea_within_group_scores.csv` для within-group результатов, а не подменяют их pooled-таблицей.
4. Убрано перезаписывание подробной методологической записки: `report/assumptions_and_method_notes.md` создаётся только при отсутствии файла.
5. Улучшен экспорт bank-level DEA stage-таблиц: добавлены `bank_name`, `bank_type`, `score`, `solution_status`, `is_efficient`, `rank_within_group` и диагностические поля.
6. Добавлен тест `tests/test_data_loader.py` для проверки безопасного поведения при `dataset_sheet: null`.
7. Выполнены технические проверки (compileall), а также предпринята попытка полного end-to-end запуска на реальном датасете (`python -m src.pipeline --stage all`), но запуск заблокирован отсутствием зависимостей (`pandas`) из-за сетевого прокси 403 при установке.
