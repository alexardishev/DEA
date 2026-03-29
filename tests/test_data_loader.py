from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.data_loader import DataLoader


def _cfg():
    return {
        "paths": {"raw_data": "dummy.xlsx"},
        "io": {"dataset_sheet": None, "id_column": "bank_id", "bank_type_column": "bank_type"},
        "column_mapping": {
            "Unternehmensname Latin alphabet": "bank_name",
            "Input 1 (=Personalaufwand)": "staff_expenses",
        },
        "bank_type_mapping": {},
    }


@patch("src.data.data_loader.pd.ExcelFile")
def test_null_sheet_uses_first_sheet(excel_file_mock):
    frame = pd.DataFrame({"A": [1]})
    xls = MagicMock()
    xls.sheet_names = ["Sheet1", "Sheet2"]
    xls.parse.return_value = frame
    excel_file_mock.return_value = xls

    df = DataLoader(_cfg()).load_raw()
    assert isinstance(df, pd.DataFrame)
    xls.parse.assert_called_once_with("Sheet1")


def test_rename_columns_normalizes_strip_and_symbols():
    df = pd.DataFrame(
        {
            " Unternehmensname Latin alphabet  ": ["X"],
            "Input 1 (=Personalaufwand) 5": [100.0],
        }
    )
    out = DataLoader(_cfg()).rename_columns(df)
    assert "bank_name" in out.columns
    assert "staff_expenses" in out.columns
