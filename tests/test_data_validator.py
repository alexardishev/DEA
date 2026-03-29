import pandas as pd

from src.data.data_validator import DataValidator


def _config() -> dict:
    return {
        "io": {"id_column": "bank_id", "bank_name_column": "bank_name"},
        "required_columns": [
            "bank_name",
            "bank_type",
            "staff_expenses",
            "operating_expenses",
            "tangible_assets",
            "deposits",
            "interest_income",
            "fee_income",
            "customer_loans",
        ],
        "optional_columns": [],
        "quality_rules": {
            "critical_nonpositive_columns": [
                "staff_expenses",
                "operating_expenses",
                "tangible_assets",
                "deposits",
                "interest_income",
                "fee_income",
                "customer_loans",
            ],
            "drop_duplicates": True,
        },
        "analysis": {
            "dea": {
                "stage1": {"inputs": ["staff_expenses", "operating_expenses", "tangible_assets"], "outputs": ["deposits"]},
                "stage2": {"inputs": ["deposits"], "outputs": ["interest_income", "fee_income", "customer_loans"]},
            }
        },
    }


def _rules() -> dict:
    return {
        "manual_exclusions": {"bank_names": [], "bank_ids": []},
        "missingness": {"max_missing_ratio_per_row": 0.3},
        "zero_rules": {"inputs": "exclude", "outputs": "flag", "allow_zero_outputs": ["fee_income"]},
        "negative_rules": {"numeric_columns": "exclude"},
        "outlier_rules": {"multiplier": 3.0, "action": "flag"},
    }


def test_zero_input_excluded_and_logged():
    df = pd.DataFrame(
        {
            "bank_id": [1, 2],
            "bank_name": ["A", "B"],
            "bank_type": ["commercial", "savings"],
            "staff_expenses": [0, 10],
            "operating_expenses": [5, 5],
            "tangible_assets": [10, 10],
            "deposits": [100, 100],
            "interest_income": [10, 10],
            "fee_income": [1, 1],
            "customer_loans": [80, 80],
        }
    )
    res = DataValidator(_config(), _rules()).run(df)
    assert len(res.cleaned_df) == 1
    assert (res.quality_report["issue_type"] == "zero_input").any()
