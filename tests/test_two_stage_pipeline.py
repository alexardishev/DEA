import pandas as pd

from src.dea.two_stage_analysis import run_two_stage_dea


def _config() -> dict:
    return {
        "io": {"id_column": "bank_id", "bank_name_column": "bank_name"},
        "analysis": {
            "dea": {
                "stage1": {
                    "returns_to_scale": "bcc",
                    "orientation": "input",
                    "inputs": ["staff_expenses", "operating_expenses", "tangible_assets"],
                    "outputs": ["deposits"],
                },
                "stage2": {
                    "returns_to_scale": "ccr",
                    "orientation": "input",
                    "inputs": ["deposits"],
                    "outputs": ["interest_income", "fee_income", "customer_loans"],
                },
            }
        },
    }


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bank_id": ["a", "b", "c", "d"],
            "bank_name": ["A", "B", "C", "D"],
            "bank_type": ["commercial", "commercial", "savings", "cooperative"],
            "staff_expenses": [10, 12, 9, 8],
            "operating_expenses": [6, 7, 6, 5],
            "tangible_assets": [20, 18, 16, 15],
            "deposits": [100, 110, 90, 80],
            "interest_income": [11, 12, 9, 8],
            "fee_income": [2, 1.5, 1.2, 0.8],
            "customer_loans": [70, 80, 60, 55],
        }
    )


def test_two_stage_outputs_have_separate_scores_and_no_loss_of_rows():
    out = run_two_stage_dea(_df(), _config())
    assert len(out.stage1_scores) == 4
    assert len(out.stage2_scores) == 4
    assert len(out.combined_summary) == 4
    assert "combined_summary_metric" in out.combined_summary.columns


def test_stage_specific_rts_applied():
    out = run_two_stage_dea(_df(), _config())
    assert set(out.stage1_scores["returns_to_scale"]) == {"bcc"}
    assert set(out.stage2_scores["returns_to_scale"]) == {"ccr"}
