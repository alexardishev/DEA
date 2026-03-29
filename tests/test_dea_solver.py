import pandas as pd

from src.dea.dea_model import DEASolver


def _toy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bank_id": ["A", "B", "C", "D"],
            "x1": [1.0, 2.0, 1.5, 1.1],
            "y1": [1.0, 2.1, 1.4, 1.0],
        }
    )


def test_efficiency_bounds_input_oriented():
    df = _toy_data()
    result = DEASolver(returns_to_scale="bcc", orientation="input").solve(df, "bank_id", ["x1"], ["y1"]).scores
    assert result["efficiency"].dropna().between(0, 1.000001).all()


def test_bcc_enforces_lambda_sum_close_to_one():
    df = _toy_data()
    result = DEASolver(returns_to_scale="bcc", orientation="input").solve(df, "bank_id", ["x1"], ["y1"]).scores
    assert (result["status"] == "Optimal").all()
    assert (result["lambda_sum"].sub(1.0).abs() < 1e-5).all()


def test_ccr_does_not_force_lambda_sum_to_one_for_all_dmus():
    df = _toy_data()
    result = DEASolver(returns_to_scale="ccr", orientation="input").solve(df, "bank_id", ["x1"], ["y1"]).scores
    # At least one DMU should typically differ from exactly 1 due to no convexity constraint.
    assert (result["lambda_sum"].sub(1.0).abs() > 1e-6).any()


def test_solver_status_column_is_exposed():
    df = _toy_data()
    out = DEASolver(returns_to_scale="bcc", orientation="output").solve(df, "bank_id", ["x1"], ["y1"]).scores
    assert "status" in out.columns
