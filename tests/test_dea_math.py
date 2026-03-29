import pandas as pd

from src.dea.dea_model import DEASolver


def test_dea_scores_bounded_for_simple_case():
    df = pd.DataFrame(
        {
            "bank_id": ["a", "b", "c"],
            "x1": [1.0, 2.0, 1.5],
            "y1": [1.0, 2.0, 1.2],
        }
    )
    solver = DEASolver(returns_to_scale="bcc", orientation="input")
    result = solver.solve(df, "bank_id", ["x1"], ["y1"]).scores
    assert result["efficiency"].between(0, 1.000001).all()
