"""DEA model implementation via linear programming (PuLP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pulp

ReturnsToScale = Literal["ccr", "bcc"]
Orientation = Literal["input", "output"]


@dataclass
class DEAResult:
    scores: pd.DataFrame
    problematic: pd.DataFrame
    metadata: dict


class DEASolver:
    """Run DEA for a set of DMUs with configurable RTS/orientation."""

    def __init__(self, returns_to_scale: ReturnsToScale = "bcc", orientation: Orientation = "input"):
        self.returns_to_scale = returns_to_scale
        self.orientation = orientation

    def _build_problem(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        target_idx: int,
    ) -> tuple[pulp.LpProblem, dict[str, pulp.LpVariable], pulp.LpVariable]:
        n = len(X)
        lambdas = {f"lambda_{j}": pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in range(n)}

        if self.orientation == "input":
            score_var = pulp.LpVariable("theta", lowBound=0)
            prob = pulp.LpProblem("DEA_Input", pulp.LpMinimize)
            prob += score_var

            for i in range(len(X.columns)):
                prob += pulp.lpSum(lambdas[f"lambda_{j}"] * X.iloc[j, i] for j in range(n)) <= score_var * X.iloc[target_idx, i]
            for r in range(len(Y.columns)):
                prob += pulp.lpSum(lambdas[f"lambda_{j}"] * Y.iloc[j, r] for j in range(n)) >= Y.iloc[target_idx, r]
        else:
            score_var = pulp.LpVariable("phi", lowBound=0)
            prob = pulp.LpProblem("DEA_Output", pulp.LpMaximize)
            prob += score_var

            for i in range(len(X.columns)):
                prob += pulp.lpSum(lambdas[f"lambda_{j}"] * X.iloc[j, i] for j in range(n)) <= X.iloc[target_idx, i]
            for r in range(len(Y.columns)):
                prob += pulp.lpSum(lambdas[f"lambda_{j}"] * Y.iloc[j, r] for j in range(n)) >= score_var * Y.iloc[target_idx, r]

        if self.returns_to_scale == "bcc":
            prob += pulp.lpSum(lambdas.values()) == 1

        return prob, lambdas, score_var

    def solve(self, data: pd.DataFrame, id_col: str, input_cols: list[str], output_cols: list[str]) -> DEAResult:
        """Solve DEA and explicitly track LP status/problematic DMUs."""
        X = data[input_cols].reset_index(drop=True)
        Y = data[output_cols].reset_index(drop=True)

        rows: list[dict] = []
        bad_rows: list[dict] = []

        for idx in range(len(data)):
            prob, lambdas, score_var = self._build_problem(X, Y, idx)
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            status_code = prob.status
            status_text = pulp.LpStatus[status_code]
            raw = score_var.value() if score_var.value() is not None else float("nan")

            if status_text != "Optimal":
                bad_rows.append(
                    {
                        id_col: data.iloc[idx][id_col],
                        "dmu_index": idx,
                        "status": status_text,
                        "reason": "LP solver did not return optimal solution",
                        "returns_to_scale": self.returns_to_scale,
                        "orientation": self.orientation,
                    }
                )
                eff = float("nan")
                lambda_sum = float("nan")
                peer_count = 0
            else:
                if self.orientation == "output":
                    eff = (1 / raw) if raw not in (None, 0) else float("nan")
                else:
                    eff = raw

                lambda_values = [var.value() or 0.0 for var in lambdas.values()]
                lambda_sum = float(sum(lambda_values))
                peer_count = int(sum(v > 1e-7 for v in lambda_values))

            rows.append(
                {
                    id_col: data.iloc[idx][id_col],
                    "dmu_index": idx,
                    "efficiency": eff,
                    "status": status_text,
                    "raw_score": raw,
                    "lambda_sum": lambda_sum,
                    "peer_count": peer_count,
                    "returns_to_scale": self.returns_to_scale,
                    "orientation": self.orientation,
                    "is_efficient": bool(pd.notna(eff) and abs(eff - 1.0) <= 1e-6),
                }
            )

        scores = pd.DataFrame(rows)
        problematic = pd.DataFrame(bad_rows)
        metadata = {
            "rts": self.returns_to_scale,
            "orientation": self.orientation,
            "supports_peer_reference": True,
            "supports_slacks": False,
            "supports_targets": False,
            "note": "Current implementation reports radial scores and peer_count only; slacks/targets are not computed.",
        }
        return DEAResult(scores=scores, problematic=problematic, metadata=metadata)
