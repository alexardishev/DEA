"""DEA model implementation via linear programming (PuLP).

This module intentionally uses transparent LP formulation instead of opaque wrappers,
so that students can inspect and defend each methodological choice.
"""

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
            theta = pulp.LpVariable("theta", lowBound=0)
            prob = pulp.LpProblem("DEA_Input", pulp.LpMinimize)
            prob += theta

            for i, col in enumerate(X.columns):
                prob += (
                    pulp.lpSum(lambdas[f"lambda_{j}"] * X.iloc[j, i] for j in range(n))
                    <= theta * X.iloc[target_idx, i]
                )
            for r, col in enumerate(Y.columns):
                prob += (
                    pulp.lpSum(lambdas[f"lambda_{j}"] * Y.iloc[j, r] for j in range(n))
                    >= Y.iloc[target_idx, r]
                )
        else:
            phi = pulp.LpVariable("phi", lowBound=0)
            prob = pulp.LpProblem("DEA_Output", pulp.LpMaximize)
            prob += phi
            for i, col in enumerate(X.columns):
                prob += (
                    pulp.lpSum(lambdas[f"lambda_{j}"] * X.iloc[j, i] for j in range(n))
                    <= X.iloc[target_idx, i]
                )
            for r, col in enumerate(Y.columns):
                prob += (
                    pulp.lpSum(lambdas[f"lambda_{j}"] * Y.iloc[j, r] for j in range(n))
                    >= phi * Y.iloc[target_idx, r]
                )
            theta = phi

        if self.returns_to_scale == "bcc":
            prob += pulp.lpSum(lambdas.values()) == 1

        return prob, lambdas, theta

    def solve(self, data: pd.DataFrame, id_col: str, input_cols: list[str], output_cols: list[str]) -> DEAResult:
        X = data[input_cols].reset_index(drop=True)
        Y = data[output_cols].reset_index(drop=True)

        rows = []
        for idx in range(len(data)):
            prob, _, score_var = self._build_problem(X, Y, idx)
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            raw = score_var.value() if score_var.value() is not None else float("nan")

            # Normalize for output orientation to [0,1] style efficiency.
            if self.orientation == "output":
                eff = (1 / raw) if raw not in (None, 0) else float("nan")
            else:
                eff = raw

            rows.append({id_col: data.iloc[idx][id_col], "efficiency": eff})

        scores = pd.DataFrame(rows)
        return DEAResult(scores=scores, metadata={"rts": self.returns_to_scale, "orientation": self.orientation})
