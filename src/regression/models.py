"""Second-stage regression models for explaining DEA scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class RegressionOutput:
    ols_summary_text: str
    fractional_glm_summary_text: str | None
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame


def _prepare_design_matrix(df: pd.DataFrame, y_col: str, predictors: list[str], include_type_dummies: bool) -> tuple[pd.Series, pd.DataFrame]:
    data = df.copy()
    cols = [c for c in predictors if c in data.columns]
    X = data[cols].copy()

    if include_type_dummies and "bank_type" in data.columns:
        dummies = pd.get_dummies(data["bank_type"], prefix="type", drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[y_col], errors="coerce")

    joined = pd.concat([y, X], axis=1).dropna()
    y = joined[y_col]
    X = joined.drop(columns=[y_col])
    X = sm.add_constant(X, has_constant="add")
    return y, X


def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    vals = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vals.append({"variable": col, "vif": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(vals)


def run_regressions(df: pd.DataFrame, config: dict) -> RegressionOutput:
    reg_cfg = config["analysis"]["regression"]
    y_col = reg_cfg["dependent_metric"]
    y, X = _prepare_design_matrix(
        df,
        y_col=y_col,
        predictors=reg_cfg["base_predictors"],
        include_type_dummies=reg_cfg.get("include_bank_type_dummies", True),
    )

    ols_model = sm.OLS(y, X).fit(cov_type="HC3")
    coef_df = ols_model.summary2().tables[1].reset_index().rename(columns={"index": "variable"})

    glm_text = None
    if reg_cfg.get("run_fractional_glm", True):
        # Fractional logit is often preferred over OLS for bounded response in [0,1].
        y_glm = y.clip(1e-6, 1 - 1e-6)
        glm_model = sm.GLM(y_glm, X, family=Binomial()).fit()
        glm_text = glm_model.summary().as_text()

    diagnostics = _compute_vif(X)
    return RegressionOutput(
        ols_summary_text=ols_model.summary().as_text(),
        fractional_glm_summary_text=glm_text,
        coefficients=coef_df,
        diagnostics=diagnostics,
    )
