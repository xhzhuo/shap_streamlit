# -*- coding: utf-8 -*-
"""
RED SI Pipeline 模型拟合
=========================

使用 Ridge / Positive Ridge 进行品牌搜索指数预测，
计算模型指标、系数表和 Dashboard 汇总。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from .config import FEATURES, SCENARIO_SIGMA_MULTIPLIER, POSITIVE_RIDGE_ALPHA
from .utils import root_mean_squared_error_safe, pct_vs


def fit_models(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    拟合诊断模型（RidgeCV）和最终模型（Positive Ridge），
    计算情景区间、模型指标、系数表和 Dashboard 汇总。
    """
    train_mask = model_df["brand_search_index_mil"].notna()
    forecast_mask = ~train_mask

    train = model_df.loc[train_mask].copy()
    forecast = model_df.loc[forecast_mask].copy()

    X_train = train[FEATURES].fillna(0)
    y_train = train["brand_search_index_mil"]

    # 诊断模型：普通 RidgeCV，可能出现负系数，仅用于观察
    alphas = np.logspace(-3, 3, 50)
    unconstrained_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    unconstrained_model.fit(X_train, y_train)
    unconstrained_pred = unconstrained_model.predict(X_train)

    # Final demo：Positive Ridge，保证标准化系数非负，增强业务解释性
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=POSITIVE_RIDGE_ALPHA, positive=True)),
    ])
    final_model.fit(X_train, y_train)
    train["fit_base"] = final_model.predict(X_train)

    # 写回整体数据
    model_df = model_df.copy()
    model_df["fit_base"] = np.nan
    model_df.loc[train_mask, "fit_base"] = train["fit_base"].values

    if len(forecast) > 0:
        model_df.loc[forecast_mask, "forecast_base"] = final_model.predict(forecast[FEATURES].fillna(0))
    else:
        model_df["forecast_base"] = np.nan

    model_df["base_line"] = model_df["fit_base"].combine_first(model_df["forecast_base"])

    # 情景区间：Base ± 0.9σ residual
    residual = train["brand_search_index_mil"] - train["fit_base"]
    sigma = float(residual.std(ddof=1))
    half_width = SCENARIO_SIGMA_MULTIPLIER * sigma
    model_df["optimistic"] = model_df["base_line"] + half_width
    model_df["pessimistic"] = model_df["base_line"] - half_width

    # 模型指标
    final_metrics = {
        "Model": "Final demo: Positive Ridge + fixed geometric adstock",
        "Training rows": int(train_mask.sum()),
        "Forecast rows": int(forecast_mask.sum()),
        "R2": r2_score(y_train, train["fit_base"]),
        "MAPE": mean_absolute_percentage_error(y_train, train["fit_base"]),
        "RMSE_mil": root_mean_squared_error_safe(y_train, train["fit_base"]),
        "Residual_sigma_mil": sigma,
        "Scenario_half_width_mil_0.9sigma": half_width,
        "Feature set": ", ".join(FEATURES),
    }

    unconstrained_metrics = {
        "Model": "Diagnostic only: standard RidgeCV, coefficients may be negative",
        "Alpha": unconstrained_model.named_steps["ridge"].alpha_,
        "R2": r2_score(y_train, unconstrained_pred),
        "MAPE": mean_absolute_percentage_error(y_train, unconstrained_pred),
        "RMSE_mil": root_mean_squared_error_safe(y_train, unconstrained_pred),
    }

    summary_metrics = pd.DataFrame([final_metrics, unconstrained_metrics])

    # 系数表：标准化后的系数，可用于比较方向和相对贡献
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coef_standardized_positive_ridge": final_model.named_steps["ridge"].coef_,
        "coef_standardized_unconstrained_ridge": unconstrained_model.named_steps["ridge"].coef_,
    })
    coef_df.loc[len(coef_df)] = [
        "intercept",
        final_model.named_steps["ridge"].intercept_,
        unconstrained_model.named_steps["ridge"].intercept_,
    ]

    # Dashboard-like 汇总
    actual_2025 = train[train.week_start.dt.year == 2025]["brand_search_index_mil"]
    last_12_actual = train.tail(12)["brand_search_index_mil"]

    fcst = model_df.loc[forecast_mask, "forecast_base"].dropna()
    opt = model_df.loc[forecast_mask, "optimistic"].dropna()
    pes = model_df.loc[forecast_mask, "pessimistic"].dropna()

    dashboard_rows = []
    if len(fcst) > 0:
        for comp_name, baseline in [
            ("Vs 2025 actual avg", actual_2025.mean()),
            ("Vs last 12 actual weeks avg", last_12_actual.mean()),
        ]:
            dashboard_rows.append({
                "comparison": comp_name,
                "pessimistic_avg_pct": pct_vs(pes.mean(), baseline),
                "base_avg_pct": pct_vs(fcst.mean(), baseline),
                "optimistic_avg_pct": pct_vs(opt.mean(), baseline),
                "baseline_value_mil": baseline,
                "forecast_base_avg_mil": fcst.mean(),
                "forecast_pessimistic_avg_mil": pes.mean(),
                "forecast_optimistic_avg_mil": opt.mean(),
            })
    dashboard_df = pd.DataFrame(dashboard_rows)

    return model_df, final_metrics, summary_metrics, coef_df, dashboard_df
