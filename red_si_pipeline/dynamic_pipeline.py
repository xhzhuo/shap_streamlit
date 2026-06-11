# -*- coding: utf-8 -*-
"""
Configuration-driven Adstock + Ridge modeling.

This module supports arbitrary user-selected field roles instead of fixed
RED SI column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .field_profiler import coerce_numeric, parse_period_start
from .utils import geometric_adstock, root_mean_squared_error_safe


@dataclass
class DynamicModelConfig:
    time_col: str
    target_col: str
    media_cols: list[str]
    control_cols: list[str]
    event_cols: list[str]
    diagnostic_cols: list[str]
    adstock_lambdas: dict[str, float]
    category_cols: list[str] = field(default_factory=list)
    unit_scales: dict[str, float] = field(default_factory=dict)
    unit_records: list[dict[str, Any]] = field(default_factory=list)
    media_missing_strategy: str = "zero"
    control_missing_strategy: str = "mean"
    event_missing_strategy: str = "zero"
    scenario_sigma_multiplier: float = 0.9
    positive_ridge_alpha: float = 5.0
    use_positive_ridge: bool = True


def build_quality_checks(df: pd.DataFrame, config: DynamicModelConfig) -> list[dict[str, str]]:
    """Run pre-modeling checks. Returns status/message dictionaries."""
    checks: list[dict[str, str]] = []

    def add(status: str, item: str, detail: str) -> None:
        checks.append({"status": status, "item": item, "detail": detail})

    if not config.time_col:
        add("阻断", "时间字段", "请选择 1 个时间字段。")
    elif config.time_col not in df.columns:
        add("阻断", "时间字段", f"字段不存在：{config.time_col}")
    else:
        parsed = parse_period_start(df[config.time_col])
        ratio = parsed.notna().mean()
        if ratio < 0.75:
            add("阻断", "时间解析", f"仅 {ratio:.0%} 的行可解析为时间。")
        else:
            add("通过", "时间解析", f"{ratio:.0%} 的行可解析为时间。")
            unique_periods = parsed.dropna().nunique()
            if unique_periods < parsed.dropna().shape[0]:
                add("警告", "重复周期", "检测到重复时间周期，建议确认是否需要聚合。")
            diffs = parsed.dropna().sort_values().diff().dropna()
            if not diffs.empty and diffs.nunique() > 3:
                add("警告", "周期连续性", "时间间隔不稳定，Adstock 连续累积可能受影响。")

    if not config.target_col:
        add("阻断", "目标变量", "请选择 1 个目标变量。")
    elif config.target_col not in df.columns:
        add("阻断", "目标变量", f"字段不存在：{config.target_col}")
    else:
        y = coerce_numeric(df[config.target_col])
        train_rows = int(y.notna().sum())
        forecast_rows = int(y.isna().sum())
        if train_rows < 12:
            add("阻断", "训练样本", f"目标变量仅 {train_rows} 行有值，至少建议 12 行。")
        elif train_rows < 30:
            add("警告", "训练样本", f"目标变量有 {train_rows} 行，样本偏少，结果适合方向判断。")
        else:
            add("通过", "训练样本", f"目标变量有 {train_rows} 行可训练。")
        add("通过" if forecast_rows > 0 else "警告", "预测期识别", f"目标变量为空的预测期行数：{forecast_rows}")

    model_cols = config.media_cols + config.control_cols + config.event_cols
    if not model_cols:
        add("阻断", "建模变量", "至少选择 1 个投放、控制或事件变量。")
    else:
        add("通过", "建模变量", f"已选择 {len(model_cols)} 个建模变量。")

    missing_cols = [c for c in model_cols if c not in df.columns]
    if missing_cols:
        add("阻断", "字段存在性", f"以下字段不存在：{', '.join(missing_cols)}")

    for col in config.category_cols:
        if col not in df.columns:
            add("阻断", "品类变量", f"字段不存在：{col}")
            continue
        values = coerce_numeric(df[col])
        ratio = values.notna().mean()
        if ratio < 0.75:
            add("警告", "品类变量", f"{col} 仅 {ratio:.0%} 的行可识别为数值，图表参考线可能不完整。")
        else:
            in_model = col in model_cols
            detail = f"{col} 将作为品类参考线展示，缺失处理复用控制变量策略（{config.control_missing_strategy}）。"
            if in_model:
                detail += " 该字段同时被选为建模变量，会按对应角色进入模型。"
            else:
                detail += " 仅选择为品类变量时不会额外进入模型。"
            add("通过", "品类变量", detail)

    for col in config.media_cols:
        values = coerce_numeric(df[col]) if col in df.columns else pd.Series(dtype=float)
        if values.notna().any() and values.dropna().lt(0).any():
            add("警告", "投放变量负值", f"{col} 存在负数，请确认是否为退款/冲销。")
        if values.fillna(0).eq(0).mean() > 0.85:
            add("警告", "投放变量稀疏", f"{col} 超过 85% 的行为空或 0，系数可能不稳定。")

    if len(model_cols) >= 2:
        numeric_x = pd.DataFrame({c: coerce_numeric(df[c]) for c in model_cols if c in df.columns})
        corr = numeric_x.corr().abs()
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if pd.notna(corr.loc[a, b]) and corr.loc[a, b] >= 0.85:
                    pairs.append(f"{a} / {b} ({corr.loc[a, b]:.2f})")
        if pairs:
            add("警告", "变量共线性", "高相关变量：" + "; ".join(pairs[:5]))

    return checks


def _apply_missing_strategy(values: pd.Series, strategy: str) -> pd.Series:
    if strategy == "zero":
        return values.fillna(0)
    if strategy == "ffill":
        return values.ffill().bfill()
    if strategy == "none":
        return values
    return values.fillna(values.mean())


def _scale_numeric(df: pd.DataFrame, col: str, config: DynamicModelConfig) -> pd.Series:
    scale = float(config.unit_scales.get(col, 1.0))
    return coerce_numeric(df[col]) * scale


def prepare_model_data(df: pd.DataFrame, config: DynamicModelConfig) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Clean fields and construct model feature columns."""
    clean = pd.DataFrame(index=df.index)
    clean["period_raw"] = df[config.time_col]
    clean["period_start"] = parse_period_start(df[config.time_col])
    clean["actual"] = _scale_numeric(df, config.target_col, config)

    feature_cols: list[str] = []
    feature_roles: dict[str, str] = {}

    for col in config.media_cols:
        base_col = f"media__{col}"
        lam = float(config.adstock_lambdas.get(col, 0.5))
        lam = min(max(lam, 0.0), 0.98)
        clean[base_col] = _apply_missing_strategy(_scale_numeric(df, col, config), config.media_missing_strategy)
        feat_col = f"adstock__{col}__l{int(round(lam * 100)):02d}"
        clean[feat_col] = geometric_adstock(clean[base_col], lam)
        if config.media_missing_strategy == "none":
            clean.loc[clean[base_col].isna(), feat_col] = np.nan
        feature_cols.append(feat_col)
        feature_roles[feat_col] = "投放变量"

    for col in config.control_cols:
        feat_col = f"control__{col}"
        clean[feat_col] = _apply_missing_strategy(_scale_numeric(df, col, config), config.control_missing_strategy)
        feature_cols.append(feat_col)
        feature_roles[feat_col] = "控制变量"

    for col in config.event_cols:
        feat_col = f"event__{col}"
        clean[feat_col] = _apply_missing_strategy(_scale_numeric(df, col, config), config.event_missing_strategy).clip(lower=0)
        feature_cols.append(feat_col)
        feature_roles[feat_col] = "事件变量"

    for col in config.diagnostic_cols:
        if col in df.columns:
            clean[f"diagnostic__{col}"] = _scale_numeric(df, col, config)

    for col in config.category_cols:
        if col in df.columns:
            clean[f"category__{col}"] = _apply_missing_strategy(_scale_numeric(df, col, config), config.control_missing_strategy)

    clean = clean.sort_values("period_start").reset_index(drop=True)
    return clean, feature_cols, feature_roles


def run_dynamic_adstock_ridge(df: pd.DataFrame, config: DynamicModelConfig) -> dict[str, Any]:
    """Fit the configured model and return all tables needed by the UI."""
    checks = build_quality_checks(df, config)
    blockers = [c for c in checks if c["status"] == "阻断"]
    if blockers:
        return {"ok": False, "checks": pd.DataFrame(checks), "error": "建模前检查存在阻断项。"}

    model_df, feature_cols, feature_roles = prepare_model_data(df, config)
    train_mask = model_df["actual"].notna()
    forecast_mask = ~train_mask

    x_all = model_df[feature_cols].copy()
    train_ready = train_mask & x_all.notna().all(axis=1)
    forecast_ready = forecast_mask & x_all.notna().all(axis=1)

    dropped_train = int(train_mask.sum() - train_ready.sum())
    if int(train_ready.sum()) < max(12, min(30, len(feature_cols) + 4)):
        checks.append({
            "status": "阻断",
            "item": "有效训练样本",
            "detail": f"清洗后仅 {int(train_ready.sum())} 行可训练，请减少缺失或调整变量。",
        })
        return {"ok": False, "checks": pd.DataFrame(checks), "error": "有效训练样本不足。"}

    X_train = x_all.loc[train_ready]
    y_train = model_df.loc[train_ready, "actual"]

    alphas = np.logspace(-3, 3, 50)
    diagnostic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    diagnostic_model.fit(X_train, y_train)
    diagnostic_pred = diagnostic_model.predict(X_train)

    ridge = Ridge(alpha=config.positive_ridge_alpha, positive=config.use_positive_ridge)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", ridge),
    ])
    final_model.fit(X_train, y_train)
    fit_pred = final_model.predict(X_train)

    model_df["fit_base"] = np.nan
    model_df.loc[train_ready, "fit_base"] = fit_pred
    model_df["forecast_base"] = np.nan
    if forecast_ready.any():
        model_df.loc[forecast_ready, "forecast_base"] = final_model.predict(x_all.loc[forecast_ready])

    model_df["base_line"] = model_df["fit_base"].combine_first(model_df["forecast_base"])
    residual = model_df.loc[train_ready, "actual"] - model_df.loc[train_ready, "fit_base"]
    sigma = float(residual.std(ddof=1)) if len(residual) > 1 else 0.0
    half_width = config.scenario_sigma_multiplier * sigma
    model_df["optimistic"] = model_df["base_line"] + half_width
    model_df["pessimistic"] = model_df["base_line"] - half_width
    model_df["residual"] = model_df["actual"] - model_df["fit_base"]
    model_df["abs_residual"] = model_df["residual"].abs()

    mape = mean_absolute_percentage_error(y_train, fit_pred)
    metrics = {
        "Model": "Positive Ridge + configurable geometric adstock" if config.use_positive_ridge else "Ridge + configurable geometric adstock",
        "Training rows": int(train_ready.sum()),
        "Forecast rows": int(forecast_ready.sum()),
        "Dropped training rows": dropped_train,
        "Feature count": len(feature_cols),
        "R2": float(r2_score(y_train, fit_pred)),
        "MAPE": float(mape),
        "RMSE": root_mean_squared_error_safe(y_train, fit_pred),
        "Residual sigma": sigma,
        "Scenario half width": half_width,
    }

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "role": [feature_roles[c] for c in feature_cols],
        "coef_standardized_final": final_model.named_steps["ridge"].coef_,
        "coef_standardized_diagnostic_ridge": diagnostic_model.named_steps["ridge"].coef_,
    })
    coef_df["abs_coef"] = coef_df["coef_standardized_final"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).reset_index(drop=True)

    diagnostics = build_diagnostics(model_df, config)
    summary_metrics = pd.DataFrame([metrics])
    checks_df = pd.DataFrame(checks)
    unit_records = pd.DataFrame(config.unit_records)

    return {
        "ok": True,
        "checks": checks_df,
        "model_df": model_df,
        "metrics": metrics,
        "summary_metrics": summary_metrics,
        "coef_df": coef_df,
        "diagnostics": diagnostics,
        "unit_records": unit_records,
        "feature_cols": feature_cols,
        "feature_roles": feature_roles,
        "model": final_model,
    }


def build_diagnostics(model_df: pd.DataFrame, config: DynamicModelConfig) -> dict[str, pd.DataFrame]:
    """Create residual, diagnostic-field, and forecast summary tables."""
    train = model_df.loc[model_df["actual"].notna()].copy()
    high_error_cols = ["period_start", "actual", "fit_base", "residual", "abs_residual"]
    diag_cols = [f"diagnostic__{c}" for c in config.diagnostic_cols if f"diagnostic__{c}" in model_df.columns]
    high_error = train.sort_values("abs_residual", ascending=False)[high_error_cols + diag_cols].head(12)

    residual_corr_rows = []
    if "residual" in train:
        for diag_col in diag_cols:
            valid = train[["residual", diag_col]].dropna()
            if len(valid) >= 8 and valid[diag_col].nunique() > 1:
                residual_corr_rows.append({
                    "diagnostic_field": diag_col.replace("diagnostic__", ""),
                    "corr_with_residual": valid["residual"].corr(valid[diag_col]),
                    "abs_corr": abs(valid["residual"].corr(valid[diag_col])),
                })
    residual_corr = pd.DataFrame(residual_corr_rows)
    if not residual_corr.empty:
        residual_corr = residual_corr.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])

    forecast = model_df.loc[model_df["forecast_base"].notna()].copy()
    forecast_summary = pd.DataFrame()
    if not forecast.empty:
        recent_actual = train.tail(12)["actual"].mean()
        forecast_summary = pd.DataFrame([{
            "forecast_base_avg": forecast["forecast_base"].mean(),
            "forecast_pessimistic_avg": forecast["pessimistic"].mean(),
            "forecast_optimistic_avg": forecast["optimistic"].mean(),
            "recent_12_actual_avg": recent_actual,
            "base_vs_recent_12": np.nan if pd.isna(recent_actual) or recent_actual == 0 else forecast["forecast_base"].mean() / recent_actual - 1,
        }])

    return {
        "high_error": high_error,
        "residual_corr": residual_corr,
        "forecast_summary": forecast_summary,
    }


def build_excel_bytes(
    profile_df: pd.DataFrame,
    role_df: pd.DataFrame,
    result: dict[str, Any],
) -> bytes:
    """Export dynamic modeling outputs to an in-memory Excel workbook."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        profile_df.to_excel(writer, sheet_name="01_Raw_Profile", index=False)
        role_df.to_excel(writer, sheet_name="02_Field_Roles", index=False)
        result["checks"].to_excel(writer, sheet_name="03_Quality_Checks", index=False)
        if result.get("ok"):
            result["summary_metrics"].to_excel(writer, sheet_name="04_Model_Metrics", index=False)
            result["coef_df"].to_excel(writer, sheet_name="05_Coefficients", index=False)
            result["model_df"].to_excel(writer, sheet_name="06_Model_Output", index=False)
            if "unit_records" in result and not result["unit_records"].empty:
                result["unit_records"].to_excel(writer, sheet_name="07_Unit_Conversion", index=False)
            for name, table in result["diagnostics"].items():
                sheet = f"08_{name}"[:31]
                table.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()
