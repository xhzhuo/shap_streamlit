# -*- coding: utf-8 -*-
"""
RED SI Pipeline 工具函数
========================

通用的数据处理、数值计算、格式化辅助函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# =========================
# 数据解析
# =========================
def parse_week_start(week_text: object) -> pd.Timestamp:
    """从类似 '2025/1/6 - 2025/1/12' 的文本中解析周起始日期。"""
    s = str(week_text).strip().replace(" ", "")
    start_text = s.split("-")[0]
    return pd.to_datetime(start_text, format="%Y/%m/%d", errors="coerce")


def to_num(series: pd.Series, fill_value=None) -> pd.Series:
    """安全转数值。"""
    out = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


# =========================
# Adstock 变换
# =========================
def geometric_adstock(series: pd.Series, lam: float) -> np.ndarray:
    """
    Geometric Adstock:
        adstock_t = x_t + lambda * adstock_{t-1}
    """
    x = np.asarray(pd.Series(series).fillna(0), dtype=float)
    out = np.zeros(len(x), dtype=float)
    for i, value in enumerate(x):
        out[i] = value + (lam * out[i - 1] if i else 0)
    return out


# =========================
# 模型评估
# =========================
def root_mean_squared_error_safe(y_true, y_pred) -> float:
    """兼容不同 sklearn 版本的 RMSE 写法。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =========================
# 数值 / 百分比计算
# =========================
def pct_vs(value: float, baseline: float) -> float:
    """计算相对变化。"""
    if pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return value / baseline - 1


def pct_change(value: float, baseline: float) -> float:
    """计算 value 相比 baseline 的变化百分比。"""
    if pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return value / baseline - 1


def fmt_pct(x: float) -> str:
    """格式化为 +12% / -8% 这种形式。"""
    if pd.isna(x):
        return "NA"
    return f"{x:+.0%}"


def get_period_avg(
    df: pd.DataFrame,
    start: str,
    end: str,
    value_col: str,
    fallback_cols: list[str] | None = None,
) -> float:
    """
    计算指定日期区间的均值。

    value_col：优先使用的字段。
    fallback_cols：如果 value_col 缺失，可以依次尝试这些字段。
    """
    mask = (df["week_start"] >= pd.to_datetime(start)) & (df["week_start"] <= pd.to_datetime(end))
    sub = df.loc[mask].copy()

    if sub.empty:
        return np.nan

    values = pd.to_numeric(sub[value_col], errors="coerce")

    if values.notna().sum() > 0:
        return float(values.mean())

    if fallback_cols:
        for col in fallback_cols:
            if col in sub.columns:
                fallback_values = pd.to_numeric(sub[col], errors="coerce")
                if fallback_values.notna().sum() > 0:
                    return float(fallback_values.mean())

    return np.nan
