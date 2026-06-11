# -*- coding: utf-8 -*-
"""
Dynamic field profiling for the Adstock Ridge workflow.

The profiler does not enforce fixed column names. It infers likely field roles
from data shape, values, and light keyword hints, then lets users decide.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


TIME_HINTS = ("date", "week", "month", "period", "时间", "日期", "周", "月")
EVENT_HINTS = ("promo", "event", "flag", "holiday", "launch", "618", "1111", "促销", "大促", "节日", "新品", "事件")


def _norm_name(name: Any) -> str:
    return str(name).strip().lower()


def _contains_any(name: str, hints: tuple[str, ...]) -> bool:
    return any(h in name for h in hints)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert common business-formatted values to numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("，", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_period_start(series: pd.Series) -> pd.Series:
    """Parse dates, week ranges, and month-like strings into period starts."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    raw = series.copy()
    parsed_direct = pd.to_datetime(raw, errors="coerce")
    if parsed_direct.notna().mean() >= 0.75:
        return parsed_direct

    text = raw.astype(str).str.strip()

    def parse_one(value: str) -> pd.Timestamp:
        value = value.strip()
        if not value or value.lower() in {"nan", "none"}:
            return pd.NaT

        # Week ranges: "2025/1/6 - 2025/1/12", "2025-01-06~2025-01-12"
        parts = re.split(r"\s*(?:至|到|~|–|—|\s-\s)\s*", value, maxsplit=1)
        first = parts[0].strip()
        out = pd.to_datetime(first, errors="coerce")
        if pd.notna(out):
            return out

        # Fallback: first yyyy/mm/dd or yyyy-mm-dd occurrence.
        match = re.search(r"\d{4}[/-]\d{1,2}(?:[/-]\d{1,2})?", value)
        if match:
            return pd.to_datetime(match.group(0), errors="coerce")
        return pd.NaT

    return text.apply(parse_one)


def infer_field_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with objective type signals."""
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        name = _norm_name(col)
        numeric = coerce_numeric(s)
        period = parse_period_start(s)
        numeric_ratio = float(numeric.notna().sum() / n)
        period_ratio = float(period.notna().sum() / n)
        missing_rate = float(s.isna().sum() / n)
        unique_count = int(s.nunique(dropna=True))
        non_negative = bool(numeric.dropna().ge(0).all()) if numeric.notna().any() else False
        low_cardinality = unique_count <= min(8, max(3, int(n * 0.1)))

        if period_ratio >= 0.75 or _contains_any(name, TIME_HINTS):
            inferred_type = "时间候选"
        elif numeric_ratio >= 0.75:
            inferred_type = "数值"
            if low_cardinality or _contains_any(name, EVENT_HINTS):
                inferred_type = "低基数数值"
        else:
            inferred_type = "文本/分类"

        rows.append({
            "字段": col,
            "推断类型": inferred_type,
            "缺失率": missing_rate,
            "数值识别率": numeric_ratio,
            "时间识别率": period_ratio,
            "唯一值数": unique_count,
            "是否非负": non_negative,
        })

    return pd.DataFrame(rows)
