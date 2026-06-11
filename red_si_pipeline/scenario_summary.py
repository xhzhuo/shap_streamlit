# -*- coding: utf-8 -*-
"""
RED SI Pipeline 情景汇总分析
==============================

计算第四张图 / Scenario Forecast Summary 所需数据：
    1. Forecast Avg vs Y25 Avg
    2. Forecast Avg vs Y26 H1 Avg
    3. Pessimistic / Base / Optimistic 三种情景的变化百分比

核心公式：
    base_vs_y25 = forecast_base_avg / y25_actual_avg - 1
    opt_vs_y25  = optimistic_avg / y25_actual_avg - 1
    pes_vs_y25  = pessimistic_avg / y25_actual_avg - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FORECAST_START,
    FORECAST_END,
    Y25_START,
    Y25_END,
    Y26_H1_START,
    Y26_H1_END,
)
from .utils import pct_change, fmt_pct, get_period_avg


def calculate_scenario_summary(
    model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于模型输出 DataFrame 计算情景汇总。

    返回:
        summary_df: 完整的情景汇总数据
        dashboard_table: 适合直接贴到 PPT/dashboard 的精简宽表
    """
    df = model_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])

    required_cols = [
        "week_start",
        "brand_search_index_mil",
        "forecast_base",
        "optimistic",
        "pessimistic",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少必要字段：{missing}")

    # 1. 定义预测期
    if FORECAST_START is not None and FORECAST_END is not None:
        forecast_mask = (
            (df["week_start"] >= pd.to_datetime(FORECAST_START))
            & (df["week_start"] <= pd.to_datetime(FORECAST_END))
        )
    else:
        # 默认逻辑：actual brand SI 为空且 forecast_base 有值的行 = 预测期
        forecast_mask = df["brand_search_index_mil"].isna() & df["forecast_base"].notna()

    forecast_df = df.loc[forecast_mask].copy()
    if forecast_df.empty:
        raise ValueError("没有识别到预测期数据。请检查 forecast_base 或手动设置 FORECAST_START / FORECAST_END。")

    # 2. 计算预测期三种情景均值
    forecast_base_avg = float(pd.to_numeric(forecast_df["forecast_base"], errors="coerce").mean())
    forecast_optimistic_avg = float(pd.to_numeric(forecast_df["optimistic"], errors="coerce").mean())
    forecast_pessimistic_avg = float(pd.to_numeric(forecast_df["pessimistic"], errors="coerce").mean())

    # 3. 计算对比基准
    y25_avg = get_period_avg(
        df,
        Y25_START,
        Y25_END,
        value_col="brand_search_index_mil",
    )

    # Y26 H1：优先用 actual，如果 actual 缺失，可 fallback 到 base_line/forecast_base
    y26_h1_avg = get_period_avg(
        df,
        Y26_H1_START,
        Y26_H1_END,
        value_col="brand_search_index_mil",
        fallback_cols=["base_line", "forecast_base"],
    )

    # 4. 计算第四张图表格数据
    rows = []
    for comparison_name, baseline_avg in [
        ("Vs Y25 avg.", y25_avg),
        ("Vs Y26 H1 avg.", y26_h1_avg),
    ]:
        pes_pct = pct_change(forecast_pessimistic_avg, baseline_avg)
        base_pct = pct_change(forecast_base_avg, baseline_avg)
        opt_pct = pct_change(forecast_optimistic_avg, baseline_avg)

        rows.append({
            "comparison": comparison_name,
            "baseline_avg_mil": baseline_avg,
            "pessimistic_avg_mil": forecast_pessimistic_avg,
            "base_avg_mil": forecast_base_avg,
            "optimistic_avg_mil": forecast_optimistic_avg,
            "pessimistic_pct": pes_pct,
            "base_pct": base_pct,
            "optimistic_pct": opt_pct,
            "pessimistic_pct_label": fmt_pct(pes_pct),
            "base_pct_label": fmt_pct(base_pct),
            "optimistic_pct_label": fmt_pct(opt_pct),
            "scenario_range_label": f"{fmt_pct(pes_pct)} to {fmt_pct(opt_pct)}",
        })

    summary_df = pd.DataFrame(rows)

    # 5. 额外输出一个适合直接贴到 PPT/dashboard 的宽表
    dashboard_table = summary_df[[
        "comparison",
        "pessimistic_pct_label",
        "base_pct_label",
        "optimistic_pct_label",
        "scenario_range_label",
    ]].rename(columns={
        "comparison": "Comparison",
        "pessimistic_pct_label": "Pessimistic",
        "base_pct_label": "Base",
        "optimistic_pct_label": "Optimistic",
        "scenario_range_label": "Scenario Range",
    })

    return summary_df, dashboard_table
