# -*- coding: utf-8 -*-
"""
RED SI Pipeline 参数与常量
==========================

集中管理所有可调参数、文件路径、特征列表等。
"""

# =========================
# 输入文件
# =========================
INPUT_FILE = "RED SI归因_Data_Input_Wyeth.xlsx"
INPUT_SHEET = "Data 原始"

# =========================
# 输出文件
# =========================
OUTPUT_XLSX = "red_si_attribution_model_demo_output.xlsx"
OUTPUT_PNG = "red_si_attribution_forecast_demo_chart.png"
OUTPUT_HTML = "red_si_attribution_forecast_demo_chart.html"

# =========================
# Adstock 参数
# =========================
ADSTOCK_LAMBDAS = {
    "red_sem_spend_mil": 0.30,
    "red_feeds_spend_mil": 0.50,
    "koc_spend_mil": 0.50,
    "red_branding_spend_mil": 0.82,
}

# =========================
# 模型特征列表
# =========================
FEATURES = [
    "sem_adstock_l030",
    "feeds_adstock_l050",
    "koc_adstock_l050",
    "branding_adstock_l082",
    "industry_total_search_index_mil",
    "nsr",
    "is_big_promo"
]

# =========================
# 情景分析参数
# =========================
SCENARIO_SIGMA_MULTIPLIER = 0.9

# =========================
# Positive Ridge alpha
# =========================
POSITIVE_RIDGE_ALPHA = 5.0

# =========================
# 情景汇总对比基准区间
# =========================
Y25_START = "2025-01-01"
Y25_END = "2025-12-31"
Y26_H1_START = "2026-01-01"
Y26_H1_END = "2026-06-30"

# 预测期：设为 None 则自动识别（actual 为空且 forecast_base 有值）
FORECAST_START = None  # 例如 "2026-01-05"
FORECAST_END = None    # 例如 "2026-03-15"
