# -*- coding: utf-8 -*-
"""
RED SI Pipeline 数据读取与清洗
===============================

负责从原始 Excel 中读取 `Data 原始` sheet，
清洗并返回结构化的 DataFrame。
"""

from __future__ import annotations

import pandas as pd

from .config import INPUT_FILE, INPUT_SHEET
from .utils import parse_week_start, to_num


def load_and_clean_raw(
    input_file: str = INPUT_FILE,
    sheet_name: str = INPUT_SHEET,
) -> pd.DataFrame:
    """读取原始 Excel，清洗为统一的 DataFrame。"""
    raw = pd.read_excel(input_file, sheet_name=sheet_name, engine="openpyxl")

    clean = pd.DataFrame()
    clean["week_raw"] = raw["Week/Month"]
    clean["week_start"] = raw["Week/Month"].apply(parse_week_start)
    clean["week_end"] = clean["week_start"] + pd.Timedelta(days=6)

    # 目标变量 Y
    clean["brand_search_index_mil"] = to_num(raw["RED Search Index (Mil.)"])

    # 核心投放/环境变量
    clean["red_sem_spend_mil"] = to_num(raw["RED SEM SPD (Mil.)"], fill_value=0)
    clean["red_feeds_spend_mil"] = to_num(raw["RED Feeds SPD (Mil.)"], fill_value=0)

    # Data 原始 中 KOC投资看起来是 RMB 级别，转成 Mil. 对齐 spend 单位
    clean["koc_spend_mil"] = to_num(raw["KOC投资"], fill_value=0) / 1_000_000

    clean["red_branding_spend_mil"] = to_num(raw["RED Branding SPD (Mil.)"], fill_value=0)
    clean["nsr"] = to_num(raw["NSR"])
    clean["industry_total_search_index_mil"] = to_num(
        raw["Industry Total Search Index (Mil.)（红书奶粉行业十大竞品主搜）"]
    )
    clean["is_big_promo"] = to_num(raw["Is_Big_Promo"], fill_value=0)

    # 诊断字段：保留，但默认不进入 final 模型
    diagnostic_mapping = {
        "RED SEM IMP": "red_sem_imp",
        "RED SEM CLICK": "red_sem_click",
        "RED SEM 回搜量": "red_sem_research",
        "RED Feeds IMP": "red_feeds_imp",
        "RED Feeds CLICK": "red_feeds_click",
        "RED Feeds 回搜量": "red_feeds_research",
        "RED Feeds Frequency（IMP）": "red_feeds_frequency_imp",
        "KOL笔记数": "kol_note_count",
        "KOC笔记数": "koc_note_count",
        "KOC阅读量": "koc_reads",
        "KOC曝光": "koc_impressions",
        "Branding IMP": "branding_imp",
        "Branding CLICK": "branding_click",
    }
    for src_col, dst_col in diagnostic_mapping.items():
        if src_col in raw.columns:
            clean[dst_col] = to_num(raw[src_col])

    return clean
