# -*- coding: utf-8 -*-
"""
RED SI Pipeline 特征工程
=========================

基于清洗后的数据构造 Geometric Adstock 特征变量。
"""

from __future__ import annotations

import pandas as pd

from .utils import geometric_adstock


def add_model_features(clean: pd.DataFrame) -> pd.DataFrame:
    """在清洗数据上追加 adstock 特征列。"""
    model_df = clean.copy()

    model_df["sem_adstock_l030"] = geometric_adstock(model_df["red_sem_spend_mil"], 0.30)
    model_df["feeds_adstock_l050"] = geometric_adstock(model_df["red_feeds_spend_mil"], 0.50)
    model_df["koc_adstock_l050"] = geometric_adstock(model_df["koc_spend_mil"], 0.50)
    model_df["branding_adstock_l082"] = geometric_adstock(model_df["red_branding_spend_mil"], 0.82)

    return model_df
