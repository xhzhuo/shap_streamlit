# -*- coding: utf-8 -*-
"""
RED SI Pipeline 主入口
=======================

编排完整流程：
    1. 读取并清洗原始数据
    2. 构造 Adstock 特征
    3. 拟合 Ridge / Positive Ridge 模型
    4. 计算情景汇总
    5. 导出 Excel、静态图、交互式图

运行方式（任选其一）：
    python -m red_si_pipeline.main
    python red_si_pipeline/main.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# 支持两种运行方式：
#   1) python -m red_si_pipeline.main   （包模式，相对导入）
#   2) python red_si_pipeline/main.py   （脚本模式，绝对导入）
try:
    from .config import INPUT_FILE, OUTPUT_XLSX, OUTPUT_PNG, OUTPUT_HTML
    from .data_loader import load_and_clean_raw
    from .feature_engineering import add_model_features
    from .modeling import fit_models
    from .plotting import plot_static_chart, plot_interactive_chart
    from .excel_export import export_excel
    from .scenario_summary import calculate_scenario_summary
except ImportError:
    # 脚本模式：把包的父目录加入 sys.path
    _pkg_dir = str(Path(__file__).resolve().parent.parent)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    from red_si_pipeline.config import INPUT_FILE, OUTPUT_XLSX, OUTPUT_PNG, OUTPUT_HTML
    from red_si_pipeline.data_loader import load_and_clean_raw
    from red_si_pipeline.feature_engineering import add_model_features
    from red_si_pipeline.modeling import fit_models
    from red_si_pipeline.plotting import plot_static_chart, plot_interactive_chart
    from red_si_pipeline.excel_export import export_excel
    from red_si_pipeline.scenario_summary import calculate_scenario_summary

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    # 1. 读取并清洗
    clean = load_and_clean_raw(INPUT_FILE)

    # 2. 特征工程
    model_df = add_model_features(clean)

    # 3. 模型拟合
    model_df, final_metrics, summary_metrics, coef_df, dashboard_df = fit_models(model_df)

    # 4. 情景汇总（基于模型输出，无需二次读取 Excel）
    scenario_summary_df, scenario_dashboard_df = calculate_scenario_summary(model_df)

    # 5. 导出
    export_excel(
        clean, model_df, summary_metrics, coef_df, dashboard_df,
        scenario_summary_df, scenario_dashboard_df,
        OUTPUT_XLSX,
    )
    plot_static_chart(model_df, OUTPUT_PNG)
    plot_interactive_chart(model_df, OUTPUT_HTML)

    print("建模完成。")
    print(f"输出 Excel: {OUTPUT_XLSX}")
    print(f"输出 PNG:   {OUTPUT_PNG}")
    print(f"输出 HTML:  {OUTPUT_HTML}")
    print("核心指标：")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
