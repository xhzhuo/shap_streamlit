"""
模型评估模块
包含模型质量评估等相关函数
"""

import numpy as np


def evaluate_model_quality(train_r2, test_r2, cv_mean, train_rmse=None, test_rmse=None):
    """
    评估模型质量 - 广告渠道投放分析专用
    输出简易评分和实用建议
    """
    # 关键指标
    r2_gap = abs(train_r2 - test_r2)  # 训练测试差异

    # 综合评分（更简化的权重）
    score = (
        test_r2 * 0.7 +           # 测试集表现最重要
        cv_mean * 0.2 +           # 稳定性
        (1 - min(r2_gap, 0.4)) * 0.1  # 一致性
    ) * 100
    score = max(0, min(round(score, 1), 100))

    # 质量等级与建议
    if score >= 85:
        level, color, icon = "优秀", "#22c55e", "✅"
        advice = "模型可靠，可直接用于渠道效果预测"
    elif score >= 70:
        level, color, icon = "良好", "#facc15", "👍" 
        advice = "模型可用，建议监控实际效果"
    elif score >= 55:
        level, color, icon = "一般", "#f97316", "⚠️"
        advice = "模型需优化，谨慎用于决策"
    else:
        level, color, icon = "需改进", "#ef4444", "❌"
        advice = "模型不可靠，建议重新构建"

    # # 误差检查提示
    # error_note = ""
    # if train_rmse and test_rmse and test_rmse > train_rmse * 1.5:
    #     error_note = "（模型可能存在过拟合）"

    return {
        "score": score,
        "level": level,
        "color": color,
        # "summary": f"{icon} 模型评分：{score}分 - {level}\n💡 {advice}{error_note}"
        "summary": f"{icon} 模型评分：{score}分 - {level}\n💡 {advice}"
    }