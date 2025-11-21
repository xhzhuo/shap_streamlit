"""
模型评估模块
包含模型质量评估等相关函数
"""

import numpy as np


def evaluate_model_quality(train_r2, test_r2, cv_mean, train_rmse=None, test_rmse=None):
    """
    评估模型质量 - 广告渠道投放分析专用
    综合考虑：拟合效果、泛化能力、稳定性、过拟合风险
    
    返回评分卡数据
    """
    # 输入验证
    train_r2 = max(0, min(float(train_r2), 1))
    test_r2 = max(0, min(float(test_r2), 1))
    cv_mean = max(0, min(float(cv_mean), 1))
    
    # === 计算关键指标 ===
    
    # 1. 泛化能力指标（最重要）
    # test_r2 越高越好，直接反映模型在新数据上的表现
    generalization_score = test_r2 * 100
    
    # 2. 过拟合风险指标
    # train_r2 - test_r2 的差异表示过拟合程度
    # 差异越大，过拟合越严重
    overfitting_gap = abs(train_r2 - test_r2)
    
    # 过拟合惩罚系数：
    # - gap <= 0.05: 没有惩罚（-0分）
    # - gap = 0.1: 中等惩罚（-3分）
    # - gap = 0.2: 严重惩罚（-10分）
    # - gap >= 0.3: 极度惩罚（-20分）
    if overfitting_gap <= 0.05:
        overfitting_penalty = 0
    elif overfitting_gap <= 0.15:
        overfitting_penalty = (overfitting_gap - 0.05) / 0.1 * 3  # 线性增长
    elif overfitting_gap <= 0.25:
        overfitting_penalty = 3 + (overfitting_gap - 0.15) / 0.1 * 7  # 3-10分
    else:
        overfitting_penalty = 10 + min(overfitting_gap - 0.25, 0.1) / 0.1 * 10  # 10-20分
    
    # 3. 稳定性指标
    # CV均值与test_r2的接近程度表示模型稳定性
    # 如果 cv_mean ≈ test_r2，说明模型在不同数据折上表现一致
    cv_stability_gap = abs(cv_mean - test_r2)
    
    # 稳定性奖励/惩罚：
    # - gap <= 0.05: 优秀（+2分）
    # - gap = 0.1: 良好（0分）
    # - gap >= 0.2: 不稳定（-5分）
    if cv_stability_gap <= 0.05:
        stability_bonus = 2
    elif cv_stability_gap <= 0.1:
        stability_bonus = 2 - (cv_stability_gap - 0.05) / 0.05 * 2  # 2-0分
    elif cv_stability_gap <= 0.2:
        stability_bonus = -(cv_stability_gap - 0.1) / 0.1 * 5  # 0 到 -5分
    else:
        stability_bonus = -5
    
    # === 综合评分 ===
    # 基础分数 = 泛化能力（最重要）
    # 调整项 = 过拟合惩罚 + 稳定性调整
    score = generalization_score - overfitting_penalty + stability_bonus
    score = max(0, min(round(score, 1), 100))
    
    # === 确定等级和建议 ===
    if score >= 90:
        level, color, icon = "优秀", "#22c55e", "🌟"
        advice = "模型性能最优，强烈推荐用于关键决策"
    elif score >= 80:
        level, color, icon = "很好", "#84cc16", "✅"
        advice = "模型表现优良，可放心用于预测分析"
    elif score >= 70:
        level, color, icon = "良好", "#facc15", "👍"
        advice = "模型可用，建议先小范围验证后再大规模应用"
    elif score >= 60:
        level, color, icon = "一般", "#f97316", "⚠️"
        advice = "模型有效但精度一般，建议优化特征或增加数据"
    elif score >= 50:
        level, color, icon = "较弱", "#ef4444", "❌"
        advice = "模型表现不佳，建议重新选择特征或重建模型"
    else:
        level, color, icon = "不可靠", "#dc2626", "⛔"
        advice = "模型不适用，强烈建议重新构建"
    
    # === 诊断建议 ===
    diagnoses = []
    if overfitting_gap > 0.15:
        diagnoses.append("⚠️ 检测到过拟合：模型在训练集表现过好，泛化能力下降")
    if cv_stability_gap > 0.15:
        diagnoses.append("⚠️ 稳定性问题：不同数据折的性能差异较大")
    if test_r2 < 0.5:
        diagnoses.append("⚠️ 拟合不足：模型未能充分学习数据模式")
    
    diagnosis_text = "\n".join(diagnoses) if diagnoses else "✓ 模型各项指标均衡，无明显问题"
    
    return {
        "score": score,
        "level": level,
        "color": color,
        "summary": f"{icon} 模型评分：{score}分 - {level}\n💡 {advice}",
        "diagnosis": diagnosis_text,
        "metrics": {
            "generalization": generalization_score,
            "overfitting_gap": overfitting_gap,
            "overfitting_penalty": overfitting_penalty,
            "stability_gap": cv_stability_gap,
            "stability_bonus": stability_bonus
        }
    }