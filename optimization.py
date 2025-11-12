"""
优化工具模块
包含反推/预算优化相关的函数
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _get_bounds(base_x, min_constraints, max_constraints, X_train_max):
    """获取优化变量的边界"""
    bounds = []
    n = len(base_x)
    for i in range(n):
        lower = 0.0
        upper = float(X_train_max[i]) if X_train_max is not None else float(base_x[i] * 2 if base_x[i] != 0 else 1.0)
        if min_constraints is not None:
            lower = max(lower, float(min_constraints[i]))
        if max_constraints is not None:
            upper = min(upper, float(max_constraints[i]))
        if lower > upper:
            lower, upper = upper, lower
        bounds.append((lower, upper))
    return bounds


def _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max):
    """应用约束条件"""
    constrained = suggested.copy()
    constraint_status = []
    for i in range(len(suggested)):
        status = "正常"
        if min_constraints is not None and suggested[i] < min_constraints[i]:
            constrained[i] = min_constraints[i]
            status = "触达下限"
        if max_constraints is not None and suggested[i] > max_constraints[i]:
            constrained[i] = max_constraints[i]
            status = "触达上限"
        if X_train_max is not None and constrained[i] > X_train_max[i]:
            constrained[i] = float(X_train_max[i])
            status = "触达上限"
        if constrained[i] < 0:
            constrained[i] = 0.0
            status = "触达下限"
        constraint_status.append(status)
    return constrained, constraint_status


def _improved_sensitivity_estimation(model, base_x, feature_names, X_train_min, X_train_max, n_samples=40):
    """改进的敏感度估计"""
    if model is None:
        return np.ones(len(feature_names), dtype=float) * 1e-6
    n = len(feature_names)
    base = base_x.astype(float).copy()
    sensitivities = np.zeros(n, dtype=float)
    ranges = np.maximum(np.array(X_train_max) - np.array(X_train_min), 1e-6)
    base_df = pd.DataFrame(base.reshape(1, -1), columns=feature_names)
    for i in range(n):
        sample_sens = []
        for _ in range(n_samples):
            eps_ratio = np.random.uniform(0.001, 0.02)
            eps = ranges[i] * eps_ratio
            if eps == 0:
                eps = 1e-6
            x_perturb = base.copy()
            x_perturb[i] += eps
            try:
                x_perturb_df = pd.DataFrame(x_perturb.reshape(1, -1), columns=feature_names)
                y1 = float(model.predict(x_perturb_df)[0])
                y0 = float(model.predict(base_df)[0])
                sens = (y1 - y0) / eps
                sample_sens.append(sens)
            except Exception:
                sample_sens.append(0.0)
        sensitivities[i] = np.median(sample_sens) if len(sample_sens)>0 else 0.0
    if np.allclose(sensitivities, 0.0):
        sensitivities = np.ones_like(sensitivities) * 1e-6
    return sensitivities


def _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints=None, max_constraints=None, X_train_max=None):
    """线性分配算法"""
    eps = 1e-8
    sensitivities_adj = np.where(np.abs(sensitivities) < eps, eps, sensitivities)
    delta_y = y_target - y_base
    delta_x = (weights * delta_y) / sensitivities_adj
    suggested = base_x + delta_x
    suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
    try:
        y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
    except Exception:
        y_pred_new = None
    return suggested, y_pred_new, constraint_status

# def _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints=None, max_constraints=None, X_train_max=None):
#     """线性分配算法"""
#     eps = 1e-8
#     sensitivities_adj = np.where(np.abs(sensitivities) < eps, eps, sensitivities)
    
#     suggested = base_x.copy()
#     max_iter = 5          # 最大迭代次数
#     tolerance = 0.01 * abs(y_target)  # 允许误差：目标的1%
    
#     for _ in range(max_iter):
#         y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
#         delta_y = y_target - y_pred_new
        
#         if abs(delta_y) < tolerance:
#             break  # 已经足够接近目标
        
#         # 按SHAP权重和敏感度分配调整量
#         delta_x = (weights * delta_y) / sensitivities_adj
#         suggested = suggested + delta_x
        
#         # 约束处理
#         suggested, _ = _apply_constraints(suggested, base_x, min_constraints, max_constraints,X_train_max)
    
#     # 最终预测
#     y_pred_final = float(model.predict(suggested.reshape(1, -1))[0])
#     suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints,X_train_max)
    
#     return suggested, y_pred_final, constraint_status


def _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max):
    """预算约束优化算法"""
    def objective(x):
        try:
            y_pred = float(model.predict(x.reshape(1, -1))[0])
            return abs(y_pred - y_target)
        except Exception:
            return np.sum((x - base_x) ** 2)
    constraints = []
    if total_budget is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    bounds = _get_bounds(base_x, min_constraints, max_constraints, X_train_max)
    if total_budget is not None and np.sum(weights) > 0:
        x0 = (weights / np.sum(weights)) * total_budget
    else:
        x0 = base_x.copy()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
        if result.success:
            suggested = result.x
            suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
            try:
                y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
            except Exception:
                y_pred_new = None
            return suggested, y_pred_new, constraint_status
        else:
            return _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints, max_constraints, X_train_max)
    except Exception:
        return _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints, max_constraints, X_train_max)


def _fully_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max):
    """全约束优化算法"""
    def objective(x):
        try:
            y_pred = float(model.predict(x.reshape(1, -1))[0])
            gmv_diff = abs(y_pred - y_target)
        except Exception:
            gmv_diff = np.sum((x - base_x)**2)
        change_penalty = np.sum((x - base_x) ** 2) * 0.01
        return gmv_diff + change_penalty
    constraints = []
    if total_budget is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    bounds = _get_bounds(base_x, min_constraints, max_constraints, X_train_max)
    if total_budget is not None and np.sum(weights) > 0:
        x0 = (weights / np.sum(weights)) * total_budget
    else:
        x0 = base_x.copy()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-6})
        if result.success:
            suggested = result.x
            suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
            try:
                y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
            except Exception:
                y_pred_new = None
            return suggested, y_pred_new, constraint_status
        else:
            return _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)
    except Exception:
        return _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)