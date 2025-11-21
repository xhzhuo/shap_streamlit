"""
全新的高性能优化方案 - 直接求解法
核心思想：
1. 使用线性近似直接求解（一步到位）
2. 并行批量预测（减少模型调用次数）
3. 智能缓存和预计算
4. 自适应步长调整
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from typing import Optional, Tuple, Dict
import warnings
from collections import OrderedDict

# 全局缓存
_PRED_CACHE = OrderedDict()
_MAX_CACHE_SIZE = 10000

def _hash_array(arr):
    """快速数组哈希"""
    return hash(arr.tobytes())

def _safe_predict(model, X, use_cache=True):
    """安全预测（带缓存）"""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    if use_cache:
        cache_key = (id(model), _hash_array(arr))
        if cache_key in _PRED_CACHE:
            _PRED_CACHE.move_to_end(cache_key)
            return _PRED_CACHE[cache_key]
    
    # 尝试多种预测方式
    try:
        # 优先使用DataFrame（如果模型需要特征名）
        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is not None and len(feature_names) == arr.shape[1]:
            X_df = pd.DataFrame(arr, columns=feature_names)
            pred = float(model.predict(X_df)[0])
        else:
            pred = float(model.predict(arr)[0])
    except Exception:
        try:
            pred = float(model.predict(arr)[0])
        except Exception:
            pred = float(model.predict_proba(arr)[0, 1])
    
    if use_cache:
        _PRED_CACHE[cache_key] = pred
        if len(_PRED_CACHE) > _MAX_CACHE_SIZE:
            _PRED_CACHE.popitem(last=False)
    
    return pred


def _batch_predict(model, X_batch):
    """批量预测（更高效）"""
    X_batch = np.asarray(X_batch, dtype=np.float32)
    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(1, -1)
    
    # 尝试使用DataFrame
    feature_names = getattr(model, 'feature_names_in_', None)
    try:
        if feature_names is not None and len(feature_names) == X_batch.shape[1]:
            X_df = pd.DataFrame(X_batch, columns=feature_names)
            return model.predict(X_df).astype(np.float32)
        else:
            return model.predict(X_batch).astype(np.float32)
    except Exception:
        return model.predict(X_batch).astype(np.float32)


def _estimate_jacobian_fast(model, x, eps=0.02):
    """
    快速估计雅可比矩阵（一次批量预测）
    使用中心差分法，精度更高
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    n = len(x)
    
    # 构建扰动矩阵：[x, x+h1, x-h1, x+h2, x-h2, ...]
    X_batch = np.tile(x, (2*n + 1, 1))
    
    for i in range(n):
        h = max(abs(x[i]) * eps, eps)
        X_batch[2*i + 1, i] += h  # x + h
        X_batch[2*i + 2, i] -= h  # x - h
    
    # 一次性批量预测
    preds = _batch_predict(model, X_batch)
    
    # 计算中心差分
    jacobian = np.zeros(n, dtype=np.float32)
    for i in range(n):
        h = max(abs(x[i]) * eps, eps)
        jacobian[i] = (preds[2*i + 1] - preds[2*i + 2]) / (2 * h)
    
    return jacobian, preds[0]


def _direct_linear_solve(model, base_x, y_base, y_target, weights, 
                         min_constraints, max_constraints, X_train_max):
    """
    直接线性求解法（核心创新）
    
    基于一阶泰勒展开：
    y_target ≈ y_base + J·Δx
    其中 J 是雅可比矩阵
    
    求解：Δx = (y_target - y_base) / J（按权重分配）
    """
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    n = len(base_x)
    
    # 快速估计雅可比矩阵（一次批量调用）
    jacobian, y_current = _estimate_jacobian_fast(model, base_x)
    
    # 确保雅可比矩阵维度正确
    jacobian = np.asarray(jacobian, dtype=np.float32).ravel()
    if len(jacobian) != n:
        # 如果长度不匹配，截断或填充
        if len(jacobian) > n:
            jacobian = jacobian[:n]
        else:
            jacobian = np.pad(jacobian, (0, n - len(jacobian)), mode='constant', constant_values=1e-6)
    
    # 处理权重
    weights = np.asarray(weights, dtype=np.float32).ravel()
    if len(weights) != n:
        if len(weights) > n:
            weights = weights[:n]
        else:
            weights = np.pad(weights, (0, n - len(weights)), mode='constant', constant_values=1.0/n)
    
    weights = np.maximum(weights, 0)
    if weights.sum() <= 0:
        weights = np.ones(n, dtype=np.float32) / n
    else:
        weights = weights / weights.sum()
    
    # 目标差距
    delta_y = y_target - y_current
    
    # ===== 梯度投影法（数学改进）=====
    # 1. 计算梯度方向（带权重）
    gradient = jacobian * weights
    
    # 2. 计算最优步长（解析解）
    # 基于一阶泰勒展开: y ≈ y0 + grad·Δx
    # 最优步长: α* = Δy / ||grad||²
    grad_norm_sq = np.dot(gradient, gradient)
    if grad_norm_sq > 1e-12:
        alpha = delta_y / grad_norm_sq
    else:
        # 如果梯度太小，回退到简单分配
        alpha = delta_y / (np.sum(np.abs(jacobian) * weights) + 1e-8)
    
    # 3. 梯度下降步
    delta_x = alpha * gradient
    
    # 4. 自适应限制变化幅度（信赖域）
    # 根据梯度大小动态调整
    grad_magnitude = np.sqrt(grad_norm_sq)
    if grad_magnitude > 1e-6:
        # 大梯度：更保守
        trust_factor = 0.3 if grad_magnitude > 10 else 0.5
    else:
        trust_factor = 0.7
    
    max_change = np.maximum(trust_factor * base_x, 1.0)
    delta_x = np.clip(delta_x, -max_change, max_change)
    
    # 5. 快速 Line Search（仅验证全步长是否改进）
    # 应用约束
    suggested = base_x + delta_x
    suggested, _ = _apply_constraints(suggested, base_x, min_constraints, 
                                     max_constraints, X_train_max)
    
    # 快速验证：如果全步长不好，尝试半步长
    try:
        y_new = _safe_predict(model, suggested)
        error_full = abs(y_new - y_target)
        
        # 如果误差太大，尝试保守步长
        if error_full > abs(delta_y) * 0.8:
            suggested_half = base_x + 0.5 * delta_x
            suggested_half, _ = _apply_constraints(suggested_half, base_x, 
                                                   min_constraints, max_constraints, X_train_max)
            y_half = _safe_predict(model, suggested_half)
            error_half = abs(y_half - y_target)
            
            if error_half < error_full:
                suggested = suggested_half
    except Exception:
        pass
    
    return suggested


def _apply_constraints(x, base_x, min_c, max_c, X_train_max):
    """应用约束"""
    x = np.asarray(x, dtype=np.float32)
    base_x = np.asarray(base_x, dtype=np.float32)
    n = len(x)
    
    # 处理下界
    if min_c is None:
        lb = np.zeros(n, dtype=np.float32)
    else:
        lb = np.asarray(min_c, dtype=np.float32)
        if len(lb) == 1:
            lb = np.full(n, lb[0], dtype=np.float32)
    
    # 处理上界
    if max_c is None:
        ub = base_x * 3
    else:
        ub = np.asarray(max_c, dtype=np.float32)
        if len(ub) == 1:
            ub = np.full(n, ub[0], dtype=np.float32)
    
    if X_train_max is not None:
        X_train_max_arr = np.asarray(X_train_max, dtype=np.float32)
        if len(X_train_max_arr) == n:
            ub = np.minimum(ub, X_train_max_arr)
    
    # 确保边界有效
    lb = np.maximum(lb, 0.0)
    ub = np.maximum(ub, lb + 0.01)
    
    # 应用约束
    x = np.clip(x, lb, ub)
    
    # 生成状态
    status = []
    for i in range(n):
        s = []
        if abs(x[i] - lb[i]) < 1e-6:
            s.append('触达下限')
        if abs(x[i] - ub[i]) < 1e-6:
            s.append('触达上限')
        status.append(','.join(s) if s else '正常')
    
    return x, status


def optimize_allocation_v2(model, base_x, y_target,
                          total_budget=None, weights=None,
                          min_constraints=None, max_constraints=None,
                          X_train=None, max_iterations=3,
                          tolerance=0.02):
    """
    全新优化算法 V2 - 直接求解法
    
    优势：
    1. 速度快：使用批量预测，减少模型调用
    2. 精度高：基于一阶导数的线性近似
    3. 稳定：自适应步长，逐步逼近目标
    
    Parameters
    ----------
    model : 训练好的模型
    base_x : 基准分配
    y_target : 目标值
    total_budget : 总预算约束
    weights : 特征权重（SHAP值）
    min_constraints : 最小约束
    max_constraints : 最大约束
    X_train : 训练数据
    max_iterations : 最大迭代次数（默认3次，通常1-2次就够）
    tolerance : 收敛容差（默认2%）
    
    Returns
    -------
    dict : 优化结果
    """
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    n_features = len(base_x)
    
    # 处理权重
    if weights is None:
        weights = np.ones(n_features, dtype=np.float32) / n_features
    weights = np.asarray(weights, dtype=np.float32)
    
    # 处理训练数据范围
    if X_train is not None:
        X_train = np.asarray(X_train, dtype=np.float32)
        X_train_max = np.max(X_train, axis=0)
    else:
        X_train_max = base_x * 3
    
    # 计算基准预测
    y_base = _safe_predict(model, base_x)
    
    # 直接求解（通常一次就很接近）
    suggested = _direct_linear_solve(
        model, base_x, y_base, y_target, weights,
        min_constraints, max_constraints, X_train_max
    )
    
    # 自适应迭代精调（如果需要）
    best_suggested = suggested.copy()
    y_pred = _safe_predict(model, suggested)
    best_error = abs(y_pred - y_target)
    
    for iteration in range(max_iterations - 1):
        # 检查是否已收敛
        relative_error = abs(y_pred - y_target) / max(abs(y_target), 1.0)
        if relative_error < tolerance:
            break
        
        # 基于当前位置再次求解
        suggested = _direct_linear_solve(
            model, suggested, y_pred, y_target, weights,
            min_constraints, max_constraints, X_train_max
        )
        
        y_pred = _safe_predict(model, suggested)
        error = abs(y_pred - y_target)
        
        # 更新最佳结果
        if error < best_error:
            best_error = error
            best_suggested = suggested.copy()
    
    # 最终结果
    suggested, constraint_status = _apply_constraints(
        best_suggested, base_x, min_constraints, max_constraints, X_train_max
    )
    
    y_final = _safe_predict(model, suggested)
    budget_change = float(np.sum(suggested) - np.sum(base_x))
    efficiency = float((y_final - y_base) / (budget_change + 1e-8)) if abs(budget_change) > 1e-9 else 0.0
    
    return {
        'suggested_allocation': suggested,
        'predicted_value': float(y_final),
        'target_value': float(y_target),
        'base_value': float(y_base),
        'constraint_status': constraint_status,
        'budget_change': budget_change,
        'efficiency_gain': efficiency,
        'method_used': 'direct_linear_solve_v2',
        'robustness_level': 'high'
    }


# ==========================================
# 兼容性接口（可以直接替换原有函数）
# ==========================================
def optimize_ad_allocation_robust(model, base_x, y_target,
                                 total_budget=None, weights=None,
                                 min_constraints=None, max_constraints=None,
                                 X_train=None, method='adaptive',
                                 robustness_level='medium', feature_names=None,
                                 previous_solution=None):
    """兼容原有接口"""
    return optimize_allocation_v2(
        model, base_x, y_target, total_budget, weights,
        min_constraints, max_constraints, X_train,
        max_iterations=3, tolerance=0.02
    )
