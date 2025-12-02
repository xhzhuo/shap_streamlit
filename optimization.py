"""
全新的高性能优化方案 - 直接求解法
核心思想：
1. 使用线性近似直接求解（一步到位）
2. 并行批量预测（减少模型调用次数）
3. 智能缓存和预计算
4. 自适应步长调整
5. 自适应 SHAP 校准（新增）
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from typing import Optional, Tuple, Dict
import warnings
from collections import OrderedDict

# SHAP 库导入（可选依赖）
try:
    import shap
    import time
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn(
        "shap 库未安装，SHAP 校准功能将不可用。\n"
        "如需使用 SHAP 校准，请运行: pip install shap",
        UserWarning
    )

# 全局缓存
_PRED_CACHE = OrderedDict()
_MAX_CACHE_SIZE = 10000

# SHAP explainer 缓存（避免重复创建）
_SHAP_EXPLAINER_CACHE = {}
_SHAP_VALUES_CACHE = OrderedDict()
_MAX_SHAP_CACHE_SIZE = 100

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


def _compute_shap_marginal(model, x, eps=0.02, feature_names=None):
    """
    计算单位化 SHAP 边际贡献（与雅可比量纲一致）
    
    Critical: 返回 (φ+ - φ-) / (2×delta)，NOT 原始 φ
    这确保 SHAP 值与雅可比具有相同的量纲（单位变化率）
    
    Parameters
    ----------
    model : 训练好的模型
    x : 基准点（1D array）
    eps : 扰动幅度（必须与 _estimate_jacobian_fast 一致，默认 0.02）
    feature_names : 特征名（可选）
    
    Returns
    -------
    dict : {
        'shap_marginal': ndarray 或 None（单位化边际 SHAP）
        'runtime_ms': float（计算时间）
        'method': str（使用的 explainer 类型：'tree'/'kernel'）
        'failure_reason': str 或 None（失败原因）
        'scale_check_passed': bool（尺度验证是否通过）
    }
    """
    if not SHAP_AVAILABLE:
        return {
            'shap_marginal': None,
            'runtime_ms': 0,
            'method': None,
            'failure_reason': 'shap_library_not_installed',
            'scale_check_passed': False
        }
    
    start_time = time.time()
    x = np.asarray(x, dtype=np.float32).ravel()
    n = len(x)
    
    try:
        # ===== 步骤1: 获取或创建 explainer（缓存优化）=====
        model_id = id(model)
        
        if model_id not in _SHAP_EXPLAINER_CACHE:
            # 尝试 TreeExplainer（快速，适合树模型）
            try:
                explainer = shap.TreeExplainer(model)
                explainer_type = 'tree'
            except Exception:
                # 降级到 KernelExplainer（慢但通用）
                try:
                    def model_predict(X):
                        return _batch_predict(model, X)
                    
                    # 使用当前点作为背景数据（简化）
                    explainer = shap.KernelExplainer(model_predict, x.reshape(1, -1))
                    explainer_type = 'kernel'
                except Exception as e:
                    return {
                        'shap_marginal': None,
                        'runtime_ms': (time.time() - start_time) * 1000,
                        'method': None,
                        'failure_reason': f'explainer_creation_failed: {str(e)[:100]}',
                        'scale_check_passed': False
                    }
            
            _SHAP_EXPLAINER_CACHE[model_id] = (explainer, explainer_type)
        else:
            explainer, explainer_type = _SHAP_EXPLAINER_CACHE[model_id]
        
        # ===== 步骤2: 构造扰动样本（批量计算优化）=====
        X_batch = np.tile(x, (2*n + 1, 1)).astype(np.float32)  # [x, x+h1, x-h1, ...]
        
        for i in range(n):
            h = max(abs(x[i]) * eps, eps)
            X_batch[2*i + 1, i] += h  # x + h
            X_batch[2*i + 2, i] -= h  # x - h
        
        # ===== 步骤3: 批量计算 SHAP 值 =====
        if explainer_type == 'tree':
            # TreeExplainer 返回 shap_values（可能是 list 或 array）
            shap_result = explainer.shap_values(X_batch)
            
            # 处理多输出情况（分类模型）
            if isinstance(shap_result, list):
                # 取最后一个类别的 SHAP 值（通常是正类）
                shap_values_batch = shap_result[-1]
            else:
                shap_values_batch = shap_result
        else:
            # KernelExplainer
            shap_values_batch = explainer.shap_values(X_batch)
        
        # ===== 步骤4: 计算单位化边际 SHAP（Critical）=====
        shap_marginal = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            h = max(abs(x[i]) * eps, eps)
            
            # 获取 x+h 和 x-h 的 SHAP 值
            phi_plus = shap_values_batch[2*i + 1]   # shape: (n,)
            phi_minus = shap_values_batch[2*i + 2]  # shape: (n,)
            
            # 单位化边际：(φ+[i] - φ-[i]) / (2h)
            # 注意：只取第 i 个特征的 SHAP 变化（对角元素）
            shap_marginal[i] = (phi_plus[i] - phi_minus[i]) / (2 * h)
        
        # ===== 步骤5: 尺度验证（Critical）=====
        shap_norm = np.linalg.norm(shap_marginal)
        
        # 验证 SHAP 是否异常（NaN/Inf/全零）
        scale_check_passed = (
            np.isfinite(shap_norm) and 
            shap_norm > 1e-12 and 
            np.all(np.isfinite(shap_marginal))
        )
        
        if not scale_check_passed:
            return {
                'shap_marginal': None,
                'runtime_ms': (time.time() - start_time) * 1000,
                'method': explainer_type,
                'failure_reason': 'shap_values_invalid_or_zero',
                'scale_check_passed': False
            }
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return {
            'shap_marginal': shap_marginal,
            'runtime_ms': runtime_ms,
            'method': explainer_type,
            'failure_reason': None,
            'scale_check_passed': True
        }
    
    except Exception as e:
        return {
            'shap_marginal': None,
            'runtime_ms': (time.time() - start_time) * 1000,
            'method': None,
            'failure_reason': f'computation_error: {str(e)[:100]}',
            'scale_check_passed': False
        }


def _detect_nonlinearity(model, base_x, jacobian, eps=0.1):
    """
    检测模型非线性程度
    
    原理：测试模型在不同点的响应是否与线性预测一致
    - 线性模型：梯度恒定，delta_y ≈ J·delta_x
    - 非线性模型：梯度变化，实际变化 ≠ 线性预测
    
    Parameters
    ----------
    model : 训练好的模型
    base_x : 基准点
    jacobian : 在base_x处的雅可比（梯度）
    eps : 测试扰动幅度（默认0.1，即10%）
    
    Returns
    -------
    float : 非线性指标（0=完全线性，>0.3表示强非线性）
    """
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    jacobian = np.asarray(jacobian, dtype=np.float32).ravel()
    
    # 获取基准预测值
    y_base = _safe_predict(model, base_x)
    
    # 测试点1：沿梯度方向移动
    direction = jacobian / (np.linalg.norm(jacobian) + 1e-8)
    x_forward = base_x + eps * np.linalg.norm(base_x) * direction
    x_backward = base_x - eps * np.linalg.norm(base_x) * direction
    
    # 实际预测
    y_forward = _safe_predict(model, x_forward)
    y_backward = _safe_predict(model, x_backward)
    
    # 线性预测（基于梯度）
    delta_x_forward = x_forward - base_x
    delta_x_backward = x_backward - base_x
    
    linear_pred_forward = y_base + np.dot(jacobian, delta_x_forward)
    linear_pred_backward = y_base + np.dot(jacobian, delta_x_backward)
    
    # 计算非线性指标：实际变化与线性预测的相对误差
    actual_change = abs(y_forward - y_backward)
    linear_change = abs(linear_pred_forward - linear_pred_backward)
    
    if linear_change < 1e-8:
        return 0.0  # 几乎无变化，认为是线性
    
    nonlinearity = abs(actual_change - linear_change) / (linear_change + 1e-8)
    
    return float(nonlinearity)


def _optimize_with_scipy(model, base_x, y_target, min_constraints, max_constraints, 
                         X_train_max, max_iterations=50):
    """
    使用SciPy优化器进行全局优化（用于非线性模型）
    
    Parameters
    ----------
    model : 训练好的模型
    base_x : 初始点（从线性求解得到）
    y_target : 目标值
    min_constraints, max_constraints : 约束
    X_train_max : 训练数据最大值
    max_iterations : 最大迭代次数
    
    Returns
    -------
    ndarray : 优化后的分配方案
    """
    from scipy.optimize import minimize
    
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    n = len(base_x)
    
    # 定义目标函数：最小化预测值与目标的平方误差
    def objective(x):
        try:
            y_pred = _safe_predict(model, x, use_cache=False)
            return (y_pred - y_target) ** 2
        except Exception:
            return 1e10  # 预测失败，返回大惩罚
    
    # 设置边界约束
    bounds = []
    for i in range(n):
        if min_constraints is not None and max_constraints is not None:
            lb = float(min_constraints[i]) if i < len(min_constraints) else 0.0
            ub = float(max_constraints[i]) if i < len(max_constraints) else base_x[i] * 3
        else:
            lb = 0.0
            ub = base_x[i] * 3 if X_train_max is None else float(X_train_max[i])
        
        bounds.append((lb, ub))
    
    # 使用L-BFGS-B优化器（适合有界约束的问题）
    try:
        result = minimize(
            objective,
            x0=base_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        if result.success or result.fun < (y_target * 0.05) ** 2:  # 误差<5%认为成功
            return result.x
        else:
            # 优化失败，返回初始值
            return base_x
    except Exception:
        return base_x


def _direct_linear_solve(model, base_x, y_base, y_target, weights, 
                         min_constraints, max_constraints, X_train_max,
                         use_shap_calibration=False,  # 新增
                         alpha=0.5,                    # 新增
                         shap_result=None):            # 新增
    """
    直接线性求解法（支持 SHAP 混合校准）
    
    核心公式：
    1. 基础：δy ≈ J·Δx
    2. SHAP校准：使用 SHAP 边际值校准梯度幅度
       Sensitivity = α·|J| + (1-α)·|Δφ|
       Gradient = sign(J) · Sensitivity
    
    Returns
    -------
    suggested : ndarray
        建议的投放方案
    diagnostics : dict
        诊断信息（包含校准详情、冲突检测等）
    """
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    n = len(base_x)
    
    # 快速估计雅可比矩阵（一次批量调用）
    jacobian, y_current = _estimate_jacobian_fast(model, base_x)
    
    # 确保雅可比矩阵维度正确
    jacobian = np.asarray(jacobian, dtype=np.float32).ravel()
    if len(jacobian) != n:
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
    
    # 初始化诊断信息
    diagnostics = {
        'shap_used': False,
        'sign_conflict_ratio': 0.0,
        'fallback_reason': None,
        'denominator_fallback': None
    }
    
    # ===== 策略选择：SHAP 校准 vs 原权重方案 =====
    calibrated_jacobian = None
    
    if use_shap_calibration and shap_result and shap_result['shap_marginal'] is not None:
        # === 方案 A: SHAP 混合校准 ===
        shap_marginal = shap_result['shap_marginal']
        
        # Critical Fix #2: 符号冲突检测
        # 检查雅可比方向与 SHAP 方向是否一致
        # 如果冲突严重（>50%），说明 SHAP 可能不可靠（或雅可比不可靠），应降级
        sign_conflict = np.sign(jacobian) != np.sign(shap_marginal)
        # 忽略接近0的项（避免噪声干扰）
        significant = (np.abs(jacobian) > 1e-6) & (np.abs(shap_marginal) > 1e-6)
        if np.any(significant):
            conflict_ratio = np.mean(sign_conflict[significant])
        else:
            conflict_ratio = 0.0
            
        diagnostics['sign_conflict_ratio'] = float(conflict_ratio)
        
        if conflict_ratio > 0.5:
            # 严重冲突：降级到原方案
            use_shap_calibration = False
            diagnostics['fallback_reason'] = 'severe_sign_conflict'
        elif conflict_ratio > 0.3:
            # 中度冲突：增加雅可比的权重（减少 SHAP 影响）
            alpha = min(0.9, alpha + 0.4)
            diagnostics['alpha_adjusted'] = float(alpha)
        
        if use_shap_calibration:
            # 执行混合校准
            # 1. 幅度混合：Sensitivity = α·|J| + (1-α)·|Δφ|
            sensitivity = alpha * np.abs(jacobian) + (1 - alpha) * np.abs(shap_marginal)
            
            # 2. 方向保持：使用雅可比的符号（通常梯度方向更可靠）
            calibrated_jacobian = np.sign(jacobian) * sensitivity
            
            # 3. 应用用户权重（可选，通常 SHAP 已经包含了重要性）
            # 这里我们保留 w^0.5 的微调，保留用户偏好但不主导
            calibrated_jacobian = calibrated_jacobian * (weights ** 0.5)
            
            diagnostics['shap_used'] = True
    
    if calibrated_jacobian is None:
        # === 方案 B: 原权重方案（Fallback）===
        # w^1.5 * J
        weight_power = 1.5
        calibrated_jacobian = (weights ** weight_power) * jacobian
    
    # ===== 求解 Δx =====
    # 公式：Δx = calibrated_J * (delta_y / denominator)
    # denominator = Σ(calibrated_J * J)
    
    # Critical Fix #3: Denominator Stability
    # 分母代表了"预测的总变化量"，必须与 delta_y 同号且非零
    
    denominator = np.dot(calibrated_jacobian, jacobian)
    
    # 安全检查
    eps = 1e-10
    if abs(denominator) < eps:
        # 分母过小，说明梯度方向与校准方向几乎正交，或者梯度消失
        diagnostics['denominator_fallback'] = 'level_1_self_dot'
        
        # Level 1: 尝试使用校准梯度的自点积（假设 J ≈ calibrated_J）
        denominator = np.dot(calibrated_jacobian, calibrated_jacobian)
        
        if abs(denominator) < eps:
            # Level 2: 完全退回原方案（纯梯度）
            diagnostics['denominator_fallback'] = 'level_2_pure_gradient'
            calibrated_jacobian = jacobian * weights
            denominator = np.dot(calibrated_jacobian, jacobian)
            
            if abs(denominator) < eps:
                # Level 3: 均匀分配（最后的救命稻草）
                diagnostics['denominator_fallback'] = 'level_3_uniform'
                calibrated_jacobian = np.sign(delta_y) * np.ones(n) / n
                denominator = 1.0  # 任意非零值，后续缩放会修正
    
    # 计算步长
    if abs(denominator) > eps:
        delta_x = calibrated_jacobian * (delta_y / denominator)
    else:
        delta_x = np.zeros(n)
    
    # Critical Fix #4: Δx Clipping (安全裁剪)
    # 防止单步变化过大导致模型进入未知区域
    if X_train_max is not None:
        max_step = 0.2 * np.asarray(X_train_max)  # 限制为训练数据的 20%
        # 确保 max_step 正数
        max_step = np.maximum(max_step, 1.0)
        delta_x = np.clip(delta_x, -max_step, max_step)
    
    # 计算建议值
    suggested = base_x + delta_x
    
    # 应用约束
    suggested, _ = _apply_constraints(suggested, base_x, min_constraints, 
                                     max_constraints, X_train_max)
    
    return suggested, diagnostics


def _apply_constraints(x, base_x, min_c, max_c, X_train_max):
    """
    应用约束
    
    约束层级：
    1. 物理约束：x >= 0（投放量不能为负）
    2. 用户约束：min_c <= x <= max_c（如果用户指定）
    3. 训练边界：x <= X_train_max（仅在有约束模式下考虑）
    
    注意：如果 min_c=None 且 max_c=None，则为真正的"无约束"模式
          此时只保留物理约束 x >= 0
    """
    x = np.asarray(x, dtype=np.float32)
    base_x = np.asarray(base_x, dtype=np.float32)
    n = len(x)
    
    # === 1. 处理下界 ===
    if min_c is None:
        # 无约束模式：只保留物理约束（非负）
        lb = np.zeros(n, dtype=np.float32)
    else:
        # 有约束模式：使用用户指定的最小值
        lb = np.asarray(min_c, dtype=np.float32)
        if len(lb) == 1:
            lb = np.full(n, lb[0], dtype=np.float32)
        # 确保不低于0（物理约束）
        lb = np.maximum(lb, 0.0)
    
    # === 2. 处理上界 ===
    if max_c is None:
        # 无约束模式：不设上限
        # 理由：用户选择"无约束"就是想探索所有可能性
        # 唯一限制是算法的数值稳定性（设一个很大的上限）
        ub = np.full(n, 1e10, dtype=np.float32)  # 实际上相当于无限制
    else:
        # 有约束模式：使用用户指定的最大值
        ub = np.asarray(max_c, dtype=np.float32)
        if len(ub) == 1:
            ub = np.full(n, ub[0], dtype=np.float32)
        
        # === 3. 训练边界约束（仅在有约束模式） ===
        # 在有约束模式下，考虑训练数据的最大值
        # 原因：超出训练范围，模型预测可能不可靠
        if X_train_max is not None:
            X_train_max_arr = np.asarray(X_train_max, dtype=np.float32)
            if len(X_train_max_arr) == n:
                ub = np.minimum(ub, X_train_max_arr)
    
    # 确保边界有效（上界必须大于下界）
    ub = np.maximum(ub, lb + 0.01)
    
    # 应用约束
    x = np.clip(x, lb, ub)
    
    # === 生成约束状态 ===
    # 注意：仅在有实际约束时才标记状态
    # 如果是"无约束"模式（min_c=None, max_c=None），
    # 则只报告是否触达物理边界（x=0）
    is_unconstrained = (min_c is None and max_c is None)
    
    status = []
    for i in range(n):
        s = []
        
        # 触达下限
        if abs(x[i] - lb[i]) < 1e-6:
            if is_unconstrained:
                # 无约束模式下，只有触达0才标记（物理边界）
                if lb[i] < 1e-6:
                    s.append('触达物理下限(0)')
            else:
                # 有约束模式下，触达用户设定的下限
                s.append('触达下限')
        
        # 触达上限
        if abs(x[i] - ub[i]) < 1e-6:
            if not is_unconstrained:
                # 只在有约束模式下标记上限
                # （无约束模式的上限是1e10，不可能触达）
                s.append('触达上限')
        
        status.append(','.join(s) if s else '正常')
    
    return x, status


def _estimate_feasible_range(model, base_x, min_constraints, max_constraints, X_train_max):
    """
    估计模型的可行输出范围
    
    通过探测不同投放规模，找到模型的实际响应上限和下限
    这对于判断目标是否可达非常关键
    
    Returns
    -------
    dict: {
        'min_output': 最小可能输出（投放归零时）,
        'max_output': 最大可能输出（投放最大时）,
        'base_output': 基准输出,
        'max_allocation': 达到最大输出时的投放方案
    }
    """
    base_x = np.asarray(base_x, dtype=np.float32).ravel()
    n = len(base_x)
    
    # 1. 基准输出
    y_base = _safe_predict(model, base_x)
    
    # 2. 最小输出（投放归零）
    x_min = np.zeros(n, dtype=np.float32)
    y_min = _safe_predict(model, x_min)
    
    # 3. 探测最大输出
    # 策略：尝试多个放大倍数，找到输出的上限
    # 考虑约束条件
    
    # 确定探测的上界
    if max_constraints is not None:
        # 有约束：使用用户指定的上限
        max_x = np.asarray(max_constraints, dtype=np.float32)
        if len(max_x) == 1:
            max_x = np.full(n, max_x[0], dtype=np.float32)
    else:
        # 无约束：使用训练最大值的2倍作为探测上限
        if X_train_max is not None:
            max_x = np.asarray(X_train_max, dtype=np.float32) * 2.0
        else:
            max_x = base_x * 5.0  # 最多探测5倍
    
    # 应用约束
    max_x, _ = _apply_constraints(max_x, base_x, min_constraints, 
                                  max_constraints, X_train_max)
    
    # 在基准和最大值之间采样探测
    # 注意：不能只用简单的线性缩放，因为优化算法会根据梯度智能分配
    # 所以这里的"最大值"应该理解为"保守估计的上限"
    scale_factors = np.linspace(1.0, 5.0, 10)
    y_max = y_base
    x_at_max = base_x.copy()
    
    for scale in scale_factors:
        x_test = base_x * scale
        x_test, _ = _apply_constraints(x_test, base_x, min_constraints,
                                       max_constraints, X_train_max)
        y_test = _safe_predict(model, x_test)
        
        if y_test > y_max:
            y_max = y_test
            x_at_max = x_test.copy()
    
    # 重要说明：这个 y_max 是简单缩放的结果，实际优化算法可能找到更好的方案
    # 因为优化会根据梯度非均匀分配，所以实际结果可能超出这个估计值 10-20%
    
    return {
        'min_output': float(y_min),
        'max_output': float(y_max),
        'base_output': float(y_base),
        'max_allocation': x_at_max,
        'is_conservative_estimate': True  # 标记这是保守估计
    }


def optimize_allocation_v2(model, base_x, y_target,
                          total_budget=None, weights=None,
                          min_constraints=None, max_constraints=None,
                          X_train=None, max_iterations=3,
                          tolerance=0.02):
    """
    全新优化算法 V2 - 直接求解法（带目标可达性分析）
    
    优势：
    1. 速度快：使用批量预测，减少模型调用
    2. 精度高：基于一阶导数的线性近似
    3. 稳定：自适应步长，逐步逼近目标
    4. 智能：自动检测目标可达性，避免无效投放
    
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
    dict : 优化结果（包含目标可达性分析）
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
    
    # === 步骤1：目标可达性分析 ===
    feasible_range = _estimate_feasible_range(
        model, base_x, min_constraints, max_constraints, X_train_max
    )
    
    y_base = feasible_range['base_output']
    y_max_possible = feasible_range['max_output']
    
    # 判断目标是否可达
    # 注意：y_max_possible 是保守估计，实际优化可能超出 10-20%
    # 所以这里用 1.15 的系数来判断（允许15%的超出空间）
    conservative_limit = y_max_possible * 1.15
    target_feasible = (y_target <= conservative_limit)
    
    # 如果目标明显超出合理范围，调整为保守估计值
    # 后续会根据实际优化的ROI再做进一步调整
    original_target = y_target
    if not target_feasible:
        y_target_adjusted = y_max_possible  # 先调整为保守估计
    else:
        y_target_adjusted = y_target
    
    # === 步骤2：非线性检测与自适应策略调整 ===
    # 检测模型非线性程度，动态调整优化策略
    jacobian_initial, _ = _estimate_jacobian_fast(model, base_x)
    nonlinearity_score = _detect_nonlinearity(model, base_x, jacobian_initial)
    
    # ===== Critical Fix #6: 平滑 Alpha 映射 =====
    def smooth_alpha_mapping(score):
        """平滑的 alpha 映射函数（避免跳变）"""
        # sigmoid(10×(score-0.3)) 在 score=0.3 附近快速变化
        # 但比阶跃函数平滑得多
        sigmoid_val = 1.0 / (1.0 + np.exp(-10 * (score - 0.3)))
        alpha = 0.95 - 0.75 * sigmoid_val  # 范围: 0.95 → 0.20
        return np.clip(alpha, 0.15, 0.95)
    
    alpha = smooth_alpha_mapping(nonlinearity_score)
    
    # ===== 自适应 SHAP 校准策略 =====
    if nonlinearity_score < 0.2:
        # 弱非线性：使用原方案（不计算 SHAP）
        use_shap_calibration = False
        adaptive_max_iterations = max_iterations
        use_scipy_fallback = False
        shap_result = None
        calibration_reason = 'weak_nonlinearity_original_scheme'
    else:
        # 中度/强非线性：启用 SHAP 校准
        use_shap_calibration = True
        adaptive_max_iterations = max_iterations + min(3, int(nonlinearity_score * 10))
        use_scipy_fallback = True
        
        # 计算 SHAP 边际值
        shap_result = _compute_shap_marginal(
            model, base_x, eps=0.02,  # 与雅可比一致的 eps
            feature_names=getattr(model, 'feature_names_in_', None)
        )
        
        if shap_result['shap_marginal'] is None:
            # SHAP 计算失败，降级到原方案
            use_shap_calibration = False
            shap_result['fallback_to_original'] = True
            calibration_reason = f"shap_failed: {shap_result['failure_reason']}"
        else:
            # SHAP 成功，使用校准方案
            calibration_reason = (
                'moderate_nonlinearity_shap' if nonlinearity_score < 0.4 
                else 'strong_nonlinearity_shap'
            )
            
            # === 尺度验证（事后检查）===
            jacobian_norm = np.linalg.norm(jacobian_initial)
            shap_norm = np.linalg.norm(shap_result['shap_marginal'])
            
            # 如果量级相差超过100倍，发出警告
            scale_ratio = shap_norm / (jacobian_norm + 1e-12)
            if scale_ratio > 100 or scale_ratio < 0.01:
                warnings.warn(
                    f"SHAP/Jacobian 尺度差异较大: {scale_ratio:.2f}, "
                    f"可能影响校准效果"
                )
    
    # 记录诊断信息
    diagnostics = {
        'nonlinearity_score': float(nonlinearity_score),
        'alpha_computed': float(alpha),
        'calibration_reason': calibration_reason,
        'shap_runtime_ms': shap_result['runtime_ms'] if shap_result else 0,
        'shap_method': shap_result['method'] if shap_result else None,
        'jacobian_norm': float(jacobian_norm) if 'jacobian_norm' in locals() else None,
        'shap_norm': float(shap_norm) if 'shap_norm' in locals() else None,
    }
    
    # === 步骤3：优化求解（快速线性方法）===
    suggested, solve_diagnostics = _direct_linear_solve(
        model, base_x, y_base, y_target_adjusted, weights,
        min_constraints, max_constraints, X_train_max,
        use_shap_calibration=use_shap_calibration,  # 新增
        alpha=alpha,                                 # 新增
        shap_result=shap_result                      # 新增
    )
    
    # 合并诊断信息
    diagnostics.update(solve_diagnostics)
    
    # 自适应迭代精调（使用根据非线性程度调整的迭代次数）
    best_suggested = suggested.copy()
    y_pred = _safe_predict(model, suggested)
    best_error = abs(y_pred - y_target_adjusted)
    
    for iteration in range(adaptive_max_iterations - 1):
        # 检查是否已收敛
        relative_error = abs(y_pred - y_target_adjusted) / max(abs(y_target_adjusted), 1.0)
        if relative_error < tolerance:
            break
        
        # 基于当前位置再次求解
        suggested, iter_diagnostics = _direct_linear_solve(
            model, suggested, y_pred, y_target_adjusted, weights,
            min_constraints, max_constraints, X_train_max,
            use_shap_calibration=use_shap_calibration,
            alpha=alpha,
            shap_result=shap_result  # 注意：迭代时复用同一个 SHAP 结果
        )
        
        # 更新诊断信息（只保留最后一次的）
        diagnostics.update(iter_diagnostics)
        
        y_pred = _safe_predict(model, suggested)
        error = abs(y_pred - y_target_adjusted)
        
        # 更新最佳结果
        if error < best_error:
            best_error = error
            best_suggested = suggested.copy()
    
    # ===== 步骤4：混合优化策略（SciPy兜底）=====
    # 如果快速方法误差仍然较大且模型非线性强，使用SciPy精调
    final_relative_error = best_error / max(abs(y_target_adjusted), 1.0)
    
    if use_scipy_fallback and final_relative_error > 0.05:  # 误差>5%时启用
        # 使用SciPy全局优化（以快速方法的结果为起点）
        scipy_suggested = _optimize_with_scipy(
            model, best_suggested, y_target_adjusted,
            min_constraints, max_constraints, X_train_max,
            max_iterations=30
        )
        
        # 检查SciPy是否真的改进了结果
        y_scipy = _safe_predict(model, scipy_suggested)
        scipy_error = abs(y_scipy - y_target_adjusted)
        
        if scipy_error < best_error:
            # SciPy更好，采用其结果
            best_suggested = scipy_suggested
            best_error = scipy_error
    
    # 最终结果
    suggested, constraint_status = _apply_constraints(
        best_suggested, base_x, min_constraints, max_constraints, X_train_max
    )
    
    y_final = _safe_predict(model, suggested)
    
    # 计算初步的投入产出比
    temp_budget_change = np.sum(suggested) - np.sum(base_x)
    temp_budget_change_pct = (temp_budget_change / np.sum(base_x)) * 100 if np.sum(base_x) > 0 else 0
    temp_output_change_pct = ((y_final - y_base) / y_base) * 100 if y_base > 0 else 0
    temp_marginal_eff = temp_output_change_pct / temp_budget_change_pct if abs(temp_budget_change_pct) > 1e-6 else 0
    
    # === ROI 合理性检查与自动调整 ===
    # 如果边际效率过低，说明进入了严重的边际递减区域
    # 应该降低目标，而不是盲目追求
    if temp_marginal_eff < 0.25 and temp_budget_change_pct > 100:
        
        # 尝试多个折中系数，找到边际效率最优的方案
        best_compromise = suggested
        best_compromise_y = y_final
        best_compromise_eff = temp_marginal_eff
        
        # 尝试 20% 到 80% 的强度
        for factor in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            compromise_suggested = base_x + (suggested - base_x) * factor
            compromise_suggested, _ = _apply_constraints(compromise_suggested, base_x,
                                                         min_constraints, max_constraints, X_train_max)
            y_compromise = _safe_predict(model, compromise_suggested)
            
            compromise_budget_pct = ((np.sum(compromise_suggested) - np.sum(base_x)) / np.sum(base_x)) * 100
            compromise_output_pct = ((y_compromise - y_base) / y_base) * 100
            compromise_marginal = compromise_output_pct / compromise_budget_pct if compromise_budget_pct > 1e-6 else 0
            
            # 选择边际效率最高的方案（至少要>0.3才算合理）
            if compromise_marginal > best_compromise_eff:
                best_compromise_eff = compromise_marginal
                best_compromise = compromise_suggested
                best_compromise_y = y_compromise
        
        # 如果找到了更合理的方案，就采用
        if best_compromise_eff > temp_marginal_eff * 1.2:  # 至少要提升20%
            suggested = best_compromise
            y_final = best_compromise_y
            # 同时降低目标值，避免后续兜底策略再次追求不合理的目标
            y_target_adjusted = y_final * 0.95  # 略低于当前预测值
    
    # ===== 最终验证：确保结果朝正确方向移动 =====
    # 核心原则：优化结果必须比基准更接近目标
    # 如果违反此原则，说明算法完全失败，需要兜底策略
    
    direction_ok = (y_target_adjusted > y_base and y_final >= y_base) or \
                   (y_target_adjusted < y_base and y_final <= y_base) or \
                   abs(y_target_adjusted - y_base) < 1e-6
    
    if not direction_ok:
        # 算法失败：结果反向移动
        # 原因可能是：
        # 1. 模型强非线性，一阶近似完全失效
        # 2. 约束过于严格，导致无法移动到正确区域
        # 3. 雅可比估计严重错误
        #
        # 兜底策略：简单线性缩放搜索
        # 数学依据：假设 y ≈ f(scale * base_x)，寻找最优 scale
        
        if y_target_adjusted > y_base:
            # 需要增加y：尝试放大投放
            # 使用对数空间搜索，覆盖更广的范围
            scale_factors = np.exp(np.linspace(0, np.log(3), 15))  # [1.0, ..., 3.0]
        else:
            # 需要减少y：尝试缩小投放
            scale_factors = np.exp(np.linspace(np.log(0.1), 0, 15))  # [0.1, ..., 1.0]
        
        best_scale = 1.0
        best_y = y_base
        best_error = abs(y_base - y_target_adjusted)
        
        for scale in scale_factors:
            scaled_x = base_x * scale
            scaled_x, _ = _apply_constraints(scaled_x, base_x, min_constraints, 
                                            max_constraints, X_train_max)
            y_scaled = _safe_predict(model, scaled_x)
            error_scaled = abs(y_scaled - y_target_adjusted)
            
            # 选择最接近目标的方案
            if error_scaled < best_error:
                best_error = error_scaled
                best_y = y_scaled
                best_scale = scale
        
        # 如果找到了更好的方案，就替换
        if best_error < abs(y_final - y_target_adjusted):
            suggested = base_x * best_scale
            suggested, constraint_status = _apply_constraints(suggested, base_x,
                                                             min_constraints, max_constraints, X_train_max)
            y_final = best_y
    
    # === 计算投入产出分析 ===
    budget_change = float(np.sum(suggested) - np.sum(base_x))
    budget_change_pct = (budget_change / np.sum(base_x)) * 100 if np.sum(base_x) > 0 else 0
    
    output_change = float(y_final - y_base)
    output_change_pct = (output_change / y_base) * 100 if y_base > 0 else 0
    
    # ROI：每增加1单位投放，产出增加多少
    roi = output_change / (budget_change + 1e-8) if abs(budget_change) > 1e-9 else 0.0
    
    # 边际效率：产出增长率 / 投放增长率
    marginal_efficiency = (output_change_pct / budget_change_pct) if abs(budget_change_pct) > 1e-6 else 0.0
    
    # 判断投入产出是否合理
    # 标准：投放增加100%，产出至少应该增加30%（根据业务调整）
    efficiency_reasonable = (marginal_efficiency >= 0.3) or (budget_change <= 0)
    
    # 生成警告信息
    warnings_list = []
    
    # 1. 目标差距警告
    final_gap_pct = abs(y_final - original_target) / max(abs(original_target), 1.0)
    if final_gap_pct > 0.2:
        warnings_list.append({
            'code': 'large_gap',
            'severity': 'warning',
            'message': f"预测结果与目标差距较大 ({final_gap_pct:.1%})",
            'suggestion': "目标可能超出模型的可行范围，建议降低目标或放宽约束"
        })
    
    # 2. 边际效率警告
    if not efficiency_reasonable and budget_change > 0:
        warnings_list.append({
            'code': 'low_efficiency',
            'severity': 'warning',
            'message': f"投入产出效率较低 (边际效率 {marginal_efficiency:.2f})",
            'suggestion': "当前已进入边际递减区域，继续增加投放可能不划算"
        })
    
    # 3. 约束触达警告
    constraints_hit = [s for s in constraint_status if s != '正常']
    if len(constraints_hit) > n_features * 0.5:
        warnings_list.append({
            'code': 'constraints_tight',
            'severity': 'info',
            'message': f"超过50%的渠道触达约束边界",
            'suggestion': "约束条件可能限制了优化空间，建议适当放宽"
        })
        
    # 4. SHAP 校准警告
    if diagnostics.get('sign_conflict_ratio', 0) > 0.3:
        warnings_list.append({
            'code': 'shap_conflict',
            'severity': 'info',
            'message': f"SHAP方向与梯度方向存在冲突 (冲突率 {diagnostics['sign_conflict_ratio']:.0%})",
            'suggestion': "模型在当前区域可能存在复杂的非线性交互"
        })
    
    return {
        'suggested_allocation': suggested,
        'predicted_value': float(y_final),
        'target_value': float(original_target),  # 返回原始目标
        'adjusted_target': float(y_target_adjusted) if not target_feasible else None,
        'base_value': float(y_base),
        'constraint_status': constraint_status,
        'budget_change': budget_change,
        'budget_change_pct': budget_change_pct,
        'output_change': output_change,
        'output_change_pct': output_change_pct,
        'roi': roi,
        'marginal_efficiency': marginal_efficiency,
        'efficiency_reasonable': efficiency_reasonable,
        'target_feasible': target_feasible,
        'feasible_range': feasible_range,
        'method_used': 'direct_linear_solve_v2_with_shap_calibration',
        'robustness_level': 'high',
        'warnings': warnings_list,
        
        # ===== 新增：SHAP 校准诊断字段 =====
        'calibration_strategy': 'shap_hybrid' if diagnostics.get('shap_used') else 'weight_scaling',
        'calibration_reason': diagnostics.get('calibration_reason'),
        'nonlinearity_score': diagnostics.get('nonlinearity_score'),
        'alpha_used': diagnostics.get('alpha_used') or diagnostics.get('alpha_adjusted'),
        
        'shap_used': diagnostics.get('shap_used', False),
        'shap_runtime_ms': diagnostics.get('shap_runtime_ms'),
        'shap_method': diagnostics.get('shap_method'),
        'shap_failure_reason': diagnostics.get('failure_reason') or shap_result.get('failure_reason') if shap_result else None,
        
        'sign_conflict_ratio': diagnostics.get('sign_conflict_ratio'),
        'fallback_reason': diagnostics.get('fallback_reason'),
        'denominator_fallback': diagnostics.get('denominator_fallback'),
        
        'jacobian_norm': diagnostics.get('jacobian_norm'),
        'shap_norm': diagnostics.get('shap_norm')
    }
    


def _generate_warnings(target_feasible, efficiency_reasonable,
                       original_target, y_max_possible, y_final,
                       budget_change_pct, output_change_pct,
                       min_constraints=None, max_constraints=None):
    """生成优化警告信息"""
    warnings = []
    
    marginal_eff = output_change_pct / budget_change_pct if abs(budget_change_pct) > 1e-6 else 0
    
    # 判断是否是无约束模式
    is_unconstrained = (min_constraints is None and max_constraints is None)
    
    if not target_feasible:
        gap = original_target - y_final
        gap_pct = (gap / original_target) * 100 if original_target > 0 else 0
        
        # 判断实际结果是否接近目标
        if abs(y_final - original_target) / original_target < 0.1:  # 10%以内
            severity = 'low'
            message = f'💡 目标 {original_target:.0f} 略超预估，但实际达到了 {y_final:.0f}（差距 {gap_pct:.1f}%）'
            suggestion = '优化算法通过智能分配找到了更优方案，实际产出接近目标。'
        else:
            severity = 'medium'
            message = f'⚠️ 目标 {original_target:.0f} 较高，当前达到 {y_final:.0f}（差距 {gap_pct:.1f}%）'
            
            # 根据约束模式给出不同的建议
            if is_unconstrained:
                # 无约束模式：不建议调整约束或预算
                suggestion = f'根据模型响应曲线分析，当前方案已接近最优平衡点。如需更高产出，建议：(1) 重新评估目标合理性；(2) 优化模型质量（增加训练数据、特征工程等）；(3) 探索新的营销渠道。'
            else:
                # 有约束模式：可以建议放宽约束
                suggestion = f'根据模型响应曲线和投入产出分析，当前方案已接近最优平衡点。如需更高产出，可能需要：(1) 放宽约束条件；(2) 增加可调配预算；(3) 优化模型质量。'
        
        warnings.append({
            'type': 'target_infeasible',
            'severity': severity,
            'message': message,
            'suggestion': suggestion
        })
    
    if not efficiency_reasonable and budget_change_pct > 50:
        warnings.append({
            'type': 'low_roi',
            'severity': 'high',
            'message': f'⚠️ 投入产出比偏低：投放增加 {budget_change_pct:.0f}%，产出增加 {output_change_pct:.0f}% (边际效率={marginal_eff:.2f})',
            'suggestion': f'模型已进入边际递减区域，继续增加投放的回报率较低。建议：(1) 接受当前方案 (产出≈{y_final:.0f})；(2) 如需更高产出，可能需要优化模型或拓展新渠道。'
        })
    
    # 新增：极端ROI警告
    if marginal_eff < 0.15 and budget_change_pct > 200:
        warnings.append({
            'type': 'extreme_low_roi',
            'severity': 'high',
            'message': f'🚨 极度不经济：投放增加 {budget_change_pct:.0f}%，但边际效率仅 {marginal_eff:.2f}',
            'suggestion': f'当前方案投入产出严重失衡。强烈建议大幅降低目标至 {y_final * 0.6:.0f} 以内，或重新审视业务策略。'
        })
    
    if abs(y_final - original_target) / max(abs(original_target), 1.0) > 0.2:
        # 根据约束模式给出不同的建议
        if is_unconstrained:
            # 无约束模式：建议重新评估目标
            suggestion = '建议重新评估目标合理性，或优化模型以提升预测能力'
        else:
            # 有约束模式：可以建议调整约束或预算
            suggestion = '可能需要调整约束条件或增加可调配预算'
        
        warnings.append({
            'type': 'large_gap',
            'severity': 'low',
            'message': f'💡 实际预测值 {y_final:.0f} 与目标 {original_target:.0f} 差距较大',
            'suggestion': suggestion
        })
    
    return warnings


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
