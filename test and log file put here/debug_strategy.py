"""
调试脚本：追踪线性分配法的路径选择
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from optimization import (
    _smoothed_predict,
    _robust_sensitivity_estimation,
    _get_bounds,
    _enhanced_linear_allocation,
    _global_optimization_fallback,
    _ensemble_objective
)

def debug_optimization_strategy():
    """调试优化策略选择"""
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X_train = np.random.rand(n_samples, n_features) * 100
    y_train = (
        X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + 
        X_train[:, 2] * 0.15 + X_train[:, 3] * 0.04 + 
        X_train[:, 4] * 0.01 + np.random.randn(n_samples) * 3
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train, y_train)
    
    # 基准和目标
    base_x = np.array([30.0, 20.0, 15.0, 10.0, 5.0])
    y_base = _smoothed_predict(model, base_x)
    y_target = y_base * 1.5  # 50%增长
    
    weights = np.ones(n_features) / n_features
    
    X_train_min = np.zeros(n_features)
    X_train_max = np.max(X_train, axis=0)
    
    sensitivities = _robust_sensitivity_estimation(
        model, base_x, [f'x{i}' for i in range(n_features)], X_train_min, X_train_max
    )
    
    print(f"基准预测: {y_base:.2f}")
    print(f"目标预测: {y_target:.2f}")
    print(f"敏感度: {sensitivities}\n")
    
    # 计算难度分数
    feature_names = [f'x{i}' for i in range(n_features)]
    current_pred = _smoothed_predict(model, base_x)
    
    sens_variation = np.std(sensitivities) / (np.mean(np.abs(sensitivities)) + 1e-8)
    bounds_obj = _get_bounds(base_x, None, None, X_train_max)
    bounds_span = bounds_obj.ub - bounds_obj.lb
    constraint_tightness = np.mean(bounds_span / (np.abs(bounds_obj.ub) + np.abs(base_x) + 1e-8))
    target_distance = abs(y_target - current_pred) / (abs(current_pred) + 1e-8)
    
    difficulty_score = 0.5 * sens_variation + 0.3 * (1 - constraint_tightness) + 0.2 * min(target_distance, 1.0)
    
    print("=== 难度分数计算 ===")
    print(f"敏感度变异: {sens_variation:.4f} (权重: 0.5)")
    print(f"约束紧密度: {constraint_tightness:.4f} (权重: 0.3)")
    print(f"目标距离: {target_distance:.4f} (权重: 0.2)")
    print(f"总难度分数: {difficulty_score:.4f}")
    print(f"阈值: 0.5")
    print(f"→ 选择: {'线性分配法' if difficulty_score < 0.5 else '全局优化'}\n")
    
    # 测试线性分配法
    print("=== 线性分配法结果 ===")
    suggested_linear, pred_linear, status_linear = _enhanced_linear_allocation(
        model, base_x, y_base, y_target, weights, sensitivities,
        None, None, X_train_max, n_starts=6
    )
    accuracy_linear = (1 - abs(pred_linear - y_target) / y_target) * 100
    print(f"预测: {pred_linear:.2f}")
    print(f"精度: {accuracy_linear:.1f}%")
    print(f"建议: {suggested_linear}\n")
    
    # 测试全局优化
    print("=== 全局优化结果 ===")
    def objective_func(x):
        return _ensemble_objective(model, x, y_target, base_x, feature_names)
    
    global_x, _, _ = _global_optimization_fallback(
        model, objective_func, bounds_obj, base_x, pop_size=20, max_iter=200
    )
    pred_global = _smoothed_predict(model, global_x)
    accuracy_global = (1 - abs(pred_global - y_target) / y_target) * 100
    print(f"预测: {pred_global:.2f}")
    print(f"精度: {accuracy_global:.1f}%")
    print(f"建议: {global_x}")

if __name__ == "__main__":
    debug_optimization_strategy()
