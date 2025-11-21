"""
诊断脚本：找出线性分配法的工作情况
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, '.')
from optimization import (
    _safe_model_predict,
    _smoothed_predict,
    _robust_sensitivity_estimation,
    _enhanced_linear_allocation
)

def diagnose_optimization():
    """诊断优化问题"""
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = np.random.rand(n_samples, n_features) * 100
    # 简单线性关系：y = 0.5*x0 + 0.3*x1 + 0.2*x2 + ...
    y_train = (X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + 
               X_train[:, 2] * 0.2 + X_train[:, 3] * 0.1 + 
               X_train[:, 4] * 0.05 + np.random.randn(n_samples) * 2)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # 基准分配
    base_x = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    
    print("=== 诊断步骤 ===\n")
    
    # 1. 测试基准预测
    y_base = _smoothed_predict(model, base_x)
    print(f"基准分配: {base_x}")
    print(f"基准预测: {y_base:.4f}\n")
    
    # 2. 测试敏感度
    X_min = np.zeros(n_features)
    X_max = np.max(X_train, axis=0)
    sensitivities = _robust_sensitivity_estimation(model, base_x, ['x0','x1','x2','x3','x4'], X_min, X_max)
    print(f"敏感度: {sensitivities}")
    print(f"敏感度范数: {np.linalg.norm(sensitivities):.6f}\n")
    
    # 3. 设置目标值
    target_delta = 0.3  # 目标增加30%
    y_target = y_base * (1 + target_delta)
    print(f"目标预测: {y_target:.4f} (基准 + {target_delta*100:.0f}%)")
    print(f"目标偏差: {y_target - y_base:.4f}\n")
    
    # 4. 测试线性分配法
    print("=== 测试线性分配法 ===")
    weights = np.ones(n_features) / n_features
    
    try:
        suggested, y_pred, constraints = _enhanced_linear_allocation(
            model, base_x, y_base, y_target, weights, sensitivities,
            min_constraints=None, max_constraints=None, X_train_max=X_max,
            n_starts=6
        )
        print(f"建议分配: {suggested}")
        print(f"预测值: {y_pred:.4f}")
        print(f"目标值: {y_target:.4f}")
        accuracy = (1 - abs(y_pred - y_target) / y_target) * 100
        print(f"达成度: {accuracy:.1f}%\n")
        
        print(f"约束状态: {constraints}\n")
        
    except Exception as e:
        print(f"线性分配法失败: {e}\n")
    
    # 5. 测试多个目标值的收敛情况
    print("=== 测试不同目标值的收敛情况 ===")
    for delta in [0.1, 0.2, 0.5, 1.0]:
        y_test = y_base * (1 + delta)
        try:
            suggested, y_pred, _ = _enhanced_linear_allocation(
                model, base_x, y_base, y_test, weights, sensitivities,
                min_constraints=None, max_constraints=None, X_train_max=X_max
            )
            accuracy = (1 - abs(y_pred - y_test) / y_test) * 100
            print(f"目标 +{delta*100:.0f}%: 预测={y_pred:.4f}, 精度={accuracy:.1f}%")
        except Exception as e:
            print(f"目标 +{delta*100:.0f}%: 失败 - {str(e)[:50]}")

if __name__ == "__main__":
    diagnose_optimization()
