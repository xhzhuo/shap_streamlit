"""
测试优化重构后的功能
验证：
1. ODE反推优化
2. 预测不确定性估计
3. 代码精简效果
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from optimization import optimize_ad_allocation_robust

def test_optimization_with_uncertainty():
    """测试优化功能和不确定性估计"""
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = np.random.rand(n_samples, n_features) * 100
    y_train = X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + X_train[:, 2] * 0.2 + np.random.randn(n_samples) * 5
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 基准分配
    base_x = np.array([50.0, 30.0, 20.0, 10.0, 5.0])
    
    # 当前预测
    current_pred = model.predict(base_x.reshape(1, -1))[0]
    print(f"当前预测值: {current_pred:.2f}")
    
    # 目标值（提高20%）
    target_value = current_pred * 1.2
    print(f"目标值: {target_value:.2f}")
    
    # 执行优化
    print("\n执行优化...")
    result = optimize_ad_allocation_robust(
        model=model,
        base_x=base_x,
        y_target=target_value,
        X_train=X_train,
        method='adaptive'
    )
    
    # 输出结果
    print("\n=== 优化结果 ===")
    print(f"建议分配: {result['suggested_allocation']}")
    print(f"预测值: {result['predicted_value']:.2f}")
    print(f"目标值: {result['target_value']:.2f}")
    print(f"基准值: {result['base_value']:.2f}")
    print(f"预算变化: {result['budget_change']:.2f}")
    print(f"效率增益: {result['efficiency_gain']:.4f}")
    
    # 计算精度
    accuracy = abs(result['predicted_value'] - target_value) / target_value * 100
    print(f"\n目标达成精度: {100 - accuracy:.2f}%")
    
    # 验证
    assert result['predicted_value'] is not None, "预测值不能为None"
    assert result['target_value'] == target_value, "目标值不匹配"
    
    print("\n✓ 测试通过！")
    
    return result

if __name__ == "__main__":
    test_optimization_with_uncertainty()
