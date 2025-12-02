"""
简单验证脚本：测试 SHAP 校准功能
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('.')

# 导入优化模块
from optimization import optimize_allocation_v2, _compute_shap_marginal

print("=" * 60)
print("SHAP 校准功能验证测试")
print("=" * 60)

# 生成测试数据
np.random.seed(42)
X_train = np.random.uniform(0, 100, (200, 5))
# 非线性函数：y = x1 + 0.01*x1² + 0.5*x2 + sin(x3/10)
y_train = (X_train[:, 0] + 
           0.01 * X_train[:, 0]**2 + 
           0.5 * X_train[:, 1] + 
           np.sin(X_train[:, 2] / 10) * 20)

# 训练模型
print("\n[1] 训练测试模型...")
model = GradientBoostingRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("✓ 模型训练完成")

# 测试点
base_x = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
base_pred = model.predict([base_x])[0]
print(f"\n[2] 基准点预测值: {base_pred:.2f}")

# 测试 SHAP 计算
print("\n[3] 测试 SHAP 边际值计算...")
shap_result = _compute_shap_marginal(model, base_x, eps=0.02)

if shap_result['shap_marginal'] is not None:
    print(f"✓ SHAP 计算成功")
    print(f"  - 方法: {shap_result['method']}")
    print(f"  - 运行时间: {shap_result['runtime_ms']:.1f} ms")
    print(f"  - SHAP 边际值: {shap_result['shap_marginal']}")
else:
    print(f"✗ SHAP 计算失败: {shap_result['failure_reason']}")

# 测试优化（弱非线性场景）
print("\n[4] 测试优化（目标值较低，应触发弱非线性）...")
y_target_low = base_pred * 1.1
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

result_low = optimize_allocation_v2(
    model, base_x, y_target_low,
    weights=weights,
    X_train=X_train,
    max_iterations=2
)

print(f"✓ 优化完成")
print(f"  - 校准策略: {result_low['calibration_strategy']}")
print(f"  - 非线性得分: {result_low.get('nonlinearity_score', 'N/A'):.3f}")
print(f"  - SHAP 是否使用: {result_low.get('shap_used', False)}")
print(f"  - 预测值: {result_low['predicted_value']:.2f}")
print(f"  - 目标值: {result_low['target_value']:.2f}")

# 测试优化（强非线性场景）
print("\n[5] 测试优化（目标值较高，应触发强非线性+SHAP）...")
y_target_high = base_pred * 1.5

result_high = optimize_allocation_v2(
    model, base_x, y_target_high,
    weights=weights,
    X_train=X_train,
    max_iterations=2
)

print(f"✓ 优化完成")
print(f"  - 校准策略: {result_high['calibration_strategy']}")
print(f"  - 校准原因: {result_high.get('calibration_reason', 'N/A')}")
print(f"  - 非线性得分: {result_high.get('nonlinearity_score', 'N/A'):.3f}")
print(f"  - SHAP 是否使用: {result_high.get('shap_used', False)}")
if result_high.get('shap_used'):
    print(f"  - SHAP 方法: {result_high.get('shap_method')}")
    print(f"  - SHAP 运行时间: {result_high.get('shap_runtime_ms', 0):.1f} ms")
    print(f"  - Alpha 使用值: {result_high.get('alpha_used', 'N/A')}")
    print(f"  - 符号冲突率: {result_high.get('sign_conflict_ratio', 0):.1%}")
print(f"  - 预测值: {result_high['predicted_value']:.2f}")
print(f"  - 目标值: {result_high['target_value']:.2f}")
print(f"  - 预算变化: {result_high['budget_change_pct']:.1f}%")

print("\n" + "=" * 60)
print("✓ 所有测试完成！")
print("=" * 60)
