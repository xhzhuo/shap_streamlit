"""快速诊断：为什么 SHAP 未启用"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('.')

from optimization import optimize_allocation_v2, _detect_nonlinearity, _estimate_jacobian_fast

# 加载数据并训练模型
df = pd.read_csv(r"test and log file put here\Advertising_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X[:240], y[:240])

# 测试点
base_x = X[10].copy()
weights = model.feature_importances_ / model.feature_importances_.sum()

print("=" * 70)
print("快速诊断：SHAP 未启用原因")
print("=" * 70)

# 检查1：非线性得分
jac, _ = _estimate_jacobian_fast(model, base_x)
nonlin = _detect_nonlinearity(model, base_x, jac)
print(f"\n[1] 非线性得分: {nonlin:.6f}")
print(f"    阈值: 0.2")
print(f"    应该启用 SHAP: {'是' if nonlin >= 0.2 else '否'}")

# 检查2：运行完整优化
print(f"\n[2] 运行完整优化...")
result = optimize_allocation_v2(
    model, base_x, y[10] * 1.5,
    weights=weights,
    X_train=X[:240],
    max_iterations=2
)

print(f"\n[3] 优化结果:")
print(f"    校准策略: {result.get('calibration_strategy')}")
print(f"    校准原因: {result.get('calibration_reason')}")
print(f"    SHAP 是否使用: {result.get('shap_used')}")
print(f"    SHAP 失败原因: {result.get('shap_failure_reason')}")
print(f"    非线性得分: {result.get('nonlinearity_score')}")

# 检查3：查看所有诊断字段
print(f"\n[4] 所有 SHAP 相关诊断:")
shap_fields = {k: v for k, v in result.items() if 'shap' in k.lower() or 'calibration' in k.lower()}
for key, value in shap_fields.items():
    print(f"    {key}: {value}")

print(f"\n" + "=" * 70)
