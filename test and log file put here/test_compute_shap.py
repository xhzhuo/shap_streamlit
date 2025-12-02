"""测试 _compute_shap_marginal 函数"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('.')

from optimization import _compute_shap_marginal

df = pd.read_csv(r"test and log file put here\Advertising_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X[:240], y[:240])

base_x = X[10].copy()

print("测试 _compute_shap_marginal 函数")
print("=" * 70)

result = _compute_shap_marginal(model, base_x, eps=0.02)

print(f"\n返回结果:")
for key, value in result.items():
    print(f"  {key}: {value}")

if result['shap_marginal'] is not None:
    print(f"\n✓ SHAP 计算成功!")
    print(f"  SHAP 边际值: {result['shap_marginal']}")
    print(f"  范数: {np.linalg.norm(result['shap_marginal']):.6f}")
else:
    print(f"\n✗ SHAP 计算失败!")
    print(f"  失败原因: {result['failure_reason']}")

print(f"\n" + "=" * 70)
