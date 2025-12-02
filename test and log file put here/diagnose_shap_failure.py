"""
SHAP 计算失败诊断脚本
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('.')

from optimization import _compute_shap_marginal

print("=" * 60)
print("SHAP 计算失败诊断")
print("=" * 60)

# 加载数据
data_path = r"test and log file put here\Advertising_Data.csv"
df = pd.read_csv(data_path)

feature_cols = df.columns[:-1].tolist()
target_col = df.columns[-1]

X = df[feature_cols].values
y = df[target_col].values

# 训练模型
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X[:240], y[:240])

print(f"\n[1] 模型类型: {type(model)}")
print(f"    模型类名: {model.__class__.__name__}")

# 测试点
test_x = X[10].copy()
print(f"\n[2] 测试点: {test_x}")
print(f"    预测值: {model.predict([test_x])[0]:.2f}")

# 尝试直接调用 SHAP 计算
print(f"\n[3] 调用 _compute_shap_marginal...")
shap_result = _compute_shap_marginal(model, test_x, eps=0.02)

print(f"\n[4] SHAP 计算结果:")
print(f"    shap_marginal: {shap_result['shap_marginal']}")
print(f"    runtime_ms: {shap_result['runtime_ms']:.1f}")
print(f"    method: {shap_result['method']}")
print(f"    failure_reason: {shap_result['failure_reason']}")
print(f"    scale_check_passed: {shap_result['scale_check_passed']}")

if shap_result['shap_marginal'] is not None:
    print(f"\n✓ SHAP 计算成功!")
    print(f"  SHAP 边际值: {shap_result['shap_marginal']}")
else:
    print(f"\n✗ SHAP 计算失败!")
    print(f"  失败原因: {shap_result['failure_reason']}")

# 尝试手动使用 SHAP
print(f"\n[5] 尝试手动创建 TreeExplainer...")
try:
    import shap
    print(f"    SHAP 库版本: {shap.__version__}")
    
    explainer = shap.TreeExplainer(model)
    print(f"    ✓ TreeExplainer 创建成功")
    
    shap_values = explainer.shap_values(test_x.reshape(1, -1))
    print(f"    SHAP 值类型: {type(shap_values)}")
    print(f"    SHAP 值形状: {np.array(shap_values).shape}")
    print(f"    SHAP 值: {shap_values}")
    
except Exception as e:
    print(f"    ✗ TreeExplainer 失败: {e}")
    print(f"    错误类型: {type(e).__name__}")
    
    import traceback
    print(f"\n完整错误堆栈:")
    traceback.print_exc()

print(f"\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
