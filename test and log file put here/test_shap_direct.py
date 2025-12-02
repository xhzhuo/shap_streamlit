"""直接测试 SHAP 计算"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 加载数据
df = pd.read_csv(r"test and log file put here\Advertising_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X[:240], y[:240])

base_x = X[10].copy()

print("直接测试 SHAP 计算")
print("=" * 60)

# 测试1: 检查 SHAP 库
try:
    import shap
    print(f"✓ SHAP 库已安装: version {shap.__version__}")
except:
    print("✗ SHAP 库未安装")
    exit(1)

# 测试2: 创建 TreeExplainer
try:
    explainer = shap.TreeExplainer(model)
    print(f"✓ TreeExplainer 创建成功")
except Exception as e:
    print(f"✗ TreeExplainer 创建失败: {e}")
    exit(1)

# 测试3: 计算 SHAP 值
try:
    shap_values = explainer.shap_values(base_x.reshape(1, -1))
    print(f"✓ SHAP 值计算成功")
    print(f"  类型: {type(shap_values)}")
    print(f"  形状: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'N/A'}")
    if isinstance(shap_values, np.ndarray):
        print(f"  值: {shap_values[0]}")
    else:
        print(f"  值类型异常: {type(shap_values)}")
except Exception as e:
    print(f"✗ SHAP 值计算失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 手动计算单位化边际 SHAP
print(f"\n手动计算单位化边际 SHAP:")
eps = 0.02
n = len(base_x)

X_batch = np.tile(base_x, (2*n + 1, 1))
for i in range(n):
    h = max(abs(base_x[i]) * eps, eps)
    X_batch[2*i + 1, i] += h
    X_batch[2*i + 2, i] -= h

try:
    shap_values_batch = explainer.shap_values(X_batch)
    print(f"  批量 SHAP 计算成功，形状: {np.array(shap_values_batch).shape}")
    
    shap_marginal = np.zeros(n)
    for i in range(n):
        h = max(abs(base_x[i]) * eps, eps)
        phi_plus = shap_values_batch[2*i + 1]
        phi_minus = shap_values_batch[2*i + 2]
        shap_marginal[i] = (phi_plus[i] - phi_minus[i]) / (2 * h)
    
    print(f"  ✓ 单位化边际 SHAP: {shap_marginal}")
    print(f"  范数: {np.linalg.norm(shap_marginal):.6f}")
    
except Exception as e:
    print(f"  ✗ 批量计算失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 60)
print("测试完成")
