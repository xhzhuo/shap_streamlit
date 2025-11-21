"""测试 V2 直接求解法的性能"""
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from optimization import optimize_ad_allocation_robust

print("="*70)
print("测试 V2 直接求解法 - 速度与精度")
print("="*70)

# 创建测试数据
np.random.seed(42)
n_samples = 1000
n_features = 6

X = np.random.rand(n_samples, n_features) * 100
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
     X[:, 3] * 0.15 + X[:, 4] * 0.1 + X[:, 5] * 0.05)

# 训练模型
print("\n训练模型...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X, y)

# 基准
base_x = X.mean(axis=0)
base_pred = model.predict([base_x])[0]
print(f"基准预测: {base_pred:.2f}")

# 测试不同目标
test_cases = [
    (base_pred * 1.1, "+10%", "小幅提升"),
    (base_pred * 1.2, "+20%", "中等提升"),
    (base_pred * 1.5, "+50%", "大幅提升"),
    (base_pred * 2.0, "+100%", "极端目标"),
]

print("\n" + "="*70)
print(f"{'场景':<15} {'目标':<10} {'耗时(秒)':<10} {'预测值':<12} {'精度(%)':<10}")
print("="*70)

total_time = 0
accuracies = []

for target, label, desc in test_cases:
    start = time.time()
    
    result = optimize_ad_allocation_robust(
        model=model,
        base_x=base_x,
        y_target=target,
        weights=np.ones(n_features) / n_features,
        X_train=X
    )
    
    elapsed = time.time() - start
    total_time += elapsed
    
    pred = result['predicted_value']
    accuracy = (1 - abs(pred - target) / abs(target)) * 100
    accuracies.append(accuracy)
    
    print(f"{desc:<15} {label:<10} {elapsed:<10.3f} {pred:<12.2f} {accuracy:<10.1f}")

print("="*70)
print(f"\n平均耗时: {total_time/len(test_cases):.3f} 秒")
print(f"平均精度: {np.mean(accuracies):.1f}%")

print("\n✨ V2 算法特点:")
print("  • 批量预测雅可比矩阵（一次调用 2*n+1 个预测）")
print("  • 基于一阶泰勒展开直接求解")
print("  • 自适应迭代精调（通常1-2次即可）")
print("  • 预期比 V1 快 5-10 倍")

print("\n📊 性能分析:")
if total_time / len(test_cases) < 0.5:
    print("  ⚡ 速度优秀！平均响应时间 < 0.5秒")
elif total_time / len(test_cases) < 1.0:
    print("  ✅ 速度良好，平均响应时间 < 1秒")
else:
    print("  ⚠️  可能需要进一步优化")

if np.mean(accuracies) > 95:
    print("  🎯 精度优秀！平均 > 95%")
elif np.mean(accuracies) > 90:
    print("  ✅ 精度良好，平均 > 90%")
else:
    print("  ⚠️  精度可能需要改进")
