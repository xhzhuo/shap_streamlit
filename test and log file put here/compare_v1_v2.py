"""对比测试：V1 vs V2 性能"""
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

# 导入两个版本
import optimization as opt_v1
import optimization_v2 as opt_v2

print("="*70)
print("优化算法对比测试：V1（迭代优化） vs V2（直接求解）")
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

# 测试场景
test_cases = [
    (base_pred * 1.1, "+10%"),
    (base_pred * 1.2, "+20%"),
    (base_pred * 1.5, "+50%"),
]

print("\n" + "="*70)
print(f"{'目标':<12} {'方法':<8} {'耗时(秒)':<12} {'预测值':<12} {'精度(%)':<12}")
print("="*70)

results_comparison = []

for target, label in test_cases:
    # V1测试
    start = time.time()
    result_v1 = opt_v1.optimize_ad_allocation_robust(
        model=model, base_x=base_x, y_target=target,
        weights=np.ones(n_features)/n_features, X_train=X
    )
    time_v1 = time.time() - start
    pred_v1 = result_v1['predicted_value']
    acc_v1 = (1 - abs(pred_v1 - target) / abs(target)) * 100
    
    # V2测试
    start = time.time()
    result_v2 = opt_v2.optimize_ad_allocation_robust(
        model=model, base_x=base_x, y_target=target,
        weights=np.ones(n_features)/n_features, X_train=X
    )
    time_v2 = time.time() - start
    pred_v2 = result_v2['predicted_value']
    acc_v2 = (1 - abs(pred_v2 - target) / abs(target)) * 100
    
    # 输出
    print(f"{label:<12} {'V1':<8} {time_v1:<12.3f} {pred_v1:<12.2f} {acc_v1:<12.1f}")
    print(f"{label:<12} {'V2':<8} {time_v2:<12.3f} {pred_v2:<12.2f} {acc_v2:<12.1f}")
    print(f"{'':<12} {'加速比':<8} {time_v1/time_v2:<12.1f}x")
    print("-"*70)
    
    results_comparison.append({
        'target': label,
        'v1_time': time_v1,
        'v2_time': time_v2,
        'speedup': time_v1/time_v2,
        'v1_acc': acc_v1,
        'v2_acc': acc_v2
    })

print("\n" + "="*70)
print("总结")
print("="*70)

avg_speedup = np.mean([r['speedup'] for r in results_comparison])
avg_v1_acc = np.mean([r['v1_acc'] for r in results_comparison])
avg_v2_acc = np.mean([r['v2_acc'] for r in results_comparison])

print(f"\n平均加速比: {avg_speedup:.1f}x")
print(f"V1平均精度: {avg_v1_acc:.1f}%")
print(f"V2平均精度: {avg_v2_acc:.1f}%")

print("\nV2 算法优势:")
print("  ✅ 批量预测：一次调用计算所有雅可比元素")
print("  ✅ 直接求解：基于一阶泰勒展开，无需迭代搜索")
print("  ✅ 自适应精调：仅在需要时进行2-3次迭代")
print("  ✅ 更少的模型调用：V1需要几十次，V2仅需3-5次")

print("\n推荐方案:")
if avg_speedup > 2 and avg_v2_acc > 90:
    print("  ⭐ 建议使用 V2（直接求解法）- 更快更稳定")
    print("  📝 修改方式: 将 optimization.py 替换为 optimization_v2.py")
else:
    print("  ⚠️ 根据具体情况选择合适的版本")
