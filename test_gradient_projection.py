"""对比测试：改进前 vs 梯度投影改进后"""
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import sys

# 先备份当前版本
import shutil
shutil.copy('optimization.py', 'optimization_gradient.py')

# 测试梯度投影版本
from optimization import optimize_ad_allocation_robust

print("="*70)
print("梯度投影法改进效果测试")
print("="*70)

# 创建测试数据
np.random.seed(42)
n_samples = 1000
n_features = 6

X = np.random.rand(n_samples, n_features) * 100
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
     X[:, 3] * 0.15 + X[:, 4] * 0.1 + X[:, 5] * 0.05)

# 训练模型
print("\n训练 Random Forest 模型...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X, y)

# 基准
base_x = X.mean(axis=0)
base_pred = model.predict([base_x])[0]
print(f"基准预测: {base_pred:.2f}")

# 测试不同目标
test_cases = [
    (base_pred * 1.05, "+5%", "微调"),
    (base_pred * 1.1, "+10%", "小幅提升"),
    (base_pred * 1.2, "+20%", "中等提升"),
    (base_pred * 1.5, "+50%", "大幅提升"),
    (base_pred * 2.0, "+100%", "极端目标"),
]

print("\n" + "="*70)
print(f"{'场景':<12} {'目标':<8} {'耗时(s)':<10} {'预测值':<12} {'精度(%)':<10} {'误差':<10}")
print("="*70)

results = []
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
    pred = result['predicted_value']
    accuracy = (1 - abs(pred - target) / abs(target)) * 100
    error = abs(pred - target)
    
    print(f"{desc:<12} {label:<8} {elapsed:<10.3f} {pred:<12.2f} {accuracy:<10.1f} {error:<10.2f}")
    
    results.append({
        'target': label,
        'time': elapsed,
        'accuracy': accuracy,
        'error': error
    })

print("="*70)

# 统计
avg_time = np.mean([r['time'] for r in results])
avg_acc = np.mean([r['accuracy'] for r in results])
avg_error = np.mean([r['error'] for r in results])

print(f"\n📊 统计结果:")
print(f"  平均耗时: {avg_time:.3f} 秒")
print(f"  平均精度: {avg_acc:.1f}%")
print(f"  平均误差: {avg_error:.2f}")

print("\n✨ 梯度投影改进点:")
print("  1. 最优步长计算: α* = Δy / ||grad||²")
print("  2. 加权梯度方向: direction = jacobian × weights")
print("  3. 自适应信赖域: 根据梯度大小动态调整")
print("  4. Line Search: 尝试多个步长，选择最优")

print("\n🎯 改进效果预期:")
print("  • 精度提升: +2-5% (93.5% → 95-98%)")
print("  • 速度影响: +0-20% (0.10s → 0.10-0.12s)")
print("  • 数学严格性: ⭐⭐ → ⭐⭐⭐")

# 性能评估
if avg_time < 0.15:
    print("\n  ⚡ 速度评估: 优秀（< 0.15秒）")
elif avg_time < 0.25:
    print("\n  ✅ 速度评估: 良好（< 0.25秒）")
else:
    print("\n  ⚠️  速度评估: 需要优化")

if avg_acc > 95:
    print("  🎯 精度评估: 优秀（> 95%）")
elif avg_acc > 90:
    print("  ✅ 精度评估: 良好（> 90%）")
else:
    print("  ⚠️  精度评估: 需要改进")

print("\n💡 数学原理:")
print("  改进前: Δx = (weights × Δy) / sensitivity  (启发式)")
print("  改进后: Δx = α* × (jacobian × weights)     (梯度下降)")
print("         其中 α* 通过解析公式计算，保证最优步长")
