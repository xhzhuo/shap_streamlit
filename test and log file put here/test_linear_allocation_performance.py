"""
完整性能测试：验证线性分配法的精准性恢复
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from optimization import optimize_ad_allocation_robust

def test_linear_allocation_accuracy():
    """测试线性分配法的精准性"""
    
    # 创建合理的模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    # 生成特征
    X_train = np.random.rand(n_samples, n_features) * 100
    
    # 生成目标：有明确的特征重要性关系
    y_train = (
        X_train[:, 0] * 0.5 +      # 特征0最重要
        X_train[:, 1] * 0.3 +      # 特征1次重要
        X_train[:, 2] * 0.15 +     # 特征2中等重要
        X_train[:, 3] * 0.04 +     # 特征3较轻
        X_train[:, 4] * 0.01 +     # 特征4最轻
        np.random.randn(n_samples) * 3  # 噪声
    )
    
    # 训练模型
    print("训练Random Forest模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # 基准分配
    base_allocation = np.array([30.0, 20.0, 15.0, 10.0, 5.0])
    base_pred = model.predict(base_allocation.reshape(1, -1))[0]
    
    print(f"\n{'='*50}")
    print(f"基准分配: {base_allocation}")
    print(f"基准预测: {base_pred:.2f}")
    print(f"{'='*50}\n")
    
    # 测试不同目标增幅
    test_cases = [
        ('保守', 0.10),    # +10%
        ('中等', 0.20),    # +20%
        ('积极', 0.50),    # +50%
    ]
    
    results = []
    for name, delta in test_cases:
        target = base_pred * (1 + delta)
        
        print(f"测试: {name}目标 (基准 + {delta*100:.0f}%)")
        print(f"目标值: {target:.2f}")
        
        # 执行优化
        result = optimize_ad_allocation_robust(
            model=model,
            base_x=base_allocation,
            y_target=target,
            X_train=X_train,
            method='adaptive'
        )
        
        pred = result['predicted_value']
        actual_delta = (pred - base_pred) / base_pred
        accuracy = (1 - abs(pred - target) / target) * 100
        
        # 预算变化分析
        budget_before = np.sum(base_allocation)
        budget_after = np.sum(result['suggested_allocation'])
        budget_change_pct = (budget_after - budget_before) / budget_before * 100 if budget_before > 0 else 0
        
        print(f"建议分配: {result['suggested_allocation']}")
        print(f"预测值: {pred:.2f}")
        print(f"实际增幅: {actual_delta*100:.1f}%")
        print(f"目标达成精度: {accuracy:.1f}%")
        print(f"预算变化: {budget_before:.1f} → {budget_after:.1f} ({budget_change_pct:+.1f}%)")
        print()
        
        results.append({
            'name': name,
            'target': target,
            'prediction': pred,
            'accuracy': accuracy,
            'budget_change_pct': budget_change_pct
        })
    
    # 总结
    print(f"{'='*50}")
    print("性能总结")
    print(f"{'='*50}")
    
    all_accurate = True
    for r in results:
        status = "✅ PASS" if r['accuracy'] > 90 else "❌ FAIL"
        print(f"{r['name']:6s}: 精度 {r['accuracy']:5.1f}% {status}")
        if r['accuracy'] <= 90:
            all_accurate = False
    
    print(f"\n整体评价: {'✅ 线性分配法恢复成功！' if all_accurate else '❌ 仍存在问题'}")
    
    return all_accurate

if __name__ == "__main__":
    success = test_linear_allocation_accuracy()
    exit(0 if success else 1)
