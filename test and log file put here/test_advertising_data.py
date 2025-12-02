"""
广告数据反推算法完整测试
测试场景：正常目标、极端目标（基准的3倍）
验证：算法合理性、SHAP校准、ROI检查、降级机制
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

from optimization import optimize_allocation_v2

print("=" * 80)
print("广告数据反推算法合理性测试")
print("=" * 80)

# ===== 1. 加载数据 =====
print("\n[步骤1] 加载广告数据...")
data_path = r"test and log file put here\Advertising_Data.csv"
df = pd.read_csv(data_path)

print(f"✓ 数据加载成功")
print(f"  - 数据形状: {df.shape}")
print(f"  - 列名: {list(df.columns)}")
print(f"\n前3行数据:")
print(df.head(3))

# 确定特征列和目标列
# 假设最后一列是目标，其余是特征
feature_cols = df.columns[:-1].tolist()
target_col = df.columns[-1]

print(f"\n特征列: {feature_cols}")
print(f"目标列: {target_col}")

X = df[feature_cols].values
y = df[target_col].values

print(f"\n特征范围:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: [{X[:, i].min():.1f}, {X[:, i].max():.1f}]")
print(f"目标范围: [{y.min():.1f}, {y.max():.1f}]")

# ===== 2. 训练模型 =====
print("\n[步骤2] 训练 GradientBoosting 模型...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✓ 模型训练完成")
print(f"  - 训练R²: {train_score:.4f}")
print(f"  - 测试R²: {test_score:.4f}")

# ===== 3. 选择测试点 =====
print("\n[步骤3] 选择测试基准点...")
# 使用测试集中的一个点作为基准
base_idx = 10
base_x = X_test[base_idx].copy()
base_pred = model.predict([base_x])[0]

print(f"基准投放: {base_x}")
print(f"基准预测值: {base_pred:.2f}")

# 计算 SHAP 权重（简化版，使用特征重要性）
feature_importance = model.feature_importances_
weights = feature_importance / feature_importance.sum()
print(f"SHAP 权重: {weights}")

# ===== 4. 测试场景 =====
test_scenarios = [
    ("正常目标 (+20%)", base_pred * 1.2, "温和增长，应该较容易达到"),
    ("较高目标 (+50%)", base_pred * 1.5, "中等难度，可能触发 SHAP 校准"),
    ("高目标 (+100%)", base_pred * 2.0, "高难度，应触发 SHAP 校准和 ROI 检查"),
    ("极端目标 (+200%)", base_pred * 3.0, "极端场景，应触发降级和警告"),
]

print("\n" + "=" * 80)
print("测试场景汇总")
print("=" * 80)

results_summary = []

for scenario_name, y_target, description in test_scenarios:
    print(f"\n{'─' * 80}")
    print(f"【{scenario_name}】")
    print(f"描述: {description}")
    print(f"目标值: {y_target:.2f} (基准值: {base_pred:.2f})")
    print(f"{'─' * 80}")
    
    # 运行优化
    result = optimize_allocation_v2(
        model=model,
        base_x=base_x,
        y_target=y_target,
        weights=weights,
        min_constraints=None,  # 无约束模式
        max_constraints=None,
        X_train=X_train,
        max_iterations=3,
        tolerance=0.02
    )
    
    # ===== 分析结果 =====
    print("\n📊 优化结果:")
    print(f"  预测值: {result['predicted_value']:.2f}")
    print(f"  目标值: {result['target_value']:.2f}")
    print(f"  预测误差: {abs(result['predicted_value'] - y_target)/y_target*100:.1f}%")
    
    print(f"\n💰 投入产出:")
    print(f"  预算变化: {result['budget_change_pct']:.1f}%")
    print(f"  产出变化: {result['output_change_pct']:.1f}%")
    print(f"  边际效率: {result['marginal_efficiency']:.3f}")
    print(f"  效率合理: {'✓ 是' if result['efficiency_reasonable'] else '✗ 否'}")
    print(f"  ROI: {result['roi']:.4f}")
    
    print(f"\n🔧 算法策略:")
    print(f"  校准方法: {result.get('calibration_strategy', 'N/A')}")
    print(f"  校准原因: {result.get('calibration_reason', 'N/A')}")
    print(f"  非线性得分: {result.get('nonlinearity_score', 0):.3f}")
    
    if result.get('shap_used'):
        print(f"\n🔬 SHAP 信息:")
        print(f"  SHAP 已启用: ✓")
        print(f"  SHAP 方法: {result.get('shap_method')}")
        print(f"  SHAP 运行时间: {result.get('shap_runtime_ms', 0):.1f} ms")
        print(f"  Alpha 使用值: {result.get('alpha_used', 'N/A')}")
        print(f"  符号冲突率: {result.get('sign_conflict_ratio', 0):.1%}")
        if result.get('fallback_reason'):
            print(f"  ⚠️ Fallback 原因: {result.get('fallback_reason')}")
    else:
        print(f"\n🔬 SHAP 信息:")
        print(f"  SHAP 已启用: ✗ (使用原权重方案)")
        if result.get('shap_failure_reason'):
            print(f"  失败原因: {result.get('shap_failure_reason')}")
    
    print(f"\n⚠️ 警告信息:")
    if result.get('warnings'):
        for warning in result['warnings']:
            print(f"  [{warning['severity'].upper()}] {warning['message']}")
            print(f"    建议: {warning['suggestion']}")
    else:
        print(f"  无警告")
    
    # 保存结果用于汇总
    results_summary.append({
        'scenario': scenario_name,
        'target': y_target,
        'predicted': result['predicted_value'],
        'error_pct': abs(result['predicted_value'] - y_target)/y_target*100,
        'budget_change_pct': result['budget_change_pct'],
        'output_change_pct': result['output_change_pct'],
        'marginal_eff': result['marginal_efficiency'],
        'shap_used': result.get('shap_used', False),
        'nonlinearity': result.get('nonlinearity_score', 0),
        'warnings_count': len(result.get('warnings', []))
    })

# ===== 5. 汇总分析 =====
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

summary_df = pd.DataFrame(results_summary)
print("\n", summary_df.to_string(index=False))

# ===== 6. 合理性分析 =====
print("\n" + "=" * 80)
print("合理性分析")
print("=" * 80)

print("\n✅ 通过检查:")
checks_passed = []
checks_failed = []

# 检查1：边际效率递减
marginal_effs = summary_df['marginal_eff'].tolist()
if all(marginal_effs[i] >= marginal_effs[i+1] or marginal_effs[i] < 0 for i in range(len(marginal_effs)-1)):
    checks_passed.append("边际效率递减规律（目标越高，效率越低）")
else:
    checks_failed.append("边际效率未呈现递减趋势")

# 检查2：SHAP 在高非线性时启用
high_target_rows = summary_df[summary_df['target'] > base_pred * 1.5]
if len(high_target_rows) > 0:
    shap_usage_rate = high_target_rows['shap_used'].mean()
    if shap_usage_rate > 0.5:
        checks_passed.append(f"高目标场景 SHAP 启用率: {shap_usage_rate:.0%}")
    else:
        checks_failed.append(f"高目标场景 SHAP 启用率过低: {shap_usage_rate:.0%}")

# 检查3：极端目标有警告
extreme_row = summary_df[summary_df['scenario'].str.contains('极端')].iloc[0]
if extreme_row['warnings_count'] > 0:
    checks_passed.append(f"极端目标触发警告 ({extreme_row['warnings_count']}条)")
else:
    checks_failed.append("极端目标未触发警告")

# 检查4：预测误差合理
if summary_df['error_pct'].max() < 20:
    checks_passed.append(f"最大预测误差 < 20% ({summary_df['error_pct'].max():.1f}%)")
else:
    checks_failed.append(f"最大预测误差过大: {summary_df['error_pct'].max():.1f}%")

for check in checks_passed:
    print(f"  ✓ {check}")

if checks_failed:
    print(f"\n⚠️ 需要关注:")
    for check in checks_failed:
        print(f"  • {check}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

# 保存结果
output_file = "test and log file put here/test_results.csv"
summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {output_file}")
