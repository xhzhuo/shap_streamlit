"""
优化逻辑合理性深度验证脚本
目标：验证新方案是否能根据 SHAP 和 敏感度（梯度）给出合理的投放建议
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import shap
import sys
import io
import warnings

# Redirect stdout to a file
sys.stdout = open('logic_test_result_utf8.txt', 'w', encoding='utf-8')

# 忽略不必要的警告
warnings.filterwarnings('ignore')

sys.path.append('.')
from optimization import optimize_allocation_v2, _estimate_jacobian_fast

def verify_logic():
    print("=" * 80)
    print("优化逻辑合理性深度验证")
    print("=" * 80)

    # 1. 准备数据与模型
    print("\n[步骤1] 准备数据与模型...")
    data_path = r"test and log file put here\Advertising_Data.csv"
    df = pd.read_csv(data_path)
    
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 2. 选择代表性样本（中位数附近）
    print("\n[步骤2] 选择代表性样本...")
    median_val = np.median(y)
    preds = model.predict(X_test)
    # 找到预测值最接近中位数的样本
    idx = np.argmin(np.abs(preds - median_val))
    base_x = X_test[idx].copy()
    base_pred = preds[idx]
    
    print(f"  选中样本索引: {idx}")
    print(f"  当前预测值: {base_pred:.2f} (全量中位数: {median_val:.2f})")
    print(f"  当前投放组合: {dict(zip(feature_cols, base_x.round(1)))}")

    # 3. 计算关键指标（SHAP 和 敏感度）
    print("\n[步骤3] 计算决策依据（SHAP & 敏感度）...")
    
    # A. 计算敏感度 (梯度)
    jacobian, _ = _estimate_jacobian_fast(model, base_x)
    
    # B. 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(base_x.reshape(1, -1))[0]
    
    # 展示各渠道指标
    metrics_df = pd.DataFrame({
        'Feature': feature_cols,
        'Base_Budget': base_x,
        'Sensitivity (Gradient)': jacobian,
        'SHAP_Value': shap_values
    })
    
    # 按敏感度排序展示
    metrics_df = metrics_df.sort_values('Sensitivity (Gradient)', ascending=False)
    print("\n各渠道指标分析 (按敏感度排序):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    print("\n  -> 理论预期: 敏感度(梯度)高且SHAP贡献正向的渠道，应该获得更多预算增加。")

    # 4. 执行优化
    print("\n[步骤4] 执行优化 (目标提升 30%)...")
    target_val = base_pred * 1.3
    
    # 使用 SHAP 值作为权重输入（模拟真实场景）
    # 注意：optimize_allocation_v2 内部也会计算 SHAP，但这里传入 weights 主要是为了初始加权
    # 我们传入归一化的特征重要性作为基础权重
    weights = model.feature_importances_
    
    result = optimize_allocation_v2(
        model=model,
        base_x=base_x,
        y_target=target_val,
        weights=weights,
        X_train=X_train,
        max_iterations=5  # 给足迭代次数
    )
    
    # 5. 结果深度分析
    print("\n[步骤5] 建议方案合理性分析...")
    
    suggested_x = result['suggested_allocation']
    budget_change = suggested_x - base_x
    budget_change_pct = (budget_change / (base_x + 1e-9)) * 100
    
    analysis_df = metrics_df.copy()
    # 映射回原始顺序以匹配结果
    analysis_df = analysis_df.set_index('Feature').reindex(feature_cols).reset_index()
    
    analysis_df['Suggested'] = suggested_x
    analysis_df['Change_Abs'] = budget_change
    analysis_df['Change_Pct'] = budget_change_pct
    
    # 再次按增量排序
    analysis_df = analysis_df.sort_values('Change_Abs', ascending=False)
    
    print("\n优化建议详情 (按预算增量排序):")
    print(analysis_df[['Feature', 'Base_Budget', 'Suggested', 'Change_Abs', 'Sensitivity (Gradient)', 'SHAP_Value']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # 6. 验证相关性
    print("\n[步骤6] 逻辑验证结论")
    
    # 计算相关系数
    corr_grad = np.corrcoef(analysis_df['Change_Abs'], analysis_df['Sensitivity (Gradient)'])[0, 1]
    corr_shap = np.corrcoef(analysis_df['Change_Abs'], analysis_df['SHAP_Value'])[0, 1]
    
    print(f"  预算增量与敏感度(梯度)的相关性: {corr_grad:.4f}")
    print(f"  预算增量与SHAP值的相关性:       {corr_shap:.4f}")
    
    if corr_grad > 0.5:
        print("  ✅ 验证通过: 算法倾向于增加高敏感度渠道的预算（符合预期）。")
    else:
        print("  ⚠️ 验证存疑: 预算分配与敏感度相关性不强，需检查是否受约束或非线性影响。")
        
    if result.get('shap_used'):
        print(f"  ✅ SHAP 校准已启用 (非线性得分: {result['nonlinearity_score']:.3f})")
    else:
        print(f"  ℹ️ SHAP 校准未启用 (非线性得分: {result['nonlinearity_score']:.3f})，当前主要依赖梯度。")

    print(f"\n  最终预测值: {result['predicted_value']:.2f} (目标: {target_val:.2f})")
    print(f"  达成率: {result['predicted_value']/target_val:.1%}")

if __name__ == "__main__":
    verify_logic()
