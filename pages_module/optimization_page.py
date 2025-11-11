"""
反推优化页面
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import safe_format, format_dataframe_numeric
from optimization import _improved_sensitivity_estimation, _linear_allocation, _budget_constrained_optimization, _fully_constrained_optimization


def page_reverse_opt(state):
    """反推与预算优化页面"""
    st.header("🔁 反推与预算优化")
    if state.get('model') is None:
        st.info("请先训练模型。")
        return
    model = state['model']
    features = state['model_features']
    base_x = state['X_train'].mean().values
    X_train_min = state['X_train'].min().values
    X_train_max = state['X_train'].max().values
    try:
        y_base = float(model.predict(base_x.reshape(1, -1))[0])
    except Exception:
        y_base = 0.0
    st.markdown(f"基准预测 {state.get('model_target','目标')}: **{safe_format(y_base, '.2f')}**")
    col1, col2 = st.columns(2)
    with col1:
        target_gmv = st.number_input(f"目标 {state.get('model_target','目标')}", value=float(y_base * 1.1), step=1.0)
    with col2:
        method = st.selectbox("优化方法", ["线性分配", "预算约束优化", "全约束优化"])
    use_budget = st.checkbox("启用总预算约束", value=False)
    total_budget = st.number_input("总预算 (总投放上限)", value=float(base_x.sum()), step=1.0) if use_budget else None
    use_channel_constraints = st.checkbox("启用渠道上下限约束（每渠道）", value=False)
    min_constraints = None
    max_constraints = None
    if use_channel_constraints:
        st.markdown("设置每个渠道的上下限（填写后生效，默认按训练范围推断）")
        min_constraints = []
        max_constraints = []
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[0]:
                min_val = st.number_input(f"{feat} 最小值", value=float(max(0.0, base_x[i] * 0.5)), key=f"min_{i}")
            with cols[1]:
                max_val = st.number_input(f"{feat} 最大值", value=float(max(base_x[i] * 2.0, X_train_max[i] if not np.isnan(X_train_max[i]) else base_x[i] * 2.0)), key=f"max_{i}")
            min_constraints.append(float(min_val))
            max_constraints.append(float(max_val))

    if st.button("🔢 计算建议方案"):
        with st.spinner("优化中..."):
            try:
                shap_abs_mean = np.abs(state['shap_values']).mean(axis=0)
                if shap_abs_mean.sum() == 0:
                    st.error("所有 SHAP 值为 0，无法按贡献分配")
                    return
                weights = shap_abs_mean / shap_abs_mean.sum()
                sensitivities = _improved_sensitivity_estimation(model, base_x, features, X_train_min, X_train_max, n_samples=40)
                if method == "线性分配":
                    suggested, y_pred_new, constraint_status = _linear_allocation(model, base_x, y_base, target_gmv, weights, sensitivities, min_constraints, max_constraints, X_train_max)
                elif method == "预算约束优化":
                    suggested, y_pred_new, constraint_status = _budget_constrained_optimization(model, base_x, y_base, target_gmv, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)
                else:
                    suggested, y_pred_new, constraint_status = _fully_constrained_optimization(model, base_x, y_base, target_gmv, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)
                results = []
                for i, feat in enumerate(features):
                    base_val = float(base_x[i])
                    sug = float(suggested[i])
                    results.append({
                        "渠道": feat,
                        "基准投放": base_val,
                        "建议投放": sug,
                        "增量": sug - base_val,
                        "SHAP权重": float(weights[i]),
                        "敏感度": float(sensitivities[i]),
                        "约束状态": constraint_status[i] if constraint_status else "正常"
                    })
                results_df = pd.DataFrame(results)
                fmt = format_dataframe_numeric(results_df)
                st.markdown("### 建议方案")
                st.dataframe(results_df.style.format(fmt), width="stretch")
                if total_budget is not None:
                    st.info(f"预算对比：基准总投放 {base_x.sum():.2f} → 建议总投放 {suggested.sum():.2f} (设定预算 {total_budget:.2f})")
                fig = go.Figure()
                fig.add_trace(go.Bar(name='基准', x=features, y=[r["基准投放"] for r in results]))
                fig.add_trace(go.Bar(name='建议', x=features, y=[r["建议投放"] for r in results]))
                fig.update_layout(barmode='group', title='基准 vs 建议投放量', plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6F0F8"))
                st.plotly_chart(fig, width="stretch")
                if y_pred_new is not None:
                    st.success(f"建议方案预测 GMV: {safe_format(y_pred_new, '.2f')}（目标：{safe_format(target_gmv, '.2f')}）")
                else:
                    st.warning("建议方案的模型预测值不可用。")
            except Exception as e:
                st.error(f"反推计算失败: {e}")