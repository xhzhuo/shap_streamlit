"""
反推优化页面（智能单入口版）
根据约束勾选自动选择最优优化方案
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import safe_format, format_dataframe_numeric
from optimization import optimize_ad_allocation_robust


def page_reverse_opt(state):
    """反推与预算优化页面（智能单入口版）"""
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
        # 使用DataFrame确保特征名称一致性
        base_x_df = pd.DataFrame([base_x], columns=features)
        y_base = float(model.predict(base_x_df)[0])
    except Exception:
        y_base = 0.0

    st.markdown(f"基准预测 {state.get('model_target','目标')}: **{safe_format(y_base, '.2f')}**")

    # ==============
    # 参数输入区域
    # ==============
    target_gmv = st.number_input(
        f"目标 {state.get('model_target','目标')}",
        value=float(y_base * 1.1),
        step=1.0
    )


    # ✅ 用户勾选约束条件
    use_budget = st.checkbox("启用总预算约束", value=False)
    total_budget = st.number_input(
        "总预算 (总投放上限)",
        value=float(base_x.sum()), step=1.0
    ) if use_budget else None

    use_channel_constraints = st.checkbox("启用渠道上下限约束（每渠道）", value=False)
    min_constraints, max_constraints = None, None
    if use_channel_constraints:
        st.markdown("设置每个渠道的上下限（默认按训练范围推断）")
        min_constraints, max_constraints = [], []
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[0]:
                min_val = st.number_input(f"{feat} 最小值", value=float(max(0.0, base_x[i] * 0.5)), key=f"min_{i}")
            with cols[1]:
                max_val = st.number_input(
                    f"{feat} 最大值",
                    value=float(max(base_x[i] * 2.0, X_train_max[i] if not np.isnan(X_train_max[i]) else base_x[i] * 2.0)),
                    key=f"max_{i}"
                )
            # 确保 min_val <= max_val
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            min_constraints.append(float(max(0.0, min_val)))
            max_constraints.append(float(max(min_constraints[-1] + 0.01, max_val)))

    # ==============
    # 优化计算按钮
    # ==============
    if st.button("🔢 计算建议方案"):
        with st.spinner("优化中..."):
            try:
                # 计算SHAP权重
                if 'shap_values' in state and state['shap_values'] is not None:
                    shap_abs_mean = np.abs(state['shap_values']).mean(axis=0)
                    if shap_abs_mean.sum() == 0:
                        st.error("所有 SHAP 值为 0，无法按贡献分配")
                        return

                    weights = shap_abs_mean / shap_abs_mean.sum()
                else:
                    # 如果没有SHAP值，则使用均匀权重
                    weights = np.ones(len(features)) / len(features)
                
                # 验证约束条件
                if max_constraints is not None:
                    for i in range(len(features)):
                        if min_constraints[i] > max_constraints[i]:
                            st.error(f"特征 {features[i]} 的最小值({min_constraints[i]:.2f})大于最大值({max_constraints[i]:.2f})")
                            return
                
                try:
                    # 使用新的统一优化接口（返回字典）
                    optimization_result = optimize_ad_allocation_robust(
                        model=model,
                        base_x=base_x,
                        y_target=target_gmv,
                        total_budget=total_budget,
                        weights=weights,
                        min_constraints=min_constraints,
                        max_constraints=max_constraints,
                        X_train=state['X_train'].values,  # 确保传递numpy数组
                        method='adaptive',  # 使用自适应优化策略
                        robustness_level='medium',  # 中等级别的鲁棒性
                        feature_names=features  # 传递特征名列表
                    )
                except ValueError as ve:
                    st.error(f"优化边界配置错误: {ve}")
                    st.info("💡 建议：请检查渠道上下限约束，确保最小值 < 最大值")
                    import traceback
                    st.error(f"详细错误:\n{traceback.format_exc()}")
                    return
                except Exception as opt_e:
                    st.error(f"优化计算错误: {opt_e}")
                    import traceback
                    st.error(f"详细错误:\n{traceback.format_exc()}")
                    return
                
                # 从结果字典中提取数据
                suggested = optimization_result['suggested_allocation']
                y_pred_new = optimization_result['predicted_value']
                constraint_status = optimization_result['constraint_status']
                method_used = optimization_result['method_used']
                robustness_level = optimization_result['robustness_level']
                
                # 计算目标达成率
                target_achieved_ratio = y_pred_new / target_gmv if target_gmv != 0 else 0
                
                # 获取敏感度信息（从优化结果获取，避免重复计算）
                from optimization import _robust_sensitivity_estimation
                feature_names = getattr(model, 'feature_names_', [f'x{i}' for i in range(len(features))])
                # 直接使用模型已计算的敏感度（缓存获取，速度快）
                sensitivities = _robust_sensitivity_estimation(model, base_x, feature_names, X_train_min, X_train_max)

                # =============================
                # 输出结果
                # =============================
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

                st.session_state["reverse_results"] = results
                st.session_state["reverse_y_base"] = y_base
                st.session_state["reverse_target_gmv"] = target_gmv

                results_df = pd.DataFrame(results)
                fmt = format_dataframe_numeric(results_df)

                st.markdown(f"### 建议方案")
                if y_pred_new is not None:
                    st.success(f"预测 {state.get('model_target','目标')}: {safe_format(y_pred_new, '.2f')}（目标：{safe_format(target_gmv, '.2f')}，达成率：{safe_format(target_achieved_ratio * 100, '.1f')}%）")
                    st.caption(f"使用的优化方法: {method_used} (鲁棒性级别: {robustness_level})")
                else:
                    st.warning("建议方案预测值不可用。")

                st.dataframe(results_df.style.format(fmt), use_container_width=True)

                if total_budget is not None:
                    st.info(f"预算对比：基准 {base_x.sum():.2f} → 建议 {np.sum(suggested):.2f} (预算 {total_budget:.2f})")

                # ==============
                # 响应曲线
                # ==============
                if "reverse_results" in st.session_state:
                    results = st.session_state["reverse_results"]
                    y_base = st.session_state["reverse_y_base"]
                    target_gmv = st.session_state["reverse_target_gmv"]

                    st.subheader("📊 模型响应分析")
                    scales = np.linspace(0.5, 2.5, 30)
                    y_preds = []

                    for scale in scales:
                        x_scaled = base_x.copy()
                        x_scaled *= scale
                        # 使用DataFrame确保特征名称一致性
                        x_scaled_df = pd.DataFrame([x_scaled], columns=features)
                        y_pred = float(model.predict(x_scaled_df)[0])
                        y_preds.append(y_pred)

                    fig = make_subplots(
                        rows=1, cols=2,
                        column_widths=[0.45, 0.55],
                        subplot_titles=("整体投放响应曲线", "基准 vs 建议投放量")
                    )

                    fig.add_trace(
                        go.Scatter(x=scales, y=y_preds, mode='lines+markers',
                                   name='预测', line=dict(color="#2E86DE", width=3)),
                        row=1, col=1
                    )
                    fig.add_hline(y=y_base, line=dict(color="gray", dash="dot"),
                                  annotation_text="基准", row=1, col=1)
                    fig.add_hline(y=target_gmv, line=dict(color="red", dash="dash"),
                                  annotation_text="目标", row=1, col=1)
                    fig.update_xaxes(title_text="投放倍数（相对基准）", row=1, col=1)
                    fig.update_yaxes(title_text="预测", row=1, col=1)

                    fig.add_trace(
                        go.Bar(name='基准', x=features, y=[r["基准投放"] for r in results], marker_color='#82CAFA'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Bar(name='建议', x=features, y=[r["建议投放"] for r in results], marker_color='#FFB347'),
                        row=1, col=2
                    )
                    fig.update_layout(
                        barmode='group',
                        title='模型响应与反推投放对比',
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E6F0F8"),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"反推计算失败: {e}")

    # ==============
    # 选型建议
    # ==============
    # st.markdown("---")
    # st.subheader("📋 优化逻辑说明")

    # st.markdown( 
    #     """
    #     系统采用统一优化算法，自动处理各种约束条件：
    #     - 无约束 → 使用基础优化策略
    #     - 预算约束 → 在优化过程中考虑总预算限制
    #     - 渠道上下限约束 → 确保各渠道的投放量在指定范围内
        
    #     系统默认使用中等级别的鲁棒性优化策略，兼顾计算效率和结果稳定性。
    #     """
    # )

    # st.info("💡 推荐先使用无约束版本评估潜力，再加预算与上下限微调执行方案。")