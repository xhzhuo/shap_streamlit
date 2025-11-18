import streamlit as st
import numpy as np
from utils import train_model, metrics_for_model, safe_format
from models import evaluate_model_quality


def page_train_and_eval(state):
    """模型训练与评估页面"""
    st.header("⚙️ 模型训练与评估")
    if state.get('df') is None:
        st.info("请先上传数据。")
        return
    df = state['df_sanitized'] if state.get('df_sanitized') is not None else state['df']
    numeric_cols = state.get('numeric_cols', df.select_dtypes(include=[np.number]).columns.tolist())
    if not numeric_cols:
        st.warning("当前数据中没有检测到数值型列，无法训练模型。")
        return

    # 检测数据是否已更新（通过检查文件名、数据形状）
    current_data_key = (state.get('filename', ''), df.shape[0], df.shape[1])
    last_data_key = state.get('_last_data_key')
    
    if last_data_key is not None and last_data_key != current_data_key:
        # 数据已更新（新上传），重置相关状态
        state['target_var'] = numeric_cols[0]
        state['selected_features'] = []
        # 清除旧的模型训练结果
        state.pop('model', None)
        state.pop('metrics', None)
        state.pop('model_features', None)
        state.pop('model_target', None)
        st.info("检测到新数据已上传，已重置特征选择和模型状态。")
    
    # 更新数据标识
    state['_last_data_key'] = current_data_key

    # 目标选择
    if 'target_var' not in state:
        state['target_var'] = numeric_cols[0]
    
    # 确保目标变量仍在可用列中
    if state['target_var'] not in numeric_cols:
        state['target_var'] = numeric_cols[0]
    
    target_idx = numeric_cols.index(state['target_var'])
    target = st.selectbox("🎯 目标变量 (Target)", options=numeric_cols, index=target_idx, key="target_select")
    
    # 当目标变量改变时，重置特征选择
    if target != state['target_var']:
        state['target_var'] = target
        state['selected_features'] = []

    # 特征选择
    available_features = [c for c in numeric_cols if c != target]
    
    # 初始化特征选择（仅在第一次或目标变量改变时）
    if 'selected_features' not in state or not state['selected_features']:
        state['selected_features'] = available_features[:6] if len(available_features) > 0 else []
    
    # 确保保存的特征都在可用列表中
    valid_features = [f for f in state['selected_features'] if f in available_features]
    if not valid_features:
        valid_features = available_features[:6] if len(available_features) > 0 else []
    
    # 使用 key 确保状态管理的独立性
    selected_features = st.multiselect(
        "选择特征变量（可以搜索并多选）",
        options=available_features,
        default=valid_features,
        key=f"features_select_{target}"  # 使用target作为key的一部分，目标变量改变时会重置
    )
    
    # 直接更新状态（Streamlit会正确处理）
    state['selected_features'] = selected_features
    
    st.markdown(f"已选特征： **{len(selected_features)}** 个")

    st.markdown("---")
    st.markdown("### 模型训练")
    run = st.button("🚀 训练模型")

    if run:
        if len(selected_features) == 0:
            st.warning("请至少选择一个特征。")
            return
        with st.spinner("模型训练中..."):
            try:
                model, X_train, X_test, y_train, y_test = train_model(
                    df, selected_features, target, n_estimators=200, random_state=42
                )
                metrics = metrics_for_model(model, X_train, X_test, y_train, y_test)
                state.update({
                    "model": model, "metrics": metrics,
                    "model_features": selected_features, "model_target": target,
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test,
                    "df": df
                })
                st.success("✅ 模型训练完成")
            except Exception as e:
                st.error(f"训练失败: {e}")

    if state.get("metrics"):
        m = state["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("训练 R²", safe_format(m.get('train_r2'), ".3f"))
        c2.metric("测试 R²", safe_format(m.get('test_r2'), ".3f"))
        c3.metric("训练 NRMSE", safe_format(m.get('train_nrmse'), ".3f"))
        c4.metric("测试 NRMSE", safe_format(m.get('test_nrmse'), ".3f"))

        st.caption(f"交叉验证 R²: {safe_format(m.get('cv_mean'), '.3f')} ± {safe_format(m.get('cv_std'), '.3f')}")

        # --- 模型评分卡 ---
        quality = evaluate_model_quality(
            m['train_r2'], m['test_r2'], m['cv_mean'],
            m['train_rmse'], m['test_rmse']
        )

        summary_parts = quality['summary'].split('\n')
        summary_title = summary_parts[0] if len(summary_parts) > 0 else quality['summary']
        summary_advice = summary_parts[1] if len(summary_parts) > 1 else ""

        with st.container():
            st.markdown(f"""
                <div style="
                    background:#ffffff;
                    color:#333333;
                    border:1px solid #e0e0e0;
                    border-left:6px solid {quality['color']};
                    border-radius:8px;
                    padding:18px 22px;
                    margin-top:20px;
                    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);
                ">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                    <div style="font-size:18px;font-weight:600;color:{quality['color']};">
                        {summary_title}
                    </div>
                    <div style="font-size:14px;margin-top:4px;color:#555;">
                        {summary_advice}
                    </div>
                    </div>
                    <div style="text-align:right;">
                    <div style="font-size:28px;font-weight:700;color:{quality['color']};line-height:1;">
                        {quality['score']}
                    </div>
                    <div style="font-size:13px;color:#777;">模型评分</div>
                    </div>
                </div>
                <hr style="border:none;border-top:1px solid #eee;margin:12px 0;">
                <div style="font-size:13.5px;color:#555;line-height:1.6;">
                    <b>📘 指标说明：</b><br>
                    • R²（决定系数）：越接近 1，拟合效果越好。<br>
                    • NRMSE（归一化均方根误差）：越小越好，表示预测值与真实值越接近。<br>
                    • 交叉验证 R²：越接近测试集 R²，说明模型稳定且能适应新数据。
                </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)

            # === 图表部分 ===
            try:
                model = state.get("model")
                X_test = state.get("X_test")
                y_test = state.get("y_test")

                if model is not None and X_test is not None and y_test is not None:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    y_test_pred = model.predict(X_test)
                    errors = y_test_pred - y_test

                    fig = make_subplots(
                        rows=1, cols=2,
                        specs=[[{"type": "scatter"}, {"type": "histogram"}]],
                        column_widths=[0.62, 0.38],
                        subplot_titles=('预测值 vs 真实值', '预测误差分布'),
                        horizontal_spacing=0.09
                    )

                    # 左侧散点图
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_test_pred,
                        mode='markers',
                        marker=dict(
                            color='rgba(64, 123, 255, 0.68)',
                            size=9,
                            line=dict(width=1, color='rgba(64, 123, 255, 0.9)')
                        ),
                        hovertemplate="真实值: %{x:.2f}<br>预测值: %{y:.2f}",
                        name="预测点"
                    ), row=1, col=1)

                    # 理想预测线
                    min_val, max_val = np.min([y_test, y_test_pred]), np.max([y_test, y_test_pred])
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='rgba(255, 99, 71, 0.9)', width=2, dash='dot'),
                        name='理想线',
                        hoverinfo='skip'
                    ), row=1, col=1)

                    # 误差分布直方图
                    fig.add_trace(go.Histogram(
                        x=errors,
                        nbinsx=30,
                        marker=dict(color='rgba(64, 123, 255, 0.6)', line=dict(color='white', width=1)),
                        opacity=0.85,
                        name='误差分布'
                    ), row=1, col=2)

                    fig.add_vline(
                        x=np.mean(errors),
                        line=dict(color='rgba(255, 99, 71, 0.9)', width=2, dash='dot'),
                        annotation_text="平均误差",
                        annotation_position="top right",
                        row=1, col=2
                    )

                    fig.update_xaxes(title_text="真实值", showgrid=True, gridcolor="rgba(230,230,230,0.6)", row=1, col=1)
                    fig.update_yaxes(title_text="预测值", showgrid=True, gridcolor="rgba(230,230,230,0.6)", row=1, col=1)
                    fig.update_xaxes(title_text="预测误差", showgrid=True, gridcolor="rgba(230,230,230,0.6)", row=1, col=2)
                    fig.update_yaxes(title_text="频次", showgrid=True, gridcolor="rgba(230,230,230,0.6)", row=1, col=2)

                    fig.update_layout(
                        title=dict(
                            text="模型预测效果分析",
                            font=dict(size=20, color="#222", family="Segoe UI"),
                            x=0.45,
                        ),
                        height=460,
                        margin=dict(l=40, r=40, t=80, b=40),
                        paper_bgcolor="#f8f9fa",
                        plot_bgcolor="#ffffff",
                        font=dict(family="Segoe UI", size=13, color="#333"),
                        hoverlabel=dict(bgcolor="white", font_size=13),
                        showlegend=False,
                        bargap=0.15,
                    )

                    st.markdown("""
                    <div style="background:white;border-radius:12px;
                                box-shadow:0 2px 10px rgba(0,0,0,0.05);
                                padding:20px 25px;margin-top:10px;">
                    """, unsafe_allow_html=True)

                    st.plotly_chart(fig, width='stretch')

                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"无法显示预测对比图表: {e}")
                
            #test