"""
可视化分析页面
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils import shap_values_from_model


def page_visual_analysis(state):
    """可视化分析页面"""
    st.header("📈 可视化分析")
    if state.get('model') is None:
        st.info("请先训练模型。")
        return
    model = state['model']
    X_test = state['X_test']
    features = state['model_features']

    if 'shap_values' not in state:
        with st.spinner("计算 SHAP 值..."):
            try:
                shap_vals, explainer = shap_values_from_model(model, state['X_train'], X_test)
                state['shap_values'] = shap_vals
                state['shap_explainer'] = explainer
            except Exception as e:
                st.error(f"SHAP 计算失败: {e}")
                return

    shap_vals = np.array(state['shap_values'])
    tab1, tab2, tab3 = st.tabs(["SHAP Summary", "特征贡献度", "相关性热力图"])

    # --- SHAP Summary ---
    with tab1:
        try:
            raw_shap = state.get('shap_values', None)
            if raw_shap is None:
                st.error("未找到 SHAP 值，请先训练模型并计算 SHAP。")
            else:
                n_samples = X_test.shape[0]
                n_features = len(features)

                def normalize_shap(raw, n_samples, n_features):
                    arr = np.array(raw)
                    if arr.ndim == 3:
                        return np.mean(arr, axis=0)
                    if arr.ndim == 2:
                        if arr.shape == (n_samples, n_features):
                            return arr
                        if arr.shape == (n_features, n_samples):
                            return arr.T
                        return arr.reshape((n_samples, n_features))
                    raise ValueError(f"不支持的 SHAP 形状: {arr.shape}")

                shap_arr = normalize_shap(raw_shap, n_samples, n_features)
                abs_mean = np.abs(shap_arr).mean(axis=0)
                sorted_idx = np.argsort(abs_mean)[::-1]
                shap_sorted = shap_arr[:, sorted_idx]
                X_sorted = pd.DataFrame(X_test, columns=features).iloc[:, sorted_idx]
                sorted_features = [features[i] for i in sorted_idx]

                long_data = []
                for i, feat in enumerate(sorted_features):
                    for j in range(len(X_sorted)):
                        long_data.append({
                            "Feature": feat,
                            "Feature value": X_sorted.iloc[j, i],
                            "SHAP value": shap_sorted[j, i]
                        })
                df_long = pd.DataFrame(long_data)

                fig = px.scatter(
                    df_long,
                    x="SHAP value",
                    y="Feature",
                    color="Feature value",
                    category_orders={"Feature": sorted_features[::-1]},
                    color_continuous_scale="RdBu_r",
                    opacity=0.6,
                    render_mode="webgl",
                    height=650,
                    title="SHAP Summary"
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(
                    yaxis=dict(title="特征", showline=True, showgrid=True, zeroline=False, autorange='reversed'),
                    xaxis=dict(title="SHAP value", showline=True, showgrid=True, zeroline=True),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E6F0F8", size=12),
                    coloraxis_colorbar=dict(title="特征值")
                )
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"绘制 SHAP Summary 失败: {e}")

    # --- 饼图 ---
    with tab2:
        try:
            importance = np.abs(state['shap_values'])
            if isinstance(importance, list):
                importance = np.mean([np.abs(v) for v in importance], axis=0)
            importance = np.mean(np.abs(importance), axis=0)

            features_list = features
            len_imp = len(importance)
            len_feat = len(features_list)
            if len_imp != len_feat:
                min_len = min(len_imp, len_feat)
                importance = importance[:min_len]
                features_list = features_list[:min_len]

            pie_df = pd.DataFrame({
                "feature": features_list,
                "importance": importance
            }).sort_values("importance", ascending=False)

            pie_fig = px.pie(
                pie_df,
                names="feature",
                values="importance",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="特征贡献占比"
            )
            pie_fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F0F8")
            )
            st.plotly_chart(pie_fig, width="stretch")
        except Exception as e:
            st.error(f"饼图绘制失败: {e}")

    # --- 相关性热力图 ---
    with tab3:
        try:
            all_cols = features + [state['model_target']]
            corr = state['df'][all_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            corr_masked = corr.where(~mask)
            hm = px.imshow(corr_masked, text_auto=".2f", aspect="auto", title="相关性矩阵")
            hm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F0F8")
            )
            st.plotly_chart(hm, width="stretch")
        except Exception as e:
            st.error(f"热力图绘制失败: {e}")