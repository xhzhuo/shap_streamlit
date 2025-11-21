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
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP Summary", "特征贡献度", "相关性热力图", "特征贡献度（含未观测因素）"])

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
                    expected_size = n_samples * n_features
                    
                    # 三维数组：取平均
                    if arr.ndim == 3:
                        return np.mean(arr, axis=0)
                    
                    # 二维数组：检查形状是否匹配
                    if arr.ndim == 2:
                        # 正确的形状
                        if arr.shape == (n_samples, n_features):
                            return arr
                        # 转置后是正确形状
                        if arr.shape == (n_features, n_samples):
                            return arr.T
                        
                        # 形状不匹配，尝试调整
                        rows, cols = arr.shape
                        
                        # 如果行数匹配样本数
                        if rows == n_samples:
                            if cols >= n_features:
                                return arr[:, :n_features]
                            else:
                                # 特征数不足，填充
                                padded = np.zeros((n_samples, n_features))
                                padded[:, :cols] = arr
                                return padded
                        
                        # 如果列数匹配特征数
                        if cols == n_features:
                            if rows >= n_samples:
                                return arr[:n_samples, :]
                            else:
                                # 样本数不足，填充
                                padded = np.zeros((n_samples, n_features))
                                padded[:rows, :] = arr
                                return padded
                        
                        # 尝试基于总大小reshape
                        if arr.size == expected_size:
                            return arr.reshape((n_samples, n_features))
                        
                        # 特殊处理：多类分类的SHAP值(60, 6, num_classes)
                        # 返回第一个类或平均
                        if rows == n_samples and cols == n_features:
                            return arr
                        
                        # 最后的尝试：取前n_samples行和n_features列
                        return arr[:min(rows, n_samples), :min(cols, n_features)]
                    
                    # 一维数组
                    if arr.ndim == 1:
                        if arr.size == expected_size:
                            return arr.reshape((n_samples, n_features))
                        elif arr.size == n_features:
                            # 只有特征维度，复制给所有样本
                            return np.tile(arr, (n_samples, 1))
                        elif arr.size == n_samples:
                            # 只有样本维度，这不合理，返回均匀值
                            return np.ones((n_samples, n_features)) * np.mean(arr)
                    
                    raise ValueError(f"无法处理的 SHAP 形状: {arr.shape}，样本数: {n_samples}, 特征数: {n_features}")

                shap_arr = normalize_shap(raw_shap, n_samples, n_features)
                abs_mean = np.abs(shap_arr).mean(axis=0)
                sorted_idx = np.argsort(abs_mean)[::-1]
                shap_sorted = shap_arr[:, sorted_idx]
                X_sorted = pd.DataFrame(X_test, columns=features).iloc[:, sorted_idx]
                sorted_features = [features[i] for i in sorted_idx]

                long_data = []
                # 确保shap_sorted和X_sorted的形状一致
                min_samples = min(shap_sorted.shape[0], X_sorted.shape[0])
                for i, feat in enumerate(sorted_features):
                    for j in range(min_samples):
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
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"绘制 SHAP Summary 失败: {e}")
            import traceback
            st.error(f"详细错误:\n{traceback.format_exc()}")

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
            st.plotly_chart(pie_fig, use_container_width=True)
        except Exception as e:
            st.error(f"饼图绘制失败: {e}")

    # --- 相关性热力图 ---
    with tab3:
        try:
            all_cols =  [state['model_target']] + features
            corr = state['df'][all_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            corr_masked = corr.where(~mask)
            hm = px.imshow(corr_masked, text_auto=".2f", aspect="auto", title="相关性矩阵")
            hm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F0F8"),
                xaxis_side="top"  # 将x轴标签移到顶部
            )
            st.plotly_chart(hm, use_container_width=True)
        except Exception as e:
            st.error(f"热力图绘制失败: {e}")

    # --- 贡献度分解 ---
    with tab4:
        try:
            # 获取测试集R²值
            test_r2 = state.get('metrics', {}).get('test_r2', 0.0)
            if test_r2 is None or np.isnan(test_r2):
                test_r2 = 0.0
            
            # 计算SHAP贡献度
            importance = np.abs(state['shap_values'])
            if isinstance(importance, list):
                importance = np.mean([np.abs(v) for v in importance], axis=0)
            importance = np.mean(np.abs(importance), axis=0)
            
            # 归一化SHAP贡献度到[0, 1]
            importance_normalized = importance / np.sum(importance) if np.sum(importance) > 0 else importance
            
            # 乘以R²值（表示能解释的方差）
            explained_importance = importance_normalized * test_r2
            
            features_list = features
            len_imp = len(explained_importance)
            len_feat = len(features_list)
            if len_imp != len_feat:
                min_len = min(len_imp, len_feat)
                explained_importance = explained_importance[:min_len]
                features_list = features_list[:min_len]
            
            # 添加未解释方差
            unexplained = 1.0 - test_r2
            feature_names = list(features_list) + ["未观测因素"]
            importance_values = list(explained_importance) + [unexplained]
            
            pie_df_adjusted = pd.DataFrame({
                "feature": feature_names,
                "importance": importance_values
            }).sort_values("importance", ascending=False)
            
            pie_fig_adjusted = px.pie(
                pie_df_adjusted,
                names="feature",
                values="importance",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title=f"特征贡献占比"
            )
            pie_fig_adjusted.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F0F8")
            )
            st.plotly_chart(pie_fig_adjusted, use_container_width=True)
        except Exception as e:
            st.error(f"贡献度分解饼图绘制失败: {e}")
            import traceback
            st.error(f"详细错误:\n{traceback.format_exc()}")