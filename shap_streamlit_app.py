# adv_shap_dashboard_datatech_v5.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------
# 页面配置 & 视觉样式
# ---------------------------
st.set_page_config(
    page_title="Ad Effect Intelligence — DataTech UI v5",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    :root{
      --bg:#071126;
      --muted:#9fb0c6;
      --accent:#00e0a1;
    }
    html, body, [class*="css"]  {
      background: linear-gradient(180deg, #041122 0%, #071126 100%);
      color: #E6F0F8;
    }
    .stApp > .main > div { padding: 1rem 1.2rem; }
    .brand { font-size:1.4rem; font-weight:700; color: var(--accent); }
    .subtitle { color: var(--muted); font-size:0.95rem }
    .small { font-size:0.85rem; color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 辅助函数：数据、训练、metrics、SHAP
# ---------------------------
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

def sanitize_numeric_df(df):
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            col = df2[c].astype(str).str.replace(',', '').str.replace('%','')
            try:
                df2[c] = pd.to_numeric(col)
            except Exception:
                pass
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    return df2, numeric_cols

def train_model(df, features, target, n_estimators=200, random_state=42):
    X = df[features].astype(float)
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def metrics_for_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    cv_scores = cross_val_score(model, X_train, y_train, cv=4, scoring='r2')
    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores))
    }

def shap_values_from_model(model, X_background, X_explain):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_explain)
    return shap_vals, explainer

def safe_format(val, fmt=".3f"):
    try:
        if val is None:
            return "—"
        v = float(val)
        return format(v, fmt)
    except Exception:
        return str(val)

def format_dataframe_numeric(df):
    fmt = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:.3f}"
    return fmt

# ---------------------------
# 反推/优化工具（完整恢复）
# ---------------------------
def _get_bounds(base_x, min_constraints, max_constraints, X_train_max):
    bounds = []
    n = len(base_x)
    for i in range(n):
        lower = 0.0
        upper = float(X_train_max[i]) if X_train_max is not None else float(base_x[i] * 2 if base_x[i] != 0 else 1.0)
        if min_constraints is not None:
            lower = max(lower, float(min_constraints[i]))
        if max_constraints is not None:
            upper = min(upper, float(max_constraints[i]))
        if lower > upper:
            lower, upper = upper, lower
        bounds.append((lower, upper))
    return bounds

def _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max):
    constrained = suggested.copy()
    constraint_status = []
    for i in range(len(suggested)):
        status = "正常"
        if min_constraints is not None and suggested[i] < min_constraints[i]:
            constrained[i] = min_constraints[i]
            status = "触达下限"
        if max_constraints is not None and suggested[i] > max_constraints[i]:
            constrained[i] = max_constraints[i]
            status = "触达上限"
        if X_train_max is not None and constrained[i] > X_train_max[i]:
            constrained[i] = float(X_train_max[i])
            status = "触达上限"
        if constrained[i] < 0:
            constrained[i] = 0.0
            status = "触达下限"
        constraint_status.append(status)
    return constrained, constraint_status

def _improved_sensitivity_estimation(model, base_x, feature_names, X_train_min, X_train_max, n_samples=40):
    if model is None:
        return np.ones(len(feature_names), dtype=float) * 1e-6
    n = len(feature_names)
    base = base_x.astype(float).copy()
    sensitivities = np.zeros(n, dtype=float)
    ranges = np.maximum(np.array(X_train_max) - np.array(X_train_min), 1e-6)
    base_df = pd.DataFrame(base.reshape(1, -1), columns=feature_names)
    for i in range(n):
        sample_sens = []
        for _ in range(n_samples):
            eps_ratio = np.random.uniform(0.001, 0.02)
            eps = ranges[i] * eps_ratio
            if eps == 0:
                eps = 1e-6
            x_perturb = base.copy()
            x_perturb[i] += eps
            try:
                x_perturb_df = pd.DataFrame(x_perturb.reshape(1, -1), columns=feature_names)
                y1 = float(model.predict(x_perturb_df)[0])
                y0 = float(model.predict(base_df)[0])
                sens = (y1 - y0) / eps
                sample_sens.append(sens)
            except Exception:
                sample_sens.append(0.0)
        sensitivities[i] = np.median(sample_sens) if len(sample_sens)>0 else 0.0
    if np.allclose(sensitivities, 0.0):
        sensitivities = np.ones_like(sensitivities) * 1e-6
    return sensitivities

def _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints=None, max_constraints=None, X_train_max=None):
    eps = 1e-8
    sensitivities_adj = np.where(np.abs(sensitivities) < eps, eps, sensitivities)
    delta_y = y_target - y_base
    delta_x = (weights * delta_y) / sensitivities_adj
    suggested = base_x + delta_x
    suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
    try:
        y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
    except Exception:
        y_pred_new = None
    return suggested, y_pred_new, constraint_status

def _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max):
    def objective(x):
        try:
            y_pred = float(model.predict(x.reshape(1, -1))[0])
            return abs(y_pred - y_target)
        except Exception:
            return np.sum((x - base_x) ** 2)
    constraints = []
    if total_budget is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    bounds = _get_bounds(base_x, min_constraints, max_constraints, X_train_max)
    if total_budget is not None and np.sum(weights) > 0:
        x0 = (weights / np.sum(weights)) * total_budget
    else:
        x0 = base_x.copy()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
        if result.success:
            suggested = result.x
            suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
            try:
                y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
            except Exception:
                y_pred_new = None
            return suggested, y_pred_new, constraint_status
        else:
            return _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints, max_constraints, X_train_max)
    except Exception:
        return _linear_allocation(model, base_x, y_base, y_target, weights, sensitivities, min_constraints, max_constraints, X_train_max)

def _fully_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max):
    def objective(x):
        try:
            y_pred = float(model.predict(x.reshape(1, -1))[0])
            gmv_diff = abs(y_pred - y_target)
        except Exception:
            gmv_diff = np.sum((x - base_x)**2)
        change_penalty = np.sum((x - base_x) ** 2) * 0.01
        return gmv_diff + change_penalty
    constraints = []
    if total_budget is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    bounds = _get_bounds(base_x, min_constraints, max_constraints, X_train_max)
    if total_budget is not None and np.sum(weights) > 0:
        x0 = (weights / np.sum(weights)) * total_budget
    else:
        x0 = base_x.copy()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-6})
        if result.success:
            suggested = result.x
            suggested, constraint_status = _apply_constraints(suggested, base_x, min_constraints, max_constraints, X_train_max)
            try:
                y_pred_new = float(model.predict(suggested.reshape(1, -1))[0])
            except Exception:
                y_pred_new = None
            return suggested, y_pred_new, constraint_status
        else:
            return _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)
    except Exception:
        return _budget_constrained_optimization(model, base_x, y_base, y_target, total_budget, weights, sensitivities, min_constraints, max_constraints, X_train_max)

# ---------------------------
# 侧栏 & 导航（保留随机种子）
# ---------------------------
with st.sidebar:
    st.markdown('<div class="brand">Ad Effect Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">DataTech UI · v5</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "导航",
        ("数据上传 & 预览", "模型训练 & 评估", "可视化分析", "反推/预算优化"),
        index=0
    )
    st.markdown("---")
    st.number_input("随机种子 (random_state)", value=42, key="seed")

# ---------------------------
# 页面：数据上传 & 预览
# ---------------------------
def page_data_upload(state):
    st.header("📂 数据上传与自动检测")
    uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=['csv','xlsx','xls'])
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        if df is not None:
            state['df'] = df
            state['filename'] = uploaded_file.name
            st.success(f"✅ 已加载：{uploaded_file.name} — {df.shape[0]} 行 × {df.shape[1]} 列")
            df2, numeric_cols = sanitize_numeric_df(df)
            state['df_sanitized'] = df2
            state['numeric_cols'] = numeric_cols
    if state.get('df') is not None:
        st.markdown("### 数据预览")
        with st.expander("📊 表格预览（前 200 行）"):
            st.dataframe(state['df'].head(200), width="stretch")
        num_cols = state.get('numeric_cols', [])
        st.write(f"检测到 {len(num_cols)} 个数值型字段：")
        st.write(", ".join(num_cols[:30]) + ("..." if len(num_cols)>30 else ""))

# ---------------------------
# 页面：模型训练 & 评估（使用 st.multiselect 选择特征）
# ---------------------------
def page_train_and_eval(state):
    st.header("⚙️ 模型训练与评估")
    if state.get('df') is None:
        st.info("请先上传数据。")
        return
    df = state['df_sanitized'] if state.get('df_sanitized') is not None else state['df']
    numeric_cols = state.get('numeric_cols', df.select_dtypes(include=[np.number]).columns.tolist())
    if not numeric_cols:
        st.warning("当前数据中没有检测到数值型列，无法训练模型。")
        return

    # 目标选择
    target = st.selectbox("🎯 目标变量 (Target)", options=numeric_cols, index=0)

    # 特征选择改为 multiselect（可搜索）
    available_features = [c for c in numeric_cols if c != target]
    default_feats = available_features[:6] if len(available_features) > 0 else []
    selected_features = st.multiselect(
        "选择特征变量（可以搜索并多选）",
        options=available_features,
        default=default_feats
    )
    st.markdown(f"已选特征： **{len(selected_features)}** 个")

    n_est = st.slider("🌲 随机森林树数量 (n_estimators)", 50, 800, 200, step=50)
    run = st.button("🚀 训练模型")

    if run:
        if len(selected_features) == 0:
            st.warning("请至少选择一个特征。")
            return
        with st.spinner("模型训练中..."):
            try:
                model, X_train, X_test, y_train, y_test = train_model(df, selected_features, target, n_est, st.session_state.seed)
                metrics = metrics_for_model(model, X_train, X_test, y_train, y_test)
                state.update({
                    "model": model, "metrics": metrics,
                    "model_features": selected_features, "model_target": target,
                    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
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
        c3.metric("训练 RMSE", safe_format(m.get('train_rmse'), ".2f"))
        c4.metric("测试 RMSE", safe_format(m.get('test_rmse'), ".2f"))
        st.caption(f"交叉验证 R²: {safe_format(m.get('cv_mean'), '.3f')} ± {safe_format(m.get('cv_std'), '.3f')}")

# ---------------------------
# 页面：可视化分析（SHAP Summary + 饼图 + 热力图）
# ---------------------------
def page_visual_analysis(state):
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
    tab1, tab2, tab3 = st.tabs(["SHAP Summary（交互）", "特征贡献（饼图）", "相关性热力图"])

    # --- SHAP Summary 修复版 ---
    with tab1:
        st.markdown("#### SHAP Summary（特征按重要度降序显示）")
        try:
            raw_shap = state.get('shap_values', None)
            if raw_shap is None:
                st.error("未找到 SHAP 值，请先训练模型并计算 SHAP。")
            else:
                n_samples = X_test.shape[0]
                n_features = len(features)

                # 规范化 SHAP 值
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

                # 排序特征（按平均绝对值降序）
                abs_mean = np.abs(shap_arr).mean(axis=0)
                sorted_idx = np.argsort(abs_mean)[::-1]  # 降序排列
                shap_sorted = shap_arr[:, sorted_idx]
                X_sorted = pd.DataFrame(X_test, columns=features).iloc[:, sorted_idx]
                sorted_features = [features[i] for i in sorted_idx]

                # 构造 long-form dataframe
                long_data = []
                for i, feat in enumerate(sorted_features):
                    for j in range(len(X_sorted)):
                        long_data.append({
                            "Feature": feat,
                            "Feature value": X_sorted.iloc[j, i],
                            "SHAP value": shap_sorted[j, i]
                        })
                df_long = pd.DataFrame(long_data)

                # 绘图
                fig = px.scatter(
                    df_long,
                    x="SHAP value",
                    y="Feature",
                    color="Feature value",
                    category_orders={"Feature": sorted_features[::-1]},  # 保持顺序，贡献度高的在上方
                    color_continuous_scale="RdBu_r",
                    opacity=0.6,
                    render_mode="webgl",
                    height=650,
                    title="SHAP Summary（特征对预测的方向与强度）"
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(
                    yaxis=dict(title="特征", showline=True, showgrid=True, zeroline=False, autorange='reversed'),  # 关键改动
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
                title="特征贡献占比（SHAP 平均绝对值）"
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
            hm = px.imshow(corr, text_auto=".2f", aspect="auto", title="相关性矩阵")
            hm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6F0F8")
            )
            st.plotly_chart(hm, width="stretch")
        except Exception as e:
            st.error(f"热力图绘制失败: {e}")

# ---------------------------
# 页面：反推 / 预算优化（完整版保留）
# ---------------------------
def page_reverse_opt(state):
    st.header("🔁 反推与预算优化（完整版）")
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
        target_gmv = st.number_input("目标 GMV", value=float(y_base * 1.1), step=1.0)
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

# ---------------------------
# 主控制流
# ---------------------------
if 'state' not in st.session_state:
    st.session_state.state = {}
state = st.session_state.state

if page == "数据上传 & 预览":
    page_data_upload(state)
elif page == "模型训练 & 评估":
    page_train_and_eval(state)
elif page == "可视化分析":
    page_visual_analysis(state)
elif page == "反推/预算优化":
    page_reverse_opt(state)

st.markdown("---")
st.markdown('<div class="small">版本：DataTech UI </div>', unsafe_allow_html=True)
