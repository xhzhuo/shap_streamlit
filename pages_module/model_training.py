"""
模型训练页面
"""

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

    st.markdown("---")
    st.markdown("### 模型训练")
    run = st.button("🚀 训练模型")

    if run:
        if len(selected_features) == 0:
            st.warning("请至少选择一个特征。")
            return
        with st.spinner("模型训练中..."):
            try:
                # 固定参数：n_estimators=200, random_state=42
                model, X_train, X_test, y_train, y_test = train_model(df, selected_features, target, n_estimators=200, random_state=42)
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

        # --- 模型评分卡 ---
        quality = evaluate_model_quality(
            m['train_r2'], m['test_r2'], m['cv_mean'],
            m['train_rmse'], m['test_rmse']
        )

        # 安全地分割字符串，避免索引越界
        summary_parts = quality['summary'].split('\n')
        summary_title = summary_parts[0] if len(summary_parts) > 0 else quality['summary']
        summary_advice = summary_parts[1] if len(summary_parts) > 1 else ""

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
                • RMSE（均方根误差）：越小越好，表示预测值与真实值越接近。<br>
                • 交叉验证 R²：越接近测试集 R²，说明模型稳定且能适应新数据。
            </div>
            </div>
            """, unsafe_allow_html=True)