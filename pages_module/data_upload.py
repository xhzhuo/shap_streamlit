"""
数据上传页面
"""

import streamlit as st
from utils import load_file, sanitize_numeric_df

def page_data_upload(state):
    """数据上传与预览页面"""

    st.caption("支持 CSV / Excel，推荐在 5 万行以内以获得最佳响应速度")
    uploaded_file = st.file_uploader("选择本地文件", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        if df is not None:
            state['df'] = df
            state['filename'] = uploaded_file.name
            df2, numeric_cols = sanitize_numeric_df(df)
            state['df_sanitized'] = df2
            state['numeric_cols'] = numeric_cols
            st.success(f"✅ 已加载：{uploaded_file.name} · {df.shape[0]} 行 × {df.shape[1]} 列")

    df_loaded = state.get('df')
    if df_loaded is None:
        st.markdown(
            """
            <div class="surface-muted">
                还没有数据——上传后将立即展示字段分布、清洗情况以及结构化预览。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    numeric_cols = state.get('numeric_cols', [])
    col1, col2, col3 = st.columns(3)
    col1.metric("样本量", f"{df_loaded.shape[0]:,}")
    col2.metric("字段数", f"{df_loaded.shape[1]:,}")
    col3.metric("数值字段", len(numeric_cols))

    tab1, tab2 = st.tabs(["结构概览", "数据预览（前 200 行）"])

    with tab1:
        st.markdown(f"**文件名：** {state.get('filename', '未命名')}")
        st.markdown(f"**数值型字段：** {len(numeric_cols)} 个")
        if numeric_cols:
            st.caption(f"字段列表: {', '.join(numeric_cols[:20])}{'...' if len(numeric_cols) > 20 else ''}")
        st.info("✓ 数据类型自动清洗完成，可直接用于建模")

    with tab2:
        st.dataframe(df_loaded.head(200), use_container_width=True)