"""
数据上传页面
"""

import streamlit as st
from utils import load_file, sanitize_numeric_df


def page_data_upload(state):
    """数据上传与预览页面"""
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
            st.dataframe(state['df'].head(200), use_container_width=True)
        num_cols = state.get('numeric_cols', [])
        st.write(f"检测到 {len(num_cols)} 个数值型字段：")
        st.write(", ".join(num_cols[:30]) + ("..." if len(num_cols)>30 else ""))