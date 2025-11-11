"""
主应用文件
整合所有模块，实现应用的主控制流
"""

import streamlit as st

# 导入配置
from config import PAGE_CONFIG, CSS_STYLE

# 设置页面配置
st.set_page_config(**PAGE_CONFIG)

# 应用样式
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# 导入页面函数
from pages_module.data_upload import page_data_upload
from pages_module.model_training import page_train_and_eval
from pages_module.visualization import page_visual_analysis
from pages_module.optimization_page import page_reverse_opt
from pages_module.dev_docs import page_dev_docs

# ---------------------------
# 页面：侧栏导航（已去除随机控件）
# ---------------------------
with st.sidebar:
    st.markdown('<div class="brand">Ad Effect Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">DataTech UI · Full</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "导航",
        ("数据上传 & 预览", "模型训练 & 评估", "可视化分析", "反推/预算优化", "说明文档"),
        index=0
    )
    st.markdown("---")
    st.caption("© DataTech · Smart Ad Analysis")

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
elif page == "说明文档":
    page_dev_docs()

st.markdown("---")
st.markdown('<div class="small">Version：V2 · Full</div>', unsafe_allow_html=True)