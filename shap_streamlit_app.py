"""
主应用文件
整合所有模块，实现应用的主控制流
"""

import streamlit as st

# 导入配置
from config import PAGE_CONFIG, CSS_STYLE, APP_PASSWORD

# 设置页面配置
st.set_page_config(**PAGE_CONFIG)

# 应用样式
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# ---------------------------
# 密码验证功能
# ---------------------------
def check_password():
    """返回 True 表示密码正确"""
    
    # 初始化 session state
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # 如果已经验证通过，直接返回
    if st.session_state.password_correct:
        return True

    # 显示密码输入界面
    st.markdown('<div class="brand">🔒 Ad Effect Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">DataTech — 请输入密码访问应用</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 创建居中的列布局
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 身份验证")
        password = st.text_input(
            "请输入密码",
            type="password",
            key="password_input",
            placeholder="输入密码后按回车"
        )
        
        if st.button("🔓 解锁", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.password_correct = True
                st.success("✅ 密码正确！正在加载应用...")
                st.rerun()
            else:
                st.error("❌ 密码错误，请重试")
        
        st.markdown("---")
    
    return False

# 检查密码
if not check_password():
    st.stop()  # 如果密码不正确，停止执行后续代码

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
    st.markdown('<div class="subtitle">DataTech </div>', unsafe_allow_html=True)
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
st.markdown('<div class="small">Version：Beta</div>', unsafe_allow_html=True)