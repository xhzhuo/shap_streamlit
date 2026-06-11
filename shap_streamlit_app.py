"""
主应用文件
整合所有模块，实现应用的主控制流
"""

import streamlit as st
import textwrap

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

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.markdown(
        """
        <div class="password-card password-card__form">
            <div class="app-hero__badge">Secure Workspace</div>
            <h2>Ad Effect Intelligence</h2>
            <p class="subtitle">DataTech · 请输入访问密码以解锁工作台</p>
            <div class="password-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("password_form"):
        password = st.text_input(
            "访问密码",
            type="password",
            key="password_input",
            placeholder="输入密码后按回车"
        )
        submitted = st.form_submit_button("进入工作台")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if password == APP_PASSWORD:
            st.session_state.password_correct = True
            st.success("✅ 密码正确！正在加载应用...")
            st.rerun()
        else:
            st.error("❌ 密码错误，请重试")

    return False

# 检查密码
if not check_password():
    st.stop()  # 如果密码不正确，停止执行后续代码

# 导入页面函数
from pages_module.data_upload import page_data_upload
from pages_module.model_training import page_train_and_eval
from pages_module.visualization import page_visual_analysis
from pages_module.adstock_ridge import page_adstock_ridge
from pages_module.optimization_page import page_reverse_opt
from pages_module.report_generator import page_report_generator
from pages_module.dev_docs import page_dev_docs
from pages_module import ai_assistant

# ---------------------------
# 页面：侧栏导航
# ---------------------------
if 'state' not in st.session_state:
    st.session_state.state = {}
state = st.session_state.state

dataset = state.get('df')
rows = f"{len(dataset):,}" if dataset is not None else "—"
cols = f"{dataset.shape[1]:,}" if dataset is not None else "—"
model_ready = "已训练" if state.get('model') is not None else "待训练"
target_name = state.get('model_target', '未选择')
adstock_result = state.get('adstock_ridge_result')
adstock_ready = "已完成" if isinstance(adstock_result, dict) and adstock_result.get("ok") else "待配置"

MAIN_PAGES = (
    "数据上传 & 预览",
    "模型训练 & 评估",
    "可视化分析",
    "反推/预算优化",
    "分析报告",
    "AI智能助手",
)
SPECIAL_TOOL_PAGES = (
    "Adstock Ridge 预测",
)
SUPPORT_PAGES = (
    "说明文档",
)
ALL_PAGES = MAIN_PAGES + SPECIAL_TOOL_PAGES + SUPPORT_PAGES

if "active_page" not in st.session_state or st.session_state.active_page not in ALL_PAGES:
    st.session_state.active_page = MAIN_PAGES[0]


def _select_sidebar_page(key: str) -> None:
    selected = st.session_state.get(key)
    if selected:
        st.session_state.active_page = selected


with st.sidebar:
    st.markdown('<div class="brand">Ad Effect Intelligence</div>', unsafe_allow_html=True)
    st.caption("Smart Ad Analysis Workspace")

    st.markdown('<div class="sidebar-section-label">Shapley & Random Forest</div>', unsafe_allow_html=True)
    main_nav_key = f"main_nav_{st.session_state.active_page in MAIN_PAGES}"
    st.radio(
        "主流程",
        MAIN_PAGES,
        index=MAIN_PAGES.index(st.session_state.active_page) if st.session_state.active_page in MAIN_PAGES else None,
        key=main_nav_key,
        label_visibility="collapsed",
        on_change=_select_sidebar_page,
        args=(main_nav_key,),
    )

    st.markdown('<div class="sidebar-section-label">Adstock & Ridge</div>', unsafe_allow_html=True)
    tool_nav_key = f"tool_nav_{st.session_state.active_page in SPECIAL_TOOL_PAGES}"
    st.radio(
        "专项建模工具",
        SPECIAL_TOOL_PAGES,
        index=SPECIAL_TOOL_PAGES.index(st.session_state.active_page) if st.session_state.active_page in SPECIAL_TOOL_PAGES else None,
        key=tool_nav_key,
        label_visibility="collapsed",
        on_change=_select_sidebar_page,
        args=(tool_nav_key,),
    )
    st.caption(f"Adstock Ridge 状态：{adstock_ready}")

    st.markdown('<div class="sidebar-section-label">支持</div>', unsafe_allow_html=True)
    support_nav_key = f"support_nav_{st.session_state.active_page in SUPPORT_PAGES}"
    st.radio(
        "支持",
        SUPPORT_PAGES,
        index=SUPPORT_PAGES.index(st.session_state.active_page) if st.session_state.active_page in SUPPORT_PAGES else None,
        key=support_nav_key,
        label_visibility="collapsed",
        on_change=_select_sidebar_page,
        args=(support_nav_key,),
    )

    page = st.session_state.active_page

    st.markdown("---")
    st.markdown("**当前数据**")
    st.metric("样本量", rows)
    st.metric("字段数", cols)
    st.metric("模型状态", model_ready)
    st.caption(f"目标变量：{target_name}")
    st.markdown("---")
    st.caption("© DataTech · Smart Ad Analysis")

flow_steps = [
    ("步骤 01", "数据上传 & 预览", "完成" if dataset is not None else "待开始"),
    ("步骤 02", "模型训练 & 评估", model_ready),
    ("步骤 03", "可视化分析", "可用" if state.get('shap_values') is not None else "待生成"),
    ("步骤 04", "反推/预算优化", "待执行" if state.get('model') is None else "准备就绪"),
    ("步骤 05", "分析报告", "可生成" if state.get('model') is not None else "待准备"),
    ("AI", "AI智能助手", "实时"),
]

if page not in ("说明文档", "Adstock Ridge 预测"):
    flow_html = "".join(
        [
            f"""
            <div class="flow-step {'active' if title == page else ''}">
                <small>{label}</small>
                <strong>{title}</strong>
                <span class="small">状态：{status}</span>
            </div>
            """
            for label, title, status in flow_steps
        ]
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="flow-steps">
                {flow_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
elif page == "Adstock Ridge 预测":
    st.markdown(
        f"""
        <div class="surface-muted">
            Adstock Ridge 预测是独立专项建模工具，不依赖主流程的随机森林训练结果。当前状态：{adstock_ready}
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "数据上传 & 预览":
    page_data_upload(state)
elif page == "模型训练 & 评估":
    page_train_and_eval(state)
elif page == "可视化分析":
    page_visual_analysis(state)
elif page == "Adstock Ridge 预测":
    page_adstock_ridge(state)
elif page == "反推/预算优化":
    page_reverse_opt(state)
elif page == "分析报告":
    page_report_generator(state)
elif page == "AI智能助手":
    ai_assistant.render()
elif page == "说明文档":
    page_dev_docs()

st.markdown("<p class='small' style='margin-top:2rem;'>Version · Beta</p>", unsafe_allow_html=True)
