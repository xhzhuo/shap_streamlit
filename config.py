"""
配置文件
包含页面配置和视觉样式
"""

import streamlit as st

# 应用密码配置
APP_PASSWORD = "datatech"  # 修改此处设置你的密码

# LLM API配置 (硅基流动)
LLM_API_CONFIG = {
    "api_url": "https://api.siliconflow.cn/v1/chat/completions",
    "api_key": "sk-unjhqxfdxfrkxzqxijkhjcqxvvhrxzvlkiyycggihucszmck",
    "model": "deepseek-ai/DeepSeek-V3",  # 改用DeepSeek-V3，速度更快
    "max_tokens": 1500,  # 减少token数量，加快响应
    "temperature": 0.7,
    "top_p": 0.7,
    "timeout": 120,  # 超时时间120秒
}

# 页面配置
PAGE_CONFIG = {
    "page_title": "Ad Effect Intelligence — DataTech",
    "page_icon": "🛰️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 视觉样式
CSS_STYLE = """
<style>
:root{
  --bg:#f6f8fb;
  --surface:#ffffff;
  --surface-alt:#eff4ff;
  --border:#e1e7f2;
  --shadow:0 18px 40px rgba(34, 61, 109, 0.08);
  --radius:18px;
  --text:#1c2340;
  --muted:#6b728e;
  --accent:#3b6ff2;
  --accent-soft:#4ac8a8;
  --accent-deep:#2b48c4;
  --accent-bright:#38c9d0;
}
html, body, .stApp{
  background:var(--bg);
  color:var(--text);
  font-family:"Segoe UI","Inter","PingFang SC","Microsoft Yahei",sans-serif;
}
.stApp .main .block-container{
  padding-top:1.25rem;
  padding-bottom:2.5rem;
  max-width:1180px;
}
section[data-testid="stSidebar"]{
  background:linear-gradient(135deg,#f9fbff 0%,#edf4ff 55%,#e1f7f0 100%) !important;
  border-right:1px solid var(--border);
  box-shadow:0 10px 35px rgba(15,23,42,0.06);
}
section[data-testid="stSidebar"]>div{
  padding:2rem 1.2rem 2.4rem;
}
.brand{font-size:1.3rem;font-weight:600;color:var(--accent);margin-bottom:0.2rem;}
.subtitle{color:var(--muted);font-size:0.9rem;margin-bottom:0.4rem;}
.small{font-size:0.82rem;color:var(--muted);}





.app-hero{
  background:linear-gradient(135deg,#f9fbff 0%,#edf4ff 55%,#e1f7f0 100%);
  border-radius:var(--radius);
  padding:2rem;
  border:1px solid rgba(255,255,255,0.7);
  box-shadow:var(--shadow);
  margin-bottom:1.5rem;
}
.app-hero__badge{
  display:inline-flex;
  align-items:center;
  padding:0.1rem 0.8rem;
  border-radius:999px;
  border:1px solid rgba(59,111,242,0.3);
  font-size:0.85rem;
  font-weight:600;
  color:var(--accent);
  margin-bottom:0.9rem;
  background:rgba(255,255,255,0.7);
}
.app-hero__title{font-size:1.9rem;font-weight:650;margin:0;color:var(--text);}
.app-hero__desc{color:var(--muted);font-size:1rem;margin-top:0.35rem;max-width:720px;}
.flow-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.8rem;margin-top:1.5rem;margin-bottom:1.5rem;width:100%;box-sizing:border-box;}
.flow-step{
  background:var(--surface);
  border-radius:14px;
  border:1px solid var(--border);
  padding:0.9rem 1rem;
  box-shadow:0 8px 24px rgba(15,23,42,0.05);
}
.flow-step small{display:block;color:var(--muted);font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;}
.flow-step strong{display:block;font-size:1rem;margin-top:0.2rem;}
.flow-step span{display:block;margin-top:0.3rem;color:var(--accent);font-size:0.85rem;font-weight:600;}
.page-hero{
  background:var(--surface);
  border-radius:var(--radius);
  padding:1.4rem 1.6rem;
  border:1px solid var(--border);
  box-shadow:var(--shadow);
  margin-bottom:1.3rem;
}
.page-hero__icon{width:48px;height:48px;border-radius:14px;background:var(--surface-alt);display:flex;align-items:center;justify-content:center;font-size:1.6rem;margin-right:1rem;}
.page-hero__body{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
.page-hero__body h1{margin:0;font-size:1.45rem;}
.page-hero__body p{margin:0;color:var(--muted);}
.section-card{
  background:var(--surface);
  border-radius:var(--radius);
  padding:1.4rem 1.5rem;
  border:1px solid var(--border);
  box-shadow:var(--shadow);
  margin-bottom:1.25rem;
}
.section-title{font-size:1.05rem;font-weight:600;margin-bottom:0.2rem;}
.section-desc{color:var(--muted);font-size:0.9rem;margin-bottom:1rem;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.9rem;margin-bottom:0.4rem;}
.stat-card{background:var(--surface-alt);padding:0.95rem 1rem;border-radius:16px;border:1px solid rgba(59,111,242,0.12);}
.stat-card small{color:var(--muted);font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;}
.stat-card span{display:block;font-size:1.2rem;font-weight:600;margin-top:0.15rem;}
.pill{display:inline-flex;align-items:center;padding:0.05rem 0.75rem;border-radius:999px;background:rgba(59,111,242,0.08);color:var(--accent);font-size:0.82rem;font-weight:500;}
.surface-muted{background:#f9fbff;border-radius:var(--radius);padding:1rem;border:1px dashed var(--border);color:var(--muted);}
/* Primary buttons (standard + form submit) */
.stButton>button,
div[data-testid="baseButton-primary"] button,
div[data-testid="stFormSubmitButton"] button,
button[kind="primary"]{
  border:none !important;
  border-radius:999px !important;
  padding:0.55rem 1.55rem !important;
  font-weight:600 !important;
  background:linear-gradient(118deg,var(--accent-deep),var(--accent),var(--accent-bright)) !important;
  color:#fff !important;
  box-shadow:0 18px 28px rgba(43,72,196,0.22) !important;
  transition:transform 0.18s ease,box-shadow 0.18s ease,filter 0.18s ease !important;
}
.stButton>button:hover,
div[data-testid="baseButton-primary"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
button[kind="primary"]:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 22px 32px rgba(43,72,196,0.28) !important;
  filter:brightness(1.03) !important;
}
.stButton>button:focus-visible,
div[data-testid="baseButton-primary"] button:focus-visible,
div[data-testid="stFormSubmitButton"] button:focus-visible,
button[kind="primary"]:focus-visible{
  outline:2px solid rgba(56,201,208,0.6) !important;
  outline-offset:2px !important;
}
/* Secondary buttons */
.stButton>button[kind="secondary"],
div[data-testid="baseButton-secondary"] button,
button[kind="secondary"]{
  background:linear-gradient(120deg,#eef2ff,#f2fbff) !important;
  color:var(--accent) !important;
  box-shadow:none !important;
  border:1px solid rgba(59,111,242,0.2) !important;
}
.stButton>button:disabled,
div[data-testid="baseButton-primary"] button:disabled,
div[data-testid="stFormSubmitButton"] button:disabled,
button:disabled{
  opacity:0.55 !important;
  box-shadow:none !important;
  background:#c6d4ff !important;
  color:#fff !important;
}

/* === Checkbox & Tooltip Overhaul === */
div[data-testid="stCheckbox"] > label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    width: 100%;
}

/* This is the container for the text and the tooltip icon */
div[data-testid="stCheckbox"] > label > div:nth-of-type(2) {
    display: flex;
    align-items: center;
    justify-content: flex-start; /* Align items to the start */
    flex-wrap: nowrap; /* Prevent wrapping */
    gap: 0.4rem;
    flex-grow: 1;
}

/* The text part of the checkbox */
div[data-testid="stCheckbox"] p {
    white-space: nowrap;
    margin: 0;
    padding: 0;
    line-height: 1.5;
}

/* The tooltip icon itself */
div[data-testid="stTooltipIcon"] {
    position: relative;
    top: 0;
    left: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-left: 0 !important; /* Override Streamlit's negative margin */
    order: 2; /* Ensure it comes after the text */
}

div[data-testid="stCheckbox"] input[type="checkbox"]{
  accent-color:var(--accent);
  width:18px;
  height:18px;
  cursor:pointer;
}
div[data-testid="stCheckbox"] svg{
  color:#fff;
}

div[data-testid="stRadio"] input[type="radio"]{
  accent-color:var(--accent);
}
.stApp input[type="checkbox"],
.stApp input[type="radio"]{
  accent-color:var(--accent);
}
div[data-testid="stHorizontalBlock"]{
  display:flex !important;
  flex-wrap:nowrap !important;
  gap:1rem !important;
  overflow-x:auto !important;
}
div[data-testid="stHorizontalBlock"]>div{
  flex:0 0 auto !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]{
  display:inline-flex !important;
  align-items:center !important;
  white-space:nowrap !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"] p{
  white-space:nowrap !important;
  margin:0 !important;
}
/* Streamlit 内部组件颜色覆盖 */

[data-baseweb="checkbox"] svg{
  fill:#fff !important;
}
/* 选中的checkbox背景 */
span[data-baseweb="checkbox"][aria-checked="true"] div{
  background-color:var(--accent) !important;
  border-color:var(--accent) !important;
}
/* MultiSelect 标签颜色 */
.stMultiSelect [data-baseweb="tag"]{
  background-color:rgba(59,111,242,0.12) !important;
  border:1px solid rgba(59,111,242,0.3) !important;
}
.stMultiSelect [data-baseweb="tag"] span{
  color:var(--accent-deep) !important;
}
.stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox select,
.stMultiSelect>div>div>div>input,
.stFileUploader div[data-testid="stDecoration"]+div{
  border-radius:12px !important;
  border:1px solid var(--border) !important;
  box-shadow:none !important;
}
.stTabs [data-baseweb="tab"]{
  border-bottom:2px solid transparent;
  font-weight:600;
  color:var(--muted);
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color:var(--accent);
  border-color:var(--accent);
}
.stExpander{border:1px solid var(--border);border-radius:14px;background:var(--surface);}
.password-card{
  max-width:460px;
  margin:4rem auto 2rem;
  background:var(--surface);
  border-radius:var(--radius);
  padding:2rem;
  text-align:center;
  border:1px solid var(--border);
  box-shadow:var(--shadow);
}
.password-card__form{
  padding:2.4rem 2.6rem 2.3rem;
}
.password-card h2{margin-bottom:0.6rem;}
.password-hint{color:var(--muted);font-size:0.9rem;margin:0.2rem 0 0.4rem;}
.password-divider{width:100%;height:1px;background:var(--border);margin:1.4rem 0 1.1rem;opacity:0.9;}
.password-card form{margin:0 auto;max-width:360px;text-align:left;}
.password-card form label{font-weight:600;color:var(--muted);}
.password-card form div[data-testid="stTextInput"]{margin-bottom:0.6rem;}
.password-card form button{width:100%;margin-top:0.2rem;}
.chat-board{background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);box-shadow:var(--shadow);padding:1rem 1.3rem;}
.chat-bubble{padding:0.8rem 1rem;border-radius:14px;margin-bottom:0.6rem;}
.chat-bubble.user{background:var(--surface-alt);}
.chat-bubble.assistant{background:#fff7ec;border:1px solid #ffe1b8;}

.flow-step.active{
  background:linear-gradient(135deg,#f0f7ff 0%,#e6f0ff 100%) !important;
  border:1.5px solid #3b6ff2 !important;
  box-shadow:0 6px 16px rgba(59,111,242,0.15) !important;
  transform:translateY(-2px);
}
.flow-step.active small{color:#2b48c4 !important;font-weight:700;}
.flow-step.active strong{color:#1c2340 !important;}
.flow-step.active span{color:#3b6ff2 !important;}

/* Styles from optimization_page.py */
.ui-card {background:#ffffff;border-radius:16px;padding:18px;margin:10px 0 16px;border:1px solid #e4e9f4;box-shadow:0 12px 30px rgba(36,63,112,0.08);}
.ui-card h4 {margin:0 0 6px 0;font-size:14px;font-weight:600;color:#2b3152;letter-spacing:0.04em;text-transform:uppercase;}
.card-info {border-left:4px solid #4c7df4;}
.card-warn {border-left:4px solid #f4b63c;}
.card-error {border-left:4px solid #f26b6b;}
.card-success {border-left:4px solid #33b27b;}
.tag {display:inline-flex;background:#f0f4ff;color:#55607a;font-size:11px;padding:2px 8px;border-radius:999px;margin-right:6px;margin-bottom:4px;border:1px solid rgba(85,96,122,0.18);}
.metrics-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:8px;}
.metric-box {background:#f7f9ff;border:1px solid #dfe5fb;border-radius:14px;padding:10px;text-align:center;}
.metric-title {font-size:11px;color:#8188a3;margin:0;font-weight:500;text-transform:uppercase;}
.metric-value {font-size:19px;font-weight:600;margin:4px 0 0 0;color:#2f3b66;}
.agg-warn-list {margin:6px 0 0;padding-left:18px;}
.agg-warn-list li {margin-bottom:6px;font-size:13px;line-height:1.4;color:#4e566f;}
.section-divider {margin:22px 0 14px;border-top:1px solid #edf1f5;}
.stMetric {padding:4px 0 0 0;}
</style>
"""
