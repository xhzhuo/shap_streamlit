"""
配置文件
包含页面配置和视觉样式
"""

import streamlit as st

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
.card { background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; }
</style>
"""