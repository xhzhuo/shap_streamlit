"""
UI 组件模块
提供可复用的 UI 组件和工具函数
"""

import streamlit as st
from typing import Optional, Literal


def render_card(
    body_html: str,
    kind: Literal["info", "warn", "error", "success"] = "info",
    title: Optional[str] = None
) -> None:
    """
    渲染统一样式的卡片组件
    
    Parameters
    ----------
    body_html : str
        卡片主体内容（HTML格式）
    kind : str
        卡片类型，可选: "info", "warn", "error", "success"
    title : str, optional
        卡片标题
        
    Examples
    --------
    >>> render_card("<p>这是一条信息</p>", kind="info", title="提示")
    >>> render_card("<p>警告内容</p>", kind="warn")
    """
    kind_class = {
        "info": "card-info",
        "warn": "card-warn",
        "error": "card-error",
        "success": "card-success",
    }.get(kind, "card-info")
    
    title_html = f"<h4>{title}</h4>" if title else ""
    st.markdown(
        f"<div class='ui-card {kind_class}'>{title_html}{body_html}</div>",
        unsafe_allow_html=True
    )


def render_empty_state(message: str) -> None:
    """
    渲染空状态提示
    
    Parameters
    ----------
    message : str
        提示信息
        
    Examples
    --------
    >>> render_empty_state("需要先上传数据才能继续")
    """
    st.markdown(
        f"""
        <div class="surface-muted">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_card(
    score: int,
    color: str,
    summary_title: str,
    summary_advice: str,
    metrics_description: Optional[str] = None
) -> None:
    """
    渲染模型质量评分卡片
    
    Parameters
    ----------
    score : int
        模型评分（0-100）
    color : str
        主题颜色（CSS颜色值）
    summary_title : str
        评分标题
    summary_advice : str
        建议文本
    metrics_description : str, optional
        指标说明（HTML格式）
        
    Examples
    --------
    >>> render_quality_card(
    ...     score=85,
    ...     color="#22c55e",
    ...     summary_title="优秀模型",
    ...     summary_advice="模型性能良好，可以投入使用"
    ... )
    """
    default_metrics = """
        <b>📘 指标说明：</b><br>
        • R²（决定系数）：越接近 1，拟合效果越好。<br>
        • NRMSE（归一化均方根误差）：越小越好，表示预测值与真实值越接近。<br>
        • 交叉验证 R²：越接近测试集 R²，说明模型稳定且能适应新数据。
    """
    
    metrics_html = metrics_description or default_metrics
    
    st.markdown(f"""
        <div style="
            background:#ffffff;
            color:#333333;
            border:1px solid #e0e0e0;
            border-left:6px solid {color};
            border-radius:8px;
            padding:18px 22px;
            margin-top:20px;
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            box-shadow:0 2px 8px rgba(0,0,0,0.05);
        ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
            <div style="font-size:18px;font-weight:600;color:{color};">
                {summary_title}
            </div>
            <div style="font-size:14px;margin-top:4px;color:#555;">
                {summary_advice}
            </div>
            </div>
            <div style="text-align:right;">
            <div style="font-size:28px;font-weight:700;color:{color};line-height:1;">
                {score}
            </div>
            <div style="font-size:13px;color:#777;">模型评分</div>
            </div>
        </div>
        <hr style="border:none;border-top:1px solid #eee;margin:12px 0;">
        <div style="font-size:13.5px;color:#555;line-height:1.6;">
            {metrics_html}
        </div>
        </div>
    """, unsafe_allow_html=True)


def render_divider(margin: str = "22px 0 14px") -> None:
    """
    渲染分隔线
    
    Parameters
    ----------
    margin : str
        CSS margin 值
        
    Examples
    --------
    >>> render_divider()
    >>> render_divider(margin="10px 0")
    """
    st.markdown(
        f'<div class="section-divider" style="margin:{margin};"></div>',
        unsafe_allow_html=True
    )


def render_metrics_grid(metrics: dict) -> None:
    """
    渲染指标网格（用于优化结果展示）
    
    Parameters
    ----------
    metrics : dict
        指标字典，包含以下键：
        - budget_change_pct: 投放变化百分比
        - output_change_pct: 产出变化百分比
        - roi: ROI值
        - marginal_efficiency: 边际效率
        
    Examples
    --------
    >>> render_metrics_grid({
    ...     'budget_change_pct': 10.5,
    ...     'output_change_pct': 15.2,
    ...     'roi': 1.45,
    ...     'marginal_efficiency': 0.68
    ... })
    """
    budget_change = metrics.get('budget_change_pct', 0)
    output_change = metrics.get('output_change_pct', 0)
    roi = metrics.get('roi', 0)
    marginal_eff = metrics.get('marginal_efficiency', 0)
    
    metric_html = (
        "<div class='ui-card card-info'><h4>投入产出指标</h4>"
        "<div class='metrics-grid'>"
        f"<div class='metric-box'><p class='metric-title'>投放变化</p><p class='metric-value'>{budget_change:+.1f}%</p></div>"
        f"<div class='metric-box'><p class='metric-title'>产出变化</p><p class='metric-value'>{output_change:+.1f}%</p></div>"
        f"<div class='metric-box'><p class='metric-title'>ROI</p><p class='metric-value'>{roi:.3f}</p></div>"
        f"<div class='metric-box'><p class='metric-title'>边际效率</p><p class='metric-value'>{marginal_eff:.2f}</p></div>"
        "</div><p style='margin:8px 0 0; font-size:12px; color:#64748b;'>边际效率 = 产出增长率 / 投放增长率 (≥0.5 优秀，0.3-0.5 合格，<0.3 需评估目标)</p></div>"
    )
    st.markdown(metric_html, unsafe_allow_html=True)


def render_info_cards_row(cards: list[dict]) -> None:
    """
    渲染一行信息卡片（适用于优化页面的适用标准展示）
    
    Parameters
    ----------
    cards : list[dict]
        卡片列表，每个卡片包含：
        - title: 标题
        - color: 边框颜色
        - content: 内容（HTML格式）
        
    Examples
    --------
    >>> render_info_cards_row([
    ...     {
    ...         'title': '✅ 特征可控',
    ...         'color': '#22c55e',
    ...         'content': '<b>可控特征：</b><br>• 投放金额<br>• 渠道分配'
    ...     },
    ...     {
    ...         'title': '✅ 因果直接',
    ...         'color': '#3b82f6',
    ...         'content': '<b>直接因果：</b><br>• 投放 → 转化'
    ...     }
    ... ])
    """
    cols = st.columns(len(cards))
    
    for col, card in zip(cols, cards):
        with col:
            st.markdown(f"""
                <div style="
                    background: white;
                    border-left: 4px solid {card['color']};
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 18px; margin-bottom: 8px;">{card['title']}</div>
                    <div style="font-size: 13px; color: #555; line-height: 1.6;">
                        {card['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
