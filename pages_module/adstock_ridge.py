# -*- coding: utf-8 -*-
"""
Adstock Ridge attribution and forecasting page.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components import render_card, render_empty_state
from red_si_pipeline.dynamic_pipeline import (
    DynamicModelConfig,
    build_excel_bytes,
    build_quality_checks,
    run_dynamic_adstock_ridge,
)
from red_si_pipeline.field_profiler import coerce_numeric, infer_field_profile


UNIT_OPTIONS = {
    "不换算 / 指数": {
        "原始值": ("原始值", 1.0),
    },
    "金额": {
        "元 → 百万元": ("百万元", 1e-6),
        "千元 → 百万元": ("百万元", 1e-3),
        "万元 → 百万元": ("百万元", 1e-2),
        "百万元": ("百万元", 1.0),
    },
    "次数": {
        "次 → 百万次": ("百万次", 1e-6),
        "千次 → 百万次": ("百万次", 1e-3),
        "万次 → 百万次": ("百万次", 1e-2),
        "百万次": ("百万次", 1.0),
    },
    "比例": {
        "比例 0-1": ("比例", 1.0),
        "百分比 0-100 → 比例": ("比例", 0.01),
    },
}


def _read_uploaded_file(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def _sheet_names(uploaded_file) -> list[str]:
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    else:
        xls = pd.ExcelFile(uploaded_file)
    return xls.sheet_names


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:+.1%}"


def _metric_value(value, fmt: str = ".3f") -> str:
    if value is None or pd.isna(value):
        return "NA"
    return format(value, fmt)


def _render_page_hero() -> None:
    st.markdown(
        """
        <div class="page-hero">
          <div class="page-hero__body">
            <div class="page-hero__icon">AR</div>
            <div>
              <h1>Adstock Ridge 预测</h1>
              <p>按字段角色配置投放、控制、事件和诊断变量，生成可解释的归因预测模型。</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_check_cards(checks_df: pd.DataFrame) -> None:
    if checks_df.empty:
        return
    status_order = {"阻断": "error", "警告": "warn", "通过": "success"}
    for status, group in checks_df.groupby("status", sort=False):
        items = "".join(
            f"<li><b>{html.escape(str(row['item']))}</b>：{html.escape(str(row['detail']))}</li>"
            for _, row in group.iterrows()
        )
        render_card(f"<ul class='agg-warn-list'>{items}</ul>", kind=status_order.get(status, "info"), title=status)


def _render_tab_guide(title: str, items: list[str]) -> None:
    body = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="surface-muted">
          <strong>{html.escape(title)}</strong>
          <ul class="agg-warn-list">{body}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_field_dictionary(title: str, definitions: dict[str, str]) -> None:
    rows = "".join(
        f"""
        <tr>
          <td>{html.escape(field)}</td>
          <td>{html.escape(desc)}</td>
        </tr>
        """
        for field, desc in definitions.items()
    )
    st.markdown(f"#### {title}")
    st.markdown(
        f"""
        <table class="field-dictionary-table">
          <thead>
            <tr>
              <th>字段</th>
              <th>作用/含义</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _display_profile_df(profile_df: pd.DataFrame) -> pd.DataFrame:
    display_profile = profile_df.drop(
        columns=["推断类型", "数值识别率", "时间识别率"],
        errors="ignore",
    ).copy()
    if "缺失率" in display_profile.columns:
        display_profile["缺失率"] = display_profile["缺失率"].map(lambda x: f"{x:.0%}")
    return display_profile


def _role_field_rows(
    target_col: str | None,
    media_cols: list[str],
    control_cols: list[str],
    event_cols: list[str],
    category_col: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if target_col:
        rows.append({"字段": target_col, "角色": "目标变量"})
    rows.extend({"字段": col, "角色": "投放变量"} for col in media_cols)
    rows.extend({"字段": col, "角色": "控制变量"} for col in control_cols)
    rows.extend({"字段": col, "角色": "事件变量"} for col in event_cols)
    if category_col:
        rows.append({"字段": category_col, "角色": "品类变量"})

    deduped: list[dict[str, str]] = []
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(row["字段"], set()).add(row["角色"])
    for field, roles in seen.items():
        deduped.append({"字段": field, "角色": " / ".join(sorted(roles))})
    return deduped


def _unit_hint(values: pd.Series) -> str:
    valid = values.dropna()
    if valid.empty:
        return "无可识别数值"
    max_abs = valid.abs().max()
    median_abs = valid.abs().median()
    if valid.between(0, 1).mean() >= 0.85:
        return "多数字段值在 0-1，可能是比例或指数"
    if valid.between(1, 100).mean() >= 0.85 and max_abs <= 100:
        return "多数值在 1-100，如为率类字段可能是百分比"
    if max_abs >= 1_000_000:
        return "量级较大，可能是元或次"
    if median_abs > 0 and max_abs / median_abs > 100:
        return "波动跨度较大，建议确认是否混入异常值或单位变化"
    return "未发现明显单位风险"


def _render_unit_config(
    df: pd.DataFrame,
    target_col: str | None,
    media_cols: list[str],
    control_cols: list[str],
    event_cols: list[str],
    category_col: str | None,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    rows = _role_field_rows(target_col, media_cols, control_cols, event_cols, category_col)
    if not rows:
        return {}, []

    st.markdown(
        """
        <div class="section-card">
          <div class="section-title">2.5 单位确认</div>
          <div class="section-desc">系统只提示疑似单位风险，不按字段名自动改单位。请选择每个字段的口径和原始单位，后台会在缺失处理和 Adstock 之前完成换算。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    unit_scales: dict[str, float] = {}
    unit_records: list[dict[str, str]] = []
    with st.expander("单位确认与换算", expanded=True):
        st.caption("未确认时保持“原始值”，即倍率 1。投放金额建议统一到百万元，曝光/点击/播放建议统一到百万次，率类字段统一到比例。")
        for row in rows:
            field = row["字段"]
            role = row["角色"]
            values = coerce_numeric(df[field]) if field in df.columns else pd.Series(dtype=float)
            valid = values.dropna()
            min_value = valid.min() if not valid.empty else np.nan
            max_value = valid.max() if not valid.empty else np.nan
            c1, c2, c3, c4 = st.columns([1.5, 1.0, 1.2, 1.4])
            c1.markdown(f"**{field}**  \n{role}")
            c2.caption(f"范围：{_metric_value(min_value)} ~ {_metric_value(max_value)}")
            kind = c3.selectbox(
                "变量口径",
                list(UNIT_OPTIONS.keys()),
                index=0,
                key=f"unit_kind_{field}",
                label_visibility="collapsed",
            )
            unit_label = c4.selectbox(
                "原始单位",
                list(UNIT_OPTIONS[kind].keys()),
                index=0,
                key=f"unit_label_{field}_{kind}",
                label_visibility="collapsed",
            )
            target_unit, scale = UNIT_OPTIONS[kind][unit_label]
            unit_scales[field] = scale
            unit_records.append({
                "字段": field,
                "角色": role,
                "变量口径": kind,
                "原始单位": unit_label,
                "统一单位": target_unit,
                "换算倍率": f"{scale:g}",
                "原始范围": f"{_metric_value(min_value)} ~ {_metric_value(max_value)}",
                "换算后范围": f"{_metric_value(min_value * scale)} ~ {_metric_value(max_value * scale)}" if pd.notna(min_value) and pd.notna(max_value) else "NA",
                "检测提示": _unit_hint(values),
            })

        records_df = pd.DataFrame(unit_records)
        st.markdown("#### 单位换算记录")
        st.dataframe(records_df, use_container_width=True, hide_index=True)

    return unit_scales, unit_records


def _render_formula_explanation(result: dict) -> None:
    st.markdown("#### 预测公式与情景区间")
    st.markdown(
        """
        <div class="formula-panel">
          <div class="formula-section">
            <div class="formula-heading">1. Adstock 变量转换</div>
            <div class="formula-block">Adstock(Media_m,t) = Media_m,t + λ_m × Adstock(Media_m,t-1)</div>
          </div>

          <div class="formula-section">
            <div class="formula-heading">2. 原始归纳公式</div>
            <div class="formula-block">
              Brand SI_t = β0<br>
              + Σ β_m × Z(Adstock(Media_m,t))<br>
              + Σ β_c × Z(Control_c,t)<br>
              + Σ β_e × Z(Event_e,t)<br>
              + ε_t
            </div>
          </div>

          <div class="formula-section">
            <div class="formula-heading">3. 情景区间公式</div>
            <div class="formula-block">
              Base Forecast_t = 当前模型公式计算值<br>
              Optimistic_t = Base Forecast_t + k × Residual sigma<br>
              Pessimistic_t = Base Forecast_t - k × Residual sigma
            </div>
          </div>

          <div class="formula-section">
            <div class="formula-heading">4. 公式解释</div>
            <ul class="agg-warn-list">
            <li>这里的 Z(...) 表示变量已先按训练期均值和标准差做标准化，因此 β 不是直接乘以原始投放金额或曝光量的原始口径系数。</li>
            <li>投放变量进入公式前会先做 Adstock，因此系数解释的是广告滞后累积后的影响，不是原始投放当期的直接影响。</li>
            <li>品类变量只有在同时选入控制变量时才进入公式；如果只选为品类变量，它只作为预测图右轴参考线展示。</li>
            <li>归纳公示的 ε_t 表示模型没有解释掉的历史波动；预测期不会直接预测 ε_t，而是用历史残差波动生成乐观/悲观情景。</li>
          </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_forecast(result: dict) -> go.Figure:
    df = result["model_df"]
    category_cols = [c for c in df.columns if c.startswith("category__")]
    fig = make_subplots(specs=[[{"secondary_y": bool(category_cols)}]])
    fig.add_trace(go.Scatter(
        x=df["period_start"], y=df["actual"], mode="lines+markers",
        name="Actual", line=dict(color="#1f4e79", width=3),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["period_start"], y=df["fit_base"], mode="lines",
        name="Historical fit", line=dict(color="#9e9e9e", width=2, dash="dash"),
    ), secondary_y=False)
    for col in category_cols:
        category_name = col.replace("category__", "")
        fig.add_trace(go.Scatter(
            x=df["period_start"],
            y=df[col],
            mode="lines",
            name=category_name,
            line=dict(color="#c7c7c7", width=3),
        ), secondary_y=True)
    if df["forecast_base"].notna().any():
        forecast = df[df["forecast_base"].notna()]
        fig.add_trace(go.Scatter(
            x=forecast["period_start"], y=forecast["forecast_base"], mode="lines+markers",
            name="Base forecast", line=dict(color="#f28e2b", width=3),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=forecast["period_start"], y=forecast["optimistic"], mode="lines",
            name="Optimistic", line=dict(color="#59a14f", width=2, dash="dot"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=forecast["period_start"], y=forecast["pessimistic"], mode="lines",
            name="Pessimistic", line=dict(color="#e15759", width=2, dash="dot"),
        ), secondary_y=False)
        fig.add_vline(x=forecast["period_start"].iloc[0], line_dash="dash", line_color="#666666")
    fig.update_layout(
        template="plotly_white",
        height=520,
        hovermode="x unified",
        title="实际值、历史拟合与预测情景",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    fig.update_xaxes(title_text="周期")
    fig.update_yaxes(title_text="目标变量", secondary_y=False)
    if category_cols:
        fig.update_yaxes(title_text="品类变量 / Category", secondary_y=True)
    return fig


def _plot_coefficients(result: dict) -> go.Figure:
    coef_df = result["coef_df"].copy()
    coef_df["direction"] = np.where(coef_df["coef_standardized_final"] >= 0, "正向", "负向")
    fig = px.bar(
        coef_df.sort_values("coef_standardized_final"),
        x="coef_standardized_final",
        y="feature",
        color="role",
        orientation="h",
        title="标准化系数贡献",
        labels={"coef_standardized_final": "标准化系数", "feature": "特征"},
        height=max(420, min(760, 80 + len(coef_df) * 34)),
    )
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=30, t=70, b=30))
    return fig


def _plot_residuals(result: dict) -> go.Figure:
    model_df = result["model_df"]
    df = model_df.loc[model_df["actual"].notna()].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["period_start"], y=df["residual"], name="Residual",
        marker_color=np.where(df["residual"] >= 0, "#59a14f", "#e15759"),
    ))
    fig.add_hline(y=0, line_color="#6b728e", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=420,
        title="残差时间序列",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    fig.update_xaxes(title_text="周期")
    fig.update_yaxes(title_text="实际值 - 拟合值")
    return fig


def _make_role_df(config: DynamicModelConfig) -> pd.DataFrame:
    rows = []
    for col in [config.time_col]:
        rows.append({"字段": col, "角色": "时间字段", "参数": ""})
    for col in [config.target_col]:
        rows.append({"字段": col, "角色": "目标变量", "参数": f"unit_scale={config.unit_scales.get(col, 1.0):g}"})
    for col in config.media_cols:
        rows.append({"字段": col, "角色": "投放变量", "参数": f"lambda={config.adstock_lambdas.get(col, 0.5):.2f}; missing={config.media_missing_strategy}; unit_scale={config.unit_scales.get(col, 1.0):g}"})
    for col in config.control_cols:
        rows.append({"字段": col, "角色": "控制变量", "参数": f"missing={config.control_missing_strategy}; unit_scale={config.unit_scales.get(col, 1.0):g}"})
    for col in config.event_cols:
        rows.append({"字段": col, "角色": "事件变量", "参数": f"missing={config.event_missing_strategy}; unit_scale={config.unit_scales.get(col, 1.0):g}"})
    model_cols = set(config.media_cols + config.control_cols + config.event_cols)
    for col in config.category_cols:
        parameter = f"右轴参考线；missing={config.control_missing_strategy}; unit_scale={config.unit_scales.get(col, 1.0):g}"
        if col in model_cols:
            parameter += "；同时按建模变量角色入模"
        else:
            parameter += "；不额外入模"
        rows.append({"字段": col, "角色": "品类变量", "参数": parameter})
    for col in config.diagnostic_cols:
        rows.append({"字段": col, "角色": "诊断字段", "参数": "not in model"})
    return pd.DataFrame(rows)


def _business_readout(result: dict) -> str:
    metrics = result["metrics"]
    diag = result["diagnostics"]
    forecast_summary = diag.get("forecast_summary", pd.DataFrame())
    r2 = metrics.get("R2", np.nan)
    mape = metrics.get("MAPE", np.nan)
    quality = "可以用于方向性预测和归因讨论"
    if pd.notna(r2) and r2 >= 0.65 and pd.notna(mape) and mape <= 0.15:
        quality = "拟合质量较好，可以作为较稳健的预测与复盘依据"
    elif pd.notna(r2) and r2 < 0.30:
        quality = "解释力偏弱，建议补充媒体触点、投放节奏、创意质量、竞品声量或外部需求变量后再用于关键决策"

    lines = [
        f"本次模型使用 {metrics['Training rows']} 个历史周期训练，{metrics['Forecast rows']} 个周期生成预测。",
        f"模型 R2 为 {_metric_value(r2)}，MAPE 为 {_fmt_pct(mape)}，{quality}。",
        f"情景区间采用 Base ± {metrics['Scenario half width']:.3f}，来自训练期残差波动，不应理解为严格统计置信区间。",
    ]
    if not forecast_summary.empty:
        row = forecast_summary.iloc[0]
        lines.append(
            f"预测期 Base 均值为 {row['forecast_base_avg']:.3f}，相对最近 12 期实际均值变化 {_fmt_pct(row['base_vs_recent_12'])}。"
        )
    residual_corr = diag.get("residual_corr", pd.DataFrame())
    if not residual_corr.empty:
        top = residual_corr.iloc[0]
        lines.append(
            f"诊断字段中，{top['diagnostic_field']} 与残差相关性最高（{top['corr_with_residual']:.2f}），建议作为下一版变量候选或异常周解释线索。"
        )
    return "\n\n".join(lines)


def page_adstock_ridge(state: dict) -> None:
    _render_page_hero()

    st.markdown(
        """
        <div class="section-card">
          <div class="section-title">1. 上传数据</div>
          <div class="section-desc">上传后系统只做客观字段画像，不按固定字段名校验，也不自动判断字段角色。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "选择 CSV / Excel 文件",
        type=["csv", "xlsx", "xls"],
        key="adstock_ridge_file",
    )

    if uploaded_file is None:
        render_empty_state("上传数据后，将进入字段画像、角色配置、建模检查和结果展示流程。")
        return

    sheet_name = None
    if not uploaded_file.name.lower().endswith(".csv"):
        try:
            sheets = _sheet_names(uploaded_file)
            sheet_name = st.selectbox("选择工作表", sheets, index=0)
        except Exception as exc:
            st.error(f"无法读取 Excel 工作表：{exc}")
            return

    try:
        df = _read_uploaded_file(uploaded_file, sheet_name)
    except Exception as exc:
        st.error(f"读取文件失败：{exc}")
        return

    if df.empty:
        render_empty_state("文件中没有可读取的数据。")
        return

    profile_df = infer_field_profile(df)
    state["adstock_ridge_df"] = df
    state["adstock_ridge_profile"] = profile_df
    data_signature = (
        uploaded_file.name,
        sheet_name,
        df.shape,
        tuple(df.columns.astype(str).tolist()),
    )
    if state.get("adstock_ridge_data_signature") != data_signature:
        state["adstock_ridge_data_signature"] = data_signature
        state.pop("adstock_ridge_result", None)
        state.pop("adstock_ridge_config", None)
        state.pop("adstock_ridge_roles", None)
        state.pop("adstock_ridge_target", None)
        st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.metric("样本量", f"{df.shape[0]:,}")
    c2.metric("字段数", f"{df.shape[1]:,}")
    c3.metric("存在缺失字段", int(profile_df["缺失率"].gt(0).sum()))

    with st.expander("字段画像", expanded=True):
        st.dataframe(_display_profile_df(profile_df), use_container_width=True, height=320)

    st.markdown(
        """
        <div class="section-card">
          <div class="section-title">2. 字段角色配置</div>
          <div class="section-desc">
            请按建模用途配置字段：时间字段用于排序和 Adstock 累积；目标变量 Y 是要预测或解释的 Brand SI；
            投放变量是媒体花费、曝光、点击等广告压力；控制变量是会影响 Brand SI、但不是本次投放动作本身的外部或业务背景变量；
            事件变量是 campaign 节点、素材上新、渠道调整等 0/1 标记；品类变量是行业/品类趋势线，只用于预测图右轴展示，也可以同时选入控制变量参与建模；
            诊断字段不参与训练，只用于解释高误差周期和排查遗漏因素。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = df.columns.astype(str).tolist()
    time_candidates = profile_df.loc[profile_df["时间识别率"].ge(0.75), "字段"].astype(str).tolist()

    left, right = st.columns(2)
    with left:
        time_col = st.selectbox("时间字段", [""] + cols, index=(cols.index(time_candidates[0]) + 1 if time_candidates else 0))
        target_col = st.selectbox("目标变量 Y", cols, index=None, placeholder="Choose an option")
        media_cols = st.multiselect("投放变量 Media / Pressure", cols, default=[])
    with right:
        control_cols = st.multiselect("控制变量 Control", cols, default=[])
        event_cols = st.multiselect("事件变量 Event Flags", cols, default=[])
        category_col = st.selectbox("品类变量 Category", cols, index=None, placeholder="Choose an option")
        diagnostic_cols = st.multiselect("诊断字段 Diagnostic", cols, default=[])

    if target_col and state.get("adstock_ridge_target") != target_col:
        state["adstock_ridge_target"] = target_col
        st.rerun()

    selected_model_cols = set(media_cols + control_cols + event_cols)
    if target_col in selected_model_cols:
        st.warning("目标变量同时出现在建模变量中，请从投放/控制/事件变量里移除，避免目标泄漏。")
    if category_col and category_col == target_col:
        st.warning("品类变量不应选择目标变量本身；请选择代表行业/品类搜索热度的字段。")
    if category_col and category_col in control_cols:
        st.info("当前品类变量也在控制变量中：它会进入模型，同时会作为右轴品类参考线展示。")

    clean_media_cols = [c for c in media_cols if c != target_col]
    clean_control_cols = [c for c in control_cols if c != target_col]
    clean_event_cols = [c for c in event_cols if c != target_col]
    clean_category_cols = [category_col] if category_col and category_col != target_col else []
    unit_scales, unit_records = _render_unit_config(
        df,
        target_col,
        clean_media_cols,
        clean_control_cols,
        clean_event_cols,
        category_col if clean_category_cols else None,
    )

    st.markdown(
        """
        <div class="section-card">
          <div class="section-title">3. 清洗与模型参数</div>
          <div class="section-desc">投放变量默认做 Geometric Adstock；控制变量按所选策略处理缺失；事件变量缺失默认按 0。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("目标变量不做自动填充：有值的历史期用于训练，目标为空的周期视为预测期；如果历史中间断点为空，会在建模检查中提示。")

    missing_labels = {
        "zero": "填 0",
        "mean": "均值填充",
        "ffill": "前后向填充",
        "none": "不填充，缺失行不参与训练/预测",
    }
    missing_col1, missing_col2, missing_col3 = st.columns(3)
    with missing_col1:
        media_missing_strategy = st.radio(
            "投放变量缺失处理",
            options=["zero", "ffill", "mean", "none"],
            format_func=lambda x: missing_labels[x],
            help="投放缺失常见含义是未投放，所以默认填 0；如果缺失代表数据未回传，请选择不填充。",
        )
    with missing_col2:
        control_missing_strategy = st.radio(
            "控制变量缺失处理",
            options=["mean", "ffill", "zero", "none"],
            format_func=lambda x: missing_labels[x],
            help="控制变量和品类变量共用这一策略。默认均值填充；如果预测期缺失且选择不填充，会影响 forecast 和品类参考线展示。",
        )
    with missing_col3:
        event_missing_strategy = st.radio(
            "事件变量缺失处理",
            options=["zero", "ffill", "none"],
            format_func=lambda x: missing_labels[x],
            help="事件变量通常是 campaign 节点、素材上新、渠道调整、平台节点等 flag，缺失默认按 0，即未发生。",
        )

    adstock_lambdas: dict[str, float] = {}
    if media_cols:
        st.caption("Decay λ 越大，投放影响残留越长。短效 0.2-0.3，中效 0.4-0.6，长效 0.7-0.9。")
        lambda_cols = st.columns(min(3, len(media_cols)))
        for i, col in enumerate(media_cols):
            with lambda_cols[i % len(lambda_cols)]:
                adstock_lambdas[col] = st.slider(
                    f"{col}",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.5,
                    step=0.05,
                    key=f"lambda_{col}",
                )

    with st.expander("高级参数", expanded=False):
        scenario_sigma_multiplier = st.slider(
            "情景区间倍数：Base ± k × residual sigma",
            0.1,
            2.0,
            0.9,
            0.1,
            help="k 越大，乐观/悲观区间越宽。默认 0.9 更接近业务情景范围，而不是严格置信区间。",
        )
        positive_ridge_alpha = st.number_input(
            "Positive Ridge alpha",
            min_value=0.01,
            max_value=1000.0,
            value=5.0,
            step=0.5,
            help="Ridge 正则化强度。值越大，系数越被压小，模型更稳但可能欠拟合；值越小，模型更贴合历史但更容易不稳定。Positive Ridge 还会约束系数非负，适合投放变量的业务解释。一般用户可保留默认值。",
        )

    config = DynamicModelConfig(
        time_col=time_col,
        target_col=target_col,
        media_cols=clean_media_cols,
        control_cols=clean_control_cols,
        event_cols=clean_event_cols,
        diagnostic_cols=diagnostic_cols,
        category_cols=clean_category_cols,
        adstock_lambdas=adstock_lambdas,
        unit_scales=unit_scales,
        unit_records=unit_records,
        media_missing_strategy=media_missing_strategy,
        control_missing_strategy=control_missing_strategy,
        event_missing_strategy=event_missing_strategy,
        scenario_sigma_multiplier=scenario_sigma_multiplier,
        positive_ridge_alpha=positive_ridge_alpha,
    )

    checks_df = pd.DataFrame(build_quality_checks(df, config))
    with st.expander("建模前检查", expanded=True):
        _render_check_cards(checks_df)

    run = st.button("运行 Adstock Ridge 建模", use_container_width=True)
    if run:
        with st.spinner("正在清洗字段、构造 Adstock 特征并训练 Ridge 模型..."):
            result = run_dynamic_adstock_ridge(df, config)
        state["adstock_ridge_result"] = result
        state["adstock_ridge_config"] = config
        state["adstock_ridge_roles"] = _make_role_df(config)

    result = state.get("adstock_ridge_result")
    if not result:
        return

    st.markdown(
        """
        <div class="section-card">
          <div class="section-title">4. 建模结果</div>
          <div class="section-desc">先看结果总览，再进入预测、贡献、诊断、数据质量和业务解读。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result.get("ok"):
        st.error(result.get("error", "建模失败。"))
        _render_check_cards(result.get("checks", pd.DataFrame()))
        return

    metrics = result["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("训练期数", metrics["Training rows"])
    m2.metric("预测期数", metrics["Forecast rows"])
    m3.metric("R2", _metric_value(metrics["R2"]))
    m4.metric("MAPE", _fmt_pct(metrics["MAPE"]))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("RMSE", _metric_value(metrics["RMSE"]))
    m6.metric("Residual sigma", _metric_value(metrics["Residual sigma"]))
    m7.metric("特征数", metrics["Feature count"])
    m8.metric("丢弃训练行", metrics["Dropped training rows"])

    _render_field_dictionary(
        "核心指标怎么阅读",
        {
            "R2": "模型对历史目标变量波动的解释程度。越接近 1，说明当前字段组合越能解释历史变化；偏低时，通常表示投放、控制或事件变量还没有覆盖关键影响因素。",
            "MAPE": "平均百分比误差，用来理解模型预测和实际值平均差多少比例。越低越好；例如 12% 可以粗略理解为历史拟合平均偏差约 12%。目标值接近 0 时，这个指标会不稳定。",
            "RMSE": "平均误差规模，单位和目标变量一致。越低越好；适合判断模型在实际业务口径下大概会偏离多少。",
            "Residual sigma": "历史残差的波动程度，也就是模型没解释掉的部分有多大。这里会用于生成乐观/悲观情景区间；数值越大，预测上下沿越宽，说明不确定性越高。",
        },
    )
    _render_formula_explanation(result)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "预测走势", "模型诊断", "诊断字段", "数据质量", "业务解读"
    ])

    with tab1:
        _render_tab_guide(
            "怎么阅读预测走势",
            [
                "Actual 是真实历史表现，Historical fit 是模型用投放、控制和事件变量拟合出来的历史走势；两条线越贴近，说明当前变量组合越能解释历史变化。",
                "品类变量是用户选择的行业/品类趋势参考线，使用右侧坐标轴展示；如果这个字段也被选为控制变量，它同时会参与模型训练。",
                "Base forecast 是按当前输入数据外推得到的基准预测；Optimistic / Pessimistic 来自历史残差波动，只用于观察预测上下沿，不是严格置信区间。",
                "下方汇总表重点看 forecast_base_avg 和 base_vs_recent_12：前者是预测期基准均值，后者表示它相对最近 12 期实际均值的变化幅度。",
            ],
        )
        st.plotly_chart(_plot_forecast(result), use_container_width=True)
        forecast_summary = result["diagnostics"].get("forecast_summary", pd.DataFrame())
        if not forecast_summary.empty:
            show = forecast_summary.copy()
            show["base_vs_recent_12"] = show["base_vs_recent_12"].map(_fmt_pct)
            _render_field_dictionary(
                "预测汇总表字段说明",
                {
                    "forecast_base_avg": "预测期 Base forecast 的平均值，用来代表模型给出的基准预测水平。",
                    "forecast_pessimistic_avg": "预测期 Pessimistic 情景的平均值，用来观察偏保守情况下的目标变量水平。",
                    "forecast_optimistic_avg": "预测期 Optimistic 情景的平均值，用来观察偏乐观情况下的目标变量水平。",
                    "recent_12_actual_avg": "最近 12 个历史训练周期的实际值平均水平，用作预测结果的近端历史参照。",
                    "base_vs_recent_12": "Base 预测均值相对最近 12 期实际均值的变化比例，正值表示预测高于近期水平，负值表示低于近期水平。",
                },
            )
            st.dataframe(show, use_container_width=True)

    with tab2:
        _render_tab_guide(
            "怎么阅读模型诊断",
            [
                "残差 = 实际值 - 拟合值；残差接近 0 说明该周期解释较充分，正残差表示实际高于模型预期，负残差表示实际低于模型预期。",
                "连续多期同方向残差通常意味着模型遗漏了某类系统性因素，单个尖峰更像是活动、异常事件、数据口径变化或突发外部因素。",
                "高误差周期表用于定位需要人工复盘的时间点；这些周期不一定代表模型错误，也可能是业务上确实发生了模型字段未覆盖的事情。",
                "下方高误差周期表建议按误差绝对值从大到小看，逐个回查当期投放排期、素材/渠道变化、数据采集口径和异常流量。",
            ],
        )
        st.plotly_chart(_plot_residuals(result), use_container_width=True)
        high_error = result["diagnostics"].get("high_error", pd.DataFrame())
        st.markdown("#### 高误差周期")
        _render_field_dictionary(
            "高误差周期表字段说明",
            {
                "period_start": "周期开始日期，用来定位是哪一个历史周期出现较大误差。",
                "actual": "该周期真实目标变量值。",
                "fit_base": "模型根据当期投放、控制和事件变量拟合出的目标变量值。",
                "residual": "实际值减拟合值。正值表示实际表现高于模型解释，负值表示实际表现低于模型解释。",
                "abs_residual": "残差绝对值，用来衡量误差大小；表格默认优先展示这个值较大的周期。",
                "diagnostic__字段名": "用户选择的诊断字段在该周期的取值，不参与训练，只帮助解释该周期为什么偏离模型预期。",
            },
        )
        st.dataframe(high_error, use_container_width=True)

    with tab3:
        _render_tab_guide(
            "怎么阅读诊断字段",
            [
                "诊断字段不参与训练，只和残差做相关性分析，用来解释模型没解释掉的部分。",
                "相关性越接近 +1 或 -1，说明该字段越可能对应模型遗漏的影响因素、异常周期线索或潜在数据口径问题。",
                "诊断字段不能直接当作因果结论；适合用于下一轮建模候选变量、业务复盘问题清单，或检查是否有目标泄漏风险。",
                "下方相关性表先看 corr_with_residual 的绝对值，再结合字段业务含义判断是否值得加入下一版模型；样本少或字段缺失多时只作为弱信号。",
            ],
        )
        residual_corr = result["diagnostics"].get("residual_corr", pd.DataFrame())
        if residual_corr.empty:
            render_empty_state("没有足够的诊断字段数据用于残差相关性分析。")
        else:
            fig = px.bar(
                residual_corr.sort_values("corr_with_residual"),
                x="corr_with_residual",
                y="diagnostic_field",
                orientation="h",
                title="诊断字段与残差相关性",
                labels={"corr_with_residual": "与残差相关性", "diagnostic_field": "诊断字段"},
            )
            fig.update_layout(template="plotly_white", height=max(360, len(residual_corr) * 34 + 120))
            st.plotly_chart(fig, use_container_width=True)
            _render_field_dictionary(
                "诊断字段相关性表字段说明",
                {
                    "diagnostic_field": "用户选择的诊断字段名称。这些字段不参与模型训练，只用于解释残差。",
                    "corr_with_residual": "诊断字段与残差的相关系数，范围为 -1 到 +1；绝对值越大，说明它越可能解释模型未覆盖的波动。",
                },
            )
            st.dataframe(residual_corr, use_container_width=True)
        st.caption("诊断字段不参与训练，用来解释高误差周期、发现遗漏变量和识别目标泄漏风险。")

    with tab4:
        st.markdown("#### 字段画像")
        st.dataframe(_display_profile_df(profile_df), use_container_width=True)
        unit_records_df = result.get("unit_records", pd.DataFrame())
        if not unit_records_df.empty:
            st.markdown("#### 单位换算记录")
            st.dataframe(unit_records_df, use_container_width=True, hide_index=True)
        st.markdown("#### 建模检查")
        st.dataframe(result["checks"], use_container_width=True)

    with tab5:
        readout = _business_readout(result)
        st.text_area("可复制业务解读", value=readout, height=220)
        excel_bytes = build_excel_bytes(_display_profile_df(profile_df), state.get("adstock_ridge_roles", _make_role_df(config)), result)
        st.download_button(
            "下载建模结果 Excel",
            data=excel_bytes,
            file_name="adstock_ridge_model_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
