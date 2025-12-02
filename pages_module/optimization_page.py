"""
反推优化页面（智能单入口版）
根据约束勾选自动选择最优优化方案
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import safe_format, format_dataframe_numeric
from optimization import optimize_ad_allocation_robust


def page_reverse_opt(state):
    """反推与预算优化页面(智能单入口版)"""
    st.header("🔁 反推与预算优化")
    # =========== 固定使用简洁视觉模式 ===========
    st.markdown(
        """
        <style>
        :root {
            --card-bg: #ffffff;
            --card-radius: 12px;
            --card-shadow: 0 1px 4px rgba(0,0,0,0.03);
            --border-soft: #e8edf2;
            --text-main: #2b3137;
            --text-sub: #67707b;
            --accent-blue: #4f8cc9;
            --accent-green: #4aa785;
            --accent-warn: #d8a54a;
            --accent-red: #d4665c;
        }
        .ui-card {background: var(--card-bg); border-radius: var(--card-radius); padding:14px 16px; margin:10px 0 16px; border:1px solid var(--border-soft); box-shadow: var(--card-shadow);}
        .ui-card h4 {margin:0 0 4px 0; font-size:14px; font-weight:600; letter-spacing:0.5px; color:var(--text-main);}
        .card-info {border:1px solid var(--border-soft);} .card-warn {border:1px solid var(--accent-warn);} .card-error {border:1px solid var(--accent-red);} .card-success {border:1px solid var(--accent-green);}        
        .ui-card.card-info h4 {color:var(--accent-blue);} .ui-card.card-warn h4 {color:var(--accent-warn);} .ui-card.card-error h4 {color:var(--accent-red);} .ui-card.card-success h4 {color:var(--accent-green);}        
        .tag {display:inline-block; background:#f3f6f9; color:#4a5562; font-size:11px; padding:2px 6px; border-radius:6px; margin-right:6px; margin-bottom:4px; border:1px solid #e1e5ea;}
        .metrics-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-top:4px;}
        .metric-box {background:#f8fafc; border:1px solid #e3e8ef; border-radius:10px; padding:8px 10px; text-align:center;}
        .metric-title {font-size:11px; color:var(--text-sub); margin:0; font-weight:500;}
        .metric-value {font-size:17px; font-weight:600; margin:2px 0 0 0; color:var(--text-main);}
        .agg-warn-list {margin:2px 0 0; padding-left:18px;}
        .agg-warn-list li {margin-bottom:3px; font-size:12px; line-height:1.35; color:var(--text-sub);}
        .section-divider {margin:22px 0 14px; border-top:1px solid #edf1f5;}
        .stMetric {padding:4px 0 0 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def render_card(body_html: str, kind: str = "info", title: str | None = None):
        kind_class = {
            "info": "card-info",
            "warn": "card-warn",
            "error": "card-error",
            "success": "card-success",
        }.get(kind, "card-info")
        title_html = f"<h4>{title}</h4>" if title else ""
        st.markdown(f"<div class='ui-card {kind_class}'>{title_html}{body_html}</div>", unsafe_allow_html=True)

    if state.get('model') is None:
        st.info("请先训练模型。")
        return

    model = state['model']
    features = state['model_features']
    base_x = state['X_train'].mean().values
    X_train_min = state['X_train'].min().values
    X_train_max = state['X_train'].max().values

    try:
        # 使用DataFrame确保特征名称一致性
        base_x_df = pd.DataFrame([base_x], columns=features)
        y_base = float(model.predict(base_x_df)[0])
    except Exception:
        y_base = 0.0

    # === 使用提醒卡片 ===
    with st.expander("💡 反推优化适用标准", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style="
                    background: white;
                    border-left: 4px solid #22c55e;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 18px; margin-bottom: 8px;">✅ 特征可控</div>
                    <div style="font-size: 13px; color: #555; line-height: 1.6;">
                        <b>可以直接调整的特征：</b><br>
                        • 投放金额<br>
                        • 渠道分配<br>
                        • 产品价格<br><br>
                        <b style="color: #ef4444;">不可控的特征：</b><br>
                        • 曝光数、点击数<br>
                        • 天气、节假日<br>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="
                    background: white;
                    border-left: 4px solid #3b82f6;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 18px; margin-bottom: 8px;">✅ 因果直接</div>
                    <div style="font-size: 13px; color: #555; line-height: 1.6;">
                        <b>直接因果关系：</b><br>
                        • 投放金额 → 转化数<br>
                        • 价格 → 销量<br>
                        • 内容类型 → 阅读量<br><br>
                        <b style="color: #f97316;">间接因果关系：</b><br>
                        • 投放 → 曝光 → 转化<br>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div style="
                    background: white;
                    border-left: 4px solid #8b5cf6;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 18px; margin-bottom: 8px;">✅ 结果可执行</div>
                    <div style="font-size: 13px; color: #555; line-height: 1.6;">
                        <b>可以立即执行：</b><br>
                        • 调整投放预算<br>
                        • 修改产品价格<br>
                        • 改变内容策略<br><br>
                        <b style="color: #ef4444;">无法执行：</b><br>
                        • "调整曝光数"<br>
                    </div>
                </div>
            """, unsafe_allow_html=True)


    # 顶部基准与目标概要（减少分散提示）
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"**基准预测 {state.get('model_target','目标')}:** {safe_format(y_base, '.2f')}")


    # ==============
    # 参数输入区域
    # ==============
    col1, col2 = st.columns([2, 1])
    with col1:
        target_gmv = st.number_input(
            f"🎯 目标 {state.get('model_target','目标')}",
            value=float(y_base * 1.1),
            step=y_base * 0.01,
            help="设置期望达到的目标值"
        )
    with col2:
        target_pct = ((target_gmv - y_base) / y_base * 100) if y_base > 0 else 0
        st.metric("目标提升", f"{target_pct:.1f}%", delta=f"{target_gmv - y_base:.2f}")

    # ============== 约束设置（重构版）==============
    st.markdown("---")
    st.subheader("⚙️ 约束条件设置")
    
    # 约束模式选择
    constraint_mode = st.radio(
        "选择约束模式",
        ["无约束（自由优化）", "智能约束（推荐）", "自定义约束"],
        index=1,
        horizontal=True,
        help="智能约束会自动设置合理的上下限"
    )
    
    min_constraints, max_constraints, total_budget = None, None, None
    
    if constraint_mode == "智能约束（推荐）":
        render_card("<p style='margin:0;'>💡 自动设置合理调整范围，防止过度偏离历史数据。</p>", kind="info", title="智能约束")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            adjust_range = st.selectbox(
                "允许调整幅度",
                ["保守（±30%）", "适中（±50%）", "激进（±100%）"],
                index=1
            )
            range_factor = {"保守（±30%）": 0.3, "适中（±50%）": 0.5, "激进（±100%）": 1.0}[adjust_range]
        
        with col2:
            respect_history = st.checkbox("尊重历史范围", value=True, help="不超过训练数据的最大值")
        
        with col3:
            allow_zero = st.checkbox("允许归零", value=False, help="允许某些渠道降为0")
        
        # 自动计算约束
        min_constraints = []
        max_constraints = []
        for i in range(len(features)):
            # 下限
            if allow_zero:
                min_val = 0.0
            else:
                min_val = max(0.0, base_x[i] * (1 - range_factor))
            
            # 上限
            max_val = base_x[i] * (1 + range_factor)
            if respect_history and not np.isnan(X_train_max[i]):
                max_val = min(max_val, X_train_max[i])
            
            min_constraints.append(float(min_val))
            max_constraints.append(float(max_val))
        
        # 显示约束预览
        with st.expander("📋 查看各渠道约束范围"):
            preview_data = []
            for i, feat in enumerate(features):
                preview_data.append({
                    "渠道": feat,
                    "基准": f"{base_x[i]:.2f}",
                    "最小值": f"{min_constraints[i]:.2f}",
                    "最大值": f"{max_constraints[i]:.2f}",
                    "可调范围": f"{((max_constraints[i] - min_constraints[i]) / base_x[i] * 100) if base_x[i] > 0 else 0:.0f}%"
                })
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
    
    elif constraint_mode == "自定义约束":
        render_card("<p style='margin:0;'>⚠️ 完全自主设置各渠道上下限，请谨慎避免极端值。</p>", kind="warn", title="自定义约束模式")
        
        # 初始化约束预设状态
        if "constraint_preset" not in st.session_state:
            st.session_state["constraint_preset"] = "⚖️ 灵活调整"
        
        # 初始化约束值（如果不存在）
        for i in range(len(features)):
            if f"custom_min_{i}" not in st.session_state:
                st.session_state[f"custom_min_{i}"] = base_x[i] * 0.5
            if f"custom_max_{i}" not in st.session_state:
                st.session_state[f"custom_max_{i}"] = base_x[i] * 1.5
        
        # 批量设置选项（使用radio自动有选中状态）
        st.markdown("**⚡ 快速批量设置**")
        
        preset_options = [
            "🔒 锁定基准",
            "📈 仅增不减", 
            "⚖️ 灵活调整",
            "📉 历史范围",
            "✏️ 自定义"
        ]
        
        preset_help = {
            "🔒 锁定基准": "最小值=最大值=基准值（不调整）",
            "📈 仅增不减": "最小值=基准，最大值=基准×2（只增不减）",
            "⚖️ 灵活调整": "最小值=基准×0.5，最大值=基准×1.5（推荐）",
            "📉 历史范围": "使用训练数据的实际最小值和最大值",
            "✏️ 自定义": "手动设置每个渠道的约束范围"
        }
        
        # 显示当前预设的说明
        st.caption(f"💡 {preset_help.get(st.session_state['constraint_preset'], '')}")
        
        # 兼容旧版本 session_state 中的名称（例如 "⚖️ 灵活"）
        current_preset = st.session_state.get("constraint_preset", "⚖️ 灵活调整")
        alias_map = {
            "⚖️ 灵活": "⚖️ 灵活调整",
            "灵活": "⚖️ 灵活调整",
            "灵活调整": "⚖️ 灵活调整",
        }
        if current_preset not in preset_options:
            mapped = alias_map.get(current_preset, "⚖️ 灵活调整")
            st.session_state["constraint_preset"] = mapped
            current_preset = mapped

        selected_preset = st.radio(
            "选择约束预设",
            preset_options,
            index=preset_options.index(current_preset),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # 如果选择变化，应用新的预设
        if selected_preset != st.session_state["constraint_preset"]:
            st.session_state["constraint_preset"] = selected_preset
            
            # 应用预设
            if selected_preset == "🔒 锁定基准":
                for i in range(len(features)):
                    st.session_state[f"custom_min_{i}"] = base_x[i]
                    st.session_state[f"custom_max_{i}"] = base_x[i]
            
            elif selected_preset == "📈 仅增不减":
                for i in range(len(features)):
                    st.session_state[f"custom_min_{i}"] = base_x[i]
                    st.session_state[f"custom_max_{i}"] = base_x[i] * 2.0
            
            elif selected_preset == "⚖️ 灵活调整":
                for i in range(len(features)):
                    st.session_state[f"custom_min_{i}"] = base_x[i] * 0.5
                    st.session_state[f"custom_max_{i}"] = base_x[i] * 1.5
            
            elif selected_preset == "📉 历史范围":
                for i in range(len(features)):
                    st.session_state[f"custom_min_{i}"] = max(0.0, X_train_min[i] if not np.isnan(X_train_min[i]) else base_x[i] * 0.5)
                    st.session_state[f"custom_max_{i}"] = X_train_max[i] if not np.isnan(X_train_max[i]) else base_x[i] * 2.0
            
            # 非自定义模式时，重新运行以更新UI
            if selected_preset != "✏️ 自定义":
                st.rerun()
        
        st.markdown("---")
        
        # 显示当前约束预览
        with st.expander("📋 当前约束范围预览", expanded=False):
            preview_data = []
            for i, feat in enumerate(features):
                min_c = st.session_state[f"custom_min_{i}"]
                max_c = st.session_state[f"custom_max_{i}"]
                preview_data.append({
                    "渠道": feat,
                    "基准": f"{base_x[i]:.2f}",
                    "最小值": f"{min_c:.2f}",
                    "最大值": f"{max_c:.2f}",
                    "可调范围": f"{((max_c - min_c) / base_x[i] * 100) if base_x[i] > 0 else 0:.0f}%"
                })
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
        
        st.markdown("**🎯 单独微调各渠道**")
        if st.session_state["constraint_preset"] != "✏️ 自定义":
            st.caption("展开渠道可查看或修改约束范围")
        else:
            st.caption("当前为自定义模式，所有约束值可自由设置")
        
        min_constraints, max_constraints = [], []
        
        for i, feat in enumerate(features):
            with st.expander(f"🔧 {feat}", expanded=(st.session_state["constraint_preset"] == "✏️ 自定义" and i == 0)):
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    st.metric("基准投放", f"{base_x[i]:.2f}")
                    if not np.isnan(X_train_min[i]) and not np.isnan(X_train_max[i]):
                        st.caption(f"历史范围: [{X_train_min[i]:.1f}, {X_train_max[i]:.1f}]")
                
                with cols[1]:
                    # 直接使用 custom_min_{i} 作为 key，实现联动
                    min_val = st.number_input(
                        "最小值", 
                        value=float(st.session_state[f"custom_min_{i}"]),
                        min_value=0.0,
                        key=f"custom_min_{i}",
                        help=f"渠道 {feat} 的投放量下限"
                    )
                
                with cols[2]:
                    # 确保最大值至少比最小值大 0.1
                    current_max = float(st.session_state[f"custom_max_{i}"])
                    safe_max = max(current_max, min_val + 0.1)
                    if safe_max != current_max:
                        st.session_state[f"custom_max_{i}"] = safe_max
                    
                    # 直接使用 custom_max_{i} 作为 key，实现联动
                    max_val = st.number_input(
                        "最大值",
                        value=safe_max,
                        min_value=min_val + 0.01,
                        key=f"custom_max_{i}",
                        help=f"渠道 {feat} 的投放量上限"
                    )
                
                min_constraints.append(float(min_val))
                max_constraints.append(float(max_val))
    
    else:  # 无约束
        render_card("<p style='margin:0;'>✅ 算法将自由寻找最优解，结果用于评估潜力。</p>", kind="success", title="无约束模式")
        min_constraints = None
        max_constraints = None

    # ==============
    # 优化计算按钮
    # ==============
    if st.button("🔢 计算建议方案"):
        with st.spinner("优化中..."):
            try:
                # 计算SHAP权重
                if 'shap_values' in state and state['shap_values'] is not None:
                    shap_abs_mean = np.abs(state['shap_values']).mean(axis=0)
                    if shap_abs_mean.sum() == 0:
                        st.error("所有 SHAP 值为 0，无法按贡献分配")
                        return

                    weights = shap_abs_mean / shap_abs_mean.sum()
                else:
                    # 如果没有SHAP值，则使用均匀权重
                    weights = np.ones(len(features)) / len(features)
                
                # 验证约束条件
                if max_constraints is not None:
                    for i in range(len(features)):
                        if min_constraints[i] > max_constraints[i]:
                            st.error(f"特征 {features[i]} 的最小值({min_constraints[i]:.2f})大于最大值({max_constraints[i]:.2f})")
                            return
                
                try:
                    # 使用新的统一优化接口（返回字典）
                    optimization_result = optimize_ad_allocation_robust(
                        model=model,
                        base_x=base_x,
                        y_target=target_gmv,
                        total_budget=total_budget,
                        weights=weights,
                        min_constraints=min_constraints,
                        max_constraints=max_constraints,
                        X_train=state['X_train'].values,  # 确保传递numpy数组
                        method='adaptive',  # 使用自适应优化策略
                        robustness_level='medium',  # 中等级别的鲁棒性
                        feature_names=features  # 传递特征名列表
                    )
                except ValueError as ve:
                    st.error(f"优化边界配置错误: {ve}")
                    st.info("💡 建议：请检查渠道上下限约束，确保最小值 < 最大值")
                    import traceback
                    st.error(f"详细错误:\n{traceback.format_exc()}")
                    return
                except Exception as opt_e:
                    st.error(f"优化计算错误: {opt_e}")
                    import traceback
                    st.error(f"详细错误:\n{traceback.format_exc()}")
                    return
                
                # 从结果字典中提取数据
                suggested = optimization_result['suggested_allocation']
                y_pred_new = optimization_result['predicted_value']
                constraint_status = optimization_result['constraint_status']
                method_used = optimization_result['method_used']
                robustness_level = optimization_result['robustness_level']
                
                # 计算目标达成率
                target_achieved_ratio = y_pred_new / target_gmv if target_gmv != 0 else 0
                
                # V2 版本使用简单的敏感度估计（避免重复计算）
                # 基于分配变化和预测变化的比率
                sensitivities = np.abs((suggested - base_x)) / (abs(y_pred_new - float(model.predict([base_x])[0])) + 1e-8)
                sensitivities = np.maximum(sensitivities, 0.01)  # 确保非零

                # =============================
                # 输出结果
                # =============================
                results = []
                # 确保suggested、weights、sensitivities长度与features一致
                n_features_actual = len(features)
                suggested_safe = np.array(suggested)[:n_features_actual] if len(suggested) >= n_features_actual else np.pad(np.array(suggested), (0, n_features_actual - len(suggested)), mode='constant')
                weights_safe = np.array(weights)[:n_features_actual] if len(weights) >= n_features_actual else np.pad(np.array(weights), (0, n_features_actual - len(weights)), mode='constant')
                sensitivities_safe = np.array(sensitivities)[:n_features_actual] if len(sensitivities) >= n_features_actual else np.pad(np.array(sensitivities), (0, n_features_actual - len(sensitivities)), mode='constant')
                
                # 规范化constraint_status列表长度
                constraint_status_safe = constraint_status
                if constraint_status and isinstance(constraint_status, (list, tuple)):
                    constraint_status_safe = list(constraint_status)[:n_features_actual] + ['正常'] * max(0, n_features_actual - len(constraint_status))
                
                for i, feat in enumerate(features):
                    if i >= len(suggested_safe):
                        break
                    base_val = float(base_x[i]) if i < len(base_x) else 0.0
                    sug = float(suggested_safe[i])
                    # 安全访问constraint_status，避免索引越界
                    constraint_info = "正常"
                    if constraint_status_safe and isinstance(constraint_status_safe, (list, tuple)) and i < len(constraint_status_safe):
                        constraint_info = constraint_status_safe[i]
                    elif constraint_status_safe and isinstance(constraint_status_safe, dict) and feat in constraint_status_safe:
                        constraint_info = constraint_status_safe[feat]
                    
                    results.append({
                        "渠道": feat,
                        "基准投放": base_val,
                        "建议投放": sug,
                        "增量": sug - base_val,
                        "SHAP权重": float(weights_safe[i]) if i < len(weights_safe) else 0.0,
                        "敏感度": float(sensitivities_safe[i]) if i < len(sensitivities_safe) else 0.0,
                        "约束状态": constraint_info
                    })

                st.session_state["reverse_results"] = results
                st.session_state["reverse_y_base"] = y_base
                st.session_state["reverse_target_gmv"] = target_gmv

                results_df = pd.DataFrame(results)
                fmt = format_dataframe_numeric(results_df)

                st.markdown(f"### 建议方案")
                
                # === 聚合显示警告信息，避免多条提示分散 ===
                warnings = optimization_result.get('warnings', [])
                if warnings:
                    items = []
                    for w in warnings:
                        sev = w.get('severity', 'low')
                        msg = w.get('message', '')
                        sug = w.get('suggestion', '')
                        sev_tag = {"high": "<span class='tag' style='background:#fee2e2;color:#b91c1c;'>高</span>",
                                   "medium": "<span class='tag' style='background:#fef9c3;color:#ca8a04;'>中</span>",
                                   "low": "<span class='tag' style='background:#dbeafe;color:#1e3a8a;'>低</span>"}.get(sev, "")
                        items.append(f"<li>{sev_tag}{msg} <span style='color:#64748b;'>💡 {sug}</span></li>")
                    render_card(f"<ul class='agg-warn-list'>{''.join(items)}</ul>", kind="warn", title="风险与提示")
                
                # === 显示投入产出分析 ===
                if y_pred_new is not None:
                    # 主要结果
                    adjusted_target = optimization_result.get('adjusted_target')
                    if adjusted_target is not None:
                        render_card(
                            f"<p style='margin:0;'>预测 {state.get('model_target','目标')}: <b>{safe_format(y_pred_new, '.2f')}</b><br/>"
                            f"目标调整：{safe_format(target_gmv, '.2f')} → {safe_format(adjusted_target, '.2f')}，达成率 {safe_format(y_pred_new/adjusted_target * 100, '.1f')}%</p>",
                            kind="success", title="预测与目标"
                        )
                    else:
                        render_card(
                            f"<p style='margin:0;'>预测 {state.get('model_target','目标')}: <b>{safe_format(y_pred_new, '.2f')}</b><br/>"
                            f"原始目标：{safe_format(target_gmv, '.2f')}，达成率 {safe_format(target_achieved_ratio * 100, '.1f')}%</p>",
                            kind="success", title="预测与目标"
                        )
                    
                    # 投入产出分析
                    budget_change_pct = optimization_result.get('budget_change_pct', 0)
                    output_change_pct = optimization_result.get('output_change_pct', 0)
                    roi = optimization_result.get('roi', 0)
                    marginal_efficiency = optimization_result.get('marginal_efficiency', 0)
                    
                    # 统一指标展示为紧凑网格
                    metric_html = (
                        "<div class='ui-card card-info'><h4>投入产出指标</h4>"
                        "<div class='metrics-grid'>"
                        f"<div class='metric-box'><p class='metric-title'>投放变化</p><p class='metric-value'>{budget_change_pct:+.1f}%</p></div>"
                        f"<div class='metric-box'><p class='metric-title'>产出变化</p><p class='metric-value'>{output_change_pct:+.1f}%</p></div>"
                        f"<div class='metric-box'><p class='metric-title'>ROI</p><p class='metric-value'>{roi:.3f}</p></div>"
                        f"<div class='metric-box'><p class='metric-title'>边际效率</p><p class='metric-value'>{marginal_efficiency:.2f}</p></div>"
                        "</div><p style='margin:8px 0 0; font-size:12px; color:#64748b;'>边际效率 = 产出增长率 / 投放增长率 (≥0.5 优秀，0.3-0.5 合格，<0.3 需评估目标)</p></div>"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                    
                else:
                    st.warning("建议方案预测值不可用。")

                st.dataframe(results_df.style.format(fmt), use_container_width=True)

                if total_budget is not None:
                    render_card(f"<p style='margin:0;'>预算对比：基准 {base_x.sum():.2f} → 建议 {np.sum(suggested):.2f} (预算 {total_budget:.2f})</p>", kind="info", title="预算变化")

                # ==============
                # 响应曲线
                # ==============
                if "reverse_results" in st.session_state:
                    results = st.session_state["reverse_results"]
                    y_base = st.session_state["reverse_y_base"]
                    target_gmv = st.session_state["reverse_target_gmv"]

                    st.subheader("📊 模型响应分析")
                    scales = np.linspace(0.5, 2.5, 50)  # 增加采样点，曲线更平滑
                    y_preds = []

                    for scale in scales:
                        x_scaled = base_x.copy()
                        x_scaled *= scale
                        # 使用DataFrame确保特征名称一致性
                        x_scaled_df = pd.DataFrame([x_scaled], columns=features)
                        y_pred = float(model.predict(x_scaled_df)[0])
                        y_preds.append(y_pred)

                    fig = make_subplots(
                        rows=1, cols=2,
                        column_widths=[0.48, 0.52],
                        subplot_titles=(
                            "📈 整体投放响应曲线",
                            "📊 渠道建议对比"
                        ),
                        horizontal_spacing=0.15
                    )

                    # =========== 左图：响应曲线（清新简约） ===========
                    fig.add_trace(
                        go.Scatter(
                            x=scales, 
                            y=y_preds, 
                            mode='lines',
                            name='响应曲线',
                            line=dict(color="#3498DB", width=2.5, shape='spline'),
                            fill='tozeroy',
                            fillcolor='rgba(52, 152, 219, 0.15)',
                            hovertemplate='<b>投放倍数</b>: %{x:.2f}x<br><b>预测值</b>: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # 基准线（虚线）
                    fig.add_hline(
                        y=y_base, 
                        line=dict(color="#95A5A6", width=1.5, dash="dash"),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=0.55, y=y_base,
                        text=f"基准: {y_base:.1f}",
                        showarrow=False,
                        font=dict(size=10, color="#7F8C8D"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="#95A5A6",
                        borderwidth=1,
                        borderpad=4,
                        xref="x", yref="y",
                        row=1, col=1
                    )
                    
                    # 目标线（粗实线，突出重点）
                    fig.add_hline(
                        y=target_gmv, 
                        line=dict(color="#E74C3C", width=2.5),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=0.55, y=target_gmv,
                        text=f"目标: {target_gmv:.1f}",
                        showarrow=False,
                        font=dict(size=10, color="#C0392B", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#E74C3C",
                        borderwidth=1.5,
                        borderpad=4,
                        xref="x", yref="y",
                        row=1, col=1
                    )
                    
                    fig.update_xaxes(
                        title_text="投放倍数 (相对基准值)",
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        zeroline=False,
                        row=1, col=1
                    )
                    fig.update_yaxes(
                        title_text="模型预测值",
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        zeroline=False,
                        row=1, col=1
                    )

                    # =========== 右图：柱状图对比（清新简约） ===========
                    fig.add_trace(
                        go.Bar(
                            name='基准投放',
                            x=features,
                            y=[r["基准投放"] for r in results],
                            marker=dict(
                                color='#3498DB',
                                opacity=0.7,
                                line=dict(color='white', width=0)
                            ),
                            hovertemplate='<b>%{x}</b><br>基准: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Bar(
                            name='建议投放',
                            x=features,
                            y=[r["建议投放"] for r in results],
                            marker=dict(
                                color='#2ECC71',
                                opacity=0.8,
                                line=dict(color='white', width=0)
                            ),
                            hovertemplate='<b>%{x}</b><br>建议: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_xaxes(
                        title_text="渠道",
                        showgrid=False,
                        row=1, col=2
                    )
                    fig.update_yaxes(
                        title_text="投放量",
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        zeroline=False,
                        row=1, col=2
                    )

                    # =========== 整体样式（清新简约） ===========
                    fig.update_layout(
                        barmode='group',
                        title=dict(
                            text='模型响应分析与投放优化方案',
                            font=dict(size=16, color="#2C3E50", family="Arial"),
                            x=0.5,
                            xanchor='center',
                            y=0.98,
                            yanchor='top'
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(color="#34495E", size=11, family="Arial"),
                        height=480,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.18,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#BDC3C7",
                            borderwidth=0.5,
                            font=dict(size=10)
                        ),
                        hovermode='x unified',
                        margin=dict(t=60, b=100, l=70, r=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"反推计算失败: {e}")

    # ==============
    # 选型建议
    # ==============
    # st.markdown("---")
    # st.subheader("📋 优化逻辑说明")

    # st.markdown( 
    #     """
    #     系统采用统一优化算法，自动处理各种约束条件：
    #     - 无约束 → 使用基础优化策略
    #     - 预算约束 → 在优化过程中考虑总预算限制
    #     - 渠道上下限约束 → 确保各渠道的投放量在指定范围内
        
    #     系统默认使用中等级别的鲁棒性优化策略，兼顾计算效率和结果稳定性。
    #     """
    # )

    # st.info("💡 推荐先使用无约束版本评估潜力，再加预算与上下限微调执行方案。")