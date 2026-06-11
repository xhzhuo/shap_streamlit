"""
分析报告生成页面
一键生成分析报告，支持Markdown预览和PDF导出
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO

# 导入LLM帮助函数
from llm_helper import LLMAssistant


def generate_report_markdown(state: dict) -> str:
    """根据state中的数据生成Markdown格式的分析报告"""
    
    sections = []
    
    # ========== 1. 报告标题 ==========
    report_date = datetime.now().strftime("%Y年%m月%d日")
    sections.append(f"""# 广告效果分析报告

**生成日期**: {report_date}

---""")
    
    # ========== 2. 数据概览 ==========
    df = state.get('df')
    if df is not None:
        n_samples = len(df)
        n_features = df.shape[1]
        target_name = state.get('model_target', '未选择')
        features = state.get('model_features', [])
        
        sections.append(f"""## 一、数据概览

| 指标 | 数值 |
|------|------|
| 样本量 | {n_samples:,} |
| 字段数 | {n_features} |
| 目标变量 | {target_name} |
| 特征变量数 | {len(features)} |

**选用特征**: {', '.join(features) if features else '未选择'}
""")
    else:
        sections.append("""## 一、数据概览

> ⚠️ 暂无数据，请先上传数据集。
""")
    
    # ========== 3. 模型训练结果 ==========
    metrics = state.get('metrics')
    if metrics:
        train_r2 = metrics.get('train_r2', 0)
        test_r2 = metrics.get('test_r2', 0)
        train_nrmse = metrics.get('train_nrmse', 0)
        test_nrmse = metrics.get('test_nrmse', 0)
        cv_mean = metrics.get('cv_mean', 0)
        cv_std = metrics.get('cv_std', 0)
        
        # 评估模型质量
        if test_r2 >= 0.85:
            quality = "优秀"
            quality_desc = "模型拟合效果很好，可信度高"
        elif test_r2 >= 0.7:
            quality = "良好"
            quality_desc = "模型表现不错，具有较好的预测能力"
        elif test_r2 >= 0.5:
            quality = "一般"
            quality_desc = "模型有一定预测能力，建议增加特征或调优"
        else:
            quality = "较弱"
            quality_desc = "模型预测能力较弱，需要优化特征或调参"
        
        sections.append(f"""## 二、模型训练结果

### 2.1 模型性能指标

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| R² (决定系数) | {train_r2:.4f} | {test_r2:.4f} |
| NRMSE (归一化误差) | {train_nrmse:.4f} | {test_nrmse:.4f} |

**交叉验证 R²**: {cv_mean:.4f} ± {cv_std:.4f}

### 2.2 模型质量评估

**综合评级**: {quality}

{quality_desc}

**指标说明**:
- **R² (决定系数)**: 越接近1表示模型解释能力越强
- **NRMSE**: 越小表示预测误差越小
- **交叉验证**: 评估模型稳定性和泛化能力
""")
    else:
        sections.append("""## 二、模型训练结果

> ⚠️ 暂无模型数据，请先完成模型训练。
""")
    
    # ========== 4. 特征重要性分析 ==========
    shap_values = state.get('shap_values')
    features = state.get('model_features', [])
    
    if shap_values is not None and features:
        # 计算特征重要性
        importance = np.abs(shap_values)
        if isinstance(importance, list):
            importance = np.mean([np.abs(v) for v in importance], axis=0)
        importance = np.mean(np.abs(importance), axis=0)
        
        # 确保长度匹配
        min_len = min(len(importance), len(features))
        importance = importance[:min_len]
        features_list = features[:min_len]
        
        # 计算百分比
        total = np.sum(importance)
        percentages = (importance / total * 100) if total > 0 else importance
        
        # 排序
        sorted_idx = np.argsort(percentages)[::-1]
        
        # 生成表格
        table_rows = []
        for i, idx in enumerate(sorted_idx[:10], 1):  # 只展示Top 10
            feat_name = features_list[idx]
            pct = percentages[idx]
            table_rows.append(f"| {i} | {feat_name} | {pct:.2f}% |")
        
        sections.append(f"""## 三、特征重要性分析

基于SHAP值计算的特征贡献度排名（Top 10）：

| 排名 | 特征名称 | 贡献度占比 |
|------|----------|------------|
{chr(10).join(table_rows)}

**关键发现**:
- 最重要特征: **{features_list[sorted_idx[0]]}** (贡献度 {percentages[sorted_idx[0]]:.1f}%)
- Top 3 特征累计贡献度: {sum(percentages[sorted_idx[:3]]):.1f}%
""")
    else:
        sections.append("""## 三、特征重要性分析

> ⚠️ 暂无SHAP分析数据，请先完成可视化分析。
""")
    
    return "\n".join(sections)


def generate_ai_insights(state: dict) -> str:
    """调用LLM生成AI分析洞察"""
    
    metrics = state.get('metrics')
    features = state.get('model_features', [])
    shap_values = state.get('shap_values')
    target_name = state.get('model_target', '目标变量')
    
    if not metrics:
        return "> 暂无足够数据生成AI洞察，请先完成模型训练。"
    
    # 构建上下文
    context_parts = [
        f"目标变量: {target_name}",
        f"测试集R²: {metrics.get('test_r2', 0):.4f}",
        f"测试集NRMSE: {metrics.get('test_nrmse', 0):.4f}",
        f"使用特征: {', '.join(features)}",
    ]
    
    # 添加特征重要性信息
    if shap_values is not None and features:
        importance = np.abs(shap_values)
        if isinstance(importance, list):
            importance = np.mean([np.abs(v) for v in importance], axis=0)
        importance = np.mean(np.abs(importance), axis=0)
        
        min_len = min(len(importance), len(features))
        importance = importance[:min_len]
        features_list = features[:min_len]
        
        total = np.sum(importance)
        percentages = (importance / total * 100) if total > 0 else importance
        sorted_idx = np.argsort(percentages)[::-1]
        
        top_features = [f"{features_list[i]}({percentages[i]:.1f}%)" for i in sorted_idx[:5]]
        context_parts.append(f"Top5特征贡献: {', '.join(top_features)}")
    
    context = "\n".join(context_parts)
    
    # 构建提示词
    prompt = f"""基于以下广告效果模型分析结果，请生成一段专业的分析洞察和优化建议。

{context}

请用中文输出，格式要求：
1. 先总结模型整体表现（2-3句话）
2. 分析关键特征的业务含义（3-4点）
3. 给出2-3条具体可执行的优化建议

保持专业简洁，不要使用emoji。"""

    try:
        llm = LLMAssistant()
        result = llm.chat(prompt)
        if result.get('success'):
            return result.get('content', '')
        else:
            return f"> AI洞察生成失败: {result.get('error', '未知错误')}"
    except Exception as e:
        return f"> AI洞察生成失败: {str(e)}"


def generate_html(markdown_content: str, ai_insights: str) -> str:
    """生成包含完整样式的HTML报告"""
    import markdown
    
    # 合并内容
    full_content = markdown_content
    if ai_insights and not ai_insights.startswith('>'):
        full_content += "\n\n## 四、AI智能洞察与建议\n\n" + ai_insights
        
    # 转换为HTML
    html_body = markdown.markdown(
        full_content,
        extensions=['tables', 'fenced_code', 'nl2br']
    )
    
    # 构建完整HTML，复用部分应用样式以保持一致性
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ad Effect Intelligence Analysis Report</title>
        <style>
            body {{
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                line-height: 1.6;
                color: #1c2340;
                max_width: 900px;
                margin: 0 auto;
                padding: 40px;
                background-color: #f6f8fb;
            }}
            .report-container {{
                background: white;
                padding: 50px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            }}
            h1 {{ color: #1c2340; border-bottom: 3px solid #3b6ff2; padding-bottom: 15px; margin-top: 0; }}
            h2 {{ color: #2b48c4; margin-top: 35px; border-left: 5px solid #3b6ff2; padding-left: 15px; }}
            h3 {{ color: #3b6ff2; margin-top: 25px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }}
            th {{ background: #f0f7ff; color: #2b48c4; padding: 12px; text-align: left; border: 1px solid #e1e7f2; }}
            td {{ padding: 10px; border: 1px solid #e1e7f2; }}
            tr:nth-child(even) {{ background: #f9fbff; }}
            blockquote {{ 
                border-left: 4px solid #4ac8a8; 
                margin: 20px 0; 
                padding: 15px 20px; 
                background: #f0fdf9; 
                color: #2d5c52;
                border-radius: 0 8px 8px 0;
            }}
            code {{ background: #eff4ff; color: #3b6ff2; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; }}
            .footer {{ text-align: center; margin-top: 50px; color: #888; font-size: 0.9em; border-top: 1px solid #e1e7f2; padding-top: 20px; }}
            @media print {{
                body {{ background: white; padding: 0; }}
                .report-container {{ box-shadow: none; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div style="text-align: center; margin-bottom: 40px; color: #6b728e; font-size: 0.9em; text-transform: uppercase; letter-spacing: 2px;">
                Ad Effect Intelligence Platform
            </div>
            {html_body}
            <div class="footer">
                Generated by Ad Effect Intelligence · {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

def page_report_generator(state: dict):
    """分析报告生成页面"""
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero__body">
            <div class="page-hero__icon">📊</div>
            <div>
                <h1>分析报告</h1>
                <p>一键生成专业分析报告，支持预览和HTML导出</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查是否有足够的数据
    has_data = state.get('df') is not None
    has_model = state.get('model') is not None
    has_shap = state.get('shap_values') is not None
    
    # 数据状态提示
    col1, col2, col3 = st.columns(3)
    with col1:
        if has_data:
            st.success("✅ 数据已上传")
        else:
            st.warning("⚠️ 请先上传数据")
    with col2:
        if has_model:
            st.success("✅ 模型已训练")
        else:
            st.warning("⚠️ 请先训练模型")
    with col3:
        if has_shap:
            st.success("✅ SHAP已计算")
        else:
            st.info("ℹ️ SHAP分析可选")
    
    st.markdown("---")
    
    # 生成报告按钮
    generate_btn = st.button("🚀 生成分析报告", use_container_width=True, disabled=not has_data)
    
    if generate_btn or st.session_state.get('report_generated'):
        st.session_state['report_generated'] = True
        
        with st.spinner("正在生成报告..."):
            # 生成Markdown报告
            report_md = generate_report_markdown(state)
            st.session_state['report_markdown'] = report_md
            
            # 生成AI洞察
            if has_model:
                with st.spinner("正在生成AI智能洞察..."):
                    ai_insights = generate_ai_insights(state)
                    st.session_state['ai_insights'] = ai_insights
            else:
                ai_insights = ""
                st.session_state['ai_insights'] = ""
    
    # 显示报告预览
    if st.session_state.get('report_markdown'):
        st.markdown("### 📄 报告预览")
        
        # 使用容器显示Markdown内容
        with st.container():
            st.markdown(st.session_state['report_markdown'])
            
            # 显示AI洞察
            ai_insights = st.session_state.get('ai_insights', '')
            if ai_insights and not ai_insights.startswith('>'):
                st.markdown("## 四、AI智能洞察与建议")
                st.markdown(ai_insights)
            elif ai_insights:
                st.markdown("## 四、AI智能洞察与建议")
                st.markdown(ai_insights)
        
        st.markdown("---")
        
        # 导出按钮区域
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 重新生成", use_container_width=True):
                st.session_state['report_generated'] = False
                st.session_state['report_markdown'] = None
                st.session_state['ai_insights'] = None
                st.rerun()
        
        with col2:
            # HTML导出
            try:
                html_content = generate_html(
                    st.session_state['report_markdown'],
                    st.session_state.get('ai_insights', '')
                )
                filename_html = f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                st.download_button(
                    label="🌐 导出HTML报告",
                    data=html_content,
                    file_name=filename_html,
                    mime="text/html",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"HTML生成失败: {str(e)}")

