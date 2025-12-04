"""
AI智能助手页面
提供基于LLM的智能问答功能，可以解读训练结果、优化建议等
"""

import streamlit as st
from llm_helper import LLMAssistant, SYSTEM_PROMPTS

def render():
    """渲染AI助手页面"""

    # 初始化会话状态
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "llm_assistant" not in st.session_state:
        st.session_state.llm_assistant = LLMAssistant()
    
    # 侧边栏配置 - 去掉空白卡片
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # 主界面 - 显示对话历史 - 去掉空白卡片
    st.subheader("💬 对话记录")
    if not st.session_state.chat_history:
        st.info("👋 你好！我是AI助手，可以帮你解读模型训练结果、优化建议、SHAP分析等。有什么问题尽管问我！")
    else:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
    
    st.markdown("---")
    
    # 输入区域 - 去掉空白卡片
    st.subheader("✍️ 提问")

    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_area(
            "输入你的问题",
            placeholder="例如: 当前模型的性能如何？有哪些可以优化的地方？",
            height=100,
            key="user_input"
        )

    with col2:
        st.write("")
        st.write("")
        send_button = st.button("🚀 发送", use_container_width=True, type="primary")
    
    # 快捷问题
    st.markdown("#### 💡 快捷问题")
    quick_questions = {
        "模型性能": "请分析当前模型的整体性能表现，包括准确率、AUC等指标",
        "特征重要性": "哪些特征对模型预测影响最大？请详细解读",
        "优化建议": "基于当前结果，你有什么具体的优化建议？",
        "投放策略": "根据分析结果，给出实际的广告投放策略建议"
    }
    
    cols = st.columns(len(quick_questions))
    for col, (label, question) in zip(cols, quick_questions.items()):
        if col.button(label, use_container_width=True):
            user_input = question
            send_button = True

    # 处理发送消息
    if send_button and user_input:
        with st.spinner("🤔 AI正在思考..."):
            # 构建上下文（默认启用）
            context = _extract_context()
            
            # 根据上下文和问题自动选择最佳模式
            selected_mode = _auto_select_mode(user_input, context)
            
            # 调用LLM
            result = st.session_state.llm_assistant.chat(
                user_message=user_input,
                context=context,
                system_prompt=SYSTEM_PROMPTS[selected_mode]
            )
            
            if result["success"]:
                # 添加到对话历史
                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": result["content"]
                })
                
                # 显示token使用情况
                if "usage" in result:
                    usage = result["usage"]
                    st.caption(f"📊 Token使用: {usage.get('total_tokens', 'N/A')} (输入: {usage.get('prompt_tokens', 'N/A')}, 输出: {usage.get('completion_tokens', 'N/A')})")
                
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

def _auto_select_mode(user_message: str, context: str) -> str:
    """
    根据用户问题和上下文自动选择最合适的助手模式
    
    Args:
        user_message: 用户问题
        context: 上下文信息
        
    Returns:
        模式名称 ('general', 'training', 'optimization', 'visualization')
    """
    message_lower = user_message.lower()
    context_lower = context.lower() if context else ""
    
    # 关键词匹配优先级
    # 1. 优化分析相关
    optimization_keywords = ['优化', '反推', '调整', '提升', '改进', '投放策略', '预算', '出价', 
                            'optimization', 'optimize', 'improve', 'budget', 'bid']
    if any(kw in message_lower for kw in optimization_keywords) and '反推优化' in context_lower:
        return 'optimization'
    
    # 2. SHAP可视化相关
    visualization_keywords = ['shap', '可视化', '图表', '特征贡献', '解释', '解读', 
                             'visualization', 'interpret', 'explain', '依赖图', '力图']
    if any(kw in message_lower for kw in visualization_keywords) and 'shap' in context_lower:
        return 'visualization'
    
    # 3. 模型训练相关
    training_keywords = ['模型', '训练', '性能', 'r²', 'rmse', 'auc', '准确率', '过拟合', 
                        '欠拟合', '特征重要性', 'model', 'training', 'accuracy', 'overfitting']
    if any(kw in message_lower for kw in training_keywords) and '模型训练' in context_lower:
        return 'training'
    
    # 4. 根据上下文自动判断（没有明确关键词时）
    if '反推优化' in context_lower:
        return 'optimization'
    elif 'shap' in context_lower:
        return 'visualization'
    elif '模型训练' in context_lower or '已加载模型' in context_lower:
        return 'training'
    
    # 5. 默认使用通用模式
    return 'general'

def _extract_context() -> str:
    """
    从session_state中提取上下文信息
    
    Returns:
        格式化的上下文字符串
    """
    context_parts = ["# 当前工作区数据上下文\n"]
    
    # 注意：应用使用 st.session_state.state 作为主存储
    state = st.session_state.get('state', {})
    
    # 提取数据集基本信息（优先显示，让AI了解数据规模）
    if "df" in state:
        df = state['df']
        context_parts.append("## 数据集概况")
        context_parts.append(f"- 数据文件: {state.get('filename', '未知')}")
        context_parts.append(f"- 总样本数: {len(df)} 行")
        context_parts.append(f"- 总特征数: {len(df.columns)} 列")
        context_parts.append(f"- 列名: {', '.join(df.columns.tolist()[:15])}{'...' if len(df.columns) > 15 else ''}")
        context_parts.append("")
    
    # 提取模型训练指标
    if "metrics" in state:
        m = state['metrics']
        context_parts.append("## 模型训练结果")
        context_parts.append(f"- 训练集R²: {m.get('train_r2', 'N/A'):.4f}" if isinstance(m.get('train_r2'), (int, float)) else f"- 训练集R²: {m.get('train_r2', 'N/A')}")
        context_parts.append(f"- 测试集R²: {m.get('test_r2', 'N/A'):.4f}" if isinstance(m.get('test_r2'), (int, float)) else f"- 测试集R²: {m.get('test_r2', 'N/A')}")
        context_parts.append(f"- 训练集RMSE: {m.get('train_rmse', 'N/A'):.4f}" if isinstance(m.get('train_rmse'), (int, float)) else f"- 训练集RMSE: {m.get('train_rmse', 'N/A')}")
        context_parts.append(f"- 测试集RMSE: {m.get('test_rmse', 'N/A'):.4f}" if isinstance(m.get('test_rmse'), (int, float)) else f"- 测试集RMSE: {m.get('test_rmse', 'N/A')}")
        context_parts.append(f"- 交叉验证R²: {m.get('cv_mean', 'N/A'):.4f} ± {m.get('cv_std', 'N/A'):.4f}" if isinstance(m.get('cv_mean'), (int, float)) else f"- 交叉验证R²: {m.get('cv_mean', 'N/A')}")
        context_parts.append("")
    
    # 提取模型对象和配置
    if "model" in state:
        model = state['model']
        context_parts.append("## 已加载模型")
        context_parts.append(f"- 模型类型: {type(model).__name__}")
        context_parts.append("- 状态: 已训练，可用于预测和分析")
        
        # 提取特征重要性（如果模型支持）
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if "selected_features" in state and len(state["selected_features"]) == len(importances):
                    features = state["selected_features"]
                    feature_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                    context_parts.append("\n前5个重要特征:")
                    for feat, imp in feature_imp[:5]:
                        context_parts.append(f"  - {feat}: {imp:.4f}")
        except Exception:
            pass
        context_parts.append("")
    
    # 提取反推优化结果
    if "reverse_results" in state:
        context_parts.append("## 反推优化结果")
        results = state['reverse_results']
        y_base = state.get('reverse_y_base', 'N/A')
        target = state.get('reverse_target_gmv', 'N/A')
        
        context_parts.append(f"- 基准预测值: {y_base:.4f}" if isinstance(y_base, (int, float)) else f"- 基准预测值: {y_base}")
        context_parts.append(f"- 目标值: {target:.4f}" if isinstance(target, (int, float)) else f"- 目标值: {target}")
        
        if results and 'best_x' in results:
            best = results['best_x']
            best_pred = results.get('best_y_pred', 'N/A')
            context_parts.append(f"- 优化后预测值: {best_pred:.4f}" if isinstance(best_pred, (int, float)) else f"- 优化后预测值: {best_pred}")
            
            # 显示特征调整（如果有base_x）
            if 'base_x' in state and "selected_features" in state:
                base_x = state['base_x']
                features = state['selected_features']
                context_parts.append("\n前5个特征调整建议:")
                changes = []
                for i, feat in enumerate(features):
                    if i < len(base_x) and i < len(best):
                        old_val = base_x[i]
                        new_val = best[i]
                        change_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                        changes.append((feat, old_val, new_val, abs(change_pct)))
                
                # 按变化幅度排序
                changes.sort(key=lambda x: x[3], reverse=True)
                for feat, old_val, new_val, _ in changes[:5]:
                    context_parts.append(f"  - {feat}: {old_val:.4f} → {new_val:.4f}")
        context_parts.append("")
    
    # 提取SHAP分析信息
    if "shap_values" in state:
        context_parts.append("## SHAP可解释性分析")
        context_parts.append("- 已完成SHAP值计算")
        context_parts.append("- 可用于分析特征贡献度和交互效应")
        context_parts.append("")
    
    # 提取训练配置信息
    if "target_var" in state:
        context_parts.append("## 模型配置")
        context_parts.append(f"- 目标变量: {state['target_var']}")
        if "selected_features" in state:
            features = state['selected_features']
            context_parts.append(f"- 已选特征数: {len(features)}")
            context_parts.append(f"- 特征列表: {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")
        context_parts.append("")
    
    # 如果没有任何数据，返回提示
    if len(context_parts) <= 1:  # 只有标题
        return """当前工作区暂无数据。建议操作流程：
1. 前往"数据上传 & 预览"页面上传广告数据
2. 前往"模型训练 & 评估"页面训练模型
3. 返回AI助手询问分析问题"""
    
    return "\n".join(context_parts)

def render_ai_sidebar_widget(
    page_type: str,
    specific_context: dict = None
):
    """
    在任意页面的侧边栏渲染AI助手小组件
    
    Args:
        page_type: 页面类型 ('training', 'optimization', 'visualization', 'general')
        specific_context: 特定页面的上下文信息
    """
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 AI助手")
        
        if "llm_assistant" not in st.session_state:
            st.session_state.llm_assistant = LLMAssistant()
        
        user_question = st.text_input(
            "快速提问",
            placeholder="询问当前页面相关问题...",
            key=f"ai_question_{page_type}"
        )
        
        if st.button("提问", key=f"ai_ask_{page_type}", use_container_width=True):
            if user_question:
                with st.spinner("思考中..."):
                    # 构建上下文（合并页面特定上下文和全局上下文）
                    context_parts = []
                    if specific_context:
                        context_parts.append("## 当前页面信息")
                        for k, v in specific_context.items():
                            context_parts.append(f"- {k}: {v}")
                    # 添加全局上下文
                    global_context = _extract_context()
                    if global_context:
                        context_parts.append("\n" + global_context)
                    context = "\n".join(context_parts) if context_parts else None
                    
                    # 调用LLM
                    result = st.session_state.llm_assistant.chat(
                        user_message=user_question,
                        context=context,
                        system_prompt=SYSTEM_PROMPTS.get(page_type, SYSTEM_PROMPTS["general"])
                    )
                    
                    if result["success"]:
                        st.success("💡 AI回答:")
                        st.write(result["content"])
                    else:
                        st.error(result["error"])
            else:
                st.warning("请先输入问题")
        
        st.markdown("或者前往 [AI助手页面](#) 进行深度对话")

if __name__ == "__main__":
    render()
