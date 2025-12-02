"""
前端展示建议：如何在 Streamlit 中展示预测不确定性

在 pages_module/optimization_page.py 中的优化结果展示部分添加以下代码：
"""

import streamlit as st
import plotly.graph_objects as go

def display_optimization_result_with_uncertainty(result):
    """
    展示优化结果和预测不确定性
    
    Args:
        result: optimize_ad_allocation_robust 返回的字典
    """
    
    # 1. 基本信息展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="预测值",
            value=f"{result['predicted_value']:.2f}",
            delta=f"{result['predicted_value'] - result['base_value']:.2f}"
        )
    
    with col2:
        st.metric(
            label="目标值",
            value=f"{result['target_value']:.2f}"
        )
    
    with col3:
        accuracy = (1 - abs(result['predicted_value'] - result['target_value']) / 
                   result['target_value']) * 100
        st.metric(
            label="目标达成度",
            value=f"{accuracy:.1f}%"
        )
    
    # 2. 新增：预测不确定性展示
    st.subheader("📊 预测可靠性分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**置信区间 (95%)**")
        st.write(f"下界: {result['prediction_lower_ci']:.2f}")
        st.write(f"预测: {result['predicted_value']:.2f}")
        st.write(f"上界: {result['prediction_upper_ci']:.2f}")
        
        # 计算置信区间宽度
        ci_width = result['prediction_upper_ci'] - result['prediction_lower_ci']
        st.write(f"区间宽度: ±{ci_width/2:.2f}")
    
    with col2:
        st.write("**不确定性指标**")
        st.write(f"标准差: {result['prediction_std']:.4f}")
        
        # 相对不确定性（百分比）
        relative_uncertainty = (result['prediction_std'] / 
                               abs(result['predicted_value']) * 100)
        st.write(f"相对不确定性: {relative_uncertainty:.2f}%")
        
        # 可靠性等级
        if relative_uncertainty < 1:
            reliability = "🟢 高"
        elif relative_uncertainty < 3:
            reliability = "🟡 中"
        else:
            reliability = "🔴 低"
        st.write(f"可靠性等级: {reliability}")
    
    # 3. 置信区间可视化
    st.subheader("置信区间可视化")
    
    fig = go.Figure()
    
    # 添加置信区间（误差条）
    fig.add_trace(go.Scatter(
        x=[result['predicted_value']],
        y=['预测值'],
        error_x=dict(
            type='data',
            symmetric=False,
            array=[result['prediction_upper_ci'] - result['predicted_value']],
            arrayminus=[result['predicted_value'] - result['prediction_lower_ci']],
            color='rgba(0, 100, 200, 0.3)',
            thickness=3,
            width=8
        ),
        mode='markers',
        marker=dict(size=12, color='blue'),
        name='预测值 (95% CI)'
    ))
    
    # 添加目标值参考线
    fig.add_vline(
        x=result['target_value'], 
        line_dash="dash", 
        line_color="red",
        annotation_text="目标值"
    )
    
    # 添加基准值参考线
    fig.add_vline(
        x=result['base_value'],
        line_dash="dot",
        line_color="gray",
        annotation_text="基准值"
    )
    
    fig.update_layout(
        title="预测值置信区间",
        xaxis_title="值",
        yaxis_title="",
        showlegend=True,
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. 风险提示
    if relative_uncertainty > 3:
        st.warning(
            "⚠️ 预测不确定性较高，建议：\n"
            "- 增加训练数据\n"
            "- 检查特征质量\n"
            "- 考虑使用更保守的策略"
        )
    elif relative_uncertainty < 1:
        st.success("✅ 预测可靠性高，可以放心采纳建议")
    
    # 5. 详细信息（可折叠）
    with st.expander("📋 查看详细信息"):
        st.write("**优化方法:**", result['method_used'])
        st.write("**鲁棒性级别:**", result['robustness_level'])
        st.write("**预算变化:**", f"{result['budget_change']:.2f}")
        st.write("**效率增益:**", f"{result['efficiency_gain']:.4f}")
        
        st.write("\n**约束状态:**")
        for i, status in enumerate(result['constraint_status']):
            if status != '正常':
                st.write(f"- 特征 {i}: {status}")


# 使用示例
if __name__ == "__main__":
    # 模拟优化结果
    mock_result = {
        'suggested_allocation': [50, 30, 20, 10, 5],
        'predicted_value': 44.17,
        'target_value': 53.00,
        'base_value': 44.17,
        'constraint_status': ['正常'] * 5,
        'budget_change': 0.0,
        'efficiency_gain': 0.0,
        'method_used': 'adaptive',
        'robustness_level': 'medium',
        'prediction_std': 0.4025,
        'prediction_lower_ci': 43.17,
        'prediction_upper_ci': 44.19,
        'confidence_level': 0.95
    }
    
    display_optimization_result_with_uncertainty(mock_result)
