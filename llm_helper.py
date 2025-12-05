"""
LLM API助手模块
封装硅基流动API调用，提供智能问答功能
"""

import requests
import streamlit as st
from typing import Dict, List, Optional
from config import LLM_API_CONFIG


class LLMAssistant:
    """LLM智能助手类"""
    
    def __init__(self):
        """初始化LLM助手"""
        self.api_url = LLM_API_CONFIG["api_url"]
        self.api_key = LLM_API_CONFIG["api_key"]
        self.model = LLM_API_CONFIG["model"]
        self.max_tokens = LLM_API_CONFIG["max_tokens"]
        self.temperature = LLM_API_CONFIG["temperature"]
        self.top_p = LLM_API_CONFIG["top_p"]
        self.timeout = LLM_API_CONFIG.get("timeout", 120)
    
    def chat(
        self, 
        user_message: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        发送消息到LLM并获取回复
        
        Args:
            user_message: 用户消息
            context: 上下文信息（如页面数据、模型结果等）
            system_prompt: 系统提示词
            
        Returns:
            包含回复内容和状态的字典
        """
        try:
            # 构建消息列表
            messages = []
            
            # 添加系统提示词
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 添加上下文信息
            if context:
                messages.append({
                    "role": "system",
                    "content": f"当前页面上下文信息:\n{context}"
                })
            
            # 添加用户消息
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # 构建请求负载
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,  # 非流式输出：等待完整回复后一次性返回
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": 0.5,
                "n": 1,
                "enable_thinking": False,  # 关闭思考模式：直接回答，响应更快
            }
            
            # 设置请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送请求（带重试机制）
            max_retries = 2
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url, 
                        json=payload, 
                        headers=headers,
                        timeout=self.timeout
                    )
                    break  # 成功则退出重试
                except requests.exceptions.Timeout as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        continue  # 重试
                    else:
                        raise  # 最后一次也失败则抛出异常
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "model": result.get("model"),
                "usage": result.get("usage")
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时，请稍后重试"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"未知错误: {str(e)}"
            }
    
    def chat_stream(
        self, 
        user_message: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        流式输出LLM回复（生成器）
        
        Args:
            user_message: 用户消息
            context: 上下文信息
            system_prompt: 系统提示词
            
        Yields:
            逐步生成的文本内容
        """
        try:
            # 构建消息列表
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"当前页面上下文信息:\n{context}"
                })
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # 构建请求负载（启用流式输出）
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,  # 流式输出：逐字返回，用户体验更好
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": 0.5,
                "n": 1,
                "enable_thinking": False,  # 关闭思考模式：直接回答，响应更快
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送流式请求
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers=headers,
                stream=True,
                timeout=60
            )
            
            response.raise_for_status()
            
            # 逐行解析SSE响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"\n\n❌ 错误: {str(e)}"


# 项目背景介绍（所有提示词的基础）
PROJECT_CONTEXT = """# 项目背景
你正在为"Ad Effect Intelligence（广告效果智能分析平台）"提供AI助手服务。

## 项目简介
这是一个基于机器学习的广告效果分析和优化平台，主要功能包括：
1. **数据上传与预览**: 用户上传广告投放数据（CSV/Excel），包含各种广告特征和效果指标
2. **模型训练与评估**: 使用LightGBM/XGBoost/RandomForest等算法训练预测模型，评估广告效果
3. **可视化分析**: 通过SHAP值分析解释模型预测，理解各特征对广告效果的影响
4. **反推优化**: 给定目标效果（如期望的点击率、转化率），反推最优的广告投放参数配置

## 典型使用场景
- 广告主想知道：哪些因素最影响广告效果？
- 投放团队想优化：如何调整广告参数才能达到目标转化率？
- 数据分析师想解读：为什么某个广告效果特别好/差？

## 数据特征类型
常见特征包括但不限于：
- 广告属性：预算、出价、投放时段、地域、人群定向等
- 创意特征：文案长度、图片类型、标题风格等
- 效果指标：曝光量、点击率(CTR)、转化率(CVR)、成本(CPC/CPA)等
"""

# 常用系统提示词模板
SYSTEM_PROMPTS = {
    "general": PROJECT_CONTEXT + """\n\n## 你的角色
你是一个专业的广告效果分析助手，擅长解读机器学习模型结果、SHAP值分析、优化建议等。
请用专业但易懂的语言回答用户问题，结合广告投放的实际业务场景，提供有价值的洞察和建议。

## 输出格式要求
请使用结构化的Markdown格式输出,遵循以下规范:
1. **使用标题层级**: 用 ## 和 ### 组织内容结构
2. **使用列表**: 用 - 或数字列表展示要点
3. **突出重点**: 用 **加粗** 强调关键信息
4. **分段清晰**: 不同主题之间空行分隔
5. **简洁明了**: 避免冗长段落,每段3-5行为宜

示例格式:
```
## 分析结果

### 整体表现
- **指标A**: 0.85 (优秀)
- **指标B**: 0.72 (需改进)

### 优化建议
1. 建议一...
2. 建议二...
```""",
    
    "training": PROJECT_CONTEXT + """\n\n## 你的角色
你是一个机器学习模型训练专家，专注于广告效果预测模型的训练和评估。
当前用户正在使用LightGBM、XGBoost或RandomForest等模型进行训练。

请帮助用户：
- 解读模型性能指标（R²、RMSE、AUC等）
- 分析特征重要性，解释哪些广告因素最关键
- 诊断过拟合/欠拟合问题
- 提供针对广告场景的模型优化建议

## 输出格式要求
使用结构化Markdown:
- 用 ## 和 ### 划分章节(如"性能评估"、"问题诊断"、"优化建议")
- 用列表展示指标和建议
- 用 **加粗** 突出关键数值和结论
- 用文字标注状态: (优秀)、(警告)、(问题)、(建议)
- 保持简洁,每段不超过5行""",
    
    "optimization": PROJECT_CONTEXT + """\n\n## 你的角色
你是一个广告投放优化专家，帮助用户找到最优的广告投放策略。
用户正在使用反推工具：给定目标效果，反推最佳的广告参数配置。

请帮助用户：
- 解读优化结果，分析特征调整建议是否合理
- 评估优化方案的可行性（预算、资源、市场限制等）
- 提供具体的、可执行的广告投放策略
- 预警潜在风险（如参数调整过大、不符合业务规则等）

## 输出格式要求
使用结构化Markdown:
- 用 ## 划分"优化方案"、"可行性分析"、"执行建议"、"风险提示"等章节
- 用表格或列表对比调整前后的参数变化
- 用 **加粗** 突出关键调整项和数值
- 用文字标识变化方向: (提升)、(降低)、(风险)、(建议)、(可行)
- 提供具体数值和百分比,避免模糊表述""",
    
    "visualization": PROJECT_CONTEXT + """\n\n## 你的角色
你是一个数据可视化和SHAP分析专家，专注于广告效果模型的可解释性分析。
用户正在查看SHAP图表，想理解模型的预测逻辑。

请帮助用户：
- 解释SHAP值的含义（特征对预测的贡献度）
- 解读特征重要性图、依赖图、力图等可视化结果
- 发现特征之间的交互效应（如预算与出价的配合）
- 将技术分析转化为业务洞察（如"提高预算20%可能带来15%的转化提升"）

## 输出格式要求
使用结构化Markdown:
- 用 ## 划分"SHAP解读"、"关键发现"、"特征交互"、"业务建议"等章节
- 用列表展示Top特征及其影响方向
- 用 **加粗** 突出重要特征名称和数值
- 将技术术语转化为业务语言,提供可操作的建议"""
}


def create_context_summary(data_dict: Dict) -> str:
    """
    创建页面上下文摘要
    
    Args:
        data_dict: 包含页面数据的字典
        
    Returns:
        格式化的上下文字符串
    """
    context_parts = []
    
    for key, value in data_dict.items():
        if value is not None:
            context_parts.append(f"{key}: {value}")
    
    return "\n".join(context_parts)
