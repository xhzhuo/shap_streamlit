# shap_streamlit

一个基于 Streamlit 和 SHAP 的交互式机器学习模型解释工具。

## 功能特性

- 数据上传与预览 (CSV/Excel)
- 机器学习模型训练与评估
- SHAP 值计算与可视化分析
- 反向优化与预算分配建议

## 项目结构

```
shap_streamlit/
├── main.py                 # 主应用入口
├── config.py               # 配置文件
├── utils.py                # 工具函数
├── models.py               # 模型相关函数
├── optimization.py         # 优化算法
├── pages/                  # 各页面实现
│   ├── data_upload.py
│   ├── model_training.py
│   ├── visualization.py
│   └── optimization_page.py
├── requirements.txt        # 依赖包
└── .devcontainer/          # 开发环境配置
```

## 安装与运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run main.py
```

访问 http://localhost:8501 查看应用。

## 使用说明

1. **数据上传**：上传包含数值型特征的 CSV 或 Excel 文件
2. **模型训练**：选择目标变量和特征变量训练随机森林模型
3. **模型评估**：查看模型评分和各项指标
4. **可视化分析**：通过 SHAP 值分析特征重要性和影响方向
5. **预算优化**：基于模型预测进行反向优化和预算分配