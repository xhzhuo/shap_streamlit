"""
开发文档页面
介绍反推页面用到的各个数据指标的含义
"""

import streamlit as st


def page_dev_docs():
    """开发文档页面"""
    st.header("📘 数据指标与数学概念说明")

    # ------------------------
    # 一、模型评估指标
    # ------------------------
    st.markdown("## 一、模型评估指标")

    st.subheader("1️⃣ R²（决定系数）")
    st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}")
    st.markdown("""
    - **含义**：衡量模型对目标变量方差的解释能力。  
    - R² 越接近 1，说明模型越能准确捕捉输入特征与目标结果的关系。  
    - **解释：**  
      - R² = 1 → 完全拟合  
      - R² = 0 → 无解释力  
      - R² < 0 → 比直接用平均值预测更差。
    """)

    st.subheader("2️⃣ NRMSE（归一化均方根误差）")
    st.markdown("**NRMSE 是对 RMSE 的归一化形式，用于不同量级或数据集之间的可比性。**")

    st.markdown("RMSE 定义：")
    st.latex(r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}")

    st.markdown("常见 NRMSE 计算方式：")
    st.markdown("- 按范围归一化（推荐）")
    st.latex(r"\mathrm{NRMSE}_{\mathrm{range}} = \frac{\mathrm{RMSE}}{y_{\max} - y_{\min}}")
    st.markdown("- 按均值归一化")
    st.latex(r"\mathrm{NRMSE}_{\mathrm{mean}} = \frac{\mathrm{RMSE}}{\bar{y}}")
    st.markdown("- 按标准差归一化")
    st.latex(r"\mathrm{NRMSE}_{\mathrm{std}} = \frac{\mathrm{RMSE}}{\sigma_y}")

    st.markdown("""
    - **解释：** NRMSE 越小越好，表示误差在目标取值范围内所占比例越低。  
    - 例如 NRMSE = 0.05 表示模型误差约占目标范围的 5%。  
    """)

    st.subheader("3️⃣ 交叉验证 R²")
    st.markdown("""
    - 将训练集划分为多折（如 4 折），多次训练与验证得到平均 R²。  
    - 用于衡量模型的稳定性与泛化能力；若交叉验证 R² 与测试集 R² 接近，说明模型表现稳定。
    """)

    # ------------------------
    # 二、SHAP 特征贡献分析
    # ------------------------
    st.markdown("---")
    st.markdown("## 二、SHAP 特征贡献分析")

    st.subheader("1️⃣ 原理概述（Shapley 值）")
    st.latex(r"\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,[f(S \cup \{i\}) - f(S)]")
    st.markdown("其中：")
    st.latex(r"f(S)\quad\text{表示仅使用特征集合 }S\text{ 时模型的预测结果}")
    st.latex(r"\phi_i\quad\text{表示特征 }i\text{ 的平均边际贡献}")

    st.subheader("2️⃣ 业务含义")
    st.markdown("""
    - SHAP > 0 ：特征推动预测结果上升（正向影响）  
    - SHAP < 0 ：特征抑制预测结果（负向影响）  
    - 平均绝对 SHAP 值越大，表示特征对模型输出的影响越强。
    """)

    st.subheader("3️⃣ 可视化")
    st.markdown("""
    - **Summary Plot**：展示各特征在样本层面的贡献方向与强度分布。  
    - **饼图**：展示特征对整体预测结果的平均贡献占比。  
    - **热力图**：显示特征间相关性，识别多重共线性。
    """)

    # ------------------------
    # 三、反推与预算优化
    # ------------------------
    st.markdown("---")
    st.markdown("## 三、反推与预算优化（核心算法部分）")

    st.subheader("1️⃣ 问题定义")
    st.latex(r"y = f(x_1, x_2, \dots, x_n)")
    st.latex(r"f(x') \approx y_{\mathrm{target}}")
    st.latex(r"L_i \le x'_i \le U_i")
    st.latex(r"\sum_{i=1}^n x'_i = B")

    st.subheader("2️⃣ 敏感度估计（Sensitivity Estimation）")
    st.markdown("敏感度衡量单个特征变化对预测结果的影响，可用有限差分法计算：")
    st.latex(r"s_i = \frac{f(x+\epsilon e_i) - f(x)}{\epsilon}")
    st.markdown("其中：")
    st.latex(r"e_i\quad\text{为仅在第 }i\text{ 个分量为 1 的单位向量}")
    st.latex(r"\epsilon\quad\text{为微小扰动（通常取特征范围的 0.1\%–2\%）}")
    st.markdown("通过多次扰动并取中位数可降低噪声，得到稳定的敏感度。")

    st.subheader("3️⃣ 线性分配法（Linear Allocation）")
    st.latex(r"\Delta x_i = \frac{w_i \cdot (y_{\mathrm{target}} - y_{\mathrm{base}})}{s_i}")
    st.latex(r"x'_i = x_i + \Delta x_i")
    st.markdown("""
    - \(w_i\)：特征 SHAP 权重  
    - \(s_i\)：特征敏感度  

    **优点：** 简单、快速、可解释性强。  
    **缺点：** 不考虑预算或上下限约束，结果可能不满足实际条件。
    """)

    st.subheader("4️⃣ 预算约束优化（Budget-Constrained Optimization）")
    st.latex(r"\min_{x} |f(x) - y_{\mathrm{target}}|")
    st.latex(r"\text{s.t. } \sum_{i=1}^n x_i = B, \quad L_i \le x_i \le U_i")
    st.markdown("""
    - 通过 **SLSQP（顺序二次规划）** 求解；  
    - 确保总投放预算固定在 B；  
    - 若优化失败，则回退至线性分配方案。
    """)

    st.subheader("5️⃣ 全约束优化（Fully Constrained Optimization）")
    st.latex(r"\min_{x} \Big( |f(x) - y_{\mathrm{target}}| + \lambda \sum_{i=1}^n (x_i - x_i^{\mathrm{base}})^2 \Big)")
    st.latex(r"\lambda\text{ 为平滑惩罚系数}")
    st.markdown(f"""
    - 兼顾目标达成与稳定性。  
    - λ 越大 → 调整更保守，变化幅度更小；  
      λ 越小 → 调整更激进，更接近目标。
    """)

    # ------------------------
    # ✳️ 约束方案比较与选型建议
    # ------------------------
    st.subheader("6️⃣ 不同约束方案的比较与选择建议")

    st.markdown("""
    | 约束方案 | 特点 | 优势 | 适用场景 |
    |:-----------|:-----------------|:----------------|:----------------|
    | **线性分配法** | 快速、无约束 | 计算速度快、可解释性强 | 初步评估、试算、趋势预估 |
    | **预算约束优化** | 固定总预算 | 保证资源总量不变、结果更可控 | 明确预算上限的场景（如广告总花费固定） |
    | **全约束优化** | 加入平滑项 | 控制单渠道波动、提升稳定性 | 实际执行阶段或逐步调优阶段 |
    """)

    st.markdown("""
    🔹 **推荐策略：**
    - 若只需快速估算目标可达性 → 选 **线性分配法**；  
    - 若预算固定，需合理分配各渠道 → 选 **预算约束优化**；  
    - 若要防止投放剧烈变化、确保执行平滑 → 选 **全约束优化**。  
    """)

    st.info("💡 实际系统中常采用组合策略：先线性分配生成初解，再用全约束优化微调。")

    # ------------------------
    # 输出结果字段详细说明
    # ------------------------
    st.markdown("---")
    st.subheader("7️⃣ 输出结果字段详细说明")

    st.markdown("""
    #### SHAP 权重
    - **定义**：特征对模型预测结果的重要性度量。  
    - **计算方式：**
    """)
    st.latex(r"w_i = \frac{\mathrm{mean}(|\phi_i|)}{\sum_j \mathrm{mean}(|\phi_j|)}")
    st.markdown("""
    - **范围**：0–1，所有特征权重之和为 1。  
    - **用途**：决定特征在优化过程中的权重比例。
    """)

    st.markdown("""
    #### 敏感度（Sensitivity）
    - **定义**：特征值微小变化对预测结果的影响程度。  
    - **计算方式：**
    """)
    st.latex(r"s_i = \frac{f(x+\epsilon e_i) - f(x)}{\epsilon}")
    st.markdown("""
    - **用途**：指导优化时每个特征调整的幅度。  
    - 敏感度越大，特征调整的影响越显著。
    """)

    st.markdown("""
    #### 基准投放（Baseline Input）
    """)
    st.latex(r"x_i^{\mathrm{base}} = \mathrm{mean}(x_i)")
    st.markdown("- 优化前的基准值，用作优化起点。")

    st.markdown("""
    #### 建议投放（Optimized Input）
    """)
    st.latex(r"x_i^{\mathrm{opt}} = \arg\min_x |f(x) - y_{\mathrm{target}}|")
    st.markdown("- 优化算法得到的推荐值，用于业务参考。")

    st.markdown("""
    #### 增量（Delta）
    """)
    st.latex(r"\Delta x_i = x_i^{\mathrm{opt}} - x_i^{\mathrm{base}}")
    st.markdown("- 反映建议与基准的差值，直观显示调整幅度。")

    st.markdown("""
    #### 基准预测值（Baseline Prediction）
    """)
    st.latex(r"y_{\mathrm{base}} = f(x^{\mathrm{base}})")
    st.markdown("- 代表当前投放下模型预测的结果，作为对比基线。")

    st.markdown("""
    #### 优化后预测值（Optimized Prediction）
    """)
    st.latex(r"y_{\mathrm{opt}} = f(x^{\mathrm{opt}})")
    st.markdown("- 优化后模型预测结果，用于验证方案效果。")

    st.markdown("""
    #### 约束状态（Constraint Status）
    - **定义**：每个特征是否触及上下限或预算边界。  
    - **取值说明**：  
      - “正常”：未触及约束  
      - “触达上限”：到达最大限制  
      - “触达下限”：到达最小限制
    """)

    st.markdown("""
    #### 预算对比（Budget Summary）
    """)
    st.latex(r"B_{\mathrm{base}} = \sum_i x_i^{\mathrm{base}} \quad , \quad B_{\mathrm{opt}} = \sum_i x_i^{\mathrm{opt}}")
    st.markdown("- 比较优化前后总预算差异，验证是否满足预算约束。")

    st.info("📊 以上字段共同构成优化结果表的核心信息，用于指导广告投放预算分配与调整。")

    # ------------------------
    # 四、模型评分逻辑
    # ------------------------
    st.markdown("---")
    st.markdown("## 四、模型评分逻辑（evaluate_model_quality）")
    st.latex(r"\mathrm{Score} = \big(0.7R^2_{\mathrm{test}} + 0.2R^2_{\mathrm{cv}} + 0.1(1 - |R^2_{\mathrm{train}} - R^2_{\mathrm{test}}|)\big) \times 100")
    st.markdown("""
    | 分数区间 | 等级 | 建议 |
    |:----------:|:------:|:----------------|
    | ≥85 | 优秀 | 可直接用于预测 |
    | 70–85 | 良好 | 建议监控效果 |
    | 55–70 | 一般 | 需优化模型 |
    | <55 | 需改进 | 不建议用于决策 |
    """)

  