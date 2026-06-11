# RED Search Index 归因预测建模方案说明

## 1. 项目目标

本方案用于基于 `RED SI归因_Data_Input_Wyeth.xlsx` 中的 **`Data 原始`** sheet，模仿前面截图里的小红书 Search Index Prediction 方法，搭建一套可以复现的品牌搜索指数归因与预测流程。

目标是预测：

```text
RED Search Index (Mil.)
```

并输出：

1. 历史拟合结果；
2. 未来预测 Base；
3. 乐观预测 Optimistic；
4. 悲观预测 Pessimistic；
5. 模型结果 Excel 与可复用 Python 脚本。

---

## 2. 输入数据来源

本方案只使用用户提供文件里的 `Data 原始` sheet。

### 2.1 目标变量

| 字段 | 说明 |
|---|---|
| `RED Search Index (Mil.)` | 品牌 RED 搜索指数，单位 Mil.，作为模型预测目标 Y |

### 2.2 核心建模输入字段

| 字段 | 处理方式 | 进入模型 |
|---|---|---|
| `RED SEM SPD (Mil.)` | Geometric Adstock，λ=0.30 | 是 |
| `RED Feeds SPD (Mil.)` | Geometric Adstock，λ=0.50 | 是 |
| `KOC投资` | 先换算为 Mil.，再 Geometric Adstock，λ=0.50 | 是 |
| `RED Branding SPD (Mil.)` | Geometric Adstock，λ=0.82 | 是 |
| `NSR` | 直接进入模型 | 是 |
| `Industry Total Search Index (Mil.)（红书奶粉行业十大竞品主搜）` | 直接进入模型 | 是 |

### 2.3 保留但不直接进入第一版模型的字段

`Data 原始` 中还有很多可用于诊断或后续模型升级的字段，例如：

```text
RED SEM IMP
RED SEM CLICK
RED SEM 回搜量
RED Feeds IMP
RED Feeds CLICK
RED Feeds 回搜量
RED Feeds Frequency（IMP）
KOL笔记数
KOC笔记数
KOC阅读量
KOC曝光
Branding IMP
Branding CLICK
```

第一版没有把这些字段全部放进模型，原因是：

1. 当前实际目标变量只有 52 周；
2. 变量过多容易过拟合；
3. `回搜量` 这类指标可能已经非常接近结果变量，容易造成解释目标自身的问题；
4. IMP、CLICK、SPD 之间高度相关，全部放入会造成系数不稳定。

---

## 3. 数据处理流程

### 3.1 解析周度时间

原始字段 `Week/Month` 的格式类似：

```text
2025/1/6 - 2025/1/12
```

处理逻辑：

```python
week_start = 左侧日期
week_end = week_start + 6 天
```

### 3.2 数值字段清洗

所有核心字段统一转为数值型：

```python
pd.to_numeric(..., errors="coerce")
```

其中投放金额缺失值按 0 处理，目标变量缺失值保留为 `NaN`，用于区分历史训练期和未来预测期。

### 3.3 KOC 投资单位处理

`KOC投资` 在 `Data 原始` 中更像是 RMB 级别原始值，因此脚本中将其除以 `1,000,000`，转为 Mil.：

```python
koc_spend_mil = KOC投资 / 1_000_000
```

这样可以和 `RED SEM SPD (Mil.)`、`RED Feeds SPD (Mil.)`、`RED Branding SPD (Mil.)` 保持单位一致。

---

## 4. 特征工程：Geometric Adstock

模型模仿截图中的 `Geometric Adstock + Ridge Regression` 逻辑。

### 4.1 Adstock 公式

```text
adstock_t = spend_t + λ × adstock_{t-1}
```

含义是：广告投放不仅影响当周搜索，也会对后续几周产生残留影响。

### 4.2 本方案使用的 λ

| 变量 | λ | 业务含义 |
|---|---:|---|
| SEM | 0.30 | 搜索投放影响较短期 |
| Feeds | 0.50 | 信息流有中短期残留 |
| KOC | 0.50 | 内容种草影响有一定延续 |
| Branding | 0.82 | 品牌资源长尾影响更强 |

脚本生成的模型字段为：

```text
sem_adstock_l030
feeds_adstock_l050
koc_adstock_l050
branding_adstock_l082
```

---

## 5. 建模方法

### 5.1 训练 / 预测切分

切分逻辑：

```text
RED Search Index (Mil.) 有值的周 = 训练期
RED Search Index (Mil.) 为空的周 = 预测期
```

在当前文件中：

```text
训练样本：52 周
预测样本：11 周
```

### 5.2 模型公式

模型目标是：

```text
RED Search Index
= f(
    SEM adstock,
    Feeds adstock,
    KOC adstock,
    Branding adstock,
    Industry SI,
    NSR
  )
```

### 5.3 为什么使用 Ridge

使用 Ridge Regression 的原因：

1. 媒体变量之间容易共线；
2. 样本量不大，普通线性回归容易过拟合；
3. Ridge 可以通过正则化让系数更稳定；
4. 模型结构清晰，适合给业务解释。

### 5.4 为什么 final demo 用 Positive Ridge

脚本中同时输出两套模型诊断：

1. `Diagnostic only: standard RidgeCV`：普通 RidgeCV，拟合可能略好，但可能出现不符合业务直觉的负系数；
2. `Final demo: Positive Ridge + fixed geometric adstock`：强制标准化系数非负，业务解释更稳。

使用 Positive Ridge 的原因是：在营销归因场景中，SEM / Feeds / KOC / Branding 这类投放压力理论上不应长期表现为负向贡献。如果普通 Ridge 出现负系数，通常说明变量共线、样本不足或口径混杂。

---

## 6. 模型结果

当前 demo 输出的 final 模型指标为：

| 指标 | 数值 |
|---|---:|
| Training rows | 52 |
| Forecast rows | 11 |
| R² | 0.3815 |
| MAPE | 9.69% |
| RMSE | 0.0280M |
| Residual σ | 0.0283M |
| Scenario half width, 0.9σ | 0.0255M |

解释：

- R² 约 0.38，说明模型能解释一部分搜索波动，但仍有不少波动来自未进入模型的内容爆发、节点、竞品事件或平台机制变化；
- MAPE 约 9.69%，作为第一版周度搜索指数模型，误差水平可用于 demo 和方法验证；
- RMSE 约 0.028M，代表平均预测误差量级约 0.028 百万搜索指数。

---

## 7. 情景预测方法

模仿截图中的情景定义：

```text
Base forecast = 模型直接输出
Optimistic forecast = Base + 0.9 × residual σ
Pessimistic forecast = Base - 0.9 × residual σ
```

本次模型计算得到：

```text
Residual σ = 0.0283M
0.9σ = 0.0255M
```

因此每个预测周的情景区间为：

```text
悲观下沿 = Base - 0.0255M
乐观上沿 = Base + 0.0255M
```

这不是严格统计意义上的置信区间，而是更接近业务情景区间：

- 乐观：内容/投放/行业表现好于模型平均误差；
- 悲观：内容/投放/行业表现弱于模型平均误差；
- 基准：按照当前输入变量和模型关系直接输出。

---

## 8. 输出文件说明

运行脚本后会生成：

### 8.1 Excel 结果文件

```text
red_si_attribution_model_demo_output.xlsx
```

包含 sheet：

| Sheet | 内容 |
|---|---|
| `01_Cleaned_Raw` | 清洗后的原始数据 |
| `02_Model_Output` | 模型输入、adstock 字段、拟合值、预测值、情景值 |
| `03_Model_Summary` | 模型指标 |
| `04_Coefficients` | Positive Ridge 和普通 Ridge 的标准化系数 |
| `05_Dashboard_Table` | 类似 dashboard 的汇总对比 |

### 8.2 静态图

```text
red_si_attribution_forecast_demo_chart.png
```

图中包含：

- 品牌实际 SI；
- 历史 Base fit；
- Base forecast；
- Optimistic forecast；
- Pessimistic forecast；
- Industry SI 右轴参考线。

### 8.3 交互式图

```text
red_si_attribution_forecast_demo_chart.html
```

如果本地环境安装 `plotly`，会生成可交互 HTML 图表。

---

## 9. 当前方案的限制

### 9.1 样本量有限

当前只有 52 周实际目标变量。若加入过多字段，容易造成过拟合和系数不稳定。

### 9.2 没有显式新品 / 大促字段

`Data 原始` 中 `Is_Big_Promo` 为空，`BHT DI` 也为空，因此本 demo 没有加入大促和 BHT 变量。后续如果补齐，可以加入：

```text
Is_Big_Promo
BHT DI
新品 launch flag
新品状态
```

### 9.3 Industry SI 的角色需要业务确认

当前把 `Industry Total Search Index` 作为环境变量直接进入模型。也可以在正式版中改为：

1. 只用于图表展示；
2. 用于品牌/品类 ratio 校准；
3. 进入模型作为外部需求变量。

具体取决于业务解释目标。

### 9.4 KOC 变量口径需要确认

`KOC投资` 在 `Data 原始` 中不是 Mil. 字段名，因此脚本假设其为 RMB 并换算成 Mil.。如果原始数据本身已经是 Mil.，需要删除 `/ 1_000_000`。

---

## 10. 后续正式版建议

### 10.1 保持第一版模型简洁

建议第一版正式模型先保留：

```text
SEM spend
Feeds spend
KOC spend
Branding spend
Industry SI
NSR
```

不要一开始加入所有 IMP / CLICK / 回搜量 / 笔记数。

### 10.2 单独做变量筛选测试

逐个测试：

```text
SEM IMP
SEM CLICK
SEM 回搜量
Feeds IMP
Feeds CLICK
Feeds 回搜量
KOL笔记数
KOC笔记数
KOC阅读量
KOC曝光
Branding IMP
Branding CLICK
```

判断标准：

1. 是否提升 out-of-sample 表现；
2. 系数方向是否合理；
3. 是否与 spend 高度共线；
4. 是否存在目标泄漏风险；
5. 预测期是否有同口径数据可用。

### 10.3 增加业务节点字段

如果后续能补齐大促、新品、舆情、BHT 等字段，建议加入：

```text
is_big_promo
is_cny
is_618
is_1111
launch_flag
weeks_since_launch
bht_di
negative_pr_flag
```

这些字段可以显著提高模型解释力。

---

## 11. 一句话总结

这套方案本质上是：

```text
用 RED SEM / Feeds / KOC / Branding 的周度投放花费构造 Geometric Adstock，
再结合 NSR 和 Industry SI，
通过 Positive Ridge 预测 RED Search Index，
最后用训练期残差的 0.9σ 生成 Base / Optimistic / Pessimistic 三条预测曲线。
```

它适合作为一版可解释、可复现、便于后续升级的 RED Search Index 归因预测模型 baseline。
