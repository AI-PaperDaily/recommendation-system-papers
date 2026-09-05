## 美团发布UniCon：重新定义CTR建模的上下文单元，AUC提升0.0139的背后逻辑

在搜索广告和推荐系统中，点击率（CTR）预测的核心任务是回答一个看似简单的问题：**用户为什么点击了某个商品？** 传统模型往往将用户历史行为序列与当前候选商品视为两类异质信号，分别用序列编码器和特征交互网络处理。然而在真实场景中，用户的每一次点击都发生在特定展示环境中——搜索结果页上，用户选择一家便宜的咖啡店，可能并非因为偏好，而是因为同屏展示的替代选项更贵或更远。

美团研究团队在KDD 2027论文中提出的 **UniCon**（Unified Context-Centric Modeling Paradigm），正是从这一观察出发，重新构建了CTR预测的输入组织方式。该方法将“上下文单元”（Context Unit）作为基本建模单位，在美团搜索广告场景下实现了离线AUC提升 **0.0139**，在线RPM提升 **3.09%**、CTR提升 **2.07%**、营收提升 **2.95%**。

## 一、核心方法：从“特征组织”到“上下文单元”

### 上下文单元的统一表达

UniCon的核心思想在于消除历史行为与当前请求之间的结构性断裂。论文指出，用户行为本质上是一串**同质的上下文单元序列**：历史曝光列表与当前候选集在结构上共享相同模式，二者的唯一区别仅在于“结果是否已被观测”。

为此，UniCon定义了一个上下文单元包含三类信息：**物品侧token**（物品ID、展示位置、静态属性）、**上下文token**（搜索意图、查询词、时间、地点、设备等），以及一个**用户token**（聚合的动态统计特征与静态属性）。所有特征经过线性tokenizer映射为统一表示：

$$\mathbf{z}_{g} = \mathrm{Tokenizer}_{g}\left([\mathbf{e}_{g,1};\ldots;\mathbf{e}_{g,n_{g}}]\right) = \mathbf{W}_{g}[\mathbf{e}_{g,1};\ldots;\mathbf{e}_{g,n_{g}}]+\mathbf{b}_{g}$$

其中 $[\cdot;\cdot]$ 表示拼接，$\mathbf{z}_{g} \in \mathbb{R}^{d}$。一个上下文单元 $C$ 的token序列为：

$$\mathbf{Z}_{C} = [\mathbf{z}^{item}_{1}, \ldots, \mathbf{z}^{item}_{N_C}, \mathbf{z}^{ctx}_{1}, \ldots, \mathbf{z}^{ctx}_{K_C}, \mathbf{z}^{usr}]$$

对于历史单元，物品token中的反馈分量填入真实点击标签的嵌入；对于目标单元，由于点击结果未知，填入可学习的占位符嵌入。这一设计保证了历史侧与目标侧共享完全相同的token schema，为后续的统一建模奠定基础。

![图1：上下文局部性与动态性的示意图](https://arxiv.org/html/2609.03290v1/intro.png)

### 层次化上下文建模：Locality与Dynamics

UniCon的模型架构由多个堆叠的UniConBlock组成。每个Block内部交替执行两个不同粒度的注意力操作：

**上下文内交互（Intra-Context Attention）** 与**上下文间交互（Inter-Context Attention）**。上下文内层将注意力限制在每个上下文单元的边界内，独立编码其中所有token，捕捉同屏展示物品之间的局部竞争与互补关系。这一层的边界约束确保了相邻展示列表不会被误判为局部竞争者：

$$\widetilde{\mathbf{H}}_{C}^{\ell} = \mathrm{IntraAttn}^{\ell}(\mathbf{H}_{C}^{\ell-1}), \quad C \in \mathcal{C}$$

随后，上下文间层将各单元的编码结果按时间顺序拼接，执行跨单元的全局注意力，建模用户兴趣和展示环境随时间的动态演化：

$$\mathbf{H}^{\ell} = \mathrm{InterAttn}^{\ell}\left([\widetilde{\mathbf{H}}_{C_{1}^{h}}^{\ell}, \ldots, \widetilde{\mathbf{H}}_{C_{M}^{h}}^{\ell}, \widetilde{\mathbf{H}}_{C^{t}}^{\ell}]\right)$$

两个层级均采用预归一化的Transformer结构，包含RMSNorm、self-attention和SwiGLU密集MoE前馈层。通过交替堆叠这两个层级，UniCon在不展平上下文边界的前提下，同时保留了局部决策结构和跨上下文的兴趣演化信息。

![图2：UniCon整体架构图](https://arxiv.org/html/2609.03290v1/method.png)

### 目标侧监督：暴露与位置预测

由于排序阶段目标展示列表尚未生成，UniCon用候选集初始化一个“目标潜在上下文单元”。为了让这个潜在单元逼近真实展示列表的上下文结构，论文引入了**暴露预测**和**绝对位置预测**两个辅助任务。暴露头在完整候选集上训练，预测每个候选是否进入最终展示列表；位置头仅对已曝光候选训练，预测其绝对展示位置。点击头的梯度不流回位置头，辅助任务通过共享上下文表示间接影响CTR预测。整体学习目标为：

$$\mathcal{L} = \mathcal{L}_{clk} + \lambda_{exp}\mathcal{L}_{exp} + \lambda_{pos}\mathcal{L}_{pos}$$

其中 $\mathcal{L}_{exp}$ 覆盖所有目标候选，$\mathcal{L}_{clk}$ 仅在已曝光候选的日志位置上计算，$\mathcal{L}_{pos}$ 为绝对位置预测损失，$\lambda_{exp}$ 和 $\lambda_{pos}$ 控制辅助任务权重。

## 二、高效计算：上下文感知序列压缩

随着历史曝光累积，上下文单元数量可能达到数百个。如果每个inter-context层都在完整token序列上执行注意力，计算复杂度将呈 $O(LN^{2})$ 增长。UniCon的解决策略是：在第一个Block中执行完整的inter-context交互，从第二个Block开始，逐步执行**上下文感知的序列压缩**。

压缩机制在每层利用目标单元的query向量与各历史单元的key向量计算相关性分数：

$$s_{m}^{\ell} = \frac{(\mathbf{q}_{t}^{\ell})^{\top}\mathbf{k}_{m}^{\ell}}{\sqrt{d}}, \quad C_{m}^{h} \in \mathcal{C}^{\ell-1}$$

$$K_{\ell} = \left\lceil r_{\ell}|\mathcal{C}^{\ell-1}_{h}|\right\rceil, \quad \mathcal{S}^{\ell} = \mathrm{TopK}(\{s_{m}^{\ell}\}, K_{\ell})$$

训练时采用Gumbel-TopK配合直通估计器，推理时直接执行硬TopK选择。论文理论分析表明，当采用固定保留比例 $r = r_1 = r_2 = \cdots = r_{L-1}$ 时，全局注意力成本上界从 $O(LN^{2})$ 降至 $O(N^{2}/(1-r^{2}))$。在实际部署中，序列压缩与可变长注意力算子（基于FlashAttention原理，暴露segment offsets表示上下文边界）结合，使工业场景下的高效推理成为可能。

## 三、实验：离线优势与在线增益

离线实验使用美团搜索广告一年期数据，训练集、验证集、测试集按时间顺序切分。表1展示了UniCon与多个基线的对比结果：

| 模型 | AUC | GAUC | LogLoss | 参数量/GFLOPs |
|------|-----|------|---------|----------------|
| Base（生产基线） | 0.8558 | 0.8076 | 0.2084 | 0.09B/9.69G |
| OneTrans+CIM | 0.8657 | 0.8162 | 0.2017 | 0.21B/208.75G |
| RankMixer+DSIN+CIM | 0.8661 | 0.8171 | 0.2013 | 0.40B/171.46G |
| UniCon-Small | 0.8683 | 0.8184 | 0.2001 | 0.09B/201.60G |
| UniCon-Large（压缩版） | **0.8697** | **0.8194** | **0.1991** | 0.33B/197.14G |

UniCon-Small在AUC上已超越所有研究基线，包括增强了CIM和DSIN的版本。压缩版UniCon-Large在生产配置下将计算量从801.29 GFLOPs降至197.14 GFLOPs，AUC几乎无损失（0.8698 vs 0.8697）。

消融实验进一步验证了核心设计的贡献：移除上下文组织使AUC从0.8697降至0.8687；移除目标侧上下文统一降至0.8690；用参数匹配的全局token级backbone替代层次化上下文建模导致最大退化（降至0.8637）；仅移除intra-context交互也降至0.8673。

![图3：不同参数规模下UniCon与基线的缩放对比](https://arxiv.org/html/2609.03290v1/scaling.png)

缩放实验显示，UniCon在约0.05B至0.33B参数范围内保持了对代表性基线的稳定优势，表明上下文单元组织是随容量持续有效的架构选择。压缩比例实验中，$r=0.5$ 的生产配置较 $r=1.0$ 减少75.4%计算量，AUC仅下降0.0001。

![图4：上下文保留比例与AUC、GFLOPs的权衡](https://arxiv.org/html/2609.03290v1/compression.png)

七天在线A/B测试中，UniCon-Large在20%流量上实现了 **3.09% RPM**、**2.07% CTR**、**2.95%营收**提升，所有指标在双侧检验下达到 $p \leq 0.01$ 显著水平。

## 四、展望

UniCon的价值不止于一组性能数字。它揭示了一个方向：**上下文结构应当从输入组织层面成为模型架构的一部分**，而非作为额外特征附加在既有backbone上。在电商货架、信息流瀑布、搜索结果列表等强上下文感知场景中，这种以展示列表为基本单位的建模范式具有直接的推广潜力。随着上下文序列压缩的计算优势在更大规模模型上进一步放大，上下文中心化建模或许将成为统一CTR架构的新默认设定。

**论文标题**：UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction

欢迎投稿！欢迎合作！