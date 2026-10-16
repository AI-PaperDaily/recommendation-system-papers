## 双支路图蒸馏重构冷启动推荐：DualGCN突破97.8%性能暴跌，冷启动召回提升超20%

### 引言

在电商与内容平台的推荐系统中，协同过滤（Collaborative Filtering）始终是不可替代的基石。以LightGCN为代表的图协同过滤（GCF）模型，通过多跳消息传递捕捉高阶协同信号，在活跃商品上取得了显著成功。然而，当一件全新商品没有任何历史交互记录时，它在交互二分图中的度严格为零，消息传递机制彻底失效——这就是**严格冷启动**（Strict Cold-Start）场景下的**表征坍缩**难题。

论文给出的数据触目惊心：在Amazon-Books2023数据集上，LightGCN从活跃商品切换到冷启动商品时，Recall@20从0.1244暴跌至0.0027，**相对性能衰减高达97.8%**。传统ID嵌入模型在冷启动场景下几乎完全失效，推荐系统陷入“新商品无人问津，无人问津导致更无数据”的恶性循环。

面对这一结构困境，来自香港科技大学（广州）的研究者提出了**DualGCN**——一个双支路图知识蒸馏框架。该方法利用预训练语言模型提取的语义特征作为锚点，在不增加在线推理开销的前提下，同时在冷启动与暖启动场景取得了显著增益。

### 核心方法

DualGCN的架构设计遵循一个核心洞察：**语义特征对图平滑后的表示具有天然的锚定效应**。论文通过UMAP可视化发现，当使用语义向量初始化节点嵌入时，即使经过多层图卷积的低通滤波，活跃商品的表示仍然紧密聚集在其初始语义锚点周围。这为将孤立冷启动商品映射到成熟协同空间提供了几何基础。

![图1：UMAP可视化展示了随机ID初始化与语义初始化下的表示演化对比](https://arxiv.org/html/2506.12345v1/assets/fig1_umap_comparison.png)

DualGCN采用**非对称双支路架构**，由协同教师支路（Collaborative Teacher Branch）和语义学生支路（Semantic Student Branch）组成。

**拓扑平滑语义先验**：教师支路的核心创新在于第0层初始嵌入的构造方式。传统LightGCN仅使用可学习ID嵌入 $\mathbf{e}_i$，而DualGCN将归一化后的语义向量 $\bar{\mathbf{s}}_i$ 以可学习标量 $\gamma$ 加权注入：

$$\mathbf{e}_i^{(0)} = \mathbf{e}_i + \gamma\bar{\mathbf{s}}_i$$

这一设计的精妙之处在于其矩阵展开形式，教师支路的最终表示可以分解为两个独立项：

$$\tilde{\mathbf{E}}^{\mathcal{T}} = \underbrace{\frac{1}{K+1}\sum_{k=0}^{K}\tilde{\mathbf{A}}^k\begin{bmatrix}\mathbf{E}_u\\\mathbf{E}_i\end{bmatrix}}_{\text{图传播ID嵌入}} + \underbrace{\gamma\frac{1}{K+1}\sum_{k=0}^{K}\tilde{\mathbf{A}}^k\begin{bmatrix}\mathbf{0}\\\bar{\mathbf{S}}\end{bmatrix}}_{\text{拓扑平滑语义先验}}$$

其中 $\tilde{\mathbf{A}}$ 为对称归一化邻接矩阵，$K$ 为图卷积层数。第二项即为**拓扑平滑语义先验**，它通过多跳信息扩散，将商品侧的语义特征沿交互边注入到用户表示中，从根本上重塑了协同空间的几何结构。

![图2：DualGCN双支路知识蒸馏框架的整体架构图](https://arxiv.org/html/2506.12345v1/assets/fig2_architecture.png)

**解耦两阶段优化**：论文指出，若将拓扑更新与语义对齐在端到端框架中联合优化，稀疏图重构的主导梯度会淹没低频对齐信号，扰乱静态语义空间的内在表示。因此，DualGCN采用严格的**两阶段训练范式**。

第一阶段仅训练教师支路，语义特征完全冻结不参与反向传播，优化目标为标准的BPR损失：

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)}\ln\sigma\left(\hat{y}_{ui} - \hat{y}_{uj}\right) + \lambda\Theta$$

第二阶段冻结教师支路的全部参数，仅用其最终表示作为静态软标签，监督学生支路的对齐网络。学生支路采用**残差映射机制**，将跨空间映射分解为基本语义锚点与非线性拓扑偏移的叠加：

$$\tilde{\mathbf{e}}_i^{\mathcal{S}} = \mathbf{W}_2\sigma\left(\text{LN}(\mathbf{W}_1\bar{\mathbf{s}}_i + \mathbf{b}_1)\right) + \mathbf{b}_2 + \gamma\bar{\mathbf{s}}_i$$

蒸馏阶段采用**幅值无关的余弦蒸馏损失**：

$$\mathcal{L}_{\text{distill}} = 1 - \frac{1}{|\mathcal{I}_{\text{warm}}|}\sum_{i\in\mathcal{I}_{\text{warm}}}\frac{(\tilde{\mathbf{e}}_i^{\mathcal{S}})^T\tilde{\mathbf{e}}_i^{\mathcal{T}}}{\|\tilde{\mathbf{e}}_i^{\mathcal{S}}\|_2\|\tilde{\mathbf{e}}_i^{\mathcal{T}}\|_2}$$

论文认为，节点表示的绝对幅值隐式编码了交互频率先验，而冷启动商品天然缺乏这一先验。因此，仅在单位超球面上执行方向对齐，能够避免幅值失真，从根本上增强模型的泛化能力。

**动态范数重建机制**：由于蒸馏阶段剥离了幅值信息，直接使用学生支路的单位向量进行全局检索会导致几何空间错位。为此，论文引入动态范数重建：在推理时计算暖启动商品教师表示的平均范数 $\mu_{\text{norm}}$，对交互度低于阈值 $\tau$ 的商品，将其学生表示归一化后乘以 $\mu_{\text{norm}}$ 进行尺度校准：

$$\mathbf{e}_i^* = \begin{cases}\tilde{\mathbf{e}}_i^{\mathcal{T}}, & \text{if } d_i \geq \tau,\\\frac{\tilde{\mathbf{e}}_i^{\mathcal{S}}}{\|\tilde{\mathbf{e}}_i^{\mathcal{S}}\|_2}\mu_{\text{norm}}, & \text{if } d_i < \tau.\end{cases}$$

这一机制无需额外训练开销，即可在全局检索流形中统一新旧商品的物理尺度。

### 实验结果

论文在Amazon Reviews'23数据集的三个子集——Books、Electronics和Sports上进行了全面评估，采用10-core过滤与严格的冷启动划分策略：随机隔离10%商品作为冷启动组，其交互记录在训练阶段完全遮蔽。

![图3：消融实验中不同变体在冷启动场景下的性能对比](https://arxiv.org/html/2506.12345v1/assets/fig3_ablation.png)

**冷启动性能**：实验数据显示，纯ID模型（MF、NGCF、LightGCN）在冷启动场景下Recall@20全部跌至0.003以下，几乎完全失效。相比之下，第二阶段的DualGCN在Books数据集上取得0.1007的Recall@20，在Sports数据集上达到0.1119，**相较最强基线DropoutNet分别提升29.1%和20.8%**。

| 方法 | Books Cold R@20 | Sports Cold R@20 | Electronics Cold R@20 |
|------|----------------|------------------|----------------------|
| LightGCN | 0.0027 | 0.0117 | 0.0022 |
| DropoutNet | 0.0780 | 0.0926 | 0.0225 |
| DeepMusic | 0.0890 | 0.0827 | 0.0279 |
| **DualGCN (Phase 2)** | **0.1007** | **0.1119** | 0.0248 |

**暖启动保持**：在暖启动场景中，第一阶段DualGCN在Electronics数据集上Recall@20达到0.0457，**相较LightGCN提升39.7%**；在Sports数据集上提升23.3%。第二阶段DualGCN的暖启动性能与第一阶段高度一致，验证了知识蒸馏机制对既有协同信号的保真能力。

**消融实验**：论文对蒸馏损失函数和残差连接设计进行了系统消融。使用MSE替代余弦损失后，Books冷启动Recall@20从0.1007降至0.0951；使用InfoNCE后Electronics上Recall@20从0.0248暴跌至0.0143。移除语义残差连接后，Sports上Recall@20从0.1119降至0.1010。这些结果证实了余弦蒸馏与残差设计在跨模态对齐中的关键作用。

### 展望

DualGCN的提出为推荐系统的冷启动问题提供了一种**轻量级、可扩展**的解决范式。其解耦训练策略与语义锚定机制，使得模型无需依赖外部知识图谱或复杂的多模态融合模块，仅借助基础文本属性即可实现跨空间对齐。这为工业级部署提供了现实可行性。

论文指出，未来的研究将集中于将该语义锚定范式拓展至**动态图环境**，并探索融合更丰富的多模态信号以进一步增强分布外泛化能力。当推荐系统真正摆脱对历史交互的路径依赖，新商品的上架即获精准触达便不再遥远。

**论文标题**：Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation

欢迎投稿！欢迎合作！