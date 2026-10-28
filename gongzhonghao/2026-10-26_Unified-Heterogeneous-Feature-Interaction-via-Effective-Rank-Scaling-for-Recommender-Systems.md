# 阿里巴巴HiRank：以有效秩视角统一异构特征交互，淘宝首页线上每用户订单数提升2.27%

## 一、引言

在淘宝这类日活达亿级的电商推荐系统中，排序模型需要同时估计购买转化率、加购率等多个与转化相关的目标，并整合用户行为历史与候选商品特征两类异构信号。围绕这一需求，既有研究大致沿两条平行路径推进：一条是行为序列建模，从用户历史中抽取与候选商品相关的兴趣信号，DIN、DIEN、LONGER 是代表性工作；另一条是特征交互，学习多字段高维特征之间的依赖关系，从 FM、DCNv2 到 Wukong、RankMixer 逐步演进。

问题在于，工业系统通常用“先编码、再交互”的流水线把两者拼接起来：序列编码器先把用户历史压缩成固定维度向量，下游交互模块再把它与非序列特征结合。当这一摘要成为后续模块唯一的行为输入时，同一前向过程中就无法再回到事件级序列状态取回证据。与此同时，序列与非序列信号的有效秩（eRank）并不相同，对所有语义 token 直接套用同质自注意力并不高效；若只用一份共享表示覆盖所有流量场景，也容易忽略场景特有模式，引发场景间的负迁移。

针对上述问题，研究者提出统一框架 **HiRank**，在同一主干内联合建模异构特征交互，并通过场景感知聚合与复合投影构造信息密度更均衡的高阶交互输入。该模型已在淘宝首页推荐部署，线上 A/B 测试中每用户订单数提升 2.27%。

## 二、核心方法

![图1：HiRank 整体架构](https://arxiv.org/html/2610.35318v1/hirank_overview.svg)

HiRank 在 token 层面维护四类表示：用户主行为序列状态、候选商品 token、只读的长期行为记忆，以及按语义分组的非序列特征 token。主干由若干重复的 HiRank Block 组成，每个 Block 内部依次完成门控序列建模、互补记忆检索和异构 token 交互；堆叠时，交互输出会回写为主序列状态与下一层的检索查询，使已经成形的特征上下文反过来指导后续的行为建模与记忆访问。

### 门控序列建模与统一 Token 布局

主行为序列按事件逐个投影为 $T\times d$ 的表示，保留事件级状态；每条长期历史（点击、加购、购买等）独立投影为一份长期记忆 $\mathbf{M}_j$，在整个前向过程中保持只读。非序列特征与池化后的行为摘要则按用户属性、商品属性、请求上下文等语义组分别投影成 token。

序列侧借鉴 HSTU 的思路，用点积聚合替代 softmax 归一化，并引入乘性门控显式建模局部特征与聚合历史之间的交互：

$$\mathcal{S}^{\prime}=\mathcal{S}+\operatorname{Linear}\!\left(\operatorname{LayerNorm}\left(\frac{\operatorname{SiLU}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}}{N}\right)\odot\mathbf{U}\right)$$

其中 $N$ 为有效主序列位置数，$\mathbf{U}$ 是经 SiLU 激活的门控分支。该式的含义是：注意力聚合结果被逐位置的门控信号缩放后残差回写，避免了 softmax 的单元和约束。候选 token 复用同一组 $\mathbf{K},\mathbf{V}$ 与共享的投影、归一化参数，只用自己的查询和门控读取序列，且不向序列写入任何键值。

### 互补记忆检索与异构 Token 交互

长期记忆的读取由两条互补路径完成。第一层中用可学习种子 token 对主序列做注意力，生成兴趣查询 $\mathbf{Q}_{u,j}$；更新后的候选 token 则初始化为候选条件查询 $\mathbf{Q}_{i,j}$。两者从同一份记忆中取回不同视角的证据：

$$\mathcal{R}_{u,j}=\operatorname{CrossAttn}\!\left(\mathbf{Q}_{u,j},\mathbf{M}_{j}\right),\qquad \mathcal{R}_{i,j}=\operatorname{CrossAttn}\!\left(\mathbf{Q}_{i,j},\mathbf{M}_{j}\right)$$

这样处理的原因是：由相对静态的画像特征构造查询，容易集中在少数表示方向上，限制跨用户的有效维度；以主序列初始化兴趣查询可以保留行为相关的差异性，候选条件路径则提供商品视角的补充证据。

随后，主序列状态、两路检索输出、语义特征 token 与候选 token 拼接到一起，先做头部维度的跨 token 混合再还原布局，最后逐 token 做 SwiGLU 变换：

$$\mathbf{X}^{\prime}=\mathbf{X}+\operatorname{Revert}\!\left(\mathbf{Z}+\operatorname{SwiGLU}_{\mathrm{head}}\!\left(\operatorname{Mix}(\mathbf{X})\right)\right),\qquad \mathbf{X}^{(l)}=\mathbf{X}^{\prime}+\operatorname{SwiGLU}_{\mathrm{token}}\!\left(\operatorname{RMSNorm}(\mathbf{X}^{\prime})\right)$$

第一阶段在每个通道组内跨 token 位置整合行为与特征证据，第二阶段用位置相关参数在 token 内部重组通道，两者共同承担异构信息的交换。这一设计也回应了一个表示层面的限制：单头注意力聚合满足 $\operatorname{rank}(\mathbf{A}\mathbf{V})\le\operatorname{rank}(\mathbf{V})$，本身无法提升值表示的代数秩，需要配合 token 混合才能获得更丰富的信息密度。

### 高秩注意力：场景感知压缩与复合投影

主干输出之后，High-Rank Attention 先用场景信息初始化的 scene token 对最后一层全部 token 做交叉注意力，把异构低阶 token 压缩成一个紧凑表示，避免对全量 token 拼接直接做复合投影带来的巨大映射矩阵。随后三组独立参数化的复合投影把这一完整表示变换为多个复合 token，每个 token 都能访问压缩状态的全部通道，但学习不同的投影方向，从而在后续多头注意力中获得互补的表示方向与更均衡的奇异值谱。

此外，模型还提供一个可选的用户—物品解耦变体：把主干 token 划分为用户侧与物品侧两条流，用户侧计算在一次请求内共享，主序列门控自注意力每次请求只算一次，其历史 K/V 被各候选复用；token 交互采用带掩码的头部 SwiGLU，只允许用户到物品的单向信息流动。

![图2：可选的用户—物品解耦 HiRank](https://arxiv.org/html/2610.35318v1/user_item_decoupling.svg)

## 三、实验

离线实验基于淘宝工业级数据集，覆盖 24 亿次曝光、8700 万商品、2.06 亿用户与 9400 万次购买正样本，整体稀疏度 3.92%；其中首页场景 4.38 亿次曝光，稀疏度 0.92%。主任务为购买转化率预测，加购预测作为辅助任务提供额外监督。

在首页与主搜索两个流量场景上，HiRank 取得最高的 AUC 与 GAUC。相对各自最强基线，首页 AUC 与 GAUC 分别提升 0.0009 和 0.0186，主搜索分别提升 0.0005 和 0.0009；首页 PCOC 为 0.9865，是各方法中最接近 1 的。在稠密参数 299.04M、单候选前向 1.530 GFLOPs 的开销下，其参数量低于参评的多个大规模基线。

消融结果显示各组件均有贡献：完整模型的 AUC 与 GAUC 最优；把门控注意力或互补检索替换为 DIN 式目标注意力都会带来下降；同时去掉 token 交互与 High-Rank Attention 时降幅最大，AUC 下降 0.0020、GAUC 下降 0.0489。

![图3：各阶段表示的有效秩变化](https://arxiv.org/html/2610.35318v1/effective_rank.svg)

有效秩诊断给出了与设计动机一致的观察：经过两个完整混合块，主序列表示的平均 eRank 分别从 96.51 升至 112.80、从 128.36 升至 134.80；用户长期记忆检索的输出相对输入查询从 24.52 升至 54.88；而在 High-Rank Attention 内部，eRank 从 228.58 变为 224.09，相对下降 1.96%，说明高阶交互阶段基本保留了压缩状态的谱多样性。

![图4：HiRank Block 与 High-Rank Attention 的缩放研究](https://arxiv.org/html/2610.35318v1/scaling.svg)

缩放实验在固定其余设置的前提下单独变化宽度、长度与深度。对 HiRank Block 而言，序列长度的影响最明显，主序列扩展到 96 个事件带来 0.23 个百分点的 AUC 提升，缩短到 24 个事件则下降 0.51 个百分点，而增宽或加深最多带来约 0.06 个百分点；对 High-Rank Attention，复合 token 宽度取 1024、token 数取 96、块数取 4 分别带来 0.08、0.05 和 0.06 个百分点的提升。

线上 A/B 测试在淘宝首页流量上进行，采用用户级持久随机化并固定非排序策略，以每用户订单数作为主要指标。相对生产基线，HiRank 的 PV 提升 0.45%，每用户订单数提升 2.27%（$p<0.01$），每用户 GMV 提升 2.86%（$p<0.05$），日购买用户数提升 2.13%（$p<0.01$）；同时首页推荐点击用户数提升 0.85%，每千次曝光订单数提升 1.81%。

**论文标题**：Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems

欢迎投稿！欢迎合作！