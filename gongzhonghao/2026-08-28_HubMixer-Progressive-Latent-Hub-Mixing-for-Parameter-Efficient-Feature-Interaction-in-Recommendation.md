## 快手推出HubMixer推荐架构：潜在枢纽混合实现参数效率跃升，转化率提升5.48%

## 一、引言

工业级推荐系统的核心挑战之一，是如何在大规模稀疏特征上高效建模特征交互。一个用户的年龄标签、一条历史行为序列、一个职位类别、一个上下文信号——这些语义完全不同的特征，如何让模型自动发现它们之间那些真正有价值的关联？传统做法倾向于把所有这些特征“拍平”，让模型在统一的token空间里自由组合。但当token数量从几十扩展到上百，token间可能的交互组合呈平方级爆发，而真正有用的交互却往往是稀疏的、样本相关的。模型把大量参数浪费在发现“哪些特征应该交互”这一基础问题上。**论文提出HubMixer，一种基于潜在枢纽的渐进式特征交互架构，以更少的参数实现了更强的排序性能和线上业务转化。**

## 二、核心方法：Induction–Interaction–Readout三阶段范式

HubMixer的核心洞察在于：与其让异构特征token在原始空间中直接互相混合，不如先把它们归纳到一个紧凑的**潜在枢纽空间**中，在那里完成高阶交互，再把交互后的语义精确写回每个token。这一设计形成“归纳—交互—读回”的完整闭环。

### 2.1 异构token与潜在枢纽的引入

论文将推荐系统的输入特征按语义组织为token集合 $\mathbf{X}=[\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_T]\in\mathbb{R}^{T\times d}$，其中 $T$ 为特征token数，$d$ 为维度。这些token来自截然不同的语义空间。HubMixer不直接对 $\mathbf{X}$ 施加token间的混合操作，而是引入一组可学习的潜在枢纽 $\tilde{\mathbf{H}}^{(l)}\in\mathbb{R}^{H\times d}$，其中 $H$ 远小于 $T$，默认为16。为了注入样本级上下文信息，每个枢纽通过全局token摘要的MLP生成一个条件残差进行增强：

$$\mathbf{H}^{(l)}=\tilde{\mathbf{H}}^{(l)}+\phi\big(\mathbf{W}^{(l)}_2\sigma(\mathbf{W}^{(l)}_1\mathbf{p}^{(l)}+\mathbf{b}^{(l)}_1)+\mathbf{b}^{(l)}_2\big)$$

其中 $\mathbf{p}^{(l)}=\frac{1}{T}\sum_{t=1}^{T}\mathbf{x}^{(l)}_t$ 为全局token摘要，$\sigma$ 为Swish激活。这一步让静态的枢纽原型获得了请求级动态调整能力。

![图1：HubMixer整体架构。异构特征token首先被归纳为紧凑的潜在枢纽，然后在枢纽空间中进行交互，最后通过token条件读回实现对每个token的精确语义注入。](https://arxiv.org/html/2608.27991v1/image/cosine_analysis.png)

### 2.2 枢纽归纳：从异构token到紧凑语义中心

枢纽归纳阶段的目标是让每个枢纽有选择地从原始token中提取与其语义偏好相关的信息。论文采用**交叉注意力**机制，以枢纽作为Query、原始token作为Key和Value：

$$\mathbf{Z}^{(l)}=\mathrm{RMSNorm}(\mathbf{H}^{(l)}+\mathrm{CrossAttn}(\mathbf{Q}=\mathbf{H}^{(l)},\mathbf{K}=\tilde{\mathbf{X}}^{(l)},\mathbf{V}=\tilde{\mathbf{X}}^{(l)}))$$

与直接对token做自注意力不同，这里注意力权重由枢纽自身的内容决定，天然形成了“哪些token信息应该流向哪个枢纽”的软路由。不同枢纽可以在训练过程中演化出不同的语义偏好——有的偏好用户兴趣信号，有的聚焦职位属性，有的捕捉上下文调制信号。由于注意力权重依赖输入内容，这一归纳过程是**样本自适应**的。

### 2.3 枢纽交互：在紧凑空间中完成高阶建模

归纳后的枢纽表征 $\mathbf{Z}^{(l)}$ 已经浓缩了各类token的语义混合物。下一步在枢纽空间内部进行自注意力交互：

$$\tilde{\mathbf{Z}}^{(l)}=\mathrm{RMSNorm}(\mathbf{Z}^{(l)}+\mathrm{SelfAttn}(\mathbf{Q}=\mathbf{Z}^{(l)},\mathbf{K}=\mathbf{Z}^{(l)},\mathbf{V}=\mathbf{Z}^{(l)}))$$

由于枢纽数量 $H$ 远小于token数量 $T$，这步交互的计算成本为 $O(H^2)$ 而非 $O(T^2)$。关键在于：枢纽交互发生在“已经完成了语义归纳”的空间中。每个枢纽不再是某个单一特征的表示，而是多个相关特征的融合体。对枢纽做高阶交互，等价于在**特征组级别**建模交叉关系，而非在原始特征级别穷举所有token对。这一设计将模型容量集中在了高价值的跨语义交互路径上。

### 2.4 Token条件读回：保留字段身份的信息写入

枢纽交互之后，如何将浓缩的全局语义返回到token层？论文的方案是让每个原始token用自己的表征作为Query，从交互后的枢纽中**选择性检索**自己需要的补充信息：

$$\Delta\mathbf{X}^{(l)}=\mathrm{CrossAttn}(\mathbf{Q}=\hat{\mathbf{X}}^{(l)},\mathbf{K}=\bar{\mathbf{Z}}^{(l)},\mathbf{V}=\bar{\mathbf{Z}}^{(l)})$$

读回的信号 $\Delta\mathbf{X}^{(l)}$ 以残差方式注入原始token流：

$$\mathbf{X}^{(l+1)}=\mathbf{X}^{(l)}+\boldsymbol{\gamma}^{(l)}\odot\Delta\mathbf{X}^{(l)}$$

其中 $\boldsymbol{\gamma}^{(l)}$ 为LayerScale参数，控制注入幅度。这种token条件化的读回机制区别于简单的全局池化广播：每个token根据自己的语义需求吸收不同的枢纽信息。用户行为token可能更多读取“兴趣枢纽”的交互结果，而职位属性token可能更多读取“内容枢纽”的语义。字段级别的身份信息得以保留，同时每个token都被注入了全局交互语境。

## 三、实验：更少参数，更强性能

### 3.1 离线性能与参数效率

论文在快手短视频招聘业务的超10亿样本数据集上进行了离线评估，覆盖四个多任务目标：plc_click、effective_view、interact、resume_submit。对比基线包括DCN、DCNv2、AutoInt、Wukong、RankMixer、TokenMixer等代表性强基线。

| 模型 | plc_click | effective_view | interact | resume_submit | Avg. AUC | #Params |
|------|-----------|----------------|----------|---------------|----------|---------|
| DCN | 0.8196 | 0.8453 | 0.8358 | 0.7818 | 0.8206 | 155.3M |
| DCNv2 | 0.8194 | 0.8468 | 0.8361 | 0.7825 | 0.8212 | 157.5M |
| AutoInt | 0.8194 | 0.8459 | 0.8363 | 0.7823 | 0.8210 | 162.6M |
| Wukong | 0.8183 | 0.8461 | 0.8365 | 0.7822 | 0.8208 | 154.6M |
| RankMixer | 0.8221 | 0.8495 | 0.8389 | 0.7845 | 0.8238 | 155.1M |
| TokenMixer | 0.8227 | 0.8491 | 0.8394 | 0.7852 | 0.8241 | 156.9M |
| **HubMixer** | **0.8253** | **0.8507** | **0.8401** | **0.7864** | **0.8256** | **142.4M** |

**HubMixer在所有四个任务上取得最优AUC，平均AUC达0.8256，同时参数量比最强基线TokenMixer减少约9.2%。** 值得关注的是，传统特征交叉模型（DCN系、Wukong）在平均AUC上已落后于近期token-mixing架构，而HubMixer的“归纳—交互—读回”机制在同类token-mixing方法中进一步拉开了差距。

### 3.2 表示质量分析

论文进一步通过表示分析揭示HubMixer超越TokenMixer的内在原因。图2展示了两种模型在token级别上的输入-输出余弦距离，该指标衡量每个mixer block对token表征方向的改变程度。

![图2：HubMixer与TokenMixer在各mixer层的token级别输入-输出余弦距离。数值越大表示block内部对token表征的方向性更新越强。](https://arxiv.org/html/2608.27991v1/image/cosine_analysis.png)

HubMixer在第一层的平均余弦距离为0.200，几乎是TokenMixer（0.105）的两倍。第二层仍维持优势（0.051 vs. 0.037）。更大的方向性更新意味着HubMixer向token中注入了更丰富的交互信息。同时，热力图显示更新在token间并非均匀分布，验证了token条件化读回带来的**选择性注入**效应。

![图3：HubMixer与TokenMixer的线性探测对比。骨干网络冻结后，在token级别或拼接的实例级别表征上训练轻量线性分类器，检验表征中任务相关信号的可访问性。](https://arxiv.org/html/2608.27991v1/image/linear_probing.png)

线性探测实验进一步提供了因果性证据：在冻结骨干网络后，HubMixer在大多数特征token以及聚合表征上的探测AUC均高于TokenMixer。这说明HubMixer产生的表征**不仅变化幅度更大，而且变化中包含更多任务相关的交互信号**，而非单纯的扰动。

### 3.3 线上A/B验证

HubMixer在快手短视频招聘分发系统中进行了为期7天的线上A/B测试，覆盖7.2%的生产流量。核心指标为简历提交转化率。结果显示**HubMixer带来了5.48%的转化率提升**，且统计显著。这一数字直接验证了离线AUC增益可以转化为业务价值的实际判断。A/B测试结束后，HubMixer已在快手短视频招聘业务中全量部署。

## 四、展望：潜在枢纽作为特征交互的新范式

HubMixer的价值不仅在于一个具体的架构，更在于它提出了一个可扩展的交互组织原则：**先归纳、后交互、再读回**。枢纽数量提供灵活的参数-性能调节旋钮，Cross-Attention的计算复杂度从 $O(T^2)$ 降为 $O(TH)$，为大规模部署留出工程优化空间。论文也指出了几个有潜力的方向——动态生成完整枢纽集、引入枢纽多样性正则化、任务条件化读回——这些思路有望进一步释放潜在枢纽范式的表达能力。当推荐系统面对持续膨胀的特征规模和日益复杂的业务目标时，如何用更少的参数捕捉更有价值的交互结构，仍然是这个领域最核心的追问。

**论文标题**：HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation

欢迎投稿！欢迎合作！