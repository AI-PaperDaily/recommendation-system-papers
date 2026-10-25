## 快手OneLatent用3个隐式Token替代显式CoT：SID@64提升17.44%，吞吐翻17倍？

## 一、引言

大型语言模型进入推荐系统后，“先思考、再推荐”的显式思维链（Chain-of-Thought, CoT）展现了更强推理能力，但也带来明显代价：简单请求仍要生成冗长解释，推理延迟居高不下；固定单一的推理路径又容易放大历史偏好，忽略短期兴趣，最终导致推荐错位。

快手在论文中提出的OneLatent给出另一种思路：不生成自然语言推理，而是插入3个可学习的隐式Token，将推理过程压缩为固定长度的**Latent-Reason-then-Answer**接口。训练阶段仍然利用高质量显式推理作为监督，但推理阶段直接解码语义ID（SID）。在工业数据集上，OneLatent将SID@64较显式Think变体提升17.44%，推理吞吐提升17.49倍；在线A/B测试中带来预估收入9.6%的提升。

![图1：显式CoT的过度思考与兴趣漂移](https://arxiv.org/html/2610.30675v1/fig/intro.png)

## 二、核心方法

OneLatent的核心逻辑是：用**多视角自适应CoT**构造教师监督，再用**三阶段隐式Token对齐**将显式推理蒸馏到3个隐式Token中，最后通过**课程后训练**激活推荐能力。

### 2.1 多视角自适应CoT：从单一路径到互补监督

推荐场景下的用户兴趣高度异构，单一推理模板很难捕获短期意图变化。为此，论文设计了三种互补的CoT视角：

- **探索视角**：只给用户画像和历史行为，要求教师模型生成商业意图列表。
- **评估视角**：引入真实正样本作为特权信息，生成匹配评分与理由。
- **归因视角**：解释哪些历史信号促成了目标交互。

三个视角分别对应需求发现、正样本匹配和结果归因，避免同一条固定推理路径主导所有推荐。此外，教师模型不依赖难度分类器，而是根据行为证据强弱自动调整推理深度。该自适应生成过程可表示为：

$$Y_{u}^{(v)}\sim p_{\mathrm{T}}\!\left(\cdot\mid\pi_{v}^{\mathrm{adapt}}(G_{u}^{(v)})\right).$$

实验显示，自适应策略将平均CoT长度降低35%，同时复杂case的语义相似度保持在约0.92，说明压缩主要削减了冗余推理。

### 2.2 三阶段隐式Token对齐：让3个Token承载推理

OneLatent在FRM词汇表中加入3个共享的可学习隐式Token：

$$\mathbf{L}=[\langle L_{1}\rangle;\langle L_{2}\rangle;\langle L_{3}\rangle].$$

这些Token不是样本特定的向量，而是固定ID。Transformer在处理序列时，会在这些位置产生上下文相关的隐式表示，后续SID解码通过因果注意力读取这些表示。

![图2：OneLatent整体框架](https://arxiv.org/html/2610.30675v1/fig/OneLatent.png)

对齐过程分三步。第一阶段用单一视角教师输出进行热身，使模型学会在隐式Token后生成对应视角的理由。第二阶段并行引入多视角输出，并对每个输出内部做token级损失平均，避免长轨迹拥有过大的训练权重：

$$\mathcal{L}_{\mathrm{multi}}=\mathbb{E}_{u}\!\left[\frac{1}{|\mathcal{V}_{u}|}\sum_{v\in\mathcal{V}_{u}}\ell_{u}^{(v)}\right].$$

第三阶段将同一个隐式Token接口连接到推荐目标，输入序列为 $X_{u}^{\mathrm{ctx}}=[P_{u};H_{u};\mathcal{C}_{u};\mathbf{L}]$，训练目标转为预测真实SID：

$$\mathcal{L}_{\mathrm{align}}=-\mathbb{E}\left[\log p_{\phi}\left(\mathcal{S}_{u}^{+}\mid X_{u}^{\mathrm{ctx}}\right)\right].$$

值得注意的是，评价与归因视角的教师使用了正样本信息，但学生模型的前缀中始终不包含正样本描述、候选集或目标SID，这保证了监督信息只通过隐式Token间接影响推荐决策。

### 2.3 课程后训练与生产推理：保持精度与效率

后训练阶段采用从高活跃用户到低活跃用户的课程学习，让模型逐步适应不同行为密度。训练样本以1:1比例混合隐式推理与标准推荐模式。在线推理时，模型只解码三级SID：

$$\widehat{\mathcal{S}}_{u}=\operatorname*{arg\,max}_{\mathcal{S}}p_{\phi}(\mathcal{S}\mid P_{u},H_{u},\mathcal{C}_{u},\mathbf{L}).$$

为了满足生产延迟要求，OneLatent采用异步服务架构。行为触发准入、容量感知调度和结果缓存共同作用，使大模型推理不进入在线请求关键路径。

![图3：生产在线推理架构](https://arxiv.org/html/2610.30675v1/fig/serving_architecture.png)

## 三、实验验证

论文在快手工业务数据集和公开Kuaishou LLM-Rec benchmark上对比了ID类、SID类和推理增强基线。OneLatent-Merge在全部指标上取得最优。工业数据集上，SID@64相对OneReason-CoT Think-OR提升17.44%，相对No-Think-OR提升9.33%；公开数据集上，SID@64相对Think-OR提升达169%。

推理效率的提升更具说服力。显式CoT需要先自回归生成完整理由，QPS仅为15.68；而OneLatent只解码3个隐式Token后的SID，QPS达到274.29，加速17.49倍。这意味着推荐精度提升不再以延迟为代价。

从不同CoT策略的角度看，OneLatent在原始OneReason CoT、自适应CoT以及多视角融合等设置下，均稳定高于对应的Think模式。

![图4：不同CoT策略下OneLatent与Think的SID@64对比](https://arxiv.org/html/2610.30675v1/sid_trend_mixed_vs_latent_side_www.png)

快手本地服务广告场景的7天在线A/B测试进一步验证了实际价值：展示量增加2.566%，广告消耗增加5.157%，预估收入提升9.6%。系统通过异步更新保持了在线延迟稳定。

## 四、展望

OneLatent展示了显式CoT作为训练监督、隐式Token作为推理接口的可行路径。该方法在不牺牲推荐精度的前提下，大幅降低了基础推荐模型的推理成本，为工业级部署提供了新选项。当模型不再输出可读的推理过程，评测体系与用户信任机制该如何随之调整，是下一步值得关注的问题。

**论文标题**：OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

欢迎投稿！欢迎合作！