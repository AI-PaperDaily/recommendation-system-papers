# 快手OneLatent：3枚潜在token替代显式CoT，SID@64提升17.44%，推理吞吐17.5倍？

推荐系统进入大模型时代后，显式 Chain-of-Thought（CoT）逐渐暴露出两类问题：简单请求生成冗长推理，造成“过思考”和低推理 ROI；固定单路径推理可能过度强调历史偏好，忽略短期上下文，产生“兴趣漂移”。快手团队提出 **OneLatent**，将显式推理压缩为三个可学习潜在 token，使推荐基础模型以固定开销完成“潜在推理后回答”，不生成自然语言 rationale。

![图1：显式CoT的局限性](https://arxiv.org/html/2610.30675v1/fig/intro.png)

## 一、核心方法

OneLatent 的训练流程包含三部分：多视角自适应 CoT 监督、三阶段潜在 token 对齐、课程式后训练。其核心思想是：显式 CoT 仅作为训练监督，不被解码；教师模型生成的推理痕迹通过若干可学习潜在 token 压缩进模型内部。

![图2：OneLatent总体架构](https://arxiv.org/html/2610.30675v1/fig/OneLatent.png)

### 1. MV-ACoT：多视角自适应 CoT 监督

MV-ACoT 从三个互补视角构建教师监督：

- **Exploration**：只给用户画像和历史行为，生成需求假设和商业意图列表；
- **Evaluation**：额外给出真实交互的正样本，生成匹配评估 rationale 及 1–5 分；
- **Attribution**：把正交互作为已知事实，解释是哪些前置信号促成转化。

对于每个视角，教师模型根据当前证据复杂度动态调节推理过程，而不是套用固定模板。其生成目标可写为：

$$
Y_{u}^{(v)}\sim p_{\mathrm{T}}\!\left(\cdot\mid\pi_{v}^{\mathrm{adapt}}(G_{u}^{(v)})\right)
$$

简单请求被要求快速收敛，复杂请求则进行多维度比较，从而减少低回报的冗余推理。

### 2. 潜在 token 对齐

OneLatent 在 FRM 词表中增加三个固定潜在 token：

$$
\mathbf{L}=[\langle L_{1}\rangle;\langle L_{2}\rangle;\langle L_{3}\rangle].
$$

三阶段对齐依次进行：

- **Stage I 单视角对齐**：用保留的单个视图输出热身，序列为 $[X_u;\langle v_u^\star\rangle;\mathbf{L};\widetilde{Y}_u^{(v_u^\star)}]$，只在输出 token 上计算损失；
- **Stage II 多视角对齐**：对每个可用视图独立构造 $[P_u;H_u;\langle v\rangle;\mathbf{L}]$，平均 token 级损失；
- **Stage III 推荐对齐**：去掉文本监督，直接建立 latent-token 条件到下一个 SID 的映射。推荐对齐阶段的输入定义为 $X_u^{\mathrm{ctx}}=[P_u;H_u;\mathcal{C}_u;\mathbf{L}]$，其目标如下：

$$
\mathcal{L}_{\mathrm{align}}=-\mathbb{E}\left[\log p_{\phi}\left(\mathcal{S}_{u}^{+}\mid X_{u}^{\mathrm{ctx}}\right)\right]
$$

### 3. 课程后训练与在线服务

后训练采用从高活跃用户到低活跃用户的课程策略，并以 1:1 比例混合标准 SFT 和潜在推理 SFT，使模型既能直接预测，也能在 latent token 条件下预测。在线推理时，模型只解码三个 SID token：

$$
\widehat{\mathcal{S}}_{u}=\operatorname*{arg\,max}_{\mathcal{S}}p_{\phi}(\mathcal{S}\mid P_{u},H_{u},\mathcal{C}_{u},\mathbf{L})
$$

![图3：生产在线推理架构](https://arxiv.org/html/2610.30675v1/fig/serving_architecture.png)

生产环境中，即使潜在推理已削减解码开销，同步调用全量流量仍超过延迟预算。OneLatent 采用行为触发准入、容量感知调度和异步结果物化，将 FRM 推理保持在在线请求关键路径之外。更新后的候选进入现有检索管道，与现有候选在精排阶段竞争。

## 二、实验

实验覆盖快手本地生活工业数据集和公开的 Kuaishou LLM-Rec 基准，主模型使用 Qwen3.5-0.8B。

**OneLatent-Merge** 在公共和工业数据集的所有指标上均取得最佳。工业 SID@64 相比 OneReason-CoT Think 提升 17.44%，相比 No-Think 提升 9.33%；推理吞吐达到 274.29 QPS，是显式 CoT 的 17.49 倍。公共基准 SID@64 达到 0.0239，相比匹配的显式 Think 配置相对提升 169%。

在 9B 骨干规模上，OneLatent 的 SID@64 达 0.2008，比同规模非隐式最优值高出 8.13%，说明收益可跨模型规模迁移。

教师 CoT 质量方面，LLM-as-a-Judge 评分显示 Evaluation 视角综合分 4.33，Exploration 为 4.16，均明显高于 OneReason-CoT 的 3.14。自适应策略将平均 CoT 长度降低 35%，复杂轨迹语义相似度约 0.92。

![图4：不同CoT策略下OneReason-Think与OneLatent的SID@64对比](https://arxiv.org/html/2610.30675v1/sid_trend_mixed_vs_latent_side_www.png)

在线 A/B 测试持续 7 天，在快手本地服务广告子场景中，OneLatent 曝光提升 2.566%，广告花费提升 5.157%，全局后验估计收入提升 9.6%。

## 三、展望

OneLatent 为推荐基础模型的大规模工业化提供了一条可行的潜在推理路径。固定 token 开销、离线 CoT 监督和异步服务架构共同降低了显式推理的部署负担。未来若将潜在 token 与强化学习或更强的多视图监督结合，有望在保持可扩展性的同时进一步提升推荐决策质量。

在推荐系统中，显式可解释性与推理效率是否必然冲突？

**论文标题**：OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

欢迎投稿！欢迎合作！