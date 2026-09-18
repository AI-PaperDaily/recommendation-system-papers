排序升级生成式推荐LIGE-GR

今天给大家带来Meta这篇把工业推荐系统从排序平滑升级到生成式推荐的LIGE-GR。它没有推倒重来，而是在原有itemwise排序模型上做列表级生成，思路很务实。

🔑 关键方法
1️⃣ 上下文感知打分：保留原context-free排序模型，只在中间表征上加一个4层causal Transformer，让候选视频打分时能看到已经选进列表的前缀。
2️⃣ 列表级价值模型ListVM：用continuation probability给后续位置加权，而不是把每个item价值简单相加，更接近用户真正刷到第N条的概率。
3️⃣ Palette解码器：基于RL式beam search逐位生成列表，用未来价值估计F̂剪枝；b=1时自动退化成传统贪心排序。

💡 核心创新
1️⃣ 升级而不是替换：LIGE-GR严格泛化现有推荐系统，CF模型、itemVM、贪心解码都能逐个回滚，技术风险低。
2️⃣ LLM范式

论文：LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era

欢迎投稿！欢迎合作！