OneLatent三段隐式推理推荐

今天给大家带来一篇把显式CoT压缩成三个可学习latent token、用多视角自适应监督做推荐基础模型隐式推理的论文OneLatent。

🔑关键方法
1️⃣ 用MV-ACoT造教师监督：从探索、评估、归因三个视角生成互补CoT，还让教师按样本复杂度自动调节推理长度，少一点模板味。
2️⃣ 做三阶段latent token对齐：先把单视角理由压进latent接口，再多视角对齐，最后接到推荐上下文和SID预测上。
3️⃣ 课程式后训练加Latent-Reason-then-Answer推理：线上只放三个latent token，直接解SID，不再自回归生成大段自然语言理由。

💡核心创新
1️⃣ 把显式推理轨迹蒸馏进固定数量的latent token，推理开销从“写长理由”变成“固定三token”，结构很轻。
2️⃣ 多视角加自适应复杂度，既缓解简单请求的过度思考，也降低单一推理路径带来的兴趣漂移。
3️⃣ 不只停在校内实验，还配了行为触发、异步物化、容量感知调度的生产服务，把FRM推理挪出在线关键路径。

📊实验效果
✅ 工业Kuaishou数据上，SID@64比显式Think变体提升17.44%，比No-Think变体提升9.33%。
✅ 相比显式CoT解码，推理吞吐约17.49倍，延迟基本回到不生成理由的水平。
✅ 9B规模也有效，SID@64到0.2008，超过同规模非latent最强结果约8.13%。
✅ 线上A/B本地生活广告场景，曝光+2.566%，广告花费+5.157%，预估收入提升9.6%。
✅ 通用能力分析里，latent监督下MMLU表现不降，推荐能力和通用推理更平衡。

你们更想看它三个latent token到底学到了什么，还是想看生产 serving 怎么扛住实时流量？

论文：OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

欢迎投稿！欢迎合作！