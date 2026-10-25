3个潜在token替换长CoT

今天给大家带来快手OneLatent，把推荐大模型里又长又慢的显式思维链，压缩成3个可学习潜在推理token，推理吞吐直接快17倍。

🔑关键方法
1️⃣ MV-ACoT多视角自适应CoT：用探索、评估、归因三个视角做教师监督，还让teacher按样本复杂度自己调整推理长度，简单case不硬编长逻辑，复杂case才展开多维度比较。
2️⃣ 三阶段潜在token对齐：先单视角、再多视角对齐rationale，最后把同一个latent接口接到SID预测，逐步把CoT知识蒸馏进3个固定token。
3️⃣ 课程式后训练：按用户活跃度从高到低做课程SFT，再混合常规推荐和潜在推理两种模式，保住直接推理能力，同时激活latent reasoning。

💡核心创新
1️⃣ 提出Latent-Reason-then-Answer范式，用固定3个潜在token替代自然语言CoT，推理开销不再随rationale长度涨。
2️⃣ 不加额外encoder/projector，潜在token直接走因果注意力，和现有FRM结构无缝兼容。
3️⃣ 整套MV-ACoT蒸馏+渐进对齐+课程训练流程，让隐式推理既可训练、又能在线部署。

📊实验效果
✅ 工业数据集SID@64比Think提升17.44%，比No-Think提升9.33%。
✅ 推理吞吐比显式CoT高17.49倍，延迟从55.59s降到3.17s，接近无CoT直接生成。
✅ 9B大骨干同样有效，SID@64达0.2008，比同规模非潜在基线提升8.13%。
✅ 快手本地服务广告线上A/B：广告消耗+5.157%，预估收入+9.6%。

你觉得这种“潜在token内化CoT”的思路，会比直接让大模型边想边推更适合推荐场景吗？

论文：OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

欢迎投稿！欢迎合作！