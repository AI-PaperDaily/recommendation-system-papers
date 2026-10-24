三枚潜在Token替代CoT

今天给大家带来快手最新的基础推荐模型OneLatent，它用三枚潜在Token替代冗长的显式CoT推理，既保效果又快得离谱。

🔑 关键方法
1️⃣ MV-ACoT多视角自适应CoT：让教师模型从意图探索、正样本评估、行为归因三个视角生成推理监督，并根据样本复杂度自适应调整推理长度，避免“过度思考”。
2️⃣ 三阶段潜在Token对齐：先单视角预热、再多视角对齐、最后接入SID预测，把显式推理轨迹逐步压缩进三个可学习的潜在推理Token。
3️⃣ 课程式后训练：按用户活跃度从高到低分课程做SFT，并1:1混合潜在推理与标准推荐样本，稳定激活潜在推理接口。

💡 核心创新
1️⃣ 提出Latent-Reason-then-Answer范式，推理开销恒定，只解码三个潜在Token后直接生成语义ID（SID），不用生成自然语言rationale。
2️⃣ 无需额外encoder或projector，直接复用FRM因果注意力，在潜在Token位置隐式编码用户上下文与多视角推理信号。
3️⃣ 配套工业级异步serving：行为触发准入+容量感知调度，FRM推理离线完成并缓存Redis，在线请求不直接打模型。

📊 实验效果
✅ 快手工业数据集SID@64提升17.44%，对比显式Think变体；对比No-Think提升9.33%
✅ 推理吞吐比显式CoT高17.49倍，延迟几乎和No-Think一致
✅ 公开LLM-Rec benchmark SID@64相对Think-OR提升169%
✅ 快手本地服务广告线上A/B：展示+2.566%、消耗+5.157%、预估收入+9.6%

你觉得大模型推荐系统应该“说出来”推理过程，还是让潜在Token偷偷想清楚？评论区聊聊你的看法～

论文：OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

欢迎投稿！欢迎合作！