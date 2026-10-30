原生VLM直出商品SID

今天给大家带来淘宝拍立淘PailitaoMMSearch2.0技术报告，它用原生VLM把图像加自由文本查询直接生成商品Semantic ID，做成统一端到端多模态生成式检索。

🔑关键方法
1️⃣ 数据扩展：一套可复用工具链做粗筛、细筛、查询生成、CoT和质量控制，按意图和生命周期拼成冷启动、真实流量精修、持续生产流程。
2️⃣ 意图扩展：先CPT补电商商品和SID词表知识，再做多意图混合预算SFT，把相似、编辑、匹配和no/low/high-think装进同一个模型，最后统一RL联合优化。
3️⃣ 场景扩展：用0.8B/2B/4B模型容量加推理预算和beam配置，切出不同质量-成本操作点，线上不用额外意图分类器选专家。

💡核心创新
1️⃣ 统一开放多模态生成式检索：查询不显式给意图，模型从图和指令里自己推断相似、编辑、匹配关系，直接解码商品SID。
2️⃣ 三维scaling框架：数据工具集、多阶段训练、场景操作点串成一条工业落地链路，不是只改SID构造。
3️⃣ Sample–Beam RL：推理前缀采样、SID部分beam解码，让RL rollout更贴近线上候选集生成；相比SFT macro ER提升5.97点。

📊实验效果
✅ 离线2B Low-think拿到macro NDCG 0.918、macro ER 55.1%，平均响应278ms、单卡；对比内部召回排序pipeline，ER高5.6点、RT低64.4%。
✅ 匹配意图很能打：2B模型匹配NDCG 0.955、ER 74.3%，比两个rewrite系统高13.2和7.0点，比内部pipeline高30.1点。
✅ Sample–Beam RL比SFT的macro ER高5.97点；标准RL先加3.67点，Sample–Beam再补2.30点。
✅ 淘宝拍立淘线上A/B：商品页浏览+17.90%，交易+27.04%，GMV+8.94%，统计显著。

你们更想看数据工具链怎么控质、Sample–Beam RL怎么对齐，还是线上A/B怎么设计？评论区聊聊。

论文：PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

欢迎投稿！欢迎合作！