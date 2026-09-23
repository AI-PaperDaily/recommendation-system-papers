早融合GradCIR分级检索

今天给大家带来一篇电商视觉搜索新论文：GradCIR，用4级相关性标签训练组合多模态检索，让“差不多”的商品排序更聪明。

🔑关键方法
1️⃣ VLM自动造数据：Gemini-2.5-pro做物体检测+修饰词合成生成查询，再当裁判打4档标签，全程无需人工标注。
2️⃣ 迭代难负挖掘：用训练中的检索器持续挖难负样本，3轮扩到350万对分级样本。
3️⃣ 层级感知AngleLoss：把Exact>NearExact>PartialMatch>Irrelevant的序关系变成角度约束，拒绝二元一刀切。

💡核心创新
1️⃣ 首次将CIR从单正目标三元组升级为4级有序相关性，更贴合真实电商目录的“部分匹配”场景。
2️⃣ 方法论backbone无关，套在PaliGemma2、SigLIP2、Qwen3-VL-Embedding上都能涨，增益来自配方本身。
3️⃣ 早期融合VLM做Siamese双塔，用[QUERY]/[ITEM]前缀区分角色，Matryoshka降维上生产。

📊实验效果
✅ 内部WVST上GradCIR-PG2全面最优，比最强基线Qwen3-VL-Emb+GradCIR平均Adj-R@5(Exact)高2.83%，NDCG@10高0.91%。
✅ 消融显示分级监督比二元监督NDCG@10提升4.9%-5.9%，细分NearExact和PartialMatch很关键。
✅ 同配方迁移其他多模态编码器，早期融合backbone最高涨8.5% NDCG@10。
✅ 公开FashionIQ有监督Avg Recall 0.6703，略超SPN4CIR的0.6641；零样本0.40，打平或超CLIP-L方法。
✅ 已上线Walmart视觉搜索，服务真实用户流量。

你觉得分级相关性是不是比传统二元正负样本更符合电商搜索体验？

论文：Graded-Relevance Composed Multimodal Retrieval for E-commerce Visual Search at Scale

欢迎投稿！欢迎合作！