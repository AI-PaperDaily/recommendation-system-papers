三维扩展统一多模态检索

今天给大家带来的是PailitaoMMSearch2.0，一篇把生成式检索做成统一端到端多模态搜索的工作：输入原图和自由文本指令，模型直接生成商品Semantic IDs，并用数据、意图、场景三个维度做扩展。

🔑关键方法
1️⃣可组合数据工具链：把粗筛、细筛、查询生成与修复、CoT生成、质量控制串成冷启合成、真实流量精炼和持续生产工作流。
2️⃣多阶段意图训练：先做电商CPT适配商品与SID词表，再做多意图混合预算SFT，最后用意图均衡的统一RL，把相似、编辑、搭配检索放进一个策略。
3️⃣场景缩放：用0.8B/2B/4B模型容量加No-think/Low-think/High-think推理预算，给不同业务暴露质量与成本操作点，不额外加意图路由。

💡核心创新
1️⃣统一开放多模态生成检索：不显式输入意图，也不靠意图分类器分专家，模型自己从图文指令推断检索关系。
2️⃣Sample-Beam Rollouts：推理前缀采样、商品SID用beam解码，让RL rollout更贴近线上候选集生成。
3️⃣三维scaling框架：数据扩监督、意图扩策略、场景扩部署，把“怎么学”和“怎么用”拆开。

📊实验效果
✅离线2B Low-think配置macro NDCG 0.918、macro ER 55.1%，平均RT 278ms，单GPU可跑。
✅Sample-Beam RL比SFT的macro ER高5.97个点，编辑和相似检索的增量尤其明显。
✅淘宝拍立淘在线A/B显著：商品页浏览+17.90%，交易+27.04%，GMV+8.94%。

你们更想看这套三维扩展怎么落地，还是Sample-Beam RL的训练细节？

论文：PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

欢迎投稿！欢迎合作！