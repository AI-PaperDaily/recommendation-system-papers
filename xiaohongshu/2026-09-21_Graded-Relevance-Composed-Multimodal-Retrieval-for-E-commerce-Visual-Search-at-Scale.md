GradCIR分级双塔检索

今天给大家带来一篇电商视觉搜索的硬核论文，GradCIR用4级相关性标签重构组合多模态检索，让排序不再只是“相关/不相关”。

🔑 关键方法
1️⃣ VLM全自动数据管线：Gemini-2.5-pro同时负责物体检测、modifier文本生成和4级相关性标注，零人工标注产出3.5M训练对。
2️⃣ 迭代硬负样本挖掘：边训练边用当前检索器挖难负样本，再丢给VLM打标，三轮收敛，线上难case覆盖更足。
3️⃣ 层次感知AngleLoss：把Exact/NearExact/PartialMatch/Irrelevant四级标签变成嵌入空间的角度序约束，而不是压成二元正负。

💡 核心创新
1️⃣ 分级相关性建模：在CIR里首次用4级有序标签替代单一正目标，保留“部分匹配”的排序信息。
2️⃣ 骨干无关训练配方：同一套GradCIR recipe可迁移到FashionCLIP、SigLIP2、GME-Qwen等，晚融合backbone最多涨8.5% NDCG@10。
3️⃣ 早期融合VLM双塔：PaliGemma2去掉生成头做encoder，role-specific前缀+last-token pooling+Matryoshka维度伸缩，兼顾效果和部署。

📊 实验效果
✅ Walmart内部WVST上，GradCIR-PG2在Home和Fashion垂直的NDCG@10全面领先，modifier查询增益最大。
✅ 分级vs二元ablation：只改监督粒度，NDCG@10提升4.9%-5.9%，证明4级标签是关键。
✅ FashionIQ监督微调平均Recall 0.6703，略超SPN4CIR，达到SOTA；zero-shot也匹配CLIP-L类方法。
✅ 多个backbone接入GradCIR配方平均Adj-R@5 Exact提升24.3%，数据管线价值与骨干解耦。

你们做检索排序还在用硬二元标签吗？还是已经尝试分级相关性了？评论区聊聊～

论文：Graded-Relevance Composed Multimodal Retrieval for E-commerce Visual Search at Scale

欢迎投稿！欢迎合作！