2x2用户画像结构大PK

今天给大家带来一篇生产级流媒体推荐论文，核心问题很直接：LLM做用户画像，到底什么时候才值回算力成本？

🔑关键方法
1️⃣ 2x2画像结构：表示类型分成聚合质心（SBERT物品embedding求均值）和LLM叙事画像（自然语言摘要再SBERT编码）；时间处理分成全历史和短期/长期注意力融合，组合出C、TC、N、TN四种策略。
2️⃣ 下游结构统一：画像向量和物品embedding拼接后进两层MLP打分，物品encoder、训练目标全部共享，只让画像构建方式变化。
3️⃣ 用户分层评估：按测试历史是否出现过训练物品，把用户分成Explorer和Non-Explorer，同时看准确率、覆盖率、新颖度等指标。

💡核心创新
1️⃣ 把LLM画像和聚合画像放进同一个2x2设计空间，干净隔离画像结构的影响。
2️⃣ 发现效果不是谁全面赢，而是分段条件性：聚合画像对习惯型用户更强，LLM画像在探索型用户上反超。
3️⃣ 揭示LLM画像的流行度吸引器效应：单用户列表多样性微升，但系统整体覆盖率和推荐新颖度大幅下降。

📊实验效果
✅ 全量用户上，Narrative和TemporalNarrative的Recall@10相对Centroid分别掉约31%和22%，聚合方法基本不输。
✅ Non-Explorer用户里，LLM画像全面吃亏：Narrative Recall@10约-32%，TemporalNarrative约-22%。
✅ Explorer用户出现反转：Narrative Recall@10约+19%，TemporalNarrative约+14%，TemporalCentroid反而约-7%。
✅ LLM画像Diversity@10约+4%，但Coverage约-80%，Novelty约-18%到-20%，推荐明显挤向热门内容。
✅ 时间窗口PCT在15%-25%稳定，30%明显退化，Explorer上TemporalNarrative最惨约-19%。

所以论文的落点很务实：聚合画像还是稳的默认选项，LLM画像只适合探索型用户，最好做路由式部署，别全量上。

你们在推荐系统里，LLM画像会全量铺开，还是也做用户分层路由？评论区聊聊～

论文：When LLM-Based User Profiling Adds Value in Production Streaming Recommendation

欢迎投稿！欢迎合作！