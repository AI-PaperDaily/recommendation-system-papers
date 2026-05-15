标题：MLP蒸馏推荐，8.7倍加速

今天给大家带来一篇用MLP替代Transformer进行蒸馏加速生成式推荐的新工作Sid-Mlp。

🔑关键方法  
1️⃣ 观察到语义ID（SID）生成中，后续token的搜索空间急剧坍缩，因此无需全注意力解码，只用简单结构即可  
2️⃣ 只用一个轻量注意力层一次性提取用户全局上下文，然后每个位置用独立的MLP头进行顺序预测，彻底消除重复注意力计算  
3️⃣ 通过知识蒸馏将教师Transformer解码器的行为迁移到MLP头上，训练时结合KL散度和交叉熵

💡核心创新  
1️⃣ 提出MLP中心的蒸馏框架，将自回归解码简化为一次上下文提取+MLP前向，无需KV缓存、无需验证步骤，实现即插即用加速  
2️⃣ 彻底移除解码器侧的自注意力和交叉注意力，beam search仅靠批量化MLP投影完成，延迟大幅降低  
3️⃣ 进一步将蒸馏扩展到编码器（Sid-Mlp++），用角色MLP替代Transformer encoder，解锁额外加速，速度提升至10.25倍

📊实验效果  
✅ 在三个Amazon数据集（Instruments、Scientific、Games）上，Sid-Mlp匹配甚至超越教师TIGER的NDCG@10，平均加速8.74倍  
✅ 峰值内存降低95.7%，batch size到512时仍低于1.5GB，而教师模型需30GB  
✅ 对多种tokenizer（RQ-KMeans、RQ-VAE、PSID）和beam size（10~50）均保持稳定，无性能回退

大家觉得在推荐系统里，用简单MLP替代复杂Transformer是不是趋势？欢迎评论区讨论。

论文：MLPs are Efficient Distilled Generative Recommenders

欢迎投稿！欢迎合作！