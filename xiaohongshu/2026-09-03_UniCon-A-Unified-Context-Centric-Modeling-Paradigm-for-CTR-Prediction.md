UniCon上下文单元重塑CTR

今天给大家带来一篇KDD’27的统一上下文中心CTR预估工作UniCon，核心是把历史曝光和当前候选都抽象成同构的context unit，让模型在上下文层级做统一建模，而不是继续拆成序列和非序列两套路径。

🔑关键方法
1️⃣ 上下文单元建模：把同屏展示的item、用户意图、环境信号打包成一个context unit，历史与目标共享同一schema，直接用target latent context unit承接候选集。
2️⃣ 双层上下文注意力：Intra-context attention在单元内部建模局部竞争与互补，即Locality；Inter-context attention跨单元建模用户兴趣与展示环境演变，即Dynamics。
3️⃣ 上下文级序列压缩：用target-aware TopK按相关性逐步保留历史context，叠加变长注意力和AOT编译，降低长历史在线推理成本。

💡核心创新
1️⃣ 把传统“行为序列 vs 当前候选”的异构输入，重构为同构上下文单元序列，从结构上消除历史与目标之间的gap。
2️⃣ 目标侧不是孤立打分，而是用候选集初始化latent context unit，并通过曝光/位置辅助任务还原最终展示结构。
3️⃣ 上下文边界直接作为计算边界，压缩、变长注意力、候选分片都围绕context unit展开，工业部署更高效。

📊实验效果
✅ 美团搜索广告离线AUC从Base的0.8558提升到0.8697，GAUC和LogLoss均为最优。
✅ 在线A/B 7天：RPM +3.09%，CTR +2.07%，Revenue +2.95%，统计显著。
✅ 开启压缩后GFLOPs从801.29降到197.14，吞吐提升258.1%，AUC只掉0.0001。
✅ 同参数量下，UniCon持续超过OneTrans、HyFormer、RankMixer及其+CIM/DSIN变体。

你们线上排序还在把历史序列和当前候选当两条独立路径吗？欢迎评论区聊聊上下文单元化改造的落地经验。

论文：UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction

欢迎投稿！欢迎合作！