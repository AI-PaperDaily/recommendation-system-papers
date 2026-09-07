UniCon上下文单元统一CTR

今天给大家带来一篇KDD工业推荐系统论文UniCon，核心是把历史曝光和当前候选都封装成“上下文单元”，用同一套层级注意力做排序。它不再把历史行为序列和当前候选当成两种异质信号，而是统一成一个结构。

🔑关键方法
1️⃣ 上下文单元构建：每次曝光里的商品、点击反馈、用户意图和环境打包成一个context unit，历史与目标候选共享schema。
1️⃣ 双层注意力：intra-context attention建模同屏商品竞争/互补的Locality；inter-context attention建模跨上下文决策状态演化的Dynamics。
1️⃣ 上下文压缩推理：逐层用target-aware TopK保留与当前目标最相关的历史上下文，配合变长注意力和AOT编译降本增效。

💡核心创新
1️⃣ 把老旧的“序列vs非序列”特征工程划分，换成统一上下文中心范式，消除历史与当前候选的结构错位。
1️⃣ 目标侧不是真实曝光列表，初始化一个latent context unit，并用曝光/绝对位置辅助任务约束它学习展示结构。
1️⃣ 工业可落地：padding-free变长注意力+上下文级序列压缩+动态shape编译，让大模型部署成本可控。

📊实验效果
✅ 离线AUC 0.8697，比生产Base提升0.0139；GAUC 0.8194，LogLoss 0.1991。
✅ 超过OneTrans、HyFormer、RankMixer及其+DSIN+CIM增强版，UniCon-Small已经超过所有研究基线。
✅ 压缩保留率r=0.5时，计算从801.29降到197.14 GFLOPs，AUC几乎不降；在线A/B RPM +3.09%、CTR +2.07%、Revenue +2.95%。
✅ 推理吞吐相比固定长度未压缩基线提升258.1%，延迟仍在生产SLA内。

你们做排序时，会把同屏曝光当成结构性上下文来建模吗？还是习惯把行为序列直接拍平？评论区聊聊~

论文：UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction

欢迎投稿！欢迎合作！