三损失分离补全监督与竞争

今天给大家带来一篇WWW'27生成式推荐工作：训练时补全生成器missed targets，看似只是多给监督，实际上同时改了权重、加了监督、还让两组候选抢softmax概率，论文用三个损失把这几个变化拆开，发现真正伤排序的是概率竞争。

🔑 关键方法
1️⃣ 设计WN/Cond/Full三损失：WN只保留已检索目标权重；Cond把补全目标单独归一化训练，不跟推理候选抢概率；Full再把两组放进同一个softmax，引入组间竞争。
2️⃣ Cond推导有点巧：给补全组加统一偏移，再解析最小化Full，刚好消掉跨组概率竞争，只保留两组内部的排序损失。
3️⃣ 固定生成器、候选池、scorer、优化器和推理池，用Cond−WN看补全监督、Cond−Full看概率竞争，变量唯一才可解释。

💡 核心创新
1️⃣ 把candidate completion拆成三项：检索目标权重被压低、额外监督、训练组与推理组概率竞争，并逐一分离。
2️⃣ 证明概率竞争会通过共享scorer参数改变推理时已返回候选的排序，哪怕补全物品推理阶段根本不在。
3️⃣ 提出按每个generator的开发集置信下界决定是否启用补全训练，不默认全局append。

📊 实验效果
✅ RecIF-Ads上60 epoch：Cond比Full的FT-NDCG高+0.0074，训练过程平均高+0.0124，7个新run方向一致。
✅ Amazon Video Games预设4项比较：去掉组间竞争提升7.8%–22.2%，用户区间全排除零。
✅ 直接偏移测试：给补全组加统一偏移，Cond数值不变，Full明显变化，锁定是概率竞争在起作用。
✅ 开发集规则在A-Cell允许2/3生成器补全，在A-Health全部拒绝，成功避免1.7%的FT-NDCG损失。

你们做生成式推荐时会自动补全missed targets吗？会不会按generator单独做开发集验证？评论区聊聊～

论文：Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition

欢迎投稿！欢迎合作！