ASC/K-ASC：自适应Swing计算

今天给大家带来一篇SIGMOD/PODS论文，针对大规模推荐系统i2i检索里的Swing相似度计算，提出ASC和K-ASC，把高degree query的耗时从秒级甚至小时级拉到毫秒级。

🔑关键方法
1️⃣ GNS（Grouped Naïve Sampling）：把重复采样的user pair先分组计数，只对去重pair做集合交，减少冗余交集操作。
2️⃣ USS（User Subset Sampling）：对高degree用户采样子集，只算交集基数不物化完整集合，绕开大集合相交。
3️⃣ ASC自适应选择：根据query item degree和成本模型，在GNS和USS间动态切换，低degree走USS，高degree走GNS。

💡核心创新
1️⃣ 提出(ε,λ)-近似Swing定义，同时给相对误差和加性误差的概率保证，让近似计算有理论可依。
2️⃣ 把GNS和USS统一到ASC框架，用analytical cost model做query-adaptive选择，替代工业界粗暴的600截断，质量不降。
3️⃣ K-ASC引入filter-refinement：先小样本粗筛c

论文：Efficient Swing Computation for Retrieval in Large-Scale Recommender Systems

欢迎投稿！欢迎合作！