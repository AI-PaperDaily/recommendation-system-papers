三损失拆穿生成推荐补全陷阱

今天给大家带来一篇WWW’27生成式推荐训练的新工作，核心是把候选补全里“多出来的监督”和“组间概率竞争”拆开看，发现训练时补进去但线上不可服务的未召回目标，可能反而伤害真正能排的候选。

🔑关键方法
1️⃣ 设计三组匹配损失：WN固定召回目标权重只排返回候选；Cond加入未召回目标但分组归一化；Full把两组塞进同一个softmax里竞争概率。
2️⃣ Cond推导很直接：给appended组加一个共享偏移，解析地最小化Full，只去掉组间概率竞争，保留两组内部的排序损失。
3️⃣ 所有对比共用同一候选池、表征、scorer、优化器和推理pool，所以Cond vs Full测竞争，Cond vs WN测附加监督。

💡核心创新
1️⃣ 把“append missed targets”这个动作拆成三项：召回目标权重变化、附加监督、组间概率竞争，不再混在一起。
2️⃣ 提出Cond损失，让训练专用目标不跟线上会出现的候选抢总概率，训练更稳定。
3️⃣ 给出按generator开发集调整下界决定是否启用补全的规则，而不是默认补全。

📊实验效果
✅ RecIF-Ads上，60 epoch下Cond比Full的FT-NDCG平均提升0.0074，7个新训练run方向一致。
✅ A-Games预注册比较中，移除组间竞争提升FT-NDCG 7.8%–22.2%，四个用户区间全部排除零，三个训练run区间排除零。
✅ 给Full初始化正确appended组概率后FT-NDCG提升0.0092，Cond/WN不变，直接证明是概率划分在起作用。
✅ A-Health开发集规则全部保持仅召回训练，避免强制补全带来的1.7%损失；A-Cell对2/3生成器启用补全，整体收益仍偏弱。

这篇让我比较受启发的是，补未召回目标不是“多给label就一定好”，训练里的概率争夺会通过共享scorer参数传到推理排序。你在召回-重排链路里会给reranker补训练未召回目标吗？如果线下有、线上不可服务，你会怎么处理？

论文：Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition

欢迎投稿！欢迎合作！