标题：UniRank：统一序列与特征交互的排名基准

今天给大家带来一个开放统一的推荐排名模型基准UniRank，专门用来标准化评估那些同时做序列建模和特征交互的先进模型，覆盖15个代表模型和5个超大规模公开数据集，最大超7亿样本和10万级行为序列。

🔑 关键方法  
1️⃣ 时间顺序点wise自回归监督：把用户所有行为（包括点击、曝光未点等）按时间排列，每个位置作为独立样本，用前面历史预测当前反馈，极大增加监督信号密度，更好利用长序列和多任务。  
2️⃣ 多任务标准化评估：统一构建点击、关注、点赞、购买等6种反馈任务的样本、标签、损失函数和指标（AUC/Logloss），确保不同模型在同一评价体系下可比。  
3️⃣ 分布式优化PyTorch工具包：集成DDP、混合精度训练、Flash Attention、算子编译（torch.compile）、激活检查点等加速技术，4卡训练提速14倍，单卡显存降低69%。

💡 核心创新  
1️⃣ 首个专门聚焦统一序列建模与特征交互排名的开放基准，解决此前私有数据、封闭实现导致的不可复现问题。  
2️⃣ 系统对比了15个模型（包括Stacked架构如HeMix/UniMixer，Layer-wise架构如TokenFormer/UltraHSTU），在QK-Video、KuaiRand、TAAC-25、淘宝、MerRec五个平台上的表现，揭示模型性能高度依赖数据集和任务。  
3️⃣ 提供“实战手册”级别的消融实验：涵盖token化策略、注意力激活函数、架构增强（如AttGate/RoPE）、优化器（SOAP/Muon）以及缩放定律研究，帮助从业者快速选型和调参。

📊 实验效果  
✅ UltraHSTU在QK-Video上4个任务（点击、关注、点赞、分享）中3个取得最佳AUC，TokenFormer在TAAC-25广告数据集上点击和转化AUC领先。  
✅ 无单一模型在所有数据集和任务上称霸：淘宝和MerRec上EST、HeMix更强，短视频场景统一模型更有优势。  
✅ UniRank的优化工具包使硬件需求大幅下降，16G显存可训练百亿级样本的长序列模型，加速比达14x。  
✅ 注意力激活推荐GeLU/SiLU，架构增强推荐AttGate（跨数据集最稳定收益），优化器推荐SOAP/LaProp。

大家觉得在推荐系统中，序列建模和特征交互应该先做哪个（Stacked）还是每层都融合（Layer-wise）？欢迎在评论区聊聊你的经验和看法！

论文：UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction

欢迎投稿！欢迎合作！