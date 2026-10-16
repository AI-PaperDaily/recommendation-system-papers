双分支图蒸馏冷启动不崩

今天给大家带来一篇用双分支图知识蒸馏做冷启动推荐的论文 DualGCN，语义学生分支不干扰协同主结构，却能给零交互新物品生成靠谱表示。

🔑 关键方法
1️⃣ 协同教师分支：在ID嵌入里注入冻结语义先验，经过图卷积后得到拓扑平滑的协同参考坐标。
2️⃣ 语义学生分支：只用标题+类目语义，过轻量MLP并加残差，把语义方向映射进协同空间。
3️⃣ 两阶段解耦训练：先训教师到收敛，再冻结教师单独蒸馏学生，梯度不打架，暖启动性能不倒退。

💡 核心创新
1️⃣ 拓扑平滑语义先验：把语义向量当静态锚点写进第0层，重塑协同流形，又避免微调带崩预训练先验。
2️⃣ 幅度无关余弦蒸馏：不硬套MSE欧氏距离，只在单位超球面拉近师生方向，绕开冷启动物品缺交互频率先验的问题。
3️⃣ 动态范数重建：推理时按交互度路由，冷启动物品用学生方向向量×暖物品平均范数，尺度对齐后再全局检索。

📊 实验效果
✅ Amazon-Books/Electronics/Sports 三个真实数据集上，严格冷启动 Recall@20 全面领先，Books 从 LightGCN 的 0.0027 拉到 0.1007。
✅ Sports 冷启动 Recall@20 比 DropoutNet 再提 20.8%，NDCG 同步提升。
✅ 暖启动不牺牲，Electronics 和 Sports 上 DualGCN 第一阶段 Recall@20 相比 LightGCN 分别提升 39.7% 和 23.3%。
✅ 消融显示余弦蒸馏、语义残差和冻结语义先验都很关键，InfoNCE

论文：Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation

欢迎投稿！欢迎合作！