生成式检索SID拓扑冻结术

今天给大家带来一篇动态电商搜索里生成式检索的硬核实战：核心不是造新模型，而是更新时别重写SID映射，旧Item冻结、新Item插入，模型热启动就好。

🔑关键方法
1️⃣ 三级SID结构：前两层用RQ-VAE做语义量化码本，第三层追加业务价值排序token，头部唯一ID、尾部碰撞桶，兼顾精度和长尾覆盖。
1️⃣ 映射策略：冻结旧Item SID，只对符合条件的新Item做非侵入式插入，对比全量重映射和阈值重映射，直接观察损伤。
1️⃣ 训练策略：从上一轮checkpoint热启动，只重训User+Query→SID阶段，不引入复杂持续学习模块。

💡核心创新
1️⃣ 首次系统量化SID语义漂移：宽beam会掩盖真实损伤，beam=1时HitRate掉41.78%-62.05%，beam=1000只掉8.33%-10.85%，很多系统可能正活在假安全区。
1️⃣ 发现旧Item SID重映射比例γ与检索质量严格单调负相关，部分重映射没有折中空间。
1️⃣ 提炼出拓扑保持更新原则：旧SID冻结、新Item插入、能热启动就热启动，别做model soup。

📊实验效果
✅ 67天大规模电商搜索日志，两轮不重叠更新周期反复验证。
✅ Frozen+热启动旧Item HR@1000从0.7043升到0.7272，新Item HR@1000从0.5127升到0.6200。
✅ 新Item HR@1相对提升约109%，旧Item HR@1也有约10%提升。
✅ Full-Re即使热启动，旧Item指标仍低于不重训的Frozen基线，说明拓扑破坏靠短预算续训补不回来。
✅ 权重混合不如直接用新checkpoint。
✅ 天猫App三周线上A/B：UCTCVR +0.29%，GMV +1.16%。

你们做生成式检索更新时，是更倾向全量重映射求“新鲜”，还是也发现旧SID最好别动？评论区聊聊~

论文：Don't Rewrite the SID Map: Topology Preservation for Generative Retrieval in Dynamic E-Commerce Search

欢迎投稿！欢迎合作！