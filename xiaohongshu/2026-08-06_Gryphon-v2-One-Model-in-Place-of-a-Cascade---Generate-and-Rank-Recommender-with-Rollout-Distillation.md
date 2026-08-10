标题：单模型终结多级瀑布流推荐

今天给大家带来一篇来自Yandex的推荐系统论文，讲的是如何用一个统一生成排序模型干掉传统多级级联架构。

🔑关键方法：
1️⃣ Shared-Encoder架构：用户历史只编码一次，SID解码器负责生成候选，轻量Ranking Module复用编码状态做最终排序，避免重复编码开销

2️⃣ Rollout Distillation：用训练专用的Teacher Ranker打分，覆盖两类候选——当前解码器beam search rollouts（占90%以上）+ 历史曝光日志，MAE loss蒸馏排序偏好

3️⃣ 联合训练目标：NTP loss更新生成路径，蒸馏loss更新Ranking Module和共享编码器，固定权重组合多任务分数得到最终排序

💡核心创新：
1️⃣ 把Reward-Based Post-training和Unified Generate-and-Rank两条技术线缝合，Teacher Ranker充当reward model角色但保持监督蒸馏范式

2️⃣ On-policy蒸馏思想引入推荐排序：用当前解码器rollout候选做蒸馏来源，不滞后、不设checkpoint lag，曝光日志作为补充anchor

3️⃣ 彻底解耦训练规模和serving成本：Teacher Ranker可看8000长度历史，学生模型只用2048，训练完teacher直接丢弃，serving路径零开销

📊实验效果：
✅ 线上A/B：单模型替换15+候选生成器+预排序+精排全链路，活跃用户+1.41%，总收听时长+1.62%，重复指令+15.25%，未完成率-9.65%，p<0.001

✅ 延迟与吞吐：端到端延迟和生产级联持平，吞吐是“生成器+Teacher Ranker在线推理”路径的4倍

✅ 离线指标：WPA 0.5892，教师Recall@10从0.04飙到0.5654，候选生成Recall@1000保持0.86不受损

一句话：生成模型负责召回，蒸馏Ranker负责排序，一个模型端到端搞定。

有个问题想请教大家：这种靠蒸馏teacher信号训练的生成式ranking，在线学习时teacher更新频率和student持续学习之间的稳定性，你们觉得怎么平衡比较好？在评论区聊聊？

论文：Gryphon-v2: One Model in Place of a Cascade - Generate-and-Rank Recommender with Rollout Distillation

欢迎投稿！欢迎合作！