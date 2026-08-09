今天给大家带来一篇能让你少写一整套推荐系统的论文——Gryphon-v2，把多级级联直接压成一个模型，在线效果还涨了1.41%活跃用户。

🔑关键方法
1️⃣ 共享编码器的Generate-and-Rank架构：一个历史编码器，同时喂给SID解码器生成候选和Ranking Module排序，彻底告别多级级联的重复编码。 
2️⃣ Rollout Distillation（滚动蒸馏）：用训练时才有的超大Teacher Ranker打分，把排序偏好蒸进轻量级Ranking Module，线上不增加任何额外推理负担。 
3️⃣ 双候选源训练：当前解码器beam search采样的rollout候选（占90%以上）+ 线上真实曝光日志，兼顾on-policy分布和真实流量覆盖。

💡核心创新
1️⃣ 把“候选生成+粗排+精排”这三级流水线（15+个候选生成器）全部塞进一个模型，编码一次用户历史，直接输出最终排序。 
2️⃣ Teacher Ranker只在训练时存在，蒸馏目标本身就是排序信号，绕开了RLHF式复杂奖励建模，纯监督学习搞定多目标排序。 
3️⃣ 对比Gryphon的next-item监督，蒸馏让T-R@10从0.039飙到0.565，直接证明Teacher的排序偏好才是关键。

📊实验效果
✅ 在线A/B：活跃用户+1.41%，总听歌时长+1.62%，重复播放+15.25%，未完成率-9.65%
✅ 排序质量WPA达到0.589，接近用一年数据训练的Production Ranker（0.614），而模型只用两周训练数据
✅ 延迟与生产级联持平，但吞吐量是“生成+Teacher”方案的4倍，候选从10000压到1200

一个模型干翻一整套级联，这个压缩效率你会给几分？感觉生成式推荐是不是要统一天下了？评论区聊聊。

（滑到最下面🈶彩蛋：论文里还对比了MSE/MAE/Huber/KL四种蒸馏损失，MSE在T-R@10上最强，但线上用的MAE也不差，细节控可以去看原文）

论文：Gryphon-v2: One Model in Place of a Cascade - Generate-and-Rank Recommender with Rollout Distillation

欢迎投稿！欢迎合作！