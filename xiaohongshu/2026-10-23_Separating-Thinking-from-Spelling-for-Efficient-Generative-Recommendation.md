Deco概念解码：重读轻写

今天给大家带来一篇把生成式推荐里“读历史”和“写SID”拆开的高效论文：重骨干只读一次出概念，轻解码器负责跑完整的trie约束beam search。

🔑关键方法
1️⃣ 历史融合：每个物品的K个SID码经MLP压成一个融合token，骨干输入从T×K降到T，先把读历史的成本打下来。
2️⃣ 概念生成：重骨干读一遍历史，把最后状态线性映射成紧凑的next-item concept z，实验中M=2，作为后续候选生成的用户状态。
3️⃣ 委托解码：轻量KV-cached因果解码器只看z和已生成前缀，在SID trie约束下跑完整beam search，SID码本和trie结构都不动。

💡核心创新
1️⃣ 把“用户建模”和“拼SID”明确分工，beam search不再反复过重骨干；候选扩展的计算从重骨干O(BK) token转到轻解码器。
2️⃣ 概念就是服务状态：z约1KiB bf16，可预计算缓存，在线只跑轻解码；相比历史KV cache小很多，长历史场景更友好。
3️⃣ 训练上可只用CE，也可从全骨干教师蒸馏；蒸馏有点像teacher-informed label smoothing，重复消费重的场景再加history copy pointer。

📊实验效果
✅ 七个电商、电影、音乐数据集上，Deco准确率与全骨干教师相当或更好，Deco-flat也匹配或超过FlatGR。
✅ 生产式流水线里，50ms p95延迟预算下吞吐约为全骨干基线的3.2–8.6倍；batch 128下比FusedGR高约2.8–5.1倍。
✅ 候选预算越大越划算：Deco生成1000个候选比教师生成100个还快；1.5B LLM骨干下吞吐追平69M教师，NDCG@10还高约11%。
✅ SID深度从K=2加到8，Deco变慢2.9倍，FusedGR慢7.3倍、FlatGR慢7.9倍；轻解码器1–4层宽128–512与默认差约3%，单层还能再降20–28%延迟。

如果做推荐系统，你会把beam search交给轻解码器吗？还是更想先看它和FusedGR的延迟对比拆解？

论文：Separating Thinking from Spelling for Efficient Generative Recommendation

欢迎投稿！欢迎合作！