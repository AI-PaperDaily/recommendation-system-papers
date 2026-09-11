100K序列低秩缓存新结构

今天给大家带来一篇RecSys’26工业级推荐论文：SequenceO1用Sketch Attention把10万级用户行为序列压缩成固定大小的sketch，再做双时间尺度STCA，缓存命中路径直接做到O(1)，并在抖音全流量上线。

🔑 关键方法
1️⃣ Sketch Attention：用可学习原型对100K历史做prototype-wise归一化，每个token分配到多个原型，聚合成固定大小的用户侧sketch，长度与原始序列无关。
2️⃣ 双时间尺度STCA：对最近10K后缀做细粒度目标注意力，对固定sketch做超长期信号目标注意力，再融合进排序网络。
3️⃣ 缓存优先系统：训练侧local KVCache复用同用户sketch，推理侧跨请求复用，配合MRLB用户级批处理和FlashSA融合kernel，避免物化k×n大矩阵。

💡 核心创新
1️⃣ 可缓存低秩sketch：SA在长度维度做隐式低秩投影，得到目标无关的用户侧表示，能跨候选、跨请求复用。
2️⃣ 缓存命中路径O(1)：只要sketch可用，原始100K特征不用再存储、传输和计算，成本与原始长度解耦。
3️⃣ 端到端保留增益：相比直接扩展STCA，SequenceO1保留了83%的AUC增益，同时取代TWIN V2两阶段模块。

📊 实验效果
✅ 离线：相比生产STCA 10K+TWIN V2，Finish AUC +0.29%，偏好类指标UAUC最高+1.47%。
✅ 线上：Douyin Finish +2.33%、Duration +1.50%、Dislike -6.98%；Douyin Lite Finish +3.49%，低活用户提升更明显。
✅ 计算效率：100K下训练侧比直接STCA便宜49.9倍，推理侧便宜63.9倍。
✅ 压缩质量：1K×128预算下，SA比KMeans、均值池化、Lightning-Attention等压缩方法提升最大

论文：SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching

欢迎投稿！欢迎合作！