低秩缓存+草图注意力

今天给大家带来字节跳动在RecSys’26的SequenceO1，用Sketch Attention把100K用户历史压成固定大小草图，再做双时间尺度STCA，配合低秩缓存直接端到端上线。

简单说，就是把10万级行为历史压缩成几百到一千个原型表示，命中缓存时基本不用再碰原始长序列。

🔑 关键方法
1️⃣ Sketch Attention：用可学习原型对100K历史做prototype-wise归一化，每个token按权重分配到多个原型上，聚合成512~1K大小的用户侧sketch，长度不再随原始序列涨。
2️⃣ 双时间尺度STCA：近期10K后缀直接做target-to-history交叉注意力抓时效；超长历史只对固定sketch做STCA，再和近期分支做轻量融合

论文：SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching

欢迎投稿！欢迎合作！