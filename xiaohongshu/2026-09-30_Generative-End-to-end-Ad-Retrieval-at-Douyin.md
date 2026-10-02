字节GEAR生成式广告检索框架

今天给大家带来字节在抖音广告场景落地的生成式端到端广告检索框架GEAR，它把推荐问题直接改成“自回归生成item token序列”，并且同时解决表征坍缩和item碰撞这两个工业级瓶颈。

🔑 关键方法
1️⃣ BasisVQ：用可学习正交基重新参数化码本，梯度全局共享，几何上相当于让整个码本空间做刚性旋转，未激活码不会畸变。
2️⃣ Prefix-aware BasisRQ：把前几层量化码信息聚合成缩放γ和偏置β，以仿射变换作用到当前层码本，提升表达力，同时用快速距离计算保持和普通RQ一样的O(bN_l d)复杂度。
3️⃣ 轻量rerank head：生成器解码完token序列后多加一步解码，复用自回归隐藏态和item tower embedding做点积打分，专门区分共享同一token序列的碰撞item

论文：Generative End-to-end Ad Retrieval at Douyin

欢迎投稿！欢迎合作！