## 快手×港大发布 OneLA：大 beam 线性注意力解码提速 2.46 倍，如何做到？

生成式推荐（GR）正在成为大规模推荐系统的核心范式，但它在部署时面临一个特殊挑战：为了生成数百个候选物品，系统必须依赖大 beam 解码。传统线性注意力虽然能压缩长上下文，但每个 beam 独立维护完整 recurrent state 的做法，让显存和带宽开销随 beam 宽度急剧膨胀。快手与香港大学联合提出 OneLA，通过共享上下文状态与紧凑转换记录，将大 beam 线性注意力解码的端到端速度提升 1.54–2.46 倍，同时大幅降低显存占用。

## 一、核心方法

### 1. 从“每 beam 一个完整状态”到“共享状态 + 紧凑记录”

生成式推荐通常把每个物品表示为一小段离散语义 ID（SID），模型自回归地生成这些 ID。由于需要同时探索大量候选项，系统会在每一步对所有活跃 beam 进行扩展，并保留全局 top-k 候选。这导致所有 beam 共享同一个长 prompt，但各自经过极短的 SID 序列分叉。线性注意力模块（如 Gated DeltaNet, GDN）会把历史压缩为一个固定大小的 recurrent state：$S_t \in \mathbb{R}^{d_v \times d_k}$。在传统服务系统中，每个 beam 都要保存自己的完整 $S_t$，并且随着 beam 的复制、重排、剪枝，状态矩阵也要被频繁拷贝或重建。

论文观察到，不同 beam 的 recurrent state 均源自相同的 prompt 预填充状态 $S_{\mathrm{ctx}}$，之后只经过很短的 GDN 转换序列。每一步转换都可以用一个紧凑元组完全表示。GDN 的状态更新可以写成：

$$S_t = \alpha_t S_{t-1} + \delta_t k_t^{\top}$$

其中 $\delta_t = \beta_t(v_t - \alpha_t S_{t-1} k_t)$。OneLA 不再为每个 beam 存储完整的 $d_v \times d_k$ 矩阵，而是只保存一次 $S_{\mathrm{ctx}}$，并为每个生成的候选追加一条 **GDN Transition Record (GTR)**：$r_t = (\alpha_t, \delta_t, k_t)$。每个 GTR 的存储开销仅为 $O(d_v + d_k)$，远小于完整状态矩阵。

![图3：OneLA 状态表示总览](https://arxiv.org/html/2506.12345/figures/fig3.png)

### 2. 投影重放：不重建矩阵，只重放向量

虽然 GTR 消除了每个 beam 的持久矩阵存储，但后续解码仍然需要历史状态与当前 token 的交互。关键观察是，GDN 层与历史状态只有两次向量收缩：key 投影 $S_{t-1} k_t$ 和 query 投影 $S_{t-1} q_t$。OneLA 通过 **投影重放** 直接计算这两个投影，而无需在每一步重建中间矩阵。

给定当前 beam 的祖先 GTR 序列 $r_0, \ldots, r_{t-1}$，投影重放的递推关系为：

$$
\begin{aligned}
u_q^{(-1)} &= S_{\mathrm{ctx}} q_t, & u_q^{(j)} &= \alpha_j u_q^{(j-1)} + \delta_j (k_j^\top q_t), \\
u_k^{(-1)} &= S_{\mathrm{ctx}} k_t, & u_k^{(j)} &= \alpha_j u_k^{(j-1)} + \delta_j (k_j^\top k_t), \quad 0 \leq j < t.
\end{aligned}
$$

完成重放后，当前 GDN 输出直接由 accumulators 计算：

$$o_t = \alpha_t u_q^{(t-1)} + \delta_t (k_t^\top q_t)$$

其中 $\delta_t = \beta_t(v_t - \alpha_t u_k^{(t-1)})$。整个过程只需要两个向量累加器，不产生任何中间 recurrent state 矩阵。

![图4：向量形式的投影重放](https://arxiv.org/html/2506.12345/figures/fig4.png)

### 3. 轻量祖先索引：动态 beam 演化不搬数据

大 beam 搜索中，全局选择会不断删除、重排或复制 beam。如果 beam 与物理状态位置绑定，状态拷贝不可避免。OneLA 通过 **祖先索引** 将逻辑 beam 演化与物理 GTR 存储解耦。GTR 一旦写入就不可变，动态 beam 选择只是更新指向历史 GTR 链的索引：

$$h_t(b,j) = \begin{cases} h_{t-1}(\pi_t(b), j), & 0 \leq j < t-1, \\ \pi_t(b), & j = t-1. \end{cases}$$

每个子 beam 只需继承父 beam 的祖先索引，并记录父 beam 在上一解码步的物理槽位。这样，剪枝只需丢弃引用，重排只改变引用顺序，fan-out 只复制索引，历史 GTR 无需移动或复制。

![图1：生成式推荐解码工作流](https://arxiv.org/html/2506.12345/figures/fig1.png)

### 4. 融合共享上下文 GPU 内核

OneLA 进一步将以上逻辑实现为一个融合 GPU kernel。由于 $S_{\mathrm{ctx}}$ 被所有 beam 共享，如果每个 beam 独立从 HBM 读取它，带宽浪费依然严重。OneLA 让每个线程块加载一次 $S_{\mathrm{ctx}}$ 的 value 维 tile，然后在 on-chip 存储中跨多个 beam 复用。投影重放、当前 GDN 更新和输出计算全部在芯片内完成，中间值不落 HBM。最终内核只将紧凑的 GTR 追加写回，彻底消除每 beam 完整状态的写回开销。

![图5：共享上下文注意力内核](https://arxiv.org/html/2506.12345/figures/fig5.png)

## 二、实验

研究者基于 0.8B Qwen3.5 模型进行端到端评估，其中包含 18 个 GDN 层。prompt 长度设为 1K 和 5K，固定 beam 宽度 $W=256$，输出 token 数从 3 到 7。控制实验表明，OneLA 相对原生 vLLM FullState 路径，端到端 decode-forward GPU 时间减少 **1.54–2.46 倍**，且在所有 prompt 长度和输出长度下均优于 vLLM 的 FullState 与 ReplaySSM 路径。

在算子级 benchmark 中，OneLA 覆盖 405 种形状，包括 4/8/16 请求、128/256/512 beam 宽度以及九种不同的 recurrent state 几何配置。与 vLLM、SGLang FullState、ReplaySSM、FlashInfer 和 TensorRT-LLM 相比，OneLA 的 GDN attention 延迟中位数加速可达 **17.5–46.3 倍**；在 Qwen 最大几何配置下，算子加速最高达到 **56.5 倍**。同时，数值误差相对 FP32 FullState 控制在 $4.88 \times 10^{-4}$ 以内。

![图7：算子级加速比与持久状态容量降低](https://arxiv.org/html/2506.12345/figures/fig7.png)

更细粒度的分析显示，OneLA 的显存收益同样显著。在 $R=4, W=512$ 的设置下，OneLA 的持久 recurrent-state 容量仅为 FullState 的 1/56.5（输出 3）到 1/20.3（输出 7）；状态写回量降低 **127 倍**。物理 DRAM 流量相对 FullState 和 ReplaySSM 分别减少 **76.9 倍** 和 **39.9 倍**。算术强度达到 40.0 FLOP/byte，分别比 FullState 和 ReplaySSM 高 45.6 倍和 20.6 倍。这些数据直接解释了 OneLA 在延迟上的提升来源。

![图6：端到端累积解码前向 GPU 时间](https://arxiv.org/html/2506.12345/figures/fig6.png)

## 三、展望

OneLA 的核心贡献在于识别了大 beam 生成式推荐中 “beam 宽度” 作为 recurrent attention 的新扩展维度，并用共享上下文状态 + 紧凑 GTR + 轻量祖先索引的表示方式，实现了显存与带宽的双重压缩。这一思路不仅适用于 GDN，也可迁移到其他线性注意力变体。随着生成式推荐在工业界的普及，这类面向动态大 beam 的推理优化将成为关键底座。

**论文标题**：OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation

欢迎投稿！欢迎合作！