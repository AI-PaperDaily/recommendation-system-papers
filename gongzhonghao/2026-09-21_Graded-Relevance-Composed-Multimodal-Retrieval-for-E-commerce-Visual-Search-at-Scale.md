# 沃尔玛GradCIR拿下FashionIQ 0.6703：4级分级监督如何带来5.9% NDCG增益？

## 一、引言

在电商视觉搜索中，用户上传一张产品图，再输入“换成灰色”“改成条纹”等修饰语，期望系统能从千万级商品目录中精准找到目标。但传统组合式图像检索（CIR）大多把候选商品简单二分为“相关”与“不相关”，忽略了真实目录中大量“部分匹配”商品。这种非黑即白的监督信号，会损失排序中最有价值的细粒度差异。

Walmart 全球技术团队提出 **GradCIR**，一种面向电商组合式多模态检索的分级相关性训练方法。研究者用视觉语言模型（VLM）自动生成查询和四级相关性标签，配合层级感知的角度损失函数与迭代硬负挖掘机制，在无需人工标注的情况下构建了 350 万对分级训练数据。该方案已在 Walmart 视觉搜索生产流量中部署，并在 FashionIQ 上以 Avg Recall 0.6703 略超当前最优监督基线 SPN4CIR。

## 二、核心方法：从二进制到四级分级监督

论文将 CIR 重新定义为四级序数标签问题。给定产品目录 $\mathcal{P}=\{p_i=(v_i,w_i)\}_{i=1}^{N_c}$，查询 $q=(v^q,w^q)$，目标是学习编码器 $f_\theta$，使余弦相似度 $\cos(e_q,e_p)$ 诱导的排序尊重 $r\in\{0,1,2,3\}$ 四级标签。四级标签依次为 **Irrelevant / PartialMatch / NearExact / Exact**。

![图1a：Exact](https://arxiv.org/html/2609.24152v1/figures/figure1_exact.png)
![图1b：Near Exact](https://arxiv.org/html/2609.24152v1/figures/figure1_near_exact.png)
![图1c：Partial Match](https://arxiv.org/html/2609.24152v1/figures/figure1_partial_match.png)
![图1d：Irrelevant](https://arxiv.org/html/2609.24152v1/figures/figure1_irrelevant.png)

### 1. VLM 驱动的分级数据管线与迭代反馈

GradCIR 的数据准备完全绕开人工标注。首先，VLM（Gemini-2.5-pro）对目录图片执行物体检测，裁剪出最多 10 个产品区域生成相似查询；同时对裁剪产品生成属性修改文本，构造修饰查询。随后，用预训练多模态模型对每个查询召回 top-K 候选，再由 VLM 判断器为所有（查询，候选）对标注四级相关性。

仅依赖预训练模型召回候选，容易遗漏对已部署检索器影响最大的困难负样本。论文引入**迭代相关性反馈循环**：用当前训练中的检索器挖掘新的困难负样本，去重后交给 VLM 标注，加入下一轮训练集。循环在离线 NDCG@10 不再上升时停止，实际三轮收敛，最终得到约 350 万对分级标注数据。

![图2：GradCIR端到端架构与检索结果示例](https://arxiv.org/html/2609.24152v1/figures/arch_figs/retrieved_exact_1.png)

### 2. 层级感知的角度损失

传统双塔检索器常用 InfoNCE 等多负样本对比损失，它们隐式地把所有非正样本视为同等错误的负样本，无法利用“NearExact 优于 PartialMatch”这类序数信息。GradCIR 采用 **AngleLoss**，将分级标签转换为嵌入空间中的角度约束。

$$\mathcal{L}_{\text{angle}}=\log\left[1+\sum_{(a,b):\,r_{a}>r_{b}}\exp\left(\frac{\Delta\theta_{q_{a},p_{a}}-\Delta\theta_{q_{b},p_{b}}}{\tau}\right)\right]$$

其中 $\Delta\theta_{q,p}$ 表示查询与产品嵌入的夹角，$\tau$ 为温度超参数。该损失枚举批次内所有严格分级对，完整利用 Exact $>$ NearExact、NearExact $>$ PartialMatch 等六类序数约束，而不是像二分类损失那样把六个约束压缩成一个。

模型侧选用 PaliGemma2 作为早期融合骨干，移除了生成头，以最后非填充 token 的隐藏状态作为序列嵌入，并为查询和商品分别添加 `[QUERY]` 与 `[ITEM]` 前缀。此外，训练采用 128、256、512、1024 四层 Matryoshka 表示，便于线上按延迟需求灵活截断维度。

## 三、实验：生产级检索与公开基准双重验证

在 Walmart 内部 WVST 测试集上，GradCIR-PG2 在相似查询与修饰查询的 Home、Fashion 四个子场景均取得最优。以 Fashion 相似查询为例，Adj-R@5 Exact 达 0.6747，NDCG@10 达 0.9216。相较下一名 Qwen3-VL-Emb(+GradCIR)，平均 Adj-R@5 Exact 提升 2.83%，NDCG@10 提升 0.91%。所有基线应用 GradCIR 配方后，平均 Adj-R@5 Exact 提升 24.3%，说明数据管线的价值与骨干选择无关。

消融实验进一步隔离标签粒度的影响。在控制骨干和训练数据不变的情况下，仅将相关性标签从二进制改为四级：

| 监督方式 | 相似查询 NDCG@10 | 修饰查询 NDCG@10 |
| :--- | :--- | :--- |
| Binary（2级） | 0.8698 | 0.8376 |
| 3级标签 | 0.8783 | 0.8450 |
| 4级标签 | 0.9125 | 0.8875 |

4级监督相较二进制在相似查询 NDCG@10 提升 4.9%，修饰查询提升 5.9%。论文特别指出，区分 NearExact 与 PartialMatch 是拉开差距的关键。

在公开 FashionIQ 基准上，GradCIR-PG2 微调后 Avg Recall 达 0.6703，超过此前最强监督基线 SPN4CIR 的 0.6641；在 zero-shot 设置下 Avg Recall 为 0.4000，与同量级 CLIP-L 类零样本 CIR 方法持平或更优。Street2Shop 上的 R@10 也达到 0.8266，说明分级训练同样适用于二元目标的相似度检索。

## 四、展望

GradCIR 展示了分级监督在组合式多模态检索中的明确增量，也验证了 VLM 大规模标注与迭代硬负挖掘的工业可行性。未来工作可延伸至多物品组合查询、显式属性级控制等方向。对于电商搜索从业者而言，从“相关/不相关”走向“多级相关性”的排序建模，或许正是下一轮体验提升的关键。

**论文标题**：Graded-Relevance Composed Multimodal Retrieval for E-commerce Visual Search at Scale

欢迎投稿！欢迎合作！