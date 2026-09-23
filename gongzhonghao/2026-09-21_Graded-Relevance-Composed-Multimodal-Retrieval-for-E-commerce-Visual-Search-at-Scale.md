# 沃尔玛GradCIR分级训练组合检索：NDCG提升5.9%，FashionIQ刷新至0.6703

## 一、引言

在电商平台的视觉搜索场景中，用户常常不满足于单纯上传图片寻找相似商品，而是希望附加文字指令进行修改——例如“把这款沙发换成灰色”“换成条纹款式”。这类由图像与文本共同构成的查询被称为**组合图像检索**（Composed Image Retrieval, CIR）。然而，学术界主流方法将检索结果简化为二分标签：要么相关，要么不相关。真实目录中的候选商品往往呈现**连续的相关性光谱**：完全匹配、相近匹配、部分匹配、完全无关。将这种分级信息压缩成二分类，会严重损害排序质量。

沃尔玛研究团队提出的**GradCIR**方法论，正是针对这一痛点：它利用视觉语言模型（VLM）自动生成4级相关性标签，训练早期融合检索器直接建模这种分级排序。在内部测试集上，分级监督带来**4.9%–5.9% 的NDCG@10提升**；在公开基准FashionIQ上，GradCIR微调后达到**0.6703平均召回**，超过此前最强监督基线SPN4CIR（0.6641），并已部署于沃尔玛生产环境。

## 二、核心方法

GradCIR由三个核心创新构成：**VLM驱动的分级数据管线**、**早期融合检索架构**、以及**层级感知角度损失**。

### 1. 分级相关性数据生成

传统CIR数据集（如FashionIQ、CIRR）每查询仅标注一个正样本，无法反映真实目录中大量部分匹配商品的排序价值。GradCIR采用两阶段自动化流程：

- **查询生成**：使用Gemini-2.5-pro对目录图像进行物体检测，裁剪出单个商品生成相似查询；同时提示VLM生成修改属性的文本（如“换成灰色”），形成修饰查询。
- **四级标注**：对于每个查询，先用预训练嵌入模型检索Top候选，再由同一VLM法官对每个候选赋予 `Exact`、`NearExact`、`PartialMatch`、`Irrelevant` 四级标签，无需人工标注。论文明确展示了四个典型例子：

![图1(a)：Exact: all visual attributes align.](https://arxiv.org/html/2609.24152v1/figures/figure1_exact.png)

![图1(b)：Near Exact: style modifier; core attributes (colour, material) retained but not identical.](https://arxiv.org/html/2609.24152v1/figures/figure1_near_exact.png)

![图1(c)：Partial Match: category and partial attribute (high-top) match, but a major feature (red stripes) is missing.](https://arxiv.org/html/2609.24152v1/figures/figure1_partial_match.png)

![图1(d)：Irrelevant: different product type and attributes.](https://arxiv.org/html/2609.24152v1/figures/figure1_irrelevant.png)

- **迭代难负样本挖掘**：初始数据由预训练模型检索产生，容易偏向其已擅长的简单负样本。论文采用**相关性反馈循环**：用当前训练的检索器挖掘更难区分的负样本，经VLM标注后加入训练集，循环三轮，最终得到**350万**查询-商品对（其中210万相似查询、140万修饰查询）。

### 2. 模型架构：早期融合PaliGemma2

GradCIR选用的骨干是**PaliGemma2**（SigLIP-400M视觉编码器 + Gemma2-2.6B语言模型），属于**早期融合**架构：图像patch token与文本token在Transformer中联合自注意力，能建模“图像是X但文字要求改Y”的跨模态交互。针对检索任务做了三点适配：

- **禁用解码**：移除语言建模头，模型仅作编码器。
- **末token池化**：取最后一个非填充位置的隐藏态作为序列嵌入，无需额外池化层或[CLS]。
- **角色前缀**：在输入文本前添加 `[QUERY]` 或 `[ITEM]` 前缀，使共享编码器能学习查询与商品的非对称表示。

训练采用**Matryoshka嵌套损失**，在128、256、512、1024维同时监督，部署时可按延迟需求截断到256维而不重新训练。

### 3. 层级感知角度损失

传统InfoNCE等对比损失将所有非正样本视为同等负样本，无法区分相近匹配与完全无关。GradCIR使用**AngleLoss**，将分级标签转化为嵌入空间中的角度约束：

$$\mathcal{L}_{\text{angle}}=\log\left[1+\sum_{(a,b):\,r_a>r_b}\exp\left(\frac{\Delta\theta_{q_a,p_a}-\Delta\theta_{q_b,p_b}}{\tau}\right)\right]$$

对于批次中任意两个相关性严格排序不同的配对 $(a,b)$，若 $r_a > r_b$，则要求高相关对的角度 $\Delta\theta_{q_a,p_a}$ 小于低相关对 $\Delta\theta_{q_b,p_b}$。4级标签产生6种有序约束（Exact>NearExact、Exact>PartialMatch、……），而二分损失只保留一种（正>负），这是分级监督提升排序质量的根本机制。

## 三、实验

### 内部测试集WVST

研究团队构建了覆盖家居与时尚两个垂直领域的内部测试集，每个查询的候选池由多个嵌入模型检索结果合并，并由VLM标注四级相关度。对比基线包括FashionCLIP、SigLIP2、GME-Qwen、Qwen3-VL-Embedding，均在冻结与微调两种设置下评估。结果显示：

- **GradCIR-PG2在所有指标上取得最佳**：家居垂直相似查询NDCG@10为0.9034，修饰查询为0.8807；时尚垂直相似查询NDCG@10为0.9216，修饰查询为0.8942。
- **微调配方独立于骨干**：四个基线应用GradCIR微调后，Adj-R@5(Exact)平均提升24.3%，验证了数据管线与损失函数的通用性。
- **修饰查询优势显著**：早期融合模型在修饰查询上对晚期融合（如SigLIP2）的领先幅度更大，与预期一致。

### 公开基准

在**FashionIQ**零样本设置下，GradCIR平均召回0.4000，超过所有已发表的CLIP-L类零样本方法（如CIReVL 0.3860、CoLLM-CLIP 0.3980）。监督微调后，GradCIR达到0.6703，超越此前最强的SPN4CIR（0.6641），证实分级监督在二元评估协议下同样有效。在**Street2Shop**相似查询基准上，GradCIR-PG2的R@10为0.8266，接近最强的Qwen3-VL-Emb(+GradCIR)（0.8304），并显著优于FashionCLIP、SigLIP2。

### 消融实验

- **分级 vs 二元**：固定骨干与数据，仅改变相关性粒度。4级对比2级，相似查询NDCG@10从0.8698升至0.9125（+4.9%），修饰查询从0.8376升至0.8875（+5.9%）。3级（合并NearExact与PartialMatch）已有提升，但仍显著弱于4级，说明区分“相近”与“部分”匹配至关重要。
- **修饰查询训练**：加入修饰查询后，修饰查询Adj-R@5(Exact)提升92.5%，相似查询也提升6.3%，表明修饰数据能增强细粒度图像-文本理解。

## 四、展望

GradCIR首次将分级相关性引入组合检索的训练范式，证明了VLM自动化标注与角度约束损失在生产规模检索中的价值。论文指出未来可扩展至**多物品组合查询**（“找这样的沙发配这样的地毯”）以及**属性级可控性**，进一步提升复杂购物意图的理解能力。当前方法已在沃尔玛视觉搜索系统上线，服务于真实用户流量，其数据与训练配方在各类目录数据上具备良好的可复现性。

**论文标题**：Graded-Relevance Composed Multimodal Retrieval for E-commerce Visual Search at Scale

欢迎投稿！欢迎合作！