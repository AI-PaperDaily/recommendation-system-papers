## 人大RecSoup模型合并框架：多目标推荐推理提速228倍，Pareto前沿质量提升8.5%

## 引言

现代推荐系统往往需要在准确性、多样性、公平性等多个相互冲突的目标之间取得平衡。可控多目标推荐（CMOR）允许在推理阶段动态调整目标权重，无需重新训练，以适应实时变化的用户需求。然而，现有方案要么受限于重排序的候选集范围，要么依赖昂贵的参数生成模块（如超网络或扩散模型），带来训练成本高、推理延迟大等问题。

近期模型合并（Model Merging）技术为CMOR提供了全新思路：直接复用多个目标专用子模型，通过参数组合实现动态权衡。但将这一范式迁移到推荐系统并非易事——推荐模型的任务向量不具备大语言模型中常见的“共享线性子空间”，直接线性插值会引发严重的跨目标干扰。

针对这一难题，中国人民大学的研究者提出了**RecSoup**：一个轻量级两阶段框架，通过**冲突感知剪枝**与**自适应系数学习**，在不重新训练的情况下高效逼近多目标Pareto前沿，推理速度比超网络方法最快提升**228.8倍**，且Pareto前沿质量在多个基准上超越现有方法。

## 一、核心方法：剪枝+自适应合并

### 1. 问题定义与线性假设失效

在模型合并范式中，每个目标的专用模型$\theta_i$与基础模型$\theta_{\text{base}}$的差定义为任务向量：

$$\boldsymbol{\tau}_{i}=\operatorname{vec}(\theta_{i})-\operatorname{vec}(\theta_{\mathrm{base}})$$

给定用户偏好向量$\boldsymbol{\mu}\in\Delta_{N-1}$，合并模型参数可表示为：

$$\bar{\theta}(\boldsymbol{\mu})=\theta_{\mathrm{base}}+\sum_{i=1}^{N}\lambda_{i}\boldsymbol{\tau}_{i}$$

其中$\boldsymbol{\lambda}=[\lambda_1,\dots,\lambda_N]^\top$为合并系数，通常由偏好$\boldsymbol{\mu}$通过映射函数$\mathcal{F}$得到。

传统任务算术（如线性插值）依赖**线性区域假设**：不同任务的参数更新应集中在共享的低维线性子空间内。然而，论文对TiSASRec模型分别优化准确性与多样性所得的任务向量进行PCA分析（见图2），发现两者散布在高维空间中，PC1仅解释25.7%的方差，前两个主成分累计解释不足45%。这表明准确性更新与多样性更新依赖根本不同质的参数方向，无法通过线性组合有效统一。

![图2：TiSASRec中准确性与多样性优化任务向量的PCA投影。轴对应第一主成分（PC1）和第二主成分（PC2），分别解释25.7%和15.8%的方差。](https://arxiv.org/html/2610.21220v1/Figures/TiSASRec_original_task_vector_pca_v2.png)

### 2. 阶段一：冲突缓解剪枝

RecSoup首先对任务向量矩阵$\mathbf{T}\in\mathbb{R}^{N\times D}$执行精细剪枝，以降低跨目标参数干扰。针对每个任务$i$的每个参数位置$j$，计算**综合重要性分数**：

$$S_{i,j}=\beta T_{\mathrm{self},\,i,j}+(1-\beta)T_{\mathrm{align},\,i,j}$$

其中，$T_{\mathrm{self},\,i,j}=\frac{T_{i,j}^{2}}{\|\mathbf{T}_{i,:}\|_{2}^{2}}$衡量参数对所在任务的相对贡献；$T_{\mathrm{align},\,i,j}=\left|T_{i,j}\cdot\bar{T}_{i,j}\right|$度量参数与其他任务平均更新的耦合强度。这里使用绝对值而非符号，是为了保留那些方向相反但对多目标都关键的参数（如提升准确性通常会降低多样性），避免因方向冲突而误剪。

随后采用**动态范数保留**策略：按分数排序后，选取前$k_i$个参数，使得保留的自重要性之和达到预设比例$\rho$：

$$k_i=\min\left\{k\;\bigg|\;\frac{\sum_{m=1}^{k}T_{\mathrm{self},\,i,\,\pi(m)}}{\sum_{j=1}^{D}T_{\mathrm{self},\,i,\,j}}\geq\rho\right\}$$

剪枝后的任务向量为$\tilde{\mathbf{T}}_{i}=\mathbf{T}_{i}\odot\mathbb{I}_{\mathcal{K}_i}$。这一过程有效解耦了不同目标间的参数重叠，为后续合并提供了干净的子空间。

### 3. 阶段二：可学习系数的自适应合并

第二阶段引入一个轻量级门控网络（即模型合并器）$f_\phi$，将用户偏好向量映射为合并系数：

$$\boldsymbol{\lambda}=f_{\phi}(\boldsymbol{\mu})$$

合并后的任务向量为$\boldsymbol{\tau}_{\mathrm{merged}}=\sum_{i=1}^{N}\lambda_i\tilde{\mathbf{T}}_{i,:}$，再与基础模型相加得到最终模型。训练时仅更新门控网络参数$\phi$，剪枝后的任务向量和基础模型均冻结。损失函数为偏好加权的多目标损失$\mathcal{L}(\theta_{\mathrm{merged}},\boldsymbol{\mu})=\sum_{i=1}^{N}\mu_i\cdot\mathcal{L}_i(\theta_{\mathrm{merged}})$。

这种两阶段解耦设计，使得门控网络只需在“净化后”的参数子空间中学习坐标，无需同时处理干扰与权衡，大幅减轻优化负担。论文在理论分析中（定理5.1）进一步证明，模型合并范式不受参数生成映射的Lipschitz平滑性限制，可以完全导航Pareto前沿，而参数生成方法可能无法覆盖全部凸组合解。

## 二、实验：全面验证性能优势

论文在MovieLens-1M、Amazon Grocery和Amazon Toys三个数据集上，使用SASRec、GRU4Rec、TiSASRec三种骨干模型进行了系统评估，对比了重排序（MMR）、参数生成（CMR、Paragon）、基础合并（Soup、TATR、TIES）等方法。

**主要结果（表1）显示：**

- **Pareto前沿质量**：RecSoup在超体积（HV）指标上一致超越所有合并类基线。例如在MovieLens-1M上，相比Soup，HV提升最高达**8.5%**（SASRec）；在Amazon Grocery上，相比Paragon等参数生成方法，HV提升约**20-50%**。
- **可控性**：RecSoup的Controllability平均接近0.9以上，与最佳方法相当，表明其能精确响应偏好变化。
- **推理效率**：表2展示了参数构造延迟对比。超网络方法平均需7.5-9.4毫秒，而RecSoup仅需0.04-0.05毫秒，加速**165.9-228.8倍**；扩散模型（Paragon）需约87.79秒，差距更为悬殊。
- **多目标扩展**：在准确率-公平性以及准确率-多样性-公平性三目标设置中，RecSoup同样获得最高的HV和均衡的可控性，证明其良好泛化能力。

![图5：RecSoup剪枝下任务向量的结构分析。(a)参数解耦：剪枝降低了目标间支持集重叠。(b)几何保持：奇异谱和主方向几乎不变，整体结构未破坏。(c)特征锐化：支持集中度上升，噪声更新被移除。(d)Pareto前沿：剪枝后可达解范围扩大。](https://arxiv.org/html/2610.21220v1/Figures/RecSoup_pruning_analysis.png)

![图7：RecSoup与TIES的超体积对比。柱状图显示绝对HV，折线显示相对提升。RecSoup在SASRec、GRU4Rec、TiSASRec上均持续优于TIES，其中在Food数据集相对提升最高达9.02%。](https://arxiv.org/html/2610.21220v1/Figures/RecSoup_vs_TIES_HV.png)

消融实验进一步表明，剪枝与自适应合并缺一不可：移除系数学习器（w/o MLP）导致HV显著下降；移除剪枝（w/o Prune）在任务向量干扰较强时性能恶化明显，验证了剪枝对鲁棒性的贡献。

## 三、技术价值与展望

RecSoup成功地将模型合并思想引入可控多目标推荐，其轻量级设计不仅大幅降低了推理延迟，还避免了参数生成方法的训练不稳定和架构复杂度，为工业界复用已有的多目标专用模型提供了高效路径。未来，该框架有望扩展到更广泛的目标集合（如新颖性、覆盖率等），并探索与LoRA等参数高效微调技术的结合，进一步提升多目标推荐的灵活性与实用性。

**论文标题**：RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation

欢迎投稿！欢迎合作！