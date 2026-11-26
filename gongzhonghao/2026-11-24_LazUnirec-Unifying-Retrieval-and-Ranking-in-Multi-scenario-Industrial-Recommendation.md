# 阿里LazUnirec：共享Transformer统一召回与排序，电商无触发场景GMV +4.01%

## 一、引言

在工业推荐中，召回负责从海量商品中筛选候选，排序再预测候选商品的点击、转化等概率。两个阶段通常分别建模，却经常需要编码同一段用户行为序列。随着序列变长、推荐场景增多，重复计算带来的开销也越来越明显。

一种思路是让召回和排序共享Transformer骨干网络，但共享参数不意味着优化目标自然一致：召回需要区分正负商品，排序则要拟合多个行为概率。两种任务的梯度方向可能相反，使共享参数难以同时朝两个目标改进。

LazUnirec针对多场景推荐设计了统一模型：在共享用户序列编码的同时，为两个任务保留专属计算能力，并根据梯度方向的历史稳定性调整训练权重。模型还让排序向召回传递候选商品之间的偏好信息。

![图1：Attention组件中召回与排序的梯度冲突分布](https://arxiv.org/html/2611.32590v1/motivation_attention_v2.svg)

![图1：FFN组件中召回与排序的梯度冲突分布](https://arxiv.org/html/2611.32590v1/motivation_ffn_v2.svg)

图1展示了直接共享骨干时的问题：Attention和前馈网络（**FFN**）中，梯度余弦相似度小于0的比例分别为45.85%和47.54%；采用小于-0.05的更严格冲突阈值时，比例仍分别达到39.75%和38.40%。

## 二、核心方法

LazUnirec以用户历史、当前场景和请求内候选商品为输入，通过同一Transformer骨干产生召回所需的用户兴趣表示，以及排序所需的候选表示。训练时，两类任务共同更新共享参数，也使用各自的专属参数；线上则分别部署召回和排序模型，通过缓存复用用户序列计算。

![图2：LazUnirec整体框架](https://arxiv.org/html/2611.32590v1/main_model_v5.svg)

图2中，历史、召回查询和排序候选被组织到同一输入序列；骨干中的共享与专属部分承载两个任务，训练目标再通过权重调整和知识蒸馏相互配合。

### 从同一段历史生成召回与排序表示

每个历史行为token由商品ID、商品属性和发生行为的场景特征组成。模型还加入一个**触发token**：有触发商品的场景编码该商品，没有触发商品的场景使用可学习占位向量。多个可学习查询与触发表示结合，形成召回token，用来并行提取多种用户兴趣。

排序侧则把同一请求中的候选商品分别构造成排序token。这样，多个候选可以共享同一段用户历史的存储、传输和编码，而不必为每个候选重复处理历史。

![图3：统一序列的Attention掩码](https://arxiv.org/html/2611.32590v1/attention_mask_v4.svg)

图3的掩码限定了信息流：历史token只能关注自身和更早的行为；触发token关注历史和自身；每个召回或排序token只关注历史、触发token及自身，不读取其他查询或候选。

训练时，召回要让正样本商品得分高于采样负样本。对用户$i$，正样本集合为$\mathcal P_i$，每个正样本对应的负样本集合为$\mathcal N_{ij}$，匹配分数为用户和商品向量的点积$r_{ij}$：

$$
\mathcal{L}_{\mathrm{retrieval}}
=-\frac{1}{|\mathcal{P}_{i}|}\sum_{j\in\mathcal{P}_{i}}
\log\frac{\exp(r_{ij})}{\exp(r_{ij})+\sum_{n\in\mathcal{N}_{ij}}\exp(r_{in})}.
$$

这个采样Softmax目标推动正样本在采样集合中获得更高匹配分数。排序则使用日志中的点击、转化标签，以二元交叉熵训练CTR、CVR预测。

### 共享骨干，也给任务留出专属空间

统一输入解决了计算复用，却没有消除两个任务更新共享参数时的冲突。LazUnirec分别在Attention和FFN中加入任务专属能力。

在Attention中，召回和排序共用基础投影矩阵，但各自拥有低秩残差。对于$Q$、$K$、$V$中的投影类型$p$和任务$\tau$，投影方式为：

$$
\bm p^\tau=(\bm W_p+\Delta\bm W_p^\tau)\bm x,\qquad
\Delta\bm W_p^\tau=\bm B_p^\tau\bm A_p^\tau.
$$

其中$\bm W_p$是共享矩阵，$\bm A_p^\tau$和$\bm B_p^\tau$是任务专属的低秩矩阵；历史和触发token只使用共享投影。这样既保留共享知识，也允许召回与排序调整各自的表示。

在FFN中，模型使用**稀疏混合专家**：每个任务有一个始终激活的专属专家，路由器再根据token、用户画像、场景和任务信息，从共享专家池中选择得分最高的$k$个专家。对任务$\tau$，输出为：

$$
\operatorname{MoE}^{\tau}(\bm x)
=\operatorname E_{\mathrm{private}}^\tau(\bm x)
+\sum_{e\in\mathcal D^\tau}w_e^\tau\operatorname E_e^{\mathrm{shared}}(\bm x).
$$

这里$\mathcal D^\tau$是选中的共享专家集合，$w_e^\tau$是对应的归一化权重；实现中$k=2$。专属专家处理任务特有模式，共享专家则提供跨任务知识交换，同时避免每次激活全部专家。

### 用稳定性调节训练，并让排序教召回

有了专属参数，共享部分仍会接收两个任务的梯度。模型因此按**场景和任务**定期探测少量共享参数的梯度，将梯度归一化后统计历史方向波动。对方向表示$\bm\gamma_s^\tau$，方差和证据计算为：

$$
(\bm\sigma_s^\tau)^2=
\left[\operatorname{EMA}((\bm\gamma_s^\tau)^2)
-(\operatorname{EMA}(\bm\gamma_s^\tau))^2\right]_+,\quad
\bm e_s^\tau=\min\!\left(e_{\max},((\bm\sigma_s^\tau)^2+\epsilon)^{-1/2}\right).
$$

方向历史波动越小，得到的证据越强；EMA是指数移动平均，$\epsilon$和$e_{\max}$用于控制数值稳定性。

模型再把平均证据转成不确定性$\mathcal E_s^\tau$，并相对于当前批次的平均可靠性计算损失权重：

$$
\mathcal E_s^\tau=\frac{1}{1+\operatorname{Mean}(\bm e_s^\tau)},\qquad
\omega_s^\tau=
\frac{1-\mathcal E_s^\tau}{\widehat{(1-\mathcal E_s)}_{\mathrm{batch}}}.
$$

因此，历史方向更稳定的任务梯度会获得相对更大的影响。权重停止梯度传播；实现中还将其限制在$[0.8,1.2]$，避免调整过大。这个机制衡量的是历史稳定性，而不是直接证明某个梯度方向一定正确。

排序还能利用候选与用户之间更细的交互，向召回提供额外监督。同一请求的候选构成一个列表，对商品$j$，召回侧用多个兴趣向量$\bm u_{ik}$和对应兴趣概率$\pi_{ik}$计算分数：

$$
\tilde r_{ij}=\max_{1\leq k\leq K}
\left(\pi_{ik}\bm u_{ik}^{\top}\bm v_j\right).
$$

该分数取各兴趣对商品的加权匹配结果中的最大值。随后，排序CTR logits与召回分数分别经过带温度参数的Softmax，形成教师分布和学生分布，并计算：

$$
\mathcal L_{\mathrm{kd}}
=\sum_{v_j\in\mathcal T_i}p_{ij}^{\mathrm{ranking}}
\log\frac{p_{ij}^{\mathrm{ranking}}}{p_{ij}^{\mathrm{retrieval}}}.
$$

这个KL散度让召回学习排序侧对同一列表的相对偏好；教师分支停止梯度，蒸馏更新召回表示和兴趣概率。

最终训练目标组合加权后的两个任务损失、蒸馏损失和专家路由正则项：

$$
\mathcal L=
\operatorname{sg}(\omega_s^{\mathrm{retrieval}})\mathcal L_{\mathrm{retrieval}}
+\operatorname{sg}(\omega_s^{\mathrm{ranking}})\mathcal L_{\mathrm{ranking}}
+\lambda_{\mathrm{kd}}\mathcal L_{\mathrm{kd}}
+\lambda_z\mathcal L_z.
$$

其中$\operatorname{sg}$表示停止梯度，两个$\lambda$控制蒸馏和路由正则项的贡献。

线上计算时，召回阶段保存用户序列逐层的Key、Value缓存，排序在同一请求中复用；用户序列和场景条件不变时，缓存也可跨请求复用。部署还结合算子融合、部分FFN计算使用FP8，以及跨场景请求合批，以降低开销并提高GPU利用率。

## 三、实验

离线实验使用电商平台连续25天日志训练、次日数据测试，覆盖无触发的场景1和有触发的场景2、3；各模型采用相同的5层Transformer骨干配置，训练一个epoch。排序指标包括衡量区分能力的AUC和按组统计的GAUC；召回使用HR@N，衡量目标商品出现在前N个结果中的命中情况。

与Vanilla Transformer及UniPinRec、UniR²、UniSGR、OneTrans-V2等基线相比，LazUnirec在三个场景的召回和排序指标上均领先。例如场景1中，HR@100从最佳基线的0.3346提升至0.3620，CTR GAUC从0.6695提升至0.6722。

消融实验用于区分各设计的作用：去掉低秩适配、稀疏专家或不确定性估计后，场景1的召回和排序指标均下降；去掉蒸馏后，HR@100从0.3620降至0.3502，而CVR指标略有上升，说明蒸馏收益主要体现在召回，而非所有排序指标都同步改善。

![图4：Attention组件的梯度冲突对比](https://arxiv.org/html/2611.32590v1/gradiant_conflict_attention_v2.svg)

![图4：FFN组件的梯度冲突对比](https://arxiv.org/html/2611.32590v1/gradiant_conflict_ffn_v2.svg)

图4比较了Vanilla Transformer和LazUnirec的梯度冲突比例。Attention中，普通冲突和严格冲突比例分别相对下降9.16%和13.23%；FFN中分别相对下降14.32%和37.76%，支持共享与专属参数结合能够缓解冲突的判断。

![图5：排序高不确定性样本的权重](https://arxiv.org/html/2611.32590v1/rank_max_uncertainty_v3.svg)

![图5：排序低不确定性样本的权重](https://arxiv.org/html/2611.32590v1/rank_min_uncertainty_v3.svg)

![图5：召回高不确定性样本的权重](https://arxiv.org/html/2611.32590v1/retrieval_max_uncertainty_v3.svg)

![图5：召回低不确定性样本的权重](https://arxiv.org/html/2611.32590v1/retrieval_min_uncertainty_v3.svg)

图5显示，召回和排序中高不确定性观测获得较低权重，低不确定性观测获得较高权重，验证了权重调整符合设计方向。多场景消融中，联合训练也优于各自的单场景训练；其中长尾场景X的HR@100从0.3086提升至0.3686。

线上A/B测试持续一周，使用10%线上流量，LazUnirec同时替换原有MIMA召回和LazFormer排序。相对原系统，无触发场景的商品详情页浏览量、订单数、购买人数和GMV分别提升5.25%、5.75%、4.57%和4.01%；有触发场景分别提升4.01%、4.64%、3.06%和3.06%。

**论文标题**：LazUnirec: Unifying Retrieval and Ranking in Multi-scenario Industrial Recommendation

欢迎投稿！欢迎合作！