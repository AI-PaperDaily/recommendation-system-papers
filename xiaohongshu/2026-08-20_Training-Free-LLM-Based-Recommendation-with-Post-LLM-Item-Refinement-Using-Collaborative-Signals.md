CoRRe：后LLM精修推荐

今天给大家带来一篇CIKM 2026的推荐系统论文，提出一个完全training-free的框架CoRRe，核心思路是把协同信号放在LLM之后做物品表示精修，而不是塞进prompt里，效果提升非常明显。

🔑关键方法
1️⃣ 先用LLM从用户历史交互里生成自然语言用户画像，再用文本编码器变成用户query embedding，负责捕捉高维兴趣。
2️⃣ 方向精修：基于用户-物品交互构建物品共购图，把语义物品embedding沿图传播，让常被一起买的物品在向量空间里方向更接近。
3️⃣ 幅值精修：根据物品流行度调整embedding的norm，让热度信息参与最终相似度打分，帮助区分语义相似但真实偏好不同的候选物品。

💡核心创新
1️⃣ 提出post-LLM范式，把协同信号注入LLM生成的物品表征，而不是像rerank/RAG那样在LLM输入前做文章，避免了候选集质量瓶颈。
2️⃣ 真正训练免费，不需要模型训练，也不需要任务微调，只有两个超参数λ和α控制精修强度。
3️⃣ 把用户意图推断和物品级检索解耦，LLM负责“懂你”，协同信号负责“懂细粒度”，各干各的活。

📊实验效果
✅ 在Sports、Toys、Beauty三个真实数据集上，12个评测指标全部超过现有training-free方法，最高相对提升132.43%。
✅ Sports和Toys上甚至超过了所有训练型基线，Beauty拿到第二，性能很能打。
✅ 消融实验显示，去掉幅值精修后性能骤降，说明流行度校准对最终检索至关重要。

你觉得这种“先让LLM猜兴趣，再用协同信号精修物品”的思路，比硬塞prompt更优雅吗？评论区聊聊～

论文：Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals

欢迎投稿！欢迎合作！