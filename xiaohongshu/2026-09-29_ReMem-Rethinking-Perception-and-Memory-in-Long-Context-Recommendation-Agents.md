ReMem：OCR+动态记忆

今天给大家带来一个长上下文推荐Agent新框架 ReMem，用“OCR多模态感知 + 时间演化记忆”解决推荐智能体看不明商品页、记不住长历史的问题，三个任务平均比最强基线高 5.16%。

🔑 关键方法
1️⃣ OCR感知：不解析原始HTML，对商品页截图做OCR，把文字、图片、交互元素统一成结构化卡片，过滤广告噪声，跨平台更稳。
2️⃣ 时间演化记忆（TEM）：长交互历史按chunk流式输入，每读完一块就更新固定长度记忆，只留偏好相关线索，上下文有界、推理线性。
3️⃣ Multi-Memory GRPO：把最终答案的advantage回传训练所有中间记忆，让模型学会“记什么”。

💡 核心创新
1️⃣ 首次在推荐Agent里联合OCR多模态感知和RL增强动态记忆，解决“怎么看”和“怎么记”。
2️⃣ 记忆直接放在token空间，不引入外部向量库、不改注意力结构，保持标准自回归生成，方便检查和控制。
3️⃣ chunk+固定记忆实现O(N)复杂度

论文：ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents

欢迎投稿！欢迎合作！