两级ETT框架榨干GPU空闲

今天给大家带来Meta这篇大尺度推荐系统训练效率的paper，核心就一个问题：几千张H100在跑，真正消化新数据的有效时间到底有多少？他们用ETT%把这件事量化和拆解了。

🔑关键方法
1️⃣ 两级ETT指标体系：第一层拆成TTS、TTR、NoF，第二层再下钻到调度、初始化、PT2编译、checkpoint、shutdown等阶段，每个阶段对应一个owner团队，定位问题不再靠玄学。
2️⃣ 初始化三连优化：去掉每张表的all_gather通信、用合成fast-batch让数据预热和PT2编译并行、组件并行初始化，TTS从31.8分钟压到23.1分钟。
3️⃣ PT2编译瘦身：动态shape标记减少重编译、裁剪Triton autotune搜索、修复hash并引入Mega-Cache缓存复用，冷缓存编译从2050秒降到254秒。

💡核心创新
1️⃣ 把“GPU空闲”变成可度量、可追责的工程指标，两级下钻直接找到负责团队。
2️⃣ 发现恢复成本比启动成本更关键：TTR降幅是TTS的1.8-8.6倍，优化会随重启复利放大。
3️⃣ 把checkpoint和模型发布从训练主路径剥离，异步保存、独立CPU发布，GPU不再等非计算环节。

📊实验效果
✅ 6个benchmark模型ETT%从59-85%提升到80-93%，平均+15.5个百分点
✅ 最大workload ETT%达到85%
✅ 部署后fleet-wide ETT%从约80%升到90%以上
✅ 恢复路径贡献58%总收益，启用独立发布的模型shutdown从25-55分钟压到3.3分钟内

你们平时训练任务里，GPU真正在跑新数据的时间占多少？有没有被PT2编译和失败恢复偷走大块时间？

论文：Optimizing Effective Training Time for Large-Scale Recommendation Systems

欢迎投稿！欢迎合作！