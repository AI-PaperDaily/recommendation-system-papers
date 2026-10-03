推荐系统ETT%分层框架

今天给大家带来Meta的推荐系统训练效率优化论文，他们用ETT%分层指标把GPU在训练任务生命周期里“摸鱼”的时间一一揪出来，把最大工作负载有效训练时间从50-60%提升到85%。

🔑关键方法
1️⃣ ETT%分层归因：L1拆成TTS启动、TTR恢复、NoF失败数，L2再定位到调度、初始化、PT2编译、有效训练、浪费训练、停机六个组件，每个组件都有对应owner团队，问题不再扯皮。
2️⃣ 初始化与编译优化：通信消除、fast-batch合成假batch、并行组件初始化，让DPP暖机、编译、模型创建并行跑；配合Dynamo动态shape、autotune裁剪和Mega-Cache缓存复用，冷缓存编译从2050秒降到254秒。
3️⃣ 异步checkpoint+standalone publishing：checkpoint保存不阻塞训练loop，发布阶段从GPU剥离到CPU独立作业，shutdown时间从25分钟+降到3.3分钟内。

💡核心创新
1️⃣ 用ETT%把“生命周期开销”变成可测量、可追责的运营指标，弥补MFU只看训练步和Goodput无法定位组件的缺陷。
2️⃣ 发现重启放大效应：恢复会重复初始化与编译，所以TTR降幅是TTS降幅的1.8-8.6倍，一次启动优化会被多次重启放大。
3️⃣ fast-batch解耦数据pipeline和编译依赖，让三路并行，直接砍掉串行等待；Mega-Cache跨重启复用编译产物，降低恢复成本。

📊实验效果
✅ 六个基准模型ETT%从59-85%提升到80-93%，平均提升15.5个百分点。
✅ 最大工作负载ETT%达到85%，机群级离线训练ETT%从约80%升到90%以上。
✅ PT2编译冷缓存降低87.6%，warm path降低89.5%；恢复路径贡献58%总时间节省。
✅ checkpoint保存阻塞时间降低77%，shutdown降至3.3分钟内。

这套框架把“训练效率”从只看单步速度，变成看整段生命周期里有多少时间真正在吃新数据。你们在大规模训练里最头疼的是编译缓存、checkpoint阻塞还是调度排队？评论区聊聊～

论文：Optimizing Effective Training Time for Large-Scale Recommendation Systems

欢迎投稿！欢迎合作！