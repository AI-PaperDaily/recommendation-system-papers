ETT%两级框架专治训练空转

今天给大家带来Meta这篇推荐系统训练效率神作，它用一个叫ETT%的指标把千卡GPU上那些“看不见的空转”全揪出来了。

🔑关键方法
1️⃣ 训练初始化去串行：干掉逐表all_gather，改用本地构建全局元数据；再用合成fast-batch喂给PT2编译，让DPP预热和编译并行跑，TTS直降27%。
2️⃣ PT2编译三连：动态shape标记防重复构图，autotune剪枝省搜索，Mega-Cache把编译产物打包复用，冷缓存编译从2050秒干到254秒。
3️⃣ 异步checkpoint加调频：用CSBT、UTT、NoS三件套公式找最优保存间隔，再上PyTorch原生DCP AsyncStager，save阻塞降了77%。

💡核心创新
1️⃣ ETT%两级分解：Level-1看TTS、NoF、TTR，Level-2把时间归到调度、初始化、编译、有效训练、浪费训练、关机六个组件，每个都对应一个团队，锅甩得明明白白。
2️⃣ 重启复利效应：发现恢复路径会重复启动开销，TTS省下的每一分钟在失败恢复时还能再省一次，TTR降幅是TTS的1.8到8.6倍。
3️⃣ 独立发布模型：训练完写anchor checkpoint就释放GPU，后面CPU单独导出serving快照，单job省约30分钟GPU占用。

📊实验效果
✅ 6个推荐模型ETT%从59–85%拉到80–93%，平均提升15.5个百分点。
✅ 恢复路径贡献了58%的总节省，不是单靠启动快。
✅ 最大规模workload ETT%达到85%，优化后全fleet从约80%爬到90%以上。
✅ 冷缓存编译从2050秒降到254秒，热路径从1089秒降到114秒。

看完是不是也想给自家训练pipeline做个“体检”？你们平台最痛的是启动、编译还是checkpoint？评论区聊聊👀

论文：Optimizing Effective Training Time for Large-Scale Recommendation Systems

欢迎投稿！欢迎合作！