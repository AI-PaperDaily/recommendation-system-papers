Meta分层组件性能剖析CB

今天给大家带来一篇Meta推荐系统性能剖析的新论文，核心是Component Benchmark（CB），一个轻量但能打的分层组件级模型剖析系统，专门解决大规模推荐模型结构异构、动态shape多、瓶颈难归因的问题。

🔑关键方法
1️⃣ 基于PyTorch module hooks抓取每个子模块的真实运行时输入，然后在单卡上对子模块递归独立profile，把整模型拆成一棵层级树。
2️⃣ 插件化设计，分Component Provider、Preprocessor、Profiler、Result Visualizer四类扩展点，能灵活接不同训练栈，内置FLOPs、MFU、memory snapshot、Kineto trace等分析。
3️⃣ 用Icicle图和结果表做交互式可视化，按MFU红绿着色，子模块算力/带宽瓶颈一眼定位，点进去还能看kernel级trace。

💡核心创新
1️⃣ 首次把“组件级”语义引入推荐系统性能分析，不再只看端到端吞吐或底层算子trace，能直接归因到具体PyTorch模块。
2️⃣ 用独立输入捕获+递归profile绕开PyTorch 2编译融合导致的模块边界丢失，父模块可由子模块聚合估算，超大模型也能跑。
3️⃣ 模型无关，推荐模型和LLM通用，只吃module tree，不依赖具体arch。

📊实验效果
✅ DLRM里sparse_arch几乎纯带宽bound，dense_arch中Linear和ReLU的MFU反差明显，层级视图直接定位热点。
✅ 实践中定位到低效GEMV改成elementwise kernel后，训练+服务吞吐提升3%。
✅ 用CB提前验证某模块开PT2后延迟83ms→50ms，最终带来34% QPS提升。
✅ 生成式推荐模型用CB定位到event model和set-transformer膨胀，恢复维度后省22%内存、QPS提升10%。

这套工具把性能优化从“看trace猜瓶颈”变成“看树定位模块”，对推荐系统工程师挺友好。你们调推荐模型时最常被哪个模块卡住？评论区聊聊～

论文：Component Benchmark: Hierarchical Model Profiling for Large-scale Recommendation Systems

欢迎投稿！欢迎合作！