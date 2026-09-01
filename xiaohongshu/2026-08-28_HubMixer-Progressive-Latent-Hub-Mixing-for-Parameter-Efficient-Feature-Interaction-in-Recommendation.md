HubMixer潜空间枢纽混合

今天给大家带来一篇快手招聘推荐场景的特征交互工作HubMixer，核心思路是不直接在异构token上做全量交叉，而是先归纳到少量可学习latent hub，再在潜空间做高阶交互，最后按token写回，参数更省效果还能打。

🔑关键方法
1️⃣ Hub Induction：用可学习hub当query，对原始特征token做cross-attention，把用户、行为、岗位、上下文等异构token压缩成少量潜向量。
2️⃣ Hub Interaction：在hub空间跑self-attention做高阶交互，因为hub数量远小于token数，计算很轻。
3️⃣ Token-Conditioned Readout：每个原始token再query交互后的hub，通过LayerScale残差注入，保留不同字段的差异化信息，适合多任务排序。

💡核心创新
1️⃣ 提出induction–interaction–readout三段式潜空间枢纽混合范式，让模型先组织语义再交互。
2️⃣ 用少量可学习latent hubs替代大矩阵token-token交互，把复杂度从T²降到T×H，参数效率更高。
3️⃣ token级条件读回机制，不是简单全局池化广播，能让不同任务各取所需，避免特征身份丢失。

📊实验效果
✅ 离线四任务AUC全面超过DCN、DCNv2、AutoInt、Wukong、RankMixer、TokenMixer，平均AUC 0.8256，参数只有142.4M，比TokenMixer还少。
✅ 消融实验显示，去掉hub interaction平均AUC掉0.0024；把token读回换成pooling注入平均AUC掉0.0009，说明三步都有用。
✅ 线上快手招聘A/B测试，简历提交转化率显著提升5.48%，已经全量上线。

这篇工作的潜空间hub路由思路还挺有意思，既保住了多任务场景下的字段信息，又避免了全量交互的参数浪费。大家觉得这种latent hub范式还能搬到哪些推荐场景？评论区聊聊～

论文：HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation

欢迎投稿！欢迎合作！