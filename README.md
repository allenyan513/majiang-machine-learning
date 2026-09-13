# 麻将 AI：从零到能打

一个用来学习的项目：先写麻将引擎，再写规则 bot，再用监督学习训练神经网络模仿它，最后（阶段 4）用强化学习超越它。

规则：简化推倒胡（万条筒字 136 张，碰/杠/胡，无吃无花无番型）。

## 教程

配套写了一份 15 章的教程，讲怎么从零把这个项目写出来（引擎 → 规则 bot → 监督学习 → PPO → 评估 → 改进老师 → 转向日麻）：[web/docs/](web/docs/README.md)。

在 GitHub 上可以直接读 Markdown；想要带目录导航的网页版：

```bash
uv run python -m web.server   # 教程 http://localhost:8000/docs/  游戏 /play/
```

## 安装

```bash
uv sync
uv run pytest -q
```

## 阶段 1–2：引擎与规则 bot

```bash
uv run python -m majiang.cli --seed 7                 # 看 4 个随机 bot 打一局
uv run python -m majiang.evaluate --agents rule,random,random,random --n 500
```

## 阶段 3：监督学习

```bash
uv run python -m majiang.ml.generate --games 10000 --out data/rule_10k.npz   # ~2 分钟，10 核
uv run python -m majiang.ml.train --data data/rule_10k.npz --epochs 8 --out models/discard.pt   # ~30 秒 (MPS)
uv run python -m majiang.evaluate --agents nn:models/discard.pt,rule,rule,rule --n 300
```

参考结果（1 万局数据，60 万参数的一维 CNN）：验证集与规则 bot 一致率 86.8%；实战对 3 个规则 bot 胜率 25.0%（打平）。

## 阶段 4：强化学习（PPO 微调）

```bash
uv run python -m majiang.ml.rl --init models/discard.pt --iters 60 --games 1024 --out models/rl.pt   # ~15 分钟
uv run python -m majiang.ml.plot models/rl_log.csv                                                   # 画训练曲线
uv run python -m majiang.ml.compare --candidates rule,nn:models/discard.pt,nn:models/rl.pt --n 2000   # 配对评估
```

从阶段 3 的模型初始化（从零 RL 在麻将上收敛不了），加价值头组成 Actor-Critic。每轮 10 进程并行打 1024 局（每局 2 个学习者座位 + 2 个规则 bot），奖励 = 终局分数 + 势函数塑形（向听数减少给小奖励），PPO 更新时对老师策略加 KL 惩罚防止跑偏。每 5 轮贪心评估一次，均分 > 0 即超过规则 bot。

**评估注意**：单局分数的标准差约 1.0，2000 局独立评估的 95% 区间是 ±0.04，分辨不出小的进步。`compare.py` 让各候选打完全相同的牌局（同 seed、同座位），比差值能把噪声压低一些。

第一次尝试（60 轮 × 1024 局，塑形 0.1，KL 0.05）：贪心评估胜率 20% → 23.5%，配对评估与规则 bot / 监督模型差距均在 ±0.04 内——**没有超过老师**。分析见 PR #3。

## 可视化

```bash
uv run python -m web.server --model models/discard.pt --agents nn,nn,rule,rule
```

打开 http://localhost:8000 ，牌桌布局仿 QQ 麻将：中间是剩余牌数和风位，四家的牌河摆在各自面前，副露放手牌旁边。两种模式：

- **上帝视角**：四家手牌全可见，轮到谁决策就显示谁的分析（神经网络的每张牌打出概率 + 规则 bot 的向听/进张表）。适合看 AI 之间对战。
- **玩家视角**：选一个座位坐下，其他三家的牌盖着。勾上"我来打"后由你操作这个座位——轮到你时点手牌打出、按钮碰/杠/胡/过，右侧的分析面板就是给你的"预测器"。

## 代码地图

```
majiang/engine/   tile 牌编码 · win 胡牌判定 · shanten 向听数 · game 状态机
majiang/agents/   random · rule（牌效率+防守）· nn（打牌用网络，碰杠胡用规则）
majiang/ml/       features 特征编码 · generate 多进程生成数据 · model 1D-CNN + ActorCritic · train 监督 · rl PPO · plot 曲线
web/              server 统一服务 · docs 教程 · play 单机游戏（一个镜像部署）
```
