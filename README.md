# 麻将 AI：从零到能打

一个用来学习的项目：先写麻将引擎，再写规则 bot，再用监督学习训练神经网络模仿它，最后（阶段 4）用强化学习超越它。

规则：简化推倒胡（万条筒字 136 张，碰/杠/胡，无吃无花无番型）——教程主线刻意保持规则简单，把注意力留给机器学习。

另有一套完整的日本立直麻将引擎（役、番符计分、立直、振听、半庄）独立保留在 `majiang/riichi/`，附一个单机牌桌：`uv run python -m majiang.riichi.server --port 8001`。它不在教程主线上，规则研究见 [第 14 章](web/docs/14-riichi-rules.md)。

## 教程

配套写了一份 16 章的教程，讲怎么从零把这个项目写出来（引擎 → 规则 bot → 监督学习 → PPO → 评估 → 改进老师 → 转向日麻）：[web/docs/](web/docs/README.md)。

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
uv run python -m majiang.ml.generate --games 30000 --out data/rule_30k.npz   # ~5 分钟，10 核
uv run python -m majiang.ml.train --data data/rule_30k.npz --epochs 8 --out models/discard.pt   # ~90 秒 (MPS)
uv run python -m majiang.evaluate --agents nn:models/discard.pt,rule,rule,rule --n 300
```

参考结果（3 万局数据，60 万参数的一维 CNN，`RON_POINTS = 3`）：验证集与规则 bot 一致率 86.9%；4000 局配对评估比老师低 0.048 ± 0.058（噪声范围内）。1 万局数据时一致率 84.3%、落后 0.10——数据量的作用见教程第 15 章。

## 阶段 4：强化学习（PPO 微调）

```bash
uv run python -m majiang.ml.rl --init models/discard.pt --iters 60 --games 1024 --out models/rl.pt   # ~15 分钟
uv run python -m majiang.ml.plot models/rl_log.csv                                                   # 画训练曲线
uv run python -m majiang.ml.compare --candidates rule,nn:models/discard.pt,nn:models/rl.pt --n 2000   # 配对评估
```

从阶段 3 的模型初始化（从零 RL 在麻将上收敛不了），加价值头组成 Actor-Critic。每轮 10 进程并行打 1024 局（每局 2 个学习者座位 + 2 个规则 bot），奖励 = 终局分数 + 势函数塑形（向听数减少给小奖励），PPO 更新时对老师策略加 KL 惩罚防止跑偏。每 5 轮贪心评估一次，均分 > 0 即超过规则 bot。

**评估注意**：单局分数的标准差约 1.0，2000 局独立评估的 95% 区间是 ±0.04，分辨不出小的进步。`compare.py` 让各候选打完全相同的牌局（同 seed、同座位），比差值能把噪声压低一些。

两次尝试（旧计分 / 新计分各 60 轮）都**没有超过老师**：配对评估与监督模型无差别。新计分下单局分数标准差 1.6，400 局的评估曲线是纯噪声。分析见教程第 10、15 章。

## 可视化

```bash
uv run python -m web.server --model models/discard.pt --agents nn,nn,rule,rule
```

打开 http://localhost:8000 ，牌桌布局仿 QQ 麻将：中间是剩余牌数和风位，四家的牌河摆在各自面前，副露放手牌旁边。两种模式：

- **上帝视角**：四家手牌全可见，轮到谁决策就显示谁的分析（神经网络的每张牌打出概率 + 规则 bot 的向听/进张表）。适合看 AI 之间对战。
- **玩家视角**：选一个座位坐下，其他三家的牌盖着。勾上"我来打"后由你操作这个座位——轮到你时点手牌打出、按钮碰/杠/胡/过，右侧的分析面板就是给你的"预测器"。

## 代码地图

```
majiang/engine/   tile 牌编码 · win 胡牌判定 · shanten 向听数 · game 状态机（简化规则，教程主线）
majiang/riichi/   日本立直麻将引擎 + agent + 独立牌桌服务器（保留，不在主线）
majiang/agents/   random · rule（牌效率+防守）· nn（打牌用网络，碰杠胡用规则）
majiang/ml/       features 特征编码 · generate 多进程生成数据 · model 1D-CNN + ActorCritic · train 监督 · rl PPO · plot 曲线
web/              server 统一服务 · docs 教程 · play 单机游戏（一个镜像部署）
```
