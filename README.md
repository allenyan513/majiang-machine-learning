# 麻将 AI：从零到能打

一个用来学习的项目：先写麻将引擎，再写规则 bot，再用监督学习训练神经网络模仿它，最后（阶段 4）用强化学习超越它。

规则：简化推倒胡（万条筒字 136 张，碰/杠/胡，无吃无花无番型）。

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

## 可视化

```bash
uv run python -m majiang.viz.server --model models/discard.pt --agents nn,nn,rule,rule
```

打开 http://localhost:8000 ，牌桌布局仿 QQ 麻将：中间是剩余牌数和风位，四家的牌河摆在各自面前，副露放手牌旁边。两种模式：

- **上帝视角**：四家手牌全可见，轮到谁决策就显示谁的分析（神经网络的每张牌打出概率 + 规则 bot 的向听/进张表）。适合看 AI 之间对战。
- **玩家视角**：选一个座位坐下，其他三家的牌盖着。勾上"我来打"后由你操作这个座位——轮到你时点手牌打出、按钮碰/杠/胡/过，右侧的分析面板就是给你的"预测器"。

## 代码地图

```
majiang/engine/   tile 牌编码 · win 胡牌判定 · shanten 向听数 · game 状态机
majiang/agents/   random · rule（牌效率+防守）· nn（打牌用网络，碰杠胡用规则）
majiang/ml/       features 特征编码 · generate 多进程生成数据 · model 1D-CNN · train
majiang/viz/      本地观战网页
```
