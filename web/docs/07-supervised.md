# 第 7 章 监督学习：让网络模仿规则 bot

> 对应代码：`majiang/ml/generate.py`、`majiang/ml/train.py`、`majiang/agents/nn_agent.py`

这一章把规则 bot 的知识"蒸馏"进神经网络。做完之后模型的水平约等于老师——听起来没意义（为什么不直接用规则 bot？），但它是阶段 4 的必要前提，而且过程中会暴露一个深刻的问题：**模仿得像，不等于打得好。**

> **计分说明**：本章的数字是在初版计分 `RON_POINTS = 1`（点炮扣 1 分）下测得的。仓库现在默认 `RON_POINTS = 3`，同样的命令跑出来数字会不同——对照表和原因见[第 15 章](15-rescoring.md)。要复现本章，把 `majiang/engine/rules.py` 里的常量改回 1。

## 原理：为什么要先模仿

### 为什么不直接 RL

第 0 章说过，从零 RL 在麻将上不收敛。具体说：

- 一局 60 次决策，只有终局一个奖励
- 随机策略的胡牌率接近 0，奖励几乎恒为 0
- 4 个玩家互相影响，你的动作和结果之间隔着大量噪声

这叫**探索问题**。RL 需要"偶尔做对然后被强化"，但麻将里"偶尔做对"的概率太低了。

解法是**行为克隆**（behavior cloning）：先用一个会打牌的老师产生数据，监督学习把策略灌进网络。网络起点就是一个会打牌的策略，RL 只需要在附近微调。AlphaGo 用人类棋谱做这一步，Suphx 用天凤高段牌谱，我们用规则 bot。

### 模型能超过老师吗

监督学习的天花板就是老师——这是定义决定的，你在最小化和老师的差异。但有两个意外之喜：

1. **平滑化**：网络会对老师的决策做插值。老师在两个选项分数接近时的选择是任意的（排序 key 的 tie-break），网络学到的是概率分布，可能更合理
2. **泛化**：网络对没见过的局面给出"看起来对"的答案，而规则 bot 在边角情况可能很怪

但也有系统性损失：网络看不到老师的**计算过程**（向听数、进张表），只看到输入输出。学不像的地方就是损失。

实测：验证集一致率 86.8%，实战和老师打平。**不赚不赔，但拿到了一个可微分的策略**——这才是重点。规则 bot 没法用梯度优化，神经网络可以。

## 实现

### 造数据：包装器模式

```python
class Recorder(Agent):
    """包住任意 agent，把它的打牌决策录下来。"""

    def __init__(self, inner: Agent, xs: list, ys: list):
        self.inner, self.xs, self.ys = inner, xs, ys

    def act(self, obs: Observation) -> Action:
        a = self.inner.act(obs)
        if obs.phase == Phase.DISCARD and a.type == ActionType.DISCARD:
            self.xs.append(encode_compact(obs))
            self.ys.append(a.tile)
        return a
```

第 5 章那个"只有一个方法的接口"在这里回本了：录制器本身是个 agent，套在任何 agent 外面都能用。不需要改引擎，不需要改规则 bot。

只录 DISCARD 动作，不录碰杠胡。原因：打牌占决策的 90% 以上，而且是最有策略含量的部分；碰杠胡样本少，且规则已经够好。这个选择贯穿全项目（`NNAgent` 和 RL 的 `LearnerAgent` 也只学打牌）。

### 多进程：吃满 CPU

```python
def generate(games: int, seed: int = 0, workers: int | None = None):
    workers = workers or os.cpu_count() or 1
    chunk = max(1, games // (workers * 4))  # 每个任务几百局，负载均衡
    tasks = [(seed + i, min(chunk, games - i)) for i in range(0, games, chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, tasks)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
```

自对局是纯 CPU、无共享状态的任务，多进程能线性加速。10 核跑一万局约 2 分钟，产出约 20 万条样本。

`chunk` 取 `games / (workers * 4)` 而不是 `games / workers`：切得比核数多几倍，快的进程能多领任务，避免一个慢任务拖住所有人。这是**动态负载均衡**的廉价做法。

每个任务用不同的 seed 段，保证对局不重复。

### 训练循环

标准的监督学习，没有花招：

```python
for ep in range(1, epochs + 1):
    model.train()
    order = rng.permutation(len(y_tr))
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        xb = expand(x_tr[idx]).to(device)          # 紧凑 -> 平面，在这里展开
        yb = torch.as_tensor(y_tr[idx], dtype=torch.int64, device=device)
        logits = model.mask_logits(model(xb), xb)  # 掩码非法动作
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    val_loss, val_acc = evaluate(model, x_val, y_val, device)
```

值得注意的三点：

**掩码在训练时就用。** `mask_logits` 让交叉熵只在手里有的牌上归一化，网络不用学"别打没有的牌"。

**余弦退火。** `CosineAnnealingLR` 让学习率从 1e-3 平滑降到 0，比固定学习率稳，比手动 step 省事。八轮训练的小任务上，它值这一行代码。

**设备自动选择。**

```python
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

MPS 是 Apple Silicon 的 GPU 后端。放在第一位是因为这个项目主要在 MacBook 上开发；有 CUDA 的机器会走第二个分支。

### 跑一遍

```bash
uv run python -m majiang.ml.generate --games 10000 --out data/rule_10k.npz   # ~2 分钟，10 核
uv run python -m majiang.ml.train --data data/rule_10k.npz --epochs 8 --out models/discard.pt
```

规则 bot 自对局每局产生约 42 条打牌样本（实测 41.85），所以一万局大约 42 万条。训练时每轮会打印训练/验证的损失和一致率，8 轮在 MPS 上约 30 秒。

最终结果：**验证集一致率 86.8%**，模型 59.5 万参数。训练和验证的指标接近，没有明显过拟合——42 万样本对 60 万参数偏少，但数据同质性很高（全部来自同一个确定性老师），所以问题不大。

### 把模型变成 agent

```python
class NNAgent(Agent):
    def act(self, obs: Observation) -> Action:
        by_type = {a.type: a for a in obs.legal_actions}
        # 胡牌、响应阶段、杠：交给规则
        if obs.phase != Phase.DISCARD or ActionType.TSUMO in by_type:
            return self.rule.act(obs)
        rule_action = self.rule.act(obs)
        if rule_action.type in (ActionType.ANKAN, ActionType.ADDKAN):
            return rule_action

        probs = self.discard_probs(obs)
        if self.temperature <= 0:
            t = int(probs.argmax())
        else:
            p = probs ** (1.0 / self.temperature)
            t = int(torch.multinomial(p / p.sum(), 1, generator=self.gen))
        return Action(ActionType.DISCARD, t)
```

这是个**混合 agent**：打牌用网络，碰杠胡用规则。刻意为之——只有打牌被训练过，其他动作用老师的逻辑。

`temperature` 参数控制随机性：0 = 取最大概率（评估用），>0 = 按概率采样（RL 采样用）。温度越高越随机。这个参数在第 9 章会用到。

## 关键发现：一致率 86.8%，实战打平

```bash
uv run python -m majiang.evaluate --agents nn:models/discard.pt,rule,rule,rule --n 300
```

结果：胜率 25.0%，正好是四人游戏的基准线，也就是**和三个规则 bot 完全打平**。

这里有个值得琢磨的问题：**一致率 86.8% 意味着 13% 的决策和老师不同，为什么实战一点不差？**

两个原因：

1. **大部分分歧无关紧要。** 当两三种打法向听数和进张都一样时，老师按 tie-break 规则选一个，网络选另一个。这类分歧占多数，对结果没影响。
2. **剩下的分歧有好有坏，互相抵消。** 网络在某些局面比老师平滑（好），在某些罕见局面犯傻（坏）。

反过来说这也是一个警告：**一致率（或任何代理指标）和真实目标之间可能没有单调关系。** 如果你只盯着一致率调参，把它从 86.8% 提到 90%，实战强度可能纹丝不动，甚至下降（过拟合老师的 tie-break 噪声）。

**始终用你真正关心的指标做最终判断。** 我们真正关心的是均分，所以每次改模型都要跑 `evaluate` 或 `compare`，而不是看验证集准确率。

## 坑

**训练/推理格式不一致。** 第 6 章那个 `batch == single` 测试就是防这个。

**忘记 `model.eval()`。** 有 dropout 或 BN 的模型，忘了切换会让推理结果随机。我们的网络没有这两种层，所以不受影响，但 `load()` 里还是写上了，防止以后加层时踩坑。

**数据里的样本不独立。** 同一局的 60 个决策高度相关（手牌逐步演化）。随机划分验证集时，验证集里的样本可能和训练集来自同一局，导致验证指标偏乐观。更严谨的做法是**按局划分**。我们没做，因为一万局的数据量下影响有限——但如果你要发论文，这是个必须处理的点。

## 验证

端到端的冒烟测试，跑完整条流水线：

```python
def test_generate_train_play(tmp_path):
    x, y = generate(games=12, seed=0, workers=2)
    assert x.shape[1] == 308 and len(x) == len(y) > 100
    data = tmp_path / "d.npz"
    np.savez(data, x=x, y=y)
    out = tmp_path / "m.pt"
    model = train(str(data), epochs=1, out=str(out), bs=64, channels=8, blocks=2)
    agent = NNAgent(load(str(out)), seed=1)
    for s in range(3):
        r = play_game([agent, RuleAgent(), RuleAgent(), agent], seed=s)
        assert sum(r.scores) == 0
```

12 局数据、1 轮训练、8 通道的小网络——训出来的模型很弱，但这个测试要验的不是强度，而是**每个接口对得上**：生成的数据能训练，训出的模型能加载，加载的模型能合法打完一局。跑完只要几秒，适合放进 CI。

**为机器学习流水线写冒烟测试**：用最小规模跑通全程。它抓不到"模型不够强"，但能抓到 90% 的形状错误、路径错误、序列化错误。

## 练习

1. 数据量消融：分别用 1k / 5k / 10k 局的数据训练，画一致率曲线。收益什么时候开始递减？
2. 按局划分验证集（上面"坑"里提到的问题），看验证一致率掉多少。这个差值就是数据泄露的程度
3. 训练一个同时预测"打哪张"和"当前向听数"的多任务模型（加一个回归头）。辅助任务能提高主任务的一致率吗？

下一章：[强化学习原理](08-rl-theory.md)
