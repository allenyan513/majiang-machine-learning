# 第 9 章 PPO 实战：接到麻将上

> 对应代码：`majiang/ml/rl.py`、`majiang/ml/model.py`、`tests/test_rl.py`

上一章的公式到这一章变成 326 行代码。这一章讲工程：采样架构怎么搭、奖励怎么设计、以及四个会让你损失一整天的坑。

> **计分说明**：本章的数字是在初版计分 `RON_POINTS = 1`（点炮扣 1 分）下测得的。仓库现在默认 `RON_POINTS = 3`，同样的命令跑出来数字会不同——对照表和原因见[第 15 章](15-rescoring.md)。要复现本章，把 `majiang/engine/rules.py` 里的常量改回 1。

## 整体架构

每一轮（iteration）做四件事：

```
1. 采样   10 个进程并行打 1024 局，每局 2 个学习者座位 + 2 个规则 bot
          记录每次打牌的 (特征, 动作, log概率, 价值估计, 奖励)
2. 计算   终局分数挂到最后一步 → GAE 算优势
3. 更新   PPO 剪切目标 + 价值损失 + 熵奖励 + 对老师的 KL 惩罚，3 轮 epoch
4. 评估   每 5 轮用贪心策略对 3 个规则 bot 打 200 局，记 CSV
```

60 轮约 15 分钟（10 核 MacBook）。瓶颈是第 1 步的自对局模拟，不是梯度计算——所以 GPU 帮助有限，多核 CPU 才是关键。

```bash
uv run python -m majiang.ml.rl --init models/discard.pt --iters 60 --games 1024 --out models/rl.pt
```

## Actor-Critic：加一个价值头

```python
class ActorCritic(nn.Module):
    """在 DiscardNet 的卷积干上加一个价值头（critic）。"""

    def __init__(self, policy: DiscardNet):
        super().__init__()
        self.policy = policy
        ch = policy.conv[0].out_channels
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * 34, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.policy.conv(x)
        return self.policy.head(h), self.value_head(h).squeeze(-1)

    def probs(self, x):
        logits, _ = self(x)
        return torch.softmax(DiscardNet.mask_logits(logits, x), dim=-1)
```

两个设计点：

**共享卷积主干。** policy 和 value 用同一组卷积特征，只有头不同。理由是"看懂局面"这件事对两个任务是共通的，共享能省参数、互相正则化。代价是两个任务的梯度会互相干扰（价值损失可能破坏策略）——用 `vf_coef` 调权重来平衡。

**保持 `probs()` 接口不变。** `NNAgent` 和可视化都只调用 `probs()`，所以它们不需要知道模型是 `DiscardNet` 还是 `ActorCritic`。`load()` 根据存盘时的 `kind` 字段自动构造正确的类：

```python
def save(model, path):
    if isinstance(model, ActorCritic):
        torch.save({"kind": "ac", "state": model.state_dict(), **_arch(model.policy)}, path)
    else:
        torch.save({"kind": "policy", "state": model.state_dict(), **_arch(model)}, path)
```

存架构参数（channels、blocks）而不只是 state_dict，这样加载时不用知道当初是怎么构造的。**存盘格式里带上重建对象所需的全部信息**——否则半年后你会发现模型加载不了。

## 采样：多进程的权重同步

自对局要并行，但每轮之后模型会更新，子进程需要拿到新权重。三种方案：

| 方案 | 问题 |
|---|---|
| 每轮重建进程池 | 进程创建开销大（每次要重新 import torch） |
| 用 `Pool` 传 state_dict | 60 万参数序列化 10 次，每轮几百 MB 的 IPC |
| **写临时文件 + 版本号** | 子进程按版本号判断是否需要重新加载 |

我们用第三种：

```python
_W: dict = {}  # 子进程全局缓存

def _get_model(path: str, version: int) -> ActorCritic:
    if _W.get("version") != version:
        _W["model"] = load(path)
        _W["version"] = version
    return _W["model"]
```

主进程每轮把权重写到临时文件，任务参数里带上版本号（就是 iteration 号）。子进程第一次见到新版本时从磁盘加载一次，同一轮内的后续任务直接复用内存里的。

磁盘 IO 每轮 10 次 × 3.5 MB（ActorCritic 约 87 万参数），可以忽略。**这个模式适用于任何"主进程训练、子进程采样"的架构**，比共享内存简单得多，也没有序列化开销。

另一个必须做的事：

```python
def _init_worker() -> None:
    torch.set_num_threads(1)
```

PyTorch 默认会用所有核心做矩阵运算。10 个子进程各自开 10 个线程 = 100 个线程抢 10 个核，上下文切换的开销会让速度**下降**。每个子进程限制单线程，并行度由进程数控制。这一行能带来 2–3 倍的实际加速。

## 奖励设计：稀疏奖励的解法

### 终局奖励

```python
t.r[-1] += g.result.scores[p]  # 终局奖励挂在最后一次决策上
```

一局打完，把分数（+3/+1/-1/0）加到最后一步的奖励上。前面 59 步的即时奖励是 0。因为 γ=1（不折扣），GAE 会把这个信号沿着轨迹往回传。

### 势函数塑形

只有终局奖励的话，学习信号太弱。我们加一个**稠密的中间奖励**：向听数减少就给一点正奖励。

```python
counts = list(obs.hand)
counts[a] -= 1
phi = -shanten(counts, len(obs.melds))
r = 0.0 if self.prev_phi is None else self.shaping * (phi - self.prev_phi)
self.prev_phi = phi
```

关键在于形式：奖励是**势函数的差分** `Φ(s') - Φ(s)`，而不是直接给 `-向听数`。

这是 Ng et al. (1999) 的**势函数塑形定理**：形如 `F(s,s') = γΦ(s') - Φ(s)` 的塑形奖励**不改变最优策略**。直觉解释：沿任何轨迹求和时，中间项全部抵消，只剩 `Φ(终点) - Φ(起点)`，是个常数。加了它只是改变了学习过程中的信号密度，不改变谁是最优解。

如果直接给 `-向听数` 当奖励（不是差分），AI 会学到"保持低向听数"而不是"胡牌"——比如听牌之后故意不胡，一直保持听牌状态刷奖励。**塑形必须用势函数差分形式**，这是个硬要求。

`shaping=0.1` 是系数。第 10 章会讲，这个值选得太大了——它是我们失败的核心原因。

## PPO 更新：四项损失

```python
logits, v = model(xb)
logits = masked(logits, xb)
logp_all = torch.log_softmax(logits, -1)
logp = logp_all.gather(1, a_all[idx].to(device)[:, None])[:, 0]
ratio = torch.exp(logp - old_logp[idx].to(device))
a_b = adv[idx].to(device)

pg = -torch.min(ratio * a_b, torch.clamp(ratio, 1 - clip, 1 + clip) * a_b).mean()   # 1. 策略
vl = ((v - ret[idx].to(device)) ** 2).mean()                                        # 2. 价值
p = logp_all.exp()
ent = -(p * logp_all).sum(-1).mean()                                                # 3. 熵
with torch.no_grad():
    t_logp = torch.log_softmax(masked(teacher(xb), xb), -1)
kl_t = (p * (logp_all - t_logp)).sum(-1).mean()                                     # 4. 对老师 KL

loss = policy_coef * (pg - ent_coef * ent + kl_coef * kl_t) + vf_coef * vl
```

前三项是标准 PPO。第四项是我们加的：

### 对老师的 KL 惩罚

```
kl_coef · KL(π_new ‖ π_teacher)
```

老师就是阶段 3 的监督模型（冻结参数，不更新）。这一项的作用是**把策略锚定在老师附近**。

为什么需要？因为微调一个已经不错的策略时，最大的风险是 RL 的噪声把它带偏。麻将的奖励噪声极大（一局分数标准差约 1.0，而我们想检测的改进是 0.05 量级），几百局的采样很可能给出错误的方向。KL 惩罚相当于说："你可以改进，但别离老师太远，除非你有充分证据。"

这个技巧在 RLHF 里是标配（那里是锚定在 SFT 模型上，防止语言模型被奖励模型带崩）。原理完全一样。

`kl_coef=0.05` 是个需要调的超参：太大学不动，太小失去保护。

### 优势归一化

```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

每个 batch 内把优势标准化。这让学习率的效果和奖励的绝对尺度解耦——不管你的奖励是 ±1 还是 ±1000，梯度的量级都差不多。几乎所有 PPO 实现都有这一行。

### 价值头预热

```python
stats = ppo_update(..., policy_coef=0.0 if it <= value_warmup else 1.0)
```

前 2 轮 `policy_coef=0`，只训练价值头。

理由：价值头是从零初始化的，一开始的 V(s) 是随机数，算出来的优势全是噪声。用这些噪声更新策略，会在一开始就破坏掉监督学习的成果。先让 critic 看几轮数据学会大致的局面评估，再开始动 actor。

这是微调预训练策略时的常见技巧，代价只有两轮。

## 四个坑

### 坑一：`-inf` 掩码产生 NaN

监督学习里用 `-inf` 做掩码没问题，因为交叉熵只取目标位置的 logit。但 RL 要算**熵**和 **KL**，它们对所有位置求和：

```
entropy = -Σ p · log p
```

被掩码的位置 p=0、log p=-inf，而 `0 × (-inf) = nan`。一个 nan 会在反向传播时污染所有参数，模型瞬间变成一堆 nan。

解法是用一个大负数代替：

```python
NEG = -1e9  # RL 里用大负数而不是 -inf，这样 KL / 熵里 0 * (-1e9) 仍是 0，不会出 nan

def masked(logits, x):
    return logits.masked_fill(~(x[:, 0] > 0), NEG)
```

`exp(-1e9) = 0`（下溢），softmax 结果和 `-inf` 一样；但 `0 × (-1e9) = 0`（正常的浮点乘法），不会产生 nan。

测试专门锁住这一点：

```python
def test_masked_keeps_finite_and_actor_critic_roundtrip(tmp_path):
    m = masked(logits, x)
    assert torch.isfinite(m).all()
```

### 坑二：GAE 跨局泄露

第 8 章讲过。采样时不同局的轨迹拼在一个数组里，必须用 `done` 标记边界：

```python
DONE.extend([False] * (len(t.a) - 1) + [True])
```

每条轨迹的最后一步标 True。GAE 循环里在 done 处截断。测试见第 8 章。

### 坑三：势函数的跨决策状态

```python
self.prev_phi: float | None = None
```

势函数差分需要记住"上一次决策后的势"。`LearnerAgent` 是有状态的对象，每局要新建——如果复用同一个实例跨局，第一步的奖励会包含上一局最后一步的势差，完全错误。

代码里每局重新创建 agent：

```python
agents = [
    LearnerAgent(model, trajs[p], rng, shaping) if p in learner_seats else RuleAgent()
    for p in range(4)
]
```

**带状态的 agent 必须明确生命周期。** 这类 bug 不会崩，只会让训练悄悄变差。

### 坑四：采样策略触发的引擎 bug

第 1 章提过：RL 按概率采样会走到规则 bot 永不涉足的局面，触发了加杠副露计数的 bug。一万局规则自对局没事，两万局 RL 采样就崩了。

教训：**上 RL 之前，用随机 agent 跑几万局压测引擎。** RL 采样比规则 bot 覆盖面广得多，任何潜伏的状态机 bug 都会被翻出来，而且是在训练跑了十分钟之后崩——比测试里崩贵得多。

## 日志：训练时要看什么

每轮记一行 CSV：

```python
fields = ["iter", "samples", "train_score", "eval_score", "eval_win", "eval_deal_in",
          "policy_loss", "value_loss", "entropy", "kl_teacher", "approx_kl", "clipfrac", "seconds"]
```

诊断用的指标：

| 指标 | 健康范围 | 异常说明 |
|---|---|---|
| `entropy` | 缓慢下降 | 暴跌 = 策略崩成确定性，探索没了 |
| `approx_kl` | 0.005–0.02 | 太大 = 学习率过高，一步迈太远 |
| `clipfrac` | 0.1–0.3 | 太高 = 更新幅度过大，clip 一直在生效 |
| `kl_teacher` | 缓慢上升 | 反映策略偏离老师多远 |
| `value_loss` | 应该下降 | 不降 = critic 没学会，优势估计不可信 |
| `eval_score` | 应该上升 | 这是唯一真正的目标 |

这些指标在下一章会派上大用场——我们就是靠它们诊断出失败原因的。

**训练脚本必须记日志。** 没有日志的失败训练是纯粹的时间浪费；有日志的失败训练是一次实验。

## 验证：RL 的冒烟测试

```python
def test_rl_smoke(tmp_path):
    init = tmp_path / "init.pt"
    save(DiscardNet(8, 2), str(init))
    out = tmp_path / "rl.pt"
    train(str(init), str(out), iters=2, games=8, eval_every=2, eval_games=8, workers=2,
          value_warmup=1, log_csv=str(tmp_path / "log.csv"))
    assert out.exists() and (tmp_path / "rl_last.pt").exists()
    rows = open(tmp_path / "log.csv").read().strip().splitlines()
    assert len(rows) == 3  # header + 2 iters
```

2 轮、8 局、8 通道的小网络，几秒跑完。它验的是整条 RL 流水线接得上：多进程能启动、权重能同步、GAE 不报错、PPO 能更新、CSV 能写。

**RL 代码尤其需要冒烟测试**，因为真实训练要跑十几分钟才知道结果。一个能在 10 秒内跑完全流程的测试，让你敢改代码。

## 练习

1. 把 KL 惩罚去掉（`--kl 0`）重跑 20 轮，看 `kl_teacher` 和 `eval_score` 怎么变。策略会跑多偏？
2. 现在只学打牌。加上碰/杠的决策（二分类头），奖励共享。样本会变少（碰杠机会不多），学得动吗？
3. 实现"对手池"：每 10 轮把当前模型存一份，采样时从历史模型里随机选对手。这能缓解多智能体的非平稳问题吗？

下一章：[评估与诚实的失败](10-evaluation.md)
