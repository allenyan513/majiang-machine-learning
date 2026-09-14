# 第 5 章 第一个对手：规则 bot 与评估框架

> 对应代码：`majiang/agents/`、`majiang/evaluate.py`、`tests/test_rule_agent.py`

> **计分说明**：本章的数字是在初版计分 `RON_POINTS = 1`（点炮扣 1 分）下测得的。仓库现在默认 `RON_POINTS = 3`，同样的命令跑出来数字会不同——对照表和原因见[第 15 章](15-rescoring.md)。要复现本章，把 `majiang/engine/rules.py` 里的常量改回 1。

有了引擎，现在需要一个会打牌的东西。这一章写两个 agent（随机、规则）和一套评估工具。规则 bot 会成为阶段 3 的**老师**和整个项目的**基线**——后面每一个模型的价值，都是用"比它强多少"来衡量的。

## Agent 接口：一个方法

```python
class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """必须从 obs.legal_actions 里选一个。"""
```

就这一个方法。随机 bot、规则 bot、神经网络、RL 采样器、人类玩家，全都实现这一个接口。这让"换个 agent 跑一局"变成一行参数：

```bash
uv run python -m majiang.evaluate --agents rule,random,random,random --n 500
```

**接口越窄，可组合性越强。** 后面第 7 章的数据录制器是一个"包住另一个 agent"的 agent，第 9 章的 RL 采样器也是——它们能存在，全靠这个接口只有一个方法。

## 随机 bot：不只是占位符

```python
class RandomAgent(Agent):
    """随机打牌。唯一的"智能"：能胡就胡——否则一局几乎永远打不完。"""

    def act(self, obs: Observation) -> Action:
        for a in obs.legal_actions:
            if a.type in (ActionType.TSUMO, ActionType.RON):
                return a
        return self.rng.choice(obs.legal_actions)
```

注意那个"能胡就胡"的特判。纯随机的话，AI 摸到胡牌的那一刻会有 1/14 的概率选择"胡"，剩下 13/14 打掉——一局能打到天荒地老。

随机 bot 有三个真实用途：

1. **性能基线**：任何 agent 打不过随机就是写错了
2. **模糊测试器**：它会走到你想不到的局面（第 1 章那个加杠 bug 就是这么发现的）
3. **引擎压测**：`test_tile_conservation_through_whole_game` 靠它遍历状态空间

## 规则 bot：牌效率 + 一点防守

> 本章讲的是规则 bot 的第一版（代码里的 `RuleAgent(defense="v1")`）。第 13 章给它加了校准式防守（`v2`），现在是默认值；打牌逻辑的骨架不变。

规则 bot 的决策逻辑按优先级排：

```
1. 能胡就胡（自摸 / 点炮）
2. 碰 / 杠：只在向听数减少（杠：不增加）时才做
3. 打牌：向听数最低 > 进张最多 > 危险度最低
```

### 打牌：直接调用第 3 章的分析

```python
visible = self._visible(obs)
options = discard_options(hand, n_melds, visible)
best_shanten = options[0][1]

danger = self._danger_map(obs)
push = best_shanten <= 0  # 自己已听牌就不防守

def key(opt):
    d, s, n = opt
    iso = 2 if is_honor(d) else 1 if is_terminal(d) else 0  # 同分时先扔字牌/幺九
    if push:
        return (s, -n, -iso)
    return (s, danger[d], -n, -iso)

best = min(options, key=key)
return Action(ActionType.DISCARD, best[0])
```

这个排序 key 就是整个 bot 的策略，值得逐项读：

- **`s` 向听数第一**：离胡近永远最重要
- **`danger[d]` 危险度第二**（未听牌时）：同样离胡近的打法里，选安全的
- **`-n` 进张数第三**：进张越多越好
- **`-iso` 孤张优先第四**：同分时先扔字牌、再扔幺九——因为它们组不成顺子，长期价值低

`push` 那一行是一个真实麻将策略的极简版："**自己听牌了就不防守**"。这叫"推"（push）——听牌的期望收益超过点炮的风险。真实的推/收决策要算期望值，这里用一条 if 近似。

### 防守：一条启发式

```python
@staticmethod
def _danger_map(obs: Observation) -> list[int]:
    """每张牌的危险度 = 有多少个"疑似听牌"的对手没打过它。"""
    danger = [0] * NUM_TILE_TYPES
    for p in range(4):
        if p == obs.player or len(obs.all_melds[p]) < DANGER_MELDS:
            continue
        seen = set(obs.rivers[p])
        for t in range(NUM_TILE_TYPES):
            if t not in seen:
                danger[t] += 1
    return danger
```

逻辑：**副露 3 组以上的人可能听牌了；他自己打过的牌相对安全。**

这是一条很弱的启发式。日麻里有"振听"规则（自己打过的牌不能荣胡），所以"他打过的牌绝对安全"成立；中国规则没有振听，所以这个假设只是概率上偏安全（他打过说明他不需要）。

**为什么明知道弱还这么写？** 因为它提供了一个真实的改进空间——第 10 章的结论正是"要超过规则 bot，需要学到它没有的防守知识"。老师留下的短板，就是学生的机会。

### 碰和杠：只在有用时做

```python
if ActionType.PON in by_type:
    t = by_type[ActionType.PON].tile
    after = self._with(hand, t, -2)   # 碰完手里少两张
    if shanten(after, n_melds + 1) < cur:
        return by_type[ActionType.PON]
return by_type[ActionType.PASS]
```

新手最常见的错误是"能碰就碰"。碰会让手牌变少、失去灵活性，有时反而变远。这里的判据很干净：**碰完之后向听数真的减少了才碰。**

杠的判据宽松一点（`<=` 而不是 `<`），因为杠还有补牌的好处。

一个容易错的细节：碰完之后手牌是 3k+2 张（可以直接打牌），所以 `shanten(after, ...)` 算的是"打掉最好的一张之后"的向听数——因为向听数函数对 3k+2 张的输入会自动取最优打法。这正好是我们想要的比较。

## 评估框架：比"胜率"多想一层

```bash
uv run python -m majiang.evaluate --agents rule,random,random,random --n 500
```

```
500 局，用时 34.4s，流局 77 (15.4%)
agent        局数      胜率     自摸率     放炮率      均分
rule        500   84.2%   20.4%    0.0%  +1.248
random     1500    0.1%    0.1%   21.3%  -0.416
```

这个表格里每一列都有用：

- **胜率**：多久胡一次。4 人游戏的基准线是 25%
- **自摸率**：主动性的指标。自摸得 3 分，点炮胡只得 1 分
- **放炮率**：防守能力。这一局里规则 bot 是 0.0%——不是它防守多好，而是三个随机 bot 基本胡不了牌（胜率 0.1%），没人能接炮
- **均分**：**唯一真正重要的指标**。零和游戏里 >0 就是赚

为什么均分最重要？因为它把胜率、自摸、放炮全部折算进了同一个尺度。一个"胜率高但老点炮"的 agent 可能均分是负的。**优化目标必须是单一标量**——这也是 RL 的奖励函数（第 9 章）。

### 座位轮转：消除位置偏差

```python
for g in range(n):
    shift = g % 4
    seat_names = names[shift:] + names[:shift]  # 轮转座位
    agents = [make_agent(nm, base_seed + g * 4 + i) for i, nm in enumerate(seat_names)]
    r = play_game(agents, seed=base_seed + g, dealer=g % 4)
```

庄家先摸牌，天然多一点优势；庄家的下家也有微弱优势。如果固定座位，你测的是"座位 + agent"的混合效果。轮转之后每个 agent 在每个位置的局数相同，偏差被平均掉。

这个细节在第 10 章会变得非常重要——我们实测出位置效应有 0.05–0.08 分，而阶段 4 想检测的改进量级也就是这么大。**不控制这个变量，实验就是白做。**

### 模型缓存

```python
_MODEL_CACHE: dict[str, object] = {}

def make_agent(name: str, seed: int) -> Agent:
    """支持 'random' / 'rule' / 'nn:models/discard.pt'。"""
    if name.startswith("nn:"):
        path = name[3:]
        if path not in _MODEL_CACHE:
            _MODEL_CACHE[path] = load(path)
        return NNAgent(_MODEL_CACHE[path], seed=seed)
    return AGENT_FACTORIES[name](seed)
```

每局造 4 个 agent，如果每次都从磁盘 load 一遍 60 万参数的模型，评估会慢十倍。缓存按路径存，多个 agent 共享同一个 `nn.Module` 对象——这没问题，因为推理时模型是无状态的（`eval()` 模式，不更新 BN 统计量）。

## 验证

规则 bot 的测试分两类。**行为测试**验证具体决策：

```python
def test_discards_isolated_honor_first():
    hand = "123m 456m 789m 23s 东东 白"
    a = RuleAgent().act(_obs(hand, legal))
    assert a.type == ActionType.DISCARD and a.tile == parse_hand("白")[0]


def test_pon_only_when_it_helps():
    # 东东 是唯一的对子（雀头），碰了反而没雀头 -> 向听不减 -> 过
    obs = _obs("123m 456m 789m 23s 东东", [pon(27), PASS])
    obs.phase = Phase.RESPOND
    assert agent.act(obs) == PASS
    # 有额外对子 55s，碰东能进一步 -> 碰
    obs = _obs("123m 456m 79m 55s 东东", [pon(27), PASS])
    obs.phase = Phase.RESPOND
    assert agent.act(obs) == pon(27)
```

第二个测试尤其值得学：**同一个动作在两种局面下的正确答案相反**。只测一种情况的话，一个"永远不碰"的 bot 也能过测试。

**统计测试**验证整体强度：

```python
def test_rule_agent_beats_random_and_never_illegal():
    rep = evaluate(["rule", "random", "random", "random"], n=60, base_seed=123)
    s = rep["stats"]
    assert s["rule"]["wins"] / s["rule"]["games"] > 0.5
    assert rep["draws"] / rep["n"] < 0.5
```

阈值（>50% 胜率）设得很宽松，因为 60 局的随机性大。**统计测试的阈值要留足余量**，否则它会变成随机失败的 flaky test，最后被人加 skip 标记。

## 检查点

现在的成果：

| 对局 | 流局率 | 结果 |
|---|---|---|
| 4 个随机 bot（1000 局） | 98.9% | 几乎打不完 |
| 1 规则 + 3 随机（500 局） | 15.4% | 规则 bot 胜率 84.2%，均分 +1.248 |
| 4 个规则 bot（300 局） | 0.0% | 各家胜率 25.0%，均分 0.000 |

最后一行是三个结论的合体：规则 bot 强到每一局都能打出胜负（流局 0）；四个相同策略对打时胜率精确地落在 25%、均分精确地是 0（零和 + 座位轮转的必然结果，同时也验证了评估代码没有偏差）。

"4 个规则 bot 对打"这个设置就是阶段 3 的数据源——每局都有胜负，训练数据里不缺正例。

## 练习

1. 给防守加一条"现物"以外的规则：如果对手打过 4m，那么 1m 和 7m 也相对安全（筋牌）。测一下胜率变化
2. `DANGER_MELDS = 3` 是拍脑袋定的。跑几组评估找最优值。注意每组要跑够局数才有统计意义（见第 10 章）
3. 写一个"只求速度不防守"的激进 bot 和现在的 bot 对打 2000 局。谁赢？如果差距在噪声内，说明什么？（做完再看第 13 章）

下一章：[特征工程与网络结构](06-features.md)
