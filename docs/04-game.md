# 第 4 章 游戏状态机：谁在什么时候能做什么

> 对应代码：`majiang/engine/game.py`、`majiang/engine/actions.py`、`majiang/run.py`、`tests/test_game.py`

前三章做的是纯函数：给一手牌，算个数。这一章要做的是**状态**——一局牌怎么从洗牌走到结算，中间谁该说话。这是整个项目里最容易写乱的部分，因为麻将的回合制和象棋不同：**打出一张牌后，另外三家可能同时想动作**。

## 原理：两个阶段和一个优先级

把一局麻将的流程抽象出来，只有两种局面：

```
DISCARD  当前玩家手里 14 张（或 3k+2），要打牌 / 自摸 / 暗杠 / 加杠
RESPOND  有人打出一张牌，其他玩家决定 碰 / 明杠 / 点炮胡 / 过
FINISHED 胡牌或流局
```

DISCARD 阶段只有一个人能动。RESPOND 阶段可能有多个人能动，而且他们的选择会冲突——两个人都想碰同一张牌，只能给一个。

冲突用**优先级**解决：

```python
PRIORITY = {
    ActionType.RON: 3,      # 胡 > 碰杠 > 过
    ActionType.KAN: 2,
    ActionType.PON: 2,
    ActionType.PASS: 0,
}
```

同优先级（比如两家都能碰）按**离放炮者的距离**排，下家优先。这是麻将的通行规则。

```python
key = (PRIORITY[a.type], -((p - discarder) % NUM_PLAYERS))
```

`-((p - discarder) % 4)` 这个表达式值得看两眼：下家的差值是 1，对家 2，上家 3，取负之后下家最大。一行搞定座次优先级。

## 实现

### 谁该动：`players_to_act()`

引擎不"回调" agent，而是**告诉调用方现在等谁**：

```python
def players_to_act(self) -> list[int]:
    if self.phase == Phase.DISCARD:
        return [self.current]
    if self.phase == Phase.RESPOND:
        return [p for p, a in self.pending.items() if a is None]
    return []
```

这个设计（引擎被动，调用方驱动）比回调好在：

- 测试时可以手动喂动作，不需要造 agent
- 可视化服务器可以"走一步就停"，等人点击
- 并行采样时可以随时把状态序列化出去

主循环长这样，全项目只有这一处：

```python
def play_game(agents, seed=None, dealer=0):
    g = Game(seed=seed, dealer=dealer)
    g.start()
    while not g.finished:
        for p in g.players_to_act():
            obs = g.observe(p)
            action = agents[p].act(obs)
            g.step(p, action)
    return g.result
```

注意 RESPOND 阶段的多人是**顺序问询**而不是并发：循环里挨个问，`step()` 把回答存进 `pending`，全部回答完才结算。所以先被问到的人不知道别人的选择——这符合真实麻将（大家同时决定），也避免了信息泄露。

### 能做什么：`legal_actions()`

每个 agent 只能从引擎给的合法动作里选。这是**引擎的核心契约**，写死在 `step()` 的第一行：

```python
def step(self, player: int, action: Action) -> None:
    if action not in self.legal_actions(player):
        raise ValueError(f"玩家{player} 非法动作 {action}（阶段 {self.phase.name}）")
```

看起来是多余的检查（agent 不是从合法列表里选的吗），但它救过好几次。神经网络 agent 会犯各种奇怪的错误，比如打一张手里没有的牌——有这行检查，bug 在第一时间以清晰的消息炸掉，而不是让游戏状态悄悄损坏，几百局之后以"牌数不守恒"的形式浮现。

**永远在状态机入口校验动作合法性。** 这是写游戏引擎的铁律。

合法动作的生成：

```python
def _legal_discard_phase(self, player: int) -> list[Action]:
    hand = self.hands[player]
    acts = []
    if is_win(hand.counts):
        acts.append(Action(ActionType.TSUMO, self.last_draw))
    if self.wall:  # 杠需要补牌，没牌了不能杠
        for t in range(34):
            if hand.counts[t] == 4:
                acts.append(Action(ActionType.ANKAN, t))
            elif hand.counts[t] == 1 and any(
                m.type == MeldType.PON and m.tile == t for m in hand.melds
            ):
                acts.append(Action(ActionType.ADDKAN, t))
    acts.extend(Action(ActionType.DISCARD, t) for t in range(34) if hand.counts[t] > 0)
    return acts
```

`if self.wall` 那个条件是规则细节：杠之后要补一张牌，牌墙空了就不能杠。这类"边界规则"如果不写进合法动作生成，就会变成 `step()` 里的崩溃。

### 看得见什么：`Observation`

这是全项目最重要的一个数据结构，因为它定义了 **AI 能看到什么**：

```python
@dataclass
class Observation:
    """某个玩家在决策时能看到的全部信息。AI 只能基于它做决定。"""
    player: int
    phase: Phase
    hand: list[int]                  # 自己的手牌计数 (34)
    melds: list[Meld]                # 自己的副露
    rivers: list[list[int]]          # 四家牌河
    all_melds: list[list[Meld]]      # 四家副露
    wall_remaining: int
    current_player: int
    last_discard: int                # RESPOND 阶段：刚打出的牌
    last_draw: int                   # DISCARD 阶段：刚摸到的牌
    legal_actions: list[Action]
```

注意里面**没有**什么：没有别人的手牌，没有牌墙的内容，没有牌墙的顺序。这就是不完美信息的边界。

把这个边界做成一个显式的数据类，而不是"agent 拿到 Game 对象自己取"，有两个好处。第一，**作弊变得不可能**——agent 物理上拿不到别人的手牌。第二，**它直接对应神经网络的输入**：第 6 章的特征编码函数签名就是 `encode(obs) -> tensor`，一一对应。

观察里的所有列表都是**拷贝**（`list(self.hands[player].counts)`），防止 agent 意外改到引擎的状态。每次 observe 有点分配开销，但换来的是"agent 不可能弄坏引擎"的保证。在几十万次调用的规模下这个开销可以接受，实测占总时间不到 5%。

### 结算

```python
def _finish_win(self, winner, tile, loser):
    scores = [0] * NUM_PLAYERS
    if loser is None:                        # 自摸
        for p in range(NUM_PLAYERS):
            if p != winner:
                scores[p] -= TSUMO_POINTS_EACH
                scores[winner] += TSUMO_POINTS_EACH
    else:                                    # 点炮
        scores[loser] -= RON_POINTS
        scores[winner] += RON_POINTS
    self.result = Result(winner, loser, tile, loser is None, scores)
    self.phase = Phase.FINISHED
```

分数是**零和**的：`sum(scores) == 0` 永远成立。这在测试里被反复断言，也在 RL 里很重要——零和意味着"我的平均分 > 0"就等价于"我比对手强"，这是第 10 章评估的基础。

自摸得 3 分（三家各付 1），点炮得 1 分。这个 3:1 的比例给了 AI 一个隐含的信号：自摸比点炮好。真麻将里差距更大，我们先用最简单的。

## 坑

**牌被碰走要从牌河里删掉。** 打出的牌先进牌河，如果被人碰了要拿出来：

```python
self.rivers[discarder].pop()  # 牌被拿走
```

漏掉这行，牌数就不守恒了——同一张牌同时在牌河和别人的副露里。这类 bug 不会立刻崩，而是让特征编码里"这张牌还剩几张"算错，最后表现为模型莫名其妙地弱。

**杠的补牌要从牌墙另一头拿。** 真麻将有"王牌区"，杠了从后面补。我们简化成从列表头部取：

```python
t = self.wall.pop(0) if replacement else self.wall.pop()
```

正常摸牌 `pop()`（尾部），杠后补牌 `pop(0)`（头部）。这不是为了还原规则细节，而是为了避免"杠了之后把下一张该摸的牌拿走"导致的顺序混乱。

**流局判定藏在摸牌里。** 牌墙空了就流局，而"发现空了"的时机是有人要摸牌的时候：

```python
def _draw(self, player: int, replacement: bool = False) -> None:
    if not self.wall:
        self._finish_draw()
        return
    ...
```

把终止条件放在唯一的出口处，而不是散在主循环各处，是状态机保持简单的关键。

## 验证：守恒律

物理学家验算用守恒律，写游戏引擎也一样。**136 张牌，一张都不能多不能少**：

```python
def test_tile_conservation_through_whole_game():
    """打完一局，136 张牌必须一张不多一张不少地分布在 牌墙+手牌+牌河+副露 里。"""
    for seed in range(20):
        g = Game(seed=seed); g.start()
        agents = [RandomAgent(seed=seed * 10 + i) for i in range(4)]
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        total = Counter(g.wall)
        for p in range(4):
            total.update(g.hands[p].tiles())
            total.update(g.rivers[p])
            for m in g.hands[p].melds:
                total[m.tile] += 3 if m.type == MeldType.PON else 4
        assert sum(total.values()) == 136, seed
        assert all(v == 4 for v in total.values()), seed
```

这一个测试覆盖了引擎里几乎所有的状态转移，因为随机 agent 会走遍碰、杠、胡、流局所有路径。20 局大概能覆盖掉 90% 的代码路径。

另一类测试是**摆牌测试**：人工构造一个局面，验证具体的规则：

```python
def _rig(g: Game, hands: list[str], wall: str) -> None:
    """人工摆牌：hands[0] 14 张（庄家已摸），其余 13 张；wall 按摸牌顺序给出。"""
    g.hands = [Hand(counts_from_tiles(parse_hand(h))) for h in hands]
    g.wall = list(reversed(parse_hand(wall)))  # pop() 从末尾取，所以反转
    ...


def test_ron_scores_and_priority_over_pon():
    g = Game()
    _rig(g, [
        "123m 456m 789m 123s 东东 1p",
        "123m 456m 789m 23s 白白",
        "111m 222m 333m 44m 11p",          # 1p 可以碰
        "123m 456m 789m 1p 东东东",          # 1p 可以荣胡（单钓）
    ], "5p 5p 5p")
    g.step(0, Action(ActionType.DISCARD, 18))
    assert sorted(g.players_to_act()) == [2, 3]
    g.step(2, Action(ActionType.PON, 18))
    g.step(3, Action(ActionType.RON, 18))
    assert g.finished and g.result.winner == 3 and g.result.loser == 0
    assert g.result.scores == [-1, 0, 0, 1]
```

第 1 章那个 `parse_hand` 在这里回本了：整个测试一眼能读懂。**能读懂的测试才会被维护。**

## 检查点：跑一局看看

```bash
uv run python -m majiang.cli --seed 7      # 4 个随机 bot 打一局，打印全过程
uv run python -m majiang.cli --n 1000      # 跑 1000 局统计
```

实测（一台普通云主机的单核）：

```
1000 局用时 5.15s (194 局/秒)
流局 989 (98.9%)   胡牌 11  其中自摸 1
各家胜局: {0: 5, 1: 1, 2: 1, 3: 4}
```

**98.9% 流局**。随机打牌几乎凑不出四个面子，1000 局里只胡了 11 次。这个数字本身就说明了第 0 章那个论点：靠随机探索学麻将是没戏的，正例太稀少了。下一章的规则 bot 会把它降到 0。

**速度很重要**。记一下你机器上"每秒多少局"这个数：阶段 3 要跑一万局生成数据，阶段 4 每轮要跑一千局。规则 bot 因为每步要算几百次向听数，会比随机 bot 慢一个数量级（我们这里是 6 局/秒），所以后面所有批量任务都要开多进程。

## 练习

1. 加"吃"（用手里两张 + 上家打的牌组顺子）。想想它对 `_responses_for` 和优先级表的影响——吃的优先级低于碰，而且只有下家能吃
2. 加"抢杠胡"：别人加杠时，如果你正好胡那张牌，可以抢。这需要在 ADDKAN 之后插一个 RESPOND 阶段
3. 现在自摸固定 3 分。改成"庄家多付一半"，看看第 10 章的评估里位置效应会怎么变

下一章：[规则 bot 与评估框架](05-agents.md)
