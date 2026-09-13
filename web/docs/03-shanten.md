# 第 3 章 向听数：给"还差多远"一个数字

> 对应代码：`majiang/engine/shanten.py`、`tests/test_shanten.py`

这一章是整个引擎里最难的一部分，也是最关键的一部分。如果说胡牌判定告诉你"到了没有"，向听数告诉你"还有多远"——而 AI 的每一个决策都建立在这个距离感上。

## 原理

### 定义

**向听数**（shanten）= 距离听牌还差几次有效换牌。

```
-1 = 已经胡了
 0 = 听牌（再摸一张对的就胡）
 1 = 差一张就听牌
 6 = 13 张全是孤张，最差情况
```

为什么需要它？因为胡牌判定是个布尔值，几乎所有时候都是 False，没法拿来做决策。你有 14 张牌要打掉一张，14 种打法都不胡，怎么选？向听数给出了梯度：打完这张之后离胡多远。**把稀疏的 0/1 目标变成稠密的距离信号**——这个思路后面在 RL 的奖励塑形里还会原封不动地再用一次。

### 公式

标准型的向听数有一个漂亮的闭式公式。设：

- **m** = 已经凑好的面子数（刻子或顺子）
- **t** = 搭子数（差一张成面子的两张牌，如 45m、13m、55m）
- **p** = 是否有雀头（对子），0 或 1

则：

```
shanten = 8 - 2m - t - p     约束 m + t ≤ 4
```

**怎么理解这个公式？** 从最坏情况倒推。手里 13 张全是孤张时，你需要凑 4 个面子 + 1 个雀头。每个面子要两次有效进张（孤张→搭子→面子），雀头要一次，总共 4×2+1 = 9 次。但你手里已经有 13 张牌可以当这些结构的起点，所以起点是 8（这是定义决定的常数，正好让全孤张 13 张 = 6 向听对上）。

然后：
- 每个**面子**省掉 2 步 → `-2m`
- 每个**搭子**省掉 1 步（只差一张就成面子）→ `-t`
- 有**雀头**省掉 1 步 → `-p`

约束 `m + t ≤ 4` 的意思是：你只需要 4 个面子，第 5 个搭子对你毫无用处。这个约束经常被漏掉，导致"手牌看起来有一堆搭子所以向听数很低"的错误。

来算几个例子：

```
123m 456m 789m 23s 东东    m=3, t=1(23s), p=1(东东)  →  8-6-1-1 = 0  听牌 ✓
147m 258s 369p 东南西北     m=0, t=0, p=0              →  8-0-0-0 = 8 ?
```

等等，第二个算出来是 8，但我们说最差是 6。问题在哪？

因为这 13 张牌里，**总有牌能凑**。147m 中的 1m 和 4m 相隔 3，不是搭子；但 13 张牌只有 13 张，`8 - 2m - t - p` 在 m=t=p=0 时给 8，超过了真实上限。真实答案是 6——这是因为公式还隐含一个条件：**手牌张数和结构数要匹配**。13 张牌最多用掉 `3m + 2t + 2p` 张，剩下的是孤张。当结构太少时，公式会高估。

标准做法是把公式和**手牌张数约束**一起用。我们的实现通过 DFS 枚举所有可能的 (m, t, p) 组合并取最优，而 DFS 只会产生手牌里真实存在的结构，所以自动满足约束。测试里明确锁了这个值：

```python
assert shanten(c("147m 258s 369p 东南西北")) == 6         # 全孤张：13 张最差
```

七对的公式简单得多：

```
shanten = 6 - 对子数
```

7 个对子就胡（-1），6 个对子就听（0）。我们的实现里四张相同算两对。

最终取两者较小值。

### 为什么需要 DFS 而不是直接数

"数一下有几个面子几个搭子"听起来很简单，但手牌的拆法不唯一。看这手：

```
11123m
```

可以拆成 `111m` 刻子 + `23m` 搭子（m=1, t=1），也可以拆成 `123m` 顺子 + `11m` 对子（m=1, p=1）。哪种更好取决于手牌其他部分有没有雀头。**局部最优不等于全局最优**，所以必须枚举。

## 实现：两层结构

直接对 34 张牌做 DFS 会很慢（规则 bot 每打一张牌要算几百次）。实现分两层：

### 第一层：单花色枚举所有 (m, t, p)

```python
def _suit_combos(seg, allow_sequence):
    """返回该花色所有可达的 (m, t, p)，已做支配剪枝：同 (p, m) 只保留最大 t。"""
    found = set()

    def dfs(i, m, t, p):
        while i < len(seg_l) and seg_l[i] == 0:
            i += 1
        if i == len(seg_l):
            found.add((m, min(t, _MAX_T), p))
            return
        n = seg_l[i]
        if n >= 3:                              # 刻子
            seg_l[i] -= 3; dfs(i, m + 1, t, p); seg_l[i] += 3
        if allow_sequence:
            if i + 2 < SUIT_SIZE and seg_l[i+1] and seg_l[i+2]:     # 顺子
                ...dfs(i, m + 1, t, p)...
            if i + 1 < SUIT_SIZE and seg_l[i+1]:                    # 两面搭子 45m
                ...dfs(i, m, t + 1, p)...
            if i + 2 < SUIT_SIZE and seg_l[i+2]:                    # 嵌张搭子 13m
                ...dfs(i, m, t + 1, p)...
        if n >= 2:                              # 对子
            seg_l[i] -= 2
            if p == 0:
                dfs(i, m, t, 1)                 # 当雀头
            dfs(i, m, t + 1, p)                 # 当搭子
            seg_l[i] += 2
        seg_l[i] -= 1; dfs(i, m, t, p); seg_l[i] += 1   # 孤张：丢弃
    dfs(0, 0, 0, 0)
```

和第 2 章的胡牌递归是同一个骨架（做选择 → 递归 → 撤销），只是分支从 2 个变成 6 个：刻子、顺子、两面搭子、嵌张搭子、对子（两种用法）、丢弃。

`allow_sequence=False` 用于字牌——它们只能组刻子和对子。

**支配剪枝**（dominance pruning）是这里的关键优化：

```python
best = {}
for m, t, p in found:
    key = (p, m)
    if best.get(key, -1) < t:
        best[key] = t
return [(m, t, p) for (p, m), t in best.items()]
```

在 m 和 p 都相同的情况下，t 越大向听数越小（公式里 `-t`），所以只保留最大的 t，其余全部丢弃。一个花色的组合数从几十个降到最多七八个。

### 第二层：四组合并的 DP

四个花色各有一组候选 (m, t, p)，要选一个组合使总向听数最小。约束是**全局只能有一个雀头**，以及 `m + t ≤ 4`。

```python
states = {(0, 0, 0)}                     # (p, m, t)
for combos in groups:                    # 万 条 筒 字
    nxt = set()
    for p, m, t in states:
        for gm, gt, gp in combos:
            if p + gp > 1:               # 雀头最多一个
                continue
            nm = m + gm
            if nm > need:                # 面子够了就不要更多
                continue
            nt = min(t + gt, need - nm)  # 搭子封顶
            nxt.add((p + gp, nm, nt))
    states = nxt
return min(8 - 2 * (m + num_melds) - t - p for p, m, t in states)
```

状态空间小得可怜：p ∈ {0,1}，m ∈ [0,4]，t ∈ [0,4]，最多 50 个状态。四轮合并就出答案。

`need = 4 - num_melds` 处理了副露：碰了两组就只需要再凑 2 个面子。而 `8 - 2*(m + num_melds)` 里把副露也算进面子总数——公式里的 m 是**总面子数**，包括已经亮出来的。

### 缓存

```python
@lru_cache(maxsize=1 << 16)
def _suit_combos_cached(seg: tuple[int, ...], allow_sequence: bool):
    return tuple(_suit_combos(seg, allow_sequence))
```

一个花色的牌型只有有限种（9 个位置，每个 0–4 张，实际可达的组合更少）。加上 LRU 缓存后，跑几百局之后命中率接近 100%，`shanten()` 基本退化成一次字典查找加一个小 DP。

注意参数必须是 `tuple` 而不是 `list`——`lru_cache` 要求可哈希。所以调用处写成 `_suit_combos_cached(tuple(counts[0:9]), True)`。

## 牌效率：向听数的直接应用

有了向听数，两个最实用的分析立刻就有了。

**进张**：13 张时摸到哪些牌能让向听数减少？

```python
def effective_tiles(counts, num_melds=0):
    base = shanten(counts, num_melds)
    out = []
    for t in range(34):
        if counts[t] >= 4:
            continue
        counts[t] += 1
        if shanten(counts, num_melds) < base:
            out.append(t)
        counts[t] -= 1
    return out
```

又是"加一张试试"的模式。

**打牌选择**：14 张时每种打法的好坏？

```python
def discard_options(counts, num_melds=0, visible=None):
    """返回 [(打出的牌, 打后向听数, 进张张数)]，按 (向听数升序, 进张数降序) 排序。"""
    out = []
    for d in range(34):
        if counts[d] == 0:
            continue
        counts[d] -= 1
        s = shanten(counts, num_melds)
        eff = effective_tiles(counts, num_melds)
        n_left = sum(max(0, 4 - counts[t] - visible[t]) for t in eff)
        counts[d] += 1
        out.append((d, s, n_left))
    out.sort(key=lambda x: (x[1], -x[2]))
    return out
```

这个函数就是第 5 章规则 bot 的大脑。注意 `n_left` 的算法：进张的**张数**而不是**种类数**。听 1s/4s（两种）但 1s 已经被打光 3 张，实际只剩 5 张有效牌——这个差别在实战中很重要。`visible` 参数统计牌河和副露里已经出现的牌，是 AI 的"数牌"能力。

计算量粗算一下：14 种打法 × 34 种进张 × 一次 shanten = 476 次 shanten 调用，每次决策。有 LRU 缓存才跑得动，这就是缓存不是可选优化的原因。

## 坑

**搭子封顶忘了写。** 少了 `m + t ≤ 4` 这个约束，一手全是搭子的牌会算出负的向听数。测试专门锁了这个：

```python
def test_taatsu_cap():
    # 5 个搭子 + 1 对子，但 m+t 最多 4 -> 8 - 0 - 4 - 1 = 3
    assert standard_shanten(c("12m 45m 78m 12s 45s 东东 白")) == 3
```

**对子的双重身份。** 一个对子既可以当雀头（p+1）也可以当搭子（t+1，等着摸第三张成刻子）。DFS 里必须两个分支都试。漏掉其中一个，`11m 22m 33m` 这类牌就会算错。

**七对与副露互斥。** 副露之后不可能七对（牌都亮出去了）。实现里返回 99 这个哨兵值而不是抛异常，因为上层 `min()` 会自动忽略它：

```python
def seven_pairs_shanten(counts, num_melds=0):
    if num_melds > 0:
        return 99
    return 6 - sum(c // 2 for c in counts)
```

## 验证：单调性是最好的不变量

向听数最难的地方是**你没法手算验证**复杂手牌。所以测试的重点不是具体数值，而是**结构性质**：

```python
def test_cross_check_with_win_and_waiting():
    for _ in range(2000):
        rng.shuffle(wall)
        h14 = counts_from_tiles(wall[:14])
        assert (shanten(h14) == -1) == is_win(h14)           # 和独立实现交叉验证
        h13 = counts_from_tiles(wall[:13])
        assert (shanten(h13) == 0) == bool(waiting_tiles(h13))
        # 向听数单调：加一张牌最多减 1，打一张最多加 1
        s13, s14 = shanten(h13), shanten(h14)
        assert s13 - 1 <= s14 <= s13 + 1
```

最后那条**单调性**断言特别有价值：摸一张牌不可能让你离胡近两步。这个性质对任何正确实现都成立，而对几乎所有的错误实现都不成立。在 2000 手随机牌上跑一遍，等于做了几万次一致性检查。

还有一条更强的：听牌时（向听 0），"进张"和"听牌"必须完全相同——因为让向听数从 0 变成 -1 就是胡。

```python
def test_effective_tiles_matches_waiting_at_tenpai():
    if shanten(h) == 0:
        assert effective_tiles(h) == waiting_tiles(h)
```

两条不同代码路径（公式法 vs 递归法）在同一批数据上必须重合。

**这是写数值算法的通用方法论**：当你无法手算期望输出时，找出结果必须满足的不变量（单调性、对称性、与另一实现的一致性），用随机输入去撞它们。

## 练习

1. 加一个 `--profile` 模式测 `shanten()` 的调用次数和命中率，看看 LRU 缓存实际帮了多少
2. `discard_options` 现在只看向听和进张。加一个"打完之后期望进张"（两步展望）：对每种打法，算它的所有进张摸到之后的平均进张数。这会慢多少？值得吗？
3. 实现十三幺的向听数，并入 `min()`。注意它的公式和另外两种完全不同

下一章：[游戏状态机](04-game.md)
