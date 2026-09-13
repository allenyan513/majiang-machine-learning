# 第 2 章 胡牌判定：一个递归就够了

> 对应代码：`majiang/engine/win.py`、`tests/test_win.py`

## 原理

胡牌的定义（简化推倒胡）：14 张牌能拆成 **4 个面子 + 1 个对子**，或者 **7 个对子**。

面子有两种：刻子（三张一样，如 555m）、顺子（三张连续同花色，如 345m）。

直觉上这像个组合爆炸问题——14 张牌拆成 5 组，有多少种拆法？但有三个结构性观察能把它变简单。

**观察一：花色互不干扰。** 顺子只能在同花色内组成，所以万、条、筒、字四组可以独立处理。这立刻把问题从"14 张牌"降到"最多 9 张一组"。

更进一步：如果某个花色的牌数不是 3 的倍数（拿掉对子之后），那它必然拆不完——因为面子都是 3 张。这是一个 O(1) 的剪枝，能砍掉绝大多数失败分支。

**观察二：最小的牌只有两个去处。** 拿花色里最小的那张牌（比如 3m），它要么是某个刻子的一部分，要么是某个顺子的**开头**。它不可能是顺子的中间或结尾——因为那需要更小的牌，而它已经是最小的了。

这个观察是整个算法的核心。它把"枚举所有拆法"变成了"每一步最多两个分支"，而且不需要回溯到已经处理过的牌。

**观察三：对子（雀头）先挑出来。** 枚举 34 种牌里哪个当对子（最多 34 次），剩下的必须全部拆成面子。这是一层外循环。

合起来，算法是：

```
for 每种可能的对子:
    拿掉这个对子
    if 剩下的能全拆成面子:  ← 按花色分治 + 最小牌两分支递归
        return True
```

## 实现

### 第一层：能否全拆成面子

```python
def _can_form_melds(counts: list[int]) -> bool:
    """counts 里的牌能否全部拆成刻子/顺子（无对子）。"""
    for start in (0, 9, 18):
        seg = counts[start : start + SUIT_SIZE]
        if sum(seg) % 3 != 0:      # 观察一的剪枝
            return False
        if not _melds_in_suit(seg, 0):
            return False
    # 字牌只能组刻子
    for t in range(HONOR_START, NUM_TILE_TYPES):
        if counts[t] % 3 != 0:
            return False
    return True
```

字牌那一段直接用取模判断——因为字牌不能组顺子，"能拆成面子"就等价于"每种牌是 3 的倍数"，连递归都不需要。

### 第二层：单花色递归

```python
def _melds_in_suit(seg: list[int], i: int) -> bool:
    """seg[i:] 能否拆完。贪心从最小的牌开始，它只能作为刻子或顺子的开头。"""
    while i < SUIT_SIZE and seg[i] == 0:
        i += 1
    if i == SUIT_SIZE:
        return True                      # 全拆完了
    n = seg[i]
    # 尝试刻子
    if n >= 3:
        seg[i] -= 3
        ok = _melds_in_suit(seg, i)
        seg[i] += 3
        if ok:
            return True
    # 尝试顺子 i, i+1, i+2
    if i + 2 < SUIT_SIZE and seg[i + 1] > 0 and seg[i + 2] > 0:
        seg[i] -= 1; seg[i + 1] -= 1; seg[i + 2] -= 1
        ok = _melds_in_suit(seg, i)
        seg[i] += 1; seg[i + 1] += 1; seg[i + 2] += 1
        if ok:
            return True
    return False
```

几个值得注意的细节：

**原地修改 + 手动恢复。** 每个分支先减、递归、再加回来。这比每次 `list(seg)` 复制快得多，而且递归深度最多 5 层（9 张牌最多 3 个面子），栈没有压力。这个模式（做选择 → 递归 → 撤销选择）是回溯算法的标准形状，你在 N 皇后、数独求解器里见过同一个骨架。

**递归传的还是 `i` 而不是 `i+1`。** 因为拿掉一个刻子后，`seg[i]` 可能还有剩（比如原本 4 张），仍然是最小的牌，要继续在 i 处处理。`while seg[i] == 0` 的前进循环会自动跳到下一个非零位置。

**先试刻子再试顺子的顺序无关正确性**，只影响平均速度。两个分支都试过才返回 False，所以是完备搜索。

### 拼起来

```python
def is_standard_win(counts: list[int]) -> bool:
    """4 面子 + 1 对子（面子数按手牌张数自动推算）。"""
    total = sum(counts)
    if total % 3 != 2:
        return False
    counts = list(counts)
    for pair in range(NUM_TILE_TYPES):
        if counts[pair] >= 2:
            counts[pair] -= 2
            ok = _can_form_melds(counts)
            counts[pair] += 2
            if ok:
                return True
    return False
```

注意 `total % 3 != 2` 这一行：合法的胡牌手牌张数永远是 3k+2（14、11、8、5、2——每副露一组少 3 张）。这一个取模同时处理了"手牌张数不对"和"副露了几组"两件事，不需要额外传 `num_melds`。

七对是特例，两行搞定：

```python
def is_seven_pairs(counts: list[int]) -> bool:
    """七对：手牌必须是完整 14 张（有副露就不可能）。"""
    if sum(counts) != 14:
        return False
    return all(n % 2 == 0 for n in counts)
```

`n % 2 == 0` 意味着四张相同的牌算两对——不同规则对此有分歧（有些地方"四张不算两对"），这里按宽松的算。规则分歧点记在注释里，以后好改。

### 听牌分析

有了 `is_win`，"听什么牌"是一行循环：

```python
def waiting_tiles(counts: list[int]) -> list[int]:
    """手牌 13 张（或 13-3k），返回摸到哪些牌能胡。"""
    if sum(counts) % 3 != 1:
        return []
    counts = list(counts)
    out = []
    for t in range(NUM_TILE_TYPES):
        if counts[t] >= 4:      # 自己手里已经 4 张，摸不到第 5 张
            continue
        counts[t] += 1
        if is_win(counts):
            out.append(t)
        counts[t] -= 1
    return out
```

这是本项目的一个反复出现的模式：**"加一张试试"式的枚举**。34 次 `is_win` 调用，每次几微秒，完全够用。第 3 章的进张分析、规则 bot 的打牌选择，全是这个模式的变体。

先写对，再优化——而且到最后我们也没优化，因为从来没成为瓶颈。

## 坑

**忘记 `list(counts)` 复制。** `is_standard_win` 里如果直接改传进来的数组，调用方的手牌就被污染了。原地修改要么全程配对恢复，要么入口处复制一次。我们两种都用了：入口复制一次，内层配对恢复。

**递归里 `i+1` vs `i`。** 前面说过，拿掉刻子后要留在原地。写成 `i+1` 的话，`4 张 + 后续` 这种情况会漏判。这个 bug 很隐蔽，只有特定手牌触发——所以需要下面的交叉验证测试。

## 验证：交叉验证比手写用例更值钱

手写几个胡牌用例很容易，但覆盖不了边角。真正有效的是**用两种独立方法算同一件事，比对结果**：

```python
def test_cross_check_with_win_and_waiting():
    """随机手牌交叉验证：向听 -1 <=> 胡；向听 0 <=> 有听牌。"""
    rng = random.Random(42)
    wall = full_wall()
    for _ in range(2000):
        rng.shuffle(wall)
        h14 = counts_from_tiles(wall[:14])
        assert (shanten(h14) == -1) == is_win(h14), h14
        h13 = counts_from_tiles(wall[:13])
        assert (shanten(h13) == 0) == bool(waiting_tiles(h13)), h13
```

`is_win`（递归拆解）和 `shanten`（第 3 章的公式法）是两套完全不同的实现。它们在 2000 手随机牌上必须一致——如果不一致，至少有一个错了。这种测试能抓到你根本想不到要测的情况。

断言里带上 `h14` 参数很重要：失败时直接打印出触发的手牌，省得再复现。

## 练习

1. 实现"十三幺"（13 种幺九牌各一张 + 其中一张成对）。注意它和标准型、七对是并列关系
2. `waiting_tiles` 没考虑"这张牌已经被打光了"。加一个 `visible` 参数，返回"理论上听 + 实际还剩几张"
3. 性能实验：把 `_can_form_melds` 的 `% 3` 剪枝去掉，测 10 万次调用的耗时差多少？（提示：差距会比你想的大）

下一章：[向听数与牌效率](03-shanten.md)
