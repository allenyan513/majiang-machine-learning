# 第 1 章 把麻将变成数组

> 对应代码：`majiang/engine/tile.py`、`majiang/engine/hand.py`、`majiang/engine/rules.py`

## 原理：数据结构决定后面所有事情

写游戏 AI，第一个决定是"一张牌怎么表示"。这个决定看起来无关紧要，但它会传染到后面每一行代码——胡牌判定的写法、向听数算得多快、神经网络的输入长什么样，全都由它决定。

最自然的写法是面向对象：

```python
class Tile:
    def __init__(self, suit, number):
        self.suit = suit      # 'm' / 's' / 'p' / 'z'
        self.number = number
hand = [Tile('m', 1), Tile('m', 2), ...]   # 13 个对象
```

这么写，判断"手里有没有 3 张五万"要遍历列表，判断胡牌要排序、分组、比较对象。而这两件事，一局要做**几万次**（后面你会看到，规则 bot 每打一张牌要算几百次向听数）。更麻烦的是喂给神经网络时还得再转一次格式。

换一种思路：**牌的种类只有 34 种**（万条筒各 9 种 + 字牌 7 种）。给每种编个号 0–33，一手牌就是一个长度 34 的计数数组：

```python
counts[5] == 3   # 手里有 3 张六万
```

这一个决定同时解决了三个问题：

1. **查询 O(1)**：有几张某牌，直接索引
2. **胡牌判定变成纯数组递归**：不用排序，不用分组，见第 2 章
3. **就是神经网络的输入**：34 长度的向量，可以直接 reshape 成卷积的一维"图像"

这是全局约定，代码里管它叫 `t34` 或 tile id：

```
0–8   万 1m–9m
9–17  条 1s–9s
18–26 筒 1p–9p
27–33 字 东 南 西 北 中 发 白
```

顺序不是随便定的：**同花色的牌在数组里必须连续且按点数递增**，因为顺子是"连续三个索引"。三万四万五万就是索引 2、3、4。这让顺子判定变成 `counts[i] and counts[i+1] and counts[i+2]`，也让后面的一维卷积（kernel=3）恰好能看到一个顺子的范围。

字牌放最后，因为它们**不能组成顺子**——放在末尾，处理时"索引 >= 27 就跳过顺子逻辑"即可，不用在中间做特判。

## 实现

`majiang/engine/tile.py` 的核心就是一组坐标换算函数：

```python
NUM_TILE_TYPES = 34
TILES_PER_TYPE = 4
TOTAL_TILES = 136

SUIT_SIZE = 9
HONOR_START = 27  # 字牌起始 id


def is_honor(t: int) -> bool:
    return t >= HONOR_START


def suit(t: int) -> int:
    """0=万 1=条 2=筒 3=字。"""
    return t // SUIT_SIZE if t < HONOR_START else 3


def number(t: int) -> int:
    """数牌的点数 1–9；字牌返回 0。"""
    return t % SUIT_SIZE + 1 if t < HONOR_START else 0
```

注意 `suit()` 里的 `t // 9`：因为每个花色恰好 9 张且连续，整除就是花色号。这种"编码即计算"的设计在后面会反复用到，比如 `counts[0:9]` 直接切出万字，`counts[9:18]` 切出条子。

### 人类可读的输入输出

调试麻将 AI 时你会盯着终端看几百局，所以牌必须能读。两个方向都要：

```python
def to_str(t: int) -> str:
    """人类可读：1m, 5s, 9p, 东。"""
    if is_honor(t):
        return _HONOR_NAMES[t - HONOR_START]
    return f"{number(t)}{_SUIT_CHARS[suit(t)]}"


def parse_hand(s: str) -> list[int]:
    """'123m 456s 789p 东东' -> [0,1,2, 12,13,14, 24,25,26, 27,27]"""
```

`parse_hand` 支持麻将圈通用的简写（数字连写后跟花色字母），这在**写测试时价值巨大**。对比一下：

```python
# 没有 parse_hand
assert shanten([0,0,0,1,1,1,0,0,0, 1,1,1,...]) == 0   # 谁看得懂

# 有 parse_hand
assert shanten(c("123m 456m 789m 23s 东东")) == 0      # 一眼看出是听 1s/4s
```

花十分钟写好 parse/format，后面每个测试都省一分钟。**给你的领域写一套人类可读的序列化，是游戏 AI 项目里投入产出比最高的事情之一。**

### 手牌容器

`Hand` 在计数数组上包了一层，多了一个"副露"列表：

```python
@dataclass
class Hand:
    counts: list[int] = field(default_factory=lambda: [0] * NUM_TILE_TYPES)
    melds: list[Meld] = field(default_factory=list)
```

**副露**（meld）是碰、杠之后亮在桌面上的那组牌。关键点：副露的牌**不在 counts 里**。碰了三张东风之后，counts[27] 变成 0，melds 里多一个 `Meld(PON, 27)`。

为什么这么分？因为副露已经是**确定的面子**，胡牌判定不用再管它们——只需要知道"还差几个面子"。这个不变量贯穿全项目：

```
手牌张数 = 13 - 3 × 副露数     （轮到自己前）
手牌张数 = 14 - 3 × 副露数     （摸牌后）
```

后面所有函数签名里的 `num_melds` 参数都来自这里。

## 坑：副露数怎么算

杠有三种，它们对"副露数"的影响不一样，这是我们真实踩过的 bug：

| 类型 | 说明 | 副露数变化 |
|---|---|---|
| PON 碰 | 别人打的牌 + 手里 2 张 | +1 |
| KAN 明杠 | 别人打的牌 + 手里 3 张 | +1 |
| ANKAN 暗杠 | 自己摸到第 4 张 | +1 |
| **ADDKAN 加杠** | **已经碰过，摸到第 4 张** | **0（碰变杠，不是新的一组）** |

加杠是把已有的碰升级成杠，副露数不变。我们最初写成 +1，后果是：一个玩家已经有 4 组副露时再加杠，向听数函数收到 `num_melds=5`，`need = 4 - 5 = -1`，数组越界崩溃。

这个 bug 在**阶段 3 的一万局规则 bot 自对局里从未触发**，却在阶段 4 的 RL 采样两万局里出现了。原因很有意思：RL 的策略是**按概率采样**打牌，会走到规则 bot 永远不会走的冷门局面。

教训有两条。第一，`num_melds` 这种"从状态推导出来的量"应该有断言保护：

```python
def standard_shanten(counts: list[int], num_melds: int = 0) -> int:
    need = 4 - num_melds
    if not 0 <= need <= 4:
        raise ValueError(f"副露数不合法: {num_melds}")
```

崩在清晰的报错上，比崩在 numpy 的越界信息上好调试十倍。第二，**随机性更强的 agent 是更好的模糊测试器**。RandomAgent 不只是基线，它是你的 fuzzer。

## 验证

```bash
uv run pytest tests/test_tile.py -q
```

这一层的测试都是往返测试（round-trip）——编码再解码回来必须一样：

```python
def test_roundtrip_str():
    for t in range(34):
        assert T.from_str(T.to_str(t)) == t


def test_full_wall():
    wall = T.full_wall()
    assert len(wall) == 136
    assert all(T.counts_from_tiles(wall)[t] == 4 for t in range(34))
```

往返测试是免费的正确性保险：你不需要手算期望值，只要求"转过去再转回来不变"。

## 练习

1. 加一个 `sort_key(t)` 让字牌按东南西北中发白而不是 id 排序，不改变其他函数
2. 支持红宝牌（0m 表示红五万）：需要第 35 种 id 吗？还是在 counts 之外单独记？想想哪种对神经网络更友好
3. `full_wall()` 返回未洗的牌墙。如果要支持"三人麻将"（去掉万字），改哪几行？这个改动会波及多少个文件？

下一章：[胡牌判定](02-win.md)
