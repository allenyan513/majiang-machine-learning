"""牌的编码。

全局约定：一张牌用 0–33 的整数表示（称为 tile id / t34）。

    0–8   万 1m–9m
    9–17  条 1s–9s
    18–26 筒 1p–9p
    27–33 字 东 南 西 北 中 发 白

手牌不用列表存对象，而是长度 34 的计数数组 counts[t] = 该牌张数。
这样胡牌判定/向听数是纯数组运算，之后也能直接喂给神经网络。
"""

NUM_TILE_TYPES = 34
TILES_PER_TYPE = 4
TOTAL_TILES = NUM_TILE_TYPES * TILES_PER_TYPE  # 136

SUIT_SIZE = 9
HONOR_START = 27  # 字牌起始 id

_SUIT_CHARS = "msp"
_HONOR_NAMES = ["东", "南", "西", "北", "中", "发", "白"]
_SUIT_NAMES = ["万", "条", "筒"]


def is_honor(t: int) -> bool:
    return t >= HONOR_START


def suit(t: int) -> int:
    """0=万 1=条 2=筒 3=字。"""
    return t // SUIT_SIZE if t < HONOR_START else 3


def number(t: int) -> int:
    """数牌的点数 1–9；字牌返回 0。"""
    return t % SUIT_SIZE + 1 if t < HONOR_START else 0


def is_terminal(t: int) -> bool:
    """幺九牌（1 或 9 的数牌）。"""
    return not is_honor(t) and number(t) in (1, 9)


def to_str(t: int) -> str:
    """人类可读：1m, 5s, 9p, 东。"""
    if is_honor(t):
        return _HONOR_NAMES[t - HONOR_START]
    return f"{number(t)}{_SUIT_CHARS[suit(t)]}"


def to_cn(t: int) -> str:
    """中文：一万 五条 九筒 东。"""
    if is_honor(t):
        return _HONOR_NAMES[t - HONOR_START]
    return "一二三四五六七八九"[number(t) - 1] + _SUIT_NAMES[suit(t)]


def from_str(s: str) -> int:
    """解析单张牌：'1m' / '东'。"""
    s = s.strip()
    if s in _HONOR_NAMES:
        return HONOR_START + _HONOR_NAMES.index(s)
    if len(s) == 2 and s[0].isdigit() and s[1] in _SUIT_CHARS:
        n, c = int(s[0]), _SUIT_CHARS.index(s[1])
        if 1 <= n <= 9:
            return c * SUIT_SIZE + n - 1
    raise ValueError(f"无法解析牌: {s!r}")


def parse_hand(s: str) -> list[int]:
    """解析简写手牌为 tile id 列表。

    支持两种写法，可混用（空格分隔）：
        '123m 456s 789p 东东'   -> 数字后跟花色；字牌直接写
        '1m 2m 3m'
    """
    tiles: list[int] = []
    for token in s.split():
        if token in _HONOR_NAMES:
            tiles.append(from_str(token))
            continue
        if all(ch in _HONOR_NAMES for ch in token):
            tiles.extend(from_str(ch) for ch in token)
            continue
        digits, c = token[:-1], token[-1]
        if c not in _SUIT_CHARS or not digits.isdigit():
            raise ValueError(f"无法解析: {token!r}")
        base = _SUIT_CHARS.index(c) * SUIT_SIZE
        tiles.extend(base + int(d) - 1 for d in digits)
    return tiles


def hand_to_str(tiles: list[int]) -> str:
    """把 tile 列表压成简写：'123m 456s 东东'。"""
    tiles = sorted(tiles)
    parts: list[str] = []
    for s_idx in range(3):
        nums = "".join(str(number(t)) for t in tiles if suit(t) == s_idx)
        if nums:
            parts.append(nums + _SUIT_CHARS[s_idx])
    honors = "".join(to_str(t) for t in tiles if is_honor(t))
    if honors:
        parts.append(honors)
    return " ".join(parts)


def counts_from_tiles(tiles: list[int]) -> list[int]:
    counts = [0] * NUM_TILE_TYPES
    for t in tiles:
        counts[t] += 1
    return counts


def tiles_from_counts(counts: list[int]) -> list[int]:
    out: list[int] = []
    for t, n in enumerate(counts):
        out.extend([t] * n)
    return out


def full_wall() -> list[int]:
    """一副完整的 136 张牌（未洗）。"""
    return [t for t in range(NUM_TILE_TYPES) for _ in range(TILES_PER_TYPE)]
