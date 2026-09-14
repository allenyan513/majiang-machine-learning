"""胡牌判定与拆分枚举（日本立直麻将）。

三种和牌形：
    标准型：4 面子 + 1 雀头（副露算面子，手牌 = 14 - 3 × 副露数）
    七对子：7 个**不同**的对子（四张相同不能算两对）
    国士无双：13 种幺九牌各一 + 其中任意一种做对子（必须门清 14 张）

役和符都依赖具体拆法，所以 decompose() 返回**所有**标准型拆法，
每个拆法 = (雀头, [(kind, tile), ...])，kind 是 "chi"（tile 为顺子起点）或 "pon"（刻子）。
"""

from functools import lru_cache

from .tile import HONOR_START, NUM_TILE_TYPES, SUIT_SIZE, is_yaochu

Set = tuple[str, int]  # ("chi", start) 或 ("pon", tile)
Decomposition = tuple[int, tuple[Set, ...]]  # (雀头, 面子们)

YAOCHU = [t for t in range(NUM_TILE_TYPES) if is_yaochu(t)]  # 13 种


# ---------------------------------------------------------------- 单花色所有拆法
@lru_cache(maxsize=1 << 15)
def _suit_ways(seg: tuple[int, ...], base: int, allow_chi: bool) -> tuple[tuple[Set, ...], ...]:
    """seg 里的牌拆成面子的所有方式；base 是该花色第一张牌的 tile id。拆不完返回空。"""
    seg_l = list(seg)
    out: list[tuple[Set, ...]] = []

    def dfs(i: int, acc: list[Set]) -> None:
        while i < len(seg_l) and seg_l[i] == 0:
            i += 1
        if i == len(seg_l):
            out.append(tuple(acc))
            return
        if seg_l[i] >= 3:
            seg_l[i] -= 3
            acc.append(("pon", base + i))
            dfs(i, acc)
            acc.pop()
            seg_l[i] += 3
        if allow_chi and i + 2 < len(seg_l) and seg_l[i + 1] and seg_l[i + 2]:
            seg_l[i] -= 1; seg_l[i + 1] -= 1; seg_l[i + 2] -= 1
            acc.append(("chi", base + i))
            dfs(i, acc)
            acc.pop()
            seg_l[i] += 1; seg_l[i + 1] += 1; seg_l[i + 2] += 1

    dfs(0, [])
    return tuple(out)


def _all_melds_ways(counts: list[int]) -> list[tuple[Set, ...]]:
    """整副牌（不含雀头）拆成面子的所有方式（各花色拆法的笛卡尔积）。"""
    groups: list[tuple[tuple[Set, ...], ...]] = []
    for start in (0, 9, 18):
        seg = tuple(counts[start : start + SUIT_SIZE])
        if sum(seg) % 3:
            return []
        ways = _suit_ways(seg, start, True)
        if not ways:
            return []
        groups.append(ways)
    honors = tuple(counts[HONOR_START:NUM_TILE_TYPES])
    if any(n % 3 for n in honors):
        return []
    groups.append(_suit_ways(honors, HONOR_START, False))
    result: list[tuple[Set, ...]] = [()]
    for ways in groups:
        result = [acc + w for acc in result for w in ways]
    return result


def decompose(counts: list[int]) -> list[Decomposition]:
    """所有标准型拆法。手牌张数须为 3k+2。"""
    if sum(counts) % 3 != 2:
        return []
    counts = list(counts)
    out: list[Decomposition] = []
    for pair in range(NUM_TILE_TYPES):
        if counts[pair] >= 2:
            counts[pair] -= 2
            for melds in _all_melds_ways(counts):
                out.append((pair, melds))
            counts[pair] += 2
    return out


def is_standard_win(counts: list[int]) -> bool:
    return bool(decompose(counts))


def is_seven_pairs(counts: list[int]) -> bool:
    """七对子：14 张、7 种不同的牌各 2 张。"""
    return sum(counts) == 14 and sum(1 for n in counts if n == 2) == 7


def is_kokushi(counts: list[int]) -> bool:
    """国士无双：13 种幺九各至少 1 张，共 14 张。"""
    return sum(counts) == 14 and all(counts[t] >= 1 for t in YAOCHU) and sum(counts[t] for t in YAOCHU) == 14


def is_win(counts: list[int]) -> bool:
    """牌型上能否和（不考虑役）。"""
    return is_seven_pairs(counts) or is_kokushi(counts) or is_standard_win(counts)


def waiting_tiles(counts: list[int]) -> list[int]:
    """听牌分析：手牌 3k+1 张，返回摸到哪些牌能和（牌型上）。"""
    if sum(counts) % 3 != 1:
        return []
    counts = list(counts)
    out: list[int] = []
    for t in range(NUM_TILE_TYPES):
        if counts[t] >= 4:
            continue
        counts[t] += 1
        if is_win(counts):
            out.append(t)
        counts[t] -= 1
    return out
