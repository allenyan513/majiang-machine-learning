"""向听数（离听牌还差几张）与牌效率分析。

向听数定义：
    -1 = 已胡牌
     0 = 听牌
     n = 至少还要换 n 张牌才能听牌

标准型公式（面子 m、搭子 t、对子 p）：
    shanten = 8 - 2*m - t - p，且约束 m + t <= 4
七对子（日麻）：
    shanten = 6 - 对子数 + max(0, 7 - 牌的种类数)   四张相同只算一对，且要 7 种不同的牌
国士无双：
    shanten = 13 - 幺九种类数 - (有幺九对子 ? 1 : 0)

实现分两层：
    1. 每个花色独立 DFS，枚举出所有可达的 (m, t, p) 三元组（带 lru_cache）
    2. 把 万/条/筒/字 四组的结果用小 DP 合并，取最优
这样规则 bot 每步几百次调用也跑得动。
"""

from functools import lru_cache

from .tile import NUM_TILE_TYPES, SUIT_SIZE, HONOR_START
from .win import YAOCHU

_MAX_T = 4  # 搭子数超过 4 没有意义，DFS 里直接封顶


# ---------------------------------------------------------------- 单花色枚举
def _suit_combos(seg: tuple[int, ...], allow_sequence: bool) -> list[tuple[int, int, int]]:
    """返回该花色所有可达的 (m, t, p)，已做支配剪枝：同 (p, m) 只保留最大 t。"""
    seg_l = list(seg)
    found: set[tuple[int, int, int]] = set()

    def dfs(i: int, m: int, t: int, p: int) -> None:
        while i < len(seg_l) and seg_l[i] == 0:
            i += 1
        if i == len(seg_l):
            found.add((m, min(t, _MAX_T), p))
            return
        n = seg_l[i]
        # 刻子
        if n >= 3:
            seg_l[i] -= 3
            dfs(i, m + 1, t, p)
            seg_l[i] += 3
        # 顺子 / 两面搭子 / 嵌张搭子（字牌不允许）
        if allow_sequence:
            if i + 2 < SUIT_SIZE and seg_l[i + 1] and seg_l[i + 2]:
                seg_l[i] -= 1; seg_l[i + 1] -= 1; seg_l[i + 2] -= 1
                dfs(i, m + 1, t, p)
                seg_l[i] += 1; seg_l[i + 1] += 1; seg_l[i + 2] += 1
            if i + 1 < SUIT_SIZE and seg_l[i + 1]:
                seg_l[i] -= 1; seg_l[i + 1] -= 1
                dfs(i, m, t + 1, p)
                seg_l[i] += 1; seg_l[i + 1] += 1
            if i + 2 < SUIT_SIZE and seg_l[i + 2]:
                seg_l[i] -= 1; seg_l[i + 2] -= 1
                dfs(i, m, t + 1, p)
                seg_l[i] += 1; seg_l[i + 2] += 1
        # 对子：当作雀头（p），或当作对子搭子（t）
        if n >= 2:
            seg_l[i] -= 2
            if p == 0:
                dfs(i, m, t, 1)
            dfs(i, m, t + 1, p)
            seg_l[i] += 2
        # 孤张：丢弃
        seg_l[i] -= 1
        dfs(i, m, t, p)
        seg_l[i] += 1

    dfs(0, 0, 0, 0)
    best: dict[tuple[int, int], int] = {}
    for m, t, p in found:
        key = (p, m)
        if best.get(key, -1) < t:
            best[key] = t
    return [(m, t, p) for (p, m), t in best.items()]


@lru_cache(maxsize=1 << 16)
def _suit_combos_cached(seg: tuple[int, ...], allow_sequence: bool) -> tuple[tuple[int, int, int], ...]:
    return tuple(_suit_combos(seg, allow_sequence))


# ---------------------------------------------------------------- 合并
def standard_shanten(counts: list[int], num_melds: int = 0) -> int:
    """标准型向听数。num_melds = 已副露的面子数（碰/杠）。"""
    need = 4 - num_melds  # 手牌还需要凑出几个面子
    if not 0 <= need <= 4:
        raise ValueError(f"副露数不合法: {num_melds}")
    groups = [
        _suit_combos_cached(tuple(counts[0:9]), True),
        _suit_combos_cached(tuple(counts[9:18]), True),
        _suit_combos_cached(tuple(counts[18:27]), True),
        _suit_combos_cached(tuple(counts[HONOR_START:NUM_TILE_TYPES]), False),
    ]
    # DP 状态 (p, m, t) -> 可达；逐组合并
    states: set[tuple[int, int, int]] = {(0, 0, 0)}
    for combos in groups:
        nxt: set[tuple[int, int, int]] = set()
        for p, m, t in states:
            for gm, gt, gp in combos:
                if p + gp > 1:
                    continue
                nm = m + gm
                if nm > need:
                    continue
                nt = min(t + gt, need - nm)
                nxt.add((p + gp, nm, nt))
        states = nxt
    return min(8 - 2 * (m + num_melds) - t - p for p, m, t in states)


def seven_pairs_shanten(counts: list[int], num_melds: int = 0) -> int:
    if num_melds > 0:
        return 99
    pairs = sum(1 for c in counts if c >= 2)
    kinds = sum(1 for c in counts if c >= 1)
    return 6 - pairs + max(0, 7 - kinds)


def kokushi_shanten(counts: list[int], num_melds: int = 0) -> int:
    if num_melds > 0:
        return 99
    kinds = sum(1 for t in YAOCHU if counts[t] >= 1)
    has_pair = any(counts[t] >= 2 for t in YAOCHU)
    return 13 - kinds - (1 if has_pair else 0)


def shanten(counts: list[int], num_melds: int = 0) -> int:
    return min(
        standard_shanten(counts, num_melds),
        seven_pairs_shanten(counts, num_melds),
        kokushi_shanten(counts, num_melds),
    )


# ---------------------------------------------------------------- 牌效率
def effective_tiles(counts: list[int], num_melds: int = 0) -> list[int]:
    """13 张状态下，摸到哪些牌能让向听数减少（进张）。"""
    base = shanten(counts, num_melds)
    counts = list(counts)
    out: list[int] = []
    for t in range(NUM_TILE_TYPES):
        if counts[t] >= 4:
            continue
        counts[t] += 1
        if shanten(counts, num_melds) < base:
            out.append(t)
        counts[t] -= 1
    return out


def discard_options(
    counts: list[int], num_melds: int = 0, visible: list[int] | None = None
) -> list[tuple[int, int, int]]:
    """14 张状态下，评估每种打法。

    返回 [(打出的牌, 打后向听数, 进张张数)]，按 (向听数升序, 进张数降序) 排序。
    进张张数 = 所有进张牌在牌池里还剩几张（4 - 手里 - 已见到 visible）。
    """
    counts = list(counts)
    visible = visible or [0] * NUM_TILE_TYPES
    out: list[tuple[int, int, int]] = []
    for d in range(NUM_TILE_TYPES):
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
