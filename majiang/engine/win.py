"""胡牌判定。

规则（简化推倒胡）：
    - 标准型：4 个面子（刻子/顺子）+ 1 个对子，共 14 张
      副露（碰/杠）已经是面子，所以手牌张数 = 14 - 3 * 副露数
    - 七对：7 个不同或相同的对子（四张相同算两对）

输入统一是 34 长度的计数数组 counts。
"""

from .tile import HONOR_START, NUM_TILE_TYPES, SUIT_SIZE


def _can_form_melds(counts: list[int]) -> bool:
    """counts 里的牌能否全部拆成刻子/顺子（无对子）。

    按花色独立处理：不同花色之间不可能组成顺子，
    所以每个花色的牌数必须是 3 的倍数，否则直接失败。
    """
    for start in (0, 9, 18):
        seg = counts[start : start + SUIT_SIZE]
        if sum(seg) % 3 != 0:
            return False
        if not _melds_in_suit(seg, 0):
            return False
    # 字牌只能组刻子
    for t in range(HONOR_START, NUM_TILE_TYPES):
        if counts[t] % 3 != 0:
            return False
    return True


def _melds_in_suit(seg: list[int], i: int) -> bool:
    """递归：seg[i:] 能否拆完。贪心从最小的牌开始，它只能作为刻子或顺子的开头。"""
    while i < SUIT_SIZE and seg[i] == 0:
        i += 1
    if i == SUIT_SIZE:
        return True
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
        seg[i] -= 1
        seg[i + 1] -= 1
        seg[i + 2] -= 1
        ok = _melds_in_suit(seg, i)
        seg[i] += 1
        seg[i + 1] += 1
        seg[i + 2] += 1
        if ok:
            return True
    return False


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


def is_seven_pairs(counts: list[int]) -> bool:
    """七对：手牌必须是完整 14 张（有副露就不可能）。"""
    if sum(counts) != 14:
        return False
    return all(n % 2 == 0 for n in counts)


def is_win(counts: list[int]) -> bool:
    return is_seven_pairs(counts) or is_standard_win(counts)


def waiting_tiles(counts: list[int]) -> list[int]:
    """听牌分析：手牌 13 张（或 13-3k），返回摸到哪些牌能胡。"""
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
