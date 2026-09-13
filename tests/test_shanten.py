import random

from majiang.engine.shanten import (
    discard_options,
    effective_tiles,
    seven_pairs_shanten,
    shanten,
    standard_shanten,
)
from majiang.engine.tile import counts_from_tiles, full_wall, parse_hand
from majiang.engine.win import is_win, waiting_tiles


def c(s: str) -> list[int]:
    return counts_from_tiles(parse_hand(s))


def test_known_values():
    assert shanten(c("123m 456m 789m 123s 东东")) == -1       # 胡
    assert shanten(c("123m 456m 789m 23s 东东")) == 0         # 听
    assert shanten(c("123m 456m 789m 23s 东南")) == 1         # 差一张
    assert shanten(c("147m 258s 369p 东南西北")) == 6         # 全孤张：13 张最差
    assert shanten(c("1112345678999m")) == 0                 # 九莲听牌


def test_taatsu_cap():
    # 5 个搭子 + 1 对子，但 m+t 最多 4 -> 8 - 0 - 4 - 1 = 3
    assert standard_shanten(c("12m 45m 78m 12s 45s 东东 白")) == 3
    # 换成 4 搭子 + 对子 + 1 面子：8 - 2 - 3(封顶 4-1) - 1 = 2
    assert standard_shanten(c("123m 45m 78m 12s 45s 东东")) == 2


def test_open_hand():
    # 碰了两次，手里 8 张：123m 45s 东东 -> 已有 3 面子 + 1 搭子 + 对子 = 0 向听
    assert shanten(c("123m 45s 东东"), num_melds=2) == 0
    assert shanten(c("123m 45s 东南"), num_melds=2) == 1
    assert seven_pairs_shanten(c("11m 22m 33m 44m"), num_melds=2) == 99


def test_seven_pairs():
    assert shanten(c("11m 22m 33s 44s 55p 66p 东")) == 0         # 七对听
    assert shanten(c("11m 22m 33s 44s 55p 66p 东东")) == -1      # 七对胡
    assert shanten(c("11m 22m 33s 44s 55p 6p 东南")) == 1
    assert seven_pairs_shanten(c("1111m 22m 33s 44s 55p 东")) == 0  # 四张算两对


def test_effective_tiles():
    p = parse_hand
    assert effective_tiles(c("123m 456m 789m 23s 东东")) == p("1s 4s")
    # 1 向听：123m 456m 789m 23s 东南 -> 摸 1s/4s 听牌(单钓)，摸 东/南 也听牌(两面)
    assert effective_tiles(c("123m 456m 789m 23s 东南")) == p("1s 4s 东 南")


def test_discard_options_prefers_efficiency():
    # 14 张：123m 456m 789m 23s 东东 + 孤张 白 -> 打白后听 1s4s
    opts = discard_options(c("123m 456m 789m 23s 东东 白"))
    best_tile, s, n = opts[0]
    assert best_tile == parse_hand("白")[0] and s == 0 and n == 8  # 1s×4 + 4s×4


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
        # 向听数单调：加一张牌最多减 1，打一张最多加 1
        s13 = shanten(h13)
        s14 = shanten(h14)
        assert s13 - 1 <= s14 <= s13 + 1


def test_effective_tiles_matches_waiting_at_tenpai():
    rng = random.Random(7)
    wall = full_wall()
    checked = 0
    for _ in range(3000):
        rng.shuffle(wall)
        h = counts_from_tiles(wall[:13])
        if shanten(h) == 0:
            assert effective_tiles(h) == waiting_tiles(h)
            checked += 1
    assert checked > 0
