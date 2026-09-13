from majiang.riichi.tile import counts_from_tiles, parse_hand
from majiang.riichi.win import decompose, is_kokushi, is_seven_pairs, is_standard_win, is_win, waiting_tiles


def c(s: str) -> list[int]:
    return counts_from_tiles(parse_hand(s))


def test_standard_win_cases():
    assert is_standard_win(c("123m 456m 789m 123s 东东"))
    assert is_standard_win(c("111m 222s 333p 东东东 白白"))
    assert is_standard_win(c("11123m 456s 789p 东东东"))
    assert is_standard_win(c("22334455m 678s 999p"))
    assert not is_standard_win(c("11122233m 456s 东东"))
    assert is_standard_win(c("1112345678999m 5m"))
    assert not is_standard_win(c("1112345678999m 5s"))
    # 副露后的短手牌
    assert is_standard_win(c("123m 456s 789p 东东"))
    assert is_standard_win(c("白白"))


def test_decompose_returns_all_ways():
    # 111222333m 可以拆成 三刻 或 三顺
    ways = decompose(c("111222333m 456s 东东"))
    kinds = {tuple(sorted(k for k, _ in melds)) for _, melds in ways}
    assert ("chi", "chi", "chi", "chi") in kinds
    assert ("chi", "pon", "pon", "pon") in kinds
    assert all(pair == 27 for pair, _ in ways)
    # 22334455m：两种雀头选择（2 或 5），拆法都有
    ways = decompose(c("22334455m 678s 999p"))
    assert {pair for pair, _ in ways} == {1, 4}


def test_not_wins():
    assert not is_win(c("123m 456m 789m 123s 东南"))
    assert not is_win(c("123m 456m 789m 124s 东东"))
    assert not is_win(c("东南西 123m 456m 789m 白白"))
    assert not is_win(c("123m 456m 789m 123s 东"))


def test_seven_pairs_riichi_rule():
    assert is_seven_pairs(c("11m 22m 33s 44s 55p 66p 东东"))
    assert not is_seven_pairs(c("1111m 22m 33s 44s 55p 东东"))  # 日麻：四张不算两对
    assert not is_seven_pairs(c("11m 22m 33s 44s 55p 66p 东南"))
    assert is_win(c("11m 33m 55s 77s 99p 东东 白白"))


def test_kokushi():
    assert is_kokushi(c("19m 19s 19p 东南西北白发中 中"))
    assert is_kokushi(c("19m 19s 19p 东南西北白发中 1m"))
    assert not is_kokushi(c("19m 19s 19p 东南西北白 发发发"))  # 少了中
    assert is_win(c("19m 19s 19p 东南西北白发中 东"))
    assert not is_standard_win(c("19m 19s 19p 东南西北白发中 东"))


def test_waiting_tiles():
    p = parse_hand
    assert waiting_tiles(c("123m 456m 789m 23s 东东")) == p("1s 4s")
    assert waiting_tiles(c("123m 456m 789m 123s 白")) == p("白")
    assert waiting_tiles(c("123m 456m 789m 东东 白白")) == p("东 白")
    assert waiting_tiles(c("1112345678999m")) == p("123456789m")
    assert waiting_tiles(c("19m 19s 19p 东南西北白发中")) == p("19m 19s 19p 东南西北白发中")  # 十三面
    assert waiting_tiles(c("147m 258s 369p 东南西北")) == []
