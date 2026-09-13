from majiang.engine.tile import counts_from_tiles, parse_hand
from majiang.engine.win import is_seven_pairs, is_standard_win, is_win, waiting_tiles


def c(s: str) -> list[int]:
    return counts_from_tiles(parse_hand(s))


def test_standard_win_cases():
    assert is_standard_win(c("123m 456m 789m 123s 东东"))
    assert is_standard_win(c("111m 222s 333p 东东东 白白"))
    assert is_standard_win(c("123m 111s 999p 中中中 55m"))
    assert is_standard_win(c("111m 123m 55m 456s 789p"))
    assert is_standard_win(c("11123m 456s 789p 东东东"))
    assert is_standard_win(c("22334455m 678s 999p"))
    assert is_standard_win(c("11122233m 456s 东东")) is False  # 111 222 33 -> 33 不是面子；123 123 12? 不行
    assert is_standard_win(c("111222333m 456s 东东"))          # 三刻 或 三顺 都成立
    assert is_standard_win(c("11112222m 345s 678p 9m"))  is False  # 15 张不合法
    assert is_standard_win(c("1112345678999m 5s")) is False  # 九莲宝灯 + 5s 不胡
    assert is_standard_win(c("1112345678999m 5m"))          # 九莲 + 5m 胡


def test_meld_count_with_open_hands():
    # 碰了一次 -> 手牌 11 张：3 面子 + 1 对
    assert is_standard_win(c("123m 456s 789p 东东"))
    # 碰了三次 -> 手牌 5 张
    assert is_standard_win(c("123m 白白"))
    # 碰了四次 -> 手牌 2 张（单钓）
    assert is_standard_win(c("白白"))


def test_not_wins():
    assert not is_win(c("123m 456m 789m 123s 东南"))   # 无对子
    assert not is_win(c("123m 456m 789m 124s 东东"))   # 124 不是顺子
    assert is_win(c("123m 456m 789m 789s 东东"))       # 两个相同顺子也合法
    assert not is_win(c("12m 456m 789m 123s 东东 中"))  # 12 + 中 不成
    assert not is_win(c("东南西北中发白 东南西北中发"))   # 13 张
    assert not is_win(c("123m 456m 789m 123s 东"))     # 13 张


def test_honor_cannot_form_sequence():
    assert not is_win(c("东南西 123m 456m 789m 白白"))


def test_seven_pairs():
    assert is_seven_pairs(c("11m 22m 33s 44s 55p 66p 东东"))
    assert is_seven_pairs(c("1111m 22m 33s 44s 55p 东东"))  # 四张算两对
    assert not is_seven_pairs(c("11m 22m 33s 44s 55p 66p 东南"))
    assert not is_seven_pairs(c("11m 22m 33s 44s 55p 66p"))  # 12 张
    # 七对但不是标准型
    assert is_win(c("11m 33m 55s 77s 99p 东东 白白"))
    assert not is_standard_win(c("11m 33m 55s 77s 99p 东东 白白"))


def test_waiting_tiles():
    from majiang.engine.tile import parse_hand as p

    # 两面听：123m 456m 789m 23s 东东 -> 听 1s 4s
    assert waiting_tiles(c("123m 456m 789m 23s 东东")) == p("1s 4s")
    # 单钓：4 面子 + 单张
    assert waiting_tiles(c("123m 456m 789m 123s 白")) == p("白")
    # 双碰：听 东 或 白
    assert waiting_tiles(c("123m 456m 789m 东东 白白")) == p("东 白")
    # 边张 + 嵌张：12m 听 3m；13m 听 2m
    assert waiting_tiles(c("12m 456m 789m 123s 东东")) == p("3m")
    # 九莲宝灯 听 9 面
    assert waiting_tiles(c("1112345678999m")) == p("123456789m")
    # 没听
    assert waiting_tiles(c("147m 258s 369p 东南西北")) == []
    # 14 张不是听牌状态
    assert waiting_tiles(c("123m 456m 789m 123s 东东")) == []
