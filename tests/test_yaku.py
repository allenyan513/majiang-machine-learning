from majiang.engine.hand import Meld, MeldType
from majiang.engine.score import base_points, payment
from majiang.engine.tile import EAST, GREEN, RED, SOUTH, WHITE, counts_from_tiles, parse_hand
from majiang.engine.yaku import WinContext, evaluate


def ctx(hand: str, win: str, tsumo=False, melds=(), seat=SOUTH, rnd=EAST, **kw) -> WinContext:
    tiles = parse_hand(hand) + parse_hand(win)
    return WinContext(counts_from_tiles(tiles), list(melds), parse_hand(win)[0], tsumo, seat, rnd, **kw)


def names(r):
    return {n for n, _ in r.yaku}


def test_no_yaku_is_not_a_win():
    # 门清、无立直、单骑 2s、有幺九（不是断幺）：牌型能和但没有役
    assert evaluate(ctx("123m 567m 345s 678p 2s", "2s")) is None


def test_pinfu_requires_ryanmen():
    r = evaluate(ctx("123m 567m 34s 678p 22s", "5s"))
    assert r is not None and names(r) == {"平和"} and r.han == 1 and r.fu == 30
    # 嵌张：没有平和 -> 无役
    assert evaluate(ctx("123m 567m 35s 678p 22s", "4s")) is None


def test_riichi_tsumo_pinfu_20fu():
    r = evaluate(ctx("123m 567m 34s 678p 22s", "5s", tsumo=True, riichi=True))
    assert names(r) == {"立直", "门清自摸", "平和"} and r.han == 3 and r.fu == 20
    assert r.base == 640
    pay = payment(r.base, winner=1, dealer=0, is_tsumo=True, loser=None, honba=0)
    assert pay.losses == {0: 1300, 2: 700, 3: 700}


def test_sanshoku_pinfu_ron():
    r = evaluate(ctx("234m 234s 234p 55p 78m", "9m"))
    assert names(r) == {"三色同顺", "平和"} and r.han == 3 and r.fu == 30
    assert payment(r.base, 1, 0, False, 2, 0).losses == {2: 3900}


def test_open_tanyao_30fu():
    melds = [Meld(MeldType.CHI, parse_hand("2m")[0], 0, parse_hand("3m")[0])]
    r = evaluate(ctx("567m 345s 678p 2s", "2s", melds=melds))  # 吃过 234m，单骑 2s
    assert r is not None and names(r) == {"断幺九"} and r.han == 1
    assert r.fu == 30  # 20 + 单骑 2 = 22 -> 30


def test_chiitoi_25fu():
    r = evaluate(ctx("11m 33m 55s 77s 99p 东东 白", "白"))
    assert names(r) == {"七对子"} and r.fu == 25 and r.han == 2 and r.base == 400
    assert payment(r.base, 1, 0, False, 3, 0).losses == {3: 1600}


def test_yakuhai_toitoi_sanankou():
    # 荣和 白：白刻是明刻，其余三个暗刻 -> 三暗刻 + 对对和 + 役牌白 + 自风东(seat=EAST) + 场风东
    r = evaluate(ctx("111m 999p 东东东 白白 22s", "白", seat=EAST, rnd=EAST))
    assert {"对对和", "三暗刻", "役牌 白", "自风", "场风"} <= names(r)
    assert r.han == 7 and r.limit == "跳满"


def test_suuankou_tanki_double_yakuman():
    r = evaluate(ctx("111m 999p 东东东 白白白 2s", "2s"))
    assert names(r) == {"四暗刻单骑"} and r.yakuman == 2 and r.base == 16000


def test_suuankou_by_tsumo_but_not_ron():
    r = evaluate(ctx("111m 999p 东东东 白白 22s", "白", tsumo=True))
    assert "四暗刻" in names(r) and r.yakuman == 1
    r = evaluate(ctx("111m 999p 东东东 白白 22s", "白", tsumo=False))
    assert r.yakuman == 0  # 荣和补刻 -> 三暗刻


def test_kokushi_13_wait():
    r = evaluate(ctx("19m 19s 19p 东南西北白发中", "中"))
    assert names(r) == {"国士无双十三面"} and r.yakuman == 2
    r = evaluate(ctx("9m 19s 19p 东南西北白发 中中", "1m"))
    assert names(r) == {"国士无双"} and r.yakuman == 1


def test_daisangen_and_flush():
    r = evaluate(ctx("白白白 发发发 中中中 123m 5m", "5m"))
    assert "大三元" in names(r) and r.yakuman >= 1
    r = evaluate(ctx("123m 456m 789m 111m 5m", "5m", tsumo=True, riichi=True))
    assert "清一色" in names(r) and r.han >= 8


def test_chuuren():
    r = evaluate(ctx("1112345678999m", "5m", tsumo=True))
    assert names(r) == {"纯正九莲宝灯"} and r.yakuman == 2
    r = evaluate(ctx("1112345678999m 5m".replace("5m", ""), "1m", tsumo=True) if False else ctx("1122345678999m", "1m", tsumo=True))
    assert names(r) == {"九莲宝灯"} and r.yakuman == 1


def test_dora_counting():
    ind = parse_hand("2m")  # 宝牌 3m
    r = evaluate(ctx("123m 567m 34s 678p 22s", "5s", riichi=True, dora_indicators=ind, ura_indicators=parse_hand("1s"), red_count=1))
    y = dict(r.yaku)
    assert y["宝牌"] == 1 and y["赤宝牌"] == 1 and y["里宝牌"] == 2  # 里宝牌 2s：手里 22s
    assert r.han == 1 + 1 + 4  # 立直 + 平和 + 四个宝牌


def test_fu_calculation_details():
    # 门清荣和 20+10，暗刻幺九 8 (111m)，暗刻中张 4 (555s)，役牌雀头 2 (中中)，嵌张 +2 -> 46 -> 50
    r = evaluate(ctx("111m 555s 46p 789p 中中", "5p", riichi=True))
    assert r.fu == 50 and "立直" in names(r)


def test_base_points_limits():
    assert base_points(4, 40) == (2000, "满贯")   # 40×64=2560 -> 满贯
    assert base_points(3, 30) == (960, "")
    assert base_points(4, 30) == (1920, "")
    assert base_points(6, 30) == (3000, "跳满")
    assert base_points(11, 30) == (6000, "三倍满")
    assert base_points(13, 30) == (8000, "累计役满")


def test_payment_dealer_tsumo_and_honba():
    p = payment(2000, winner=0, dealer=0, is_tsumo=True, loser=None, honba=2)
    assert p.losses == {1: 4200, 2: 4200, 3: 4200} and p.winner_gain == 12600
    p = payment(2000, winner=2, dealer=0, is_tsumo=False, loser=1, honba=1)
    assert p.losses == {1: 8300}
    p = payment(960, winner=0, dealer=0, is_tsumo=False, loser=3, honba=0)
    assert p.losses == {3: 5800}  # 庄家 3 番 30 符 = 5800
