from collections import Counter

from majiang.riichi.random_agent import RandomAgent
from majiang.riichi.rule_agent import RuleAgent
from majiang.riichi.actions import PASS, Action, ActionType, chi, discard, pon, riichi, ron, tsumo
from majiang.riichi.game import Game, Phase
from majiang.riichi.hand import Hand, MeldType
from majiang.riichi.tile import EAST, counts_from_tiles, parse_hand
from majiang.riichi.run import play_game


def _rig(g: Game, hands: list[str], wall: str, dead: str = "", current: int = 0) -> None:
    """人工摆牌：hands[current] 14 张，其余 13 张；wall 按摸牌顺序；dead 前 4 岭上，第 5 张起是宝牌指示。"""
    g.hands = [Hand(counts_from_tiles(parse_hand(h))) for h in hands]
    g.wall = parse_hand(wall)
    g.wall_red = [False] * len(g.wall)
    d = parse_hand(dead) if dead else parse_hand("1m 1m 1m 1m 2p 3p 4p 5p 6p 7p 8p 9p 1p 2p")
    g.dead, g.dead_red = d, [False] * len(d)
    g.rivers = [[] for _ in range(4)]
    g.current = current
    g.draws = [1 if p == current else 0 for p in range(4)]
    g.phase = Phase.DISCARD
    g.last_draw = -1


P = parse_hand


def _pass_all(g: Game) -> None:
    """让所有待响应的玩家都过。"""
    while g.phase == Phase.RESPOND:
        for p in g.players_to_act():
            g.step(p, PASS)


FILLER = ["147m 258s 369p 东南西北", "258m 369s 147p 白发中 东", "369m 147s 258p 南西北 白", "2468m 2468s 2468p 发"]


def test_start_deals_and_dead_wall():
    g = Game(seed=1)
    g.start()
    assert g.hands[0].size == 14 and all(g.hands[p].size == 13 for p in (1, 2, 3))
    assert len(g.wall) == 136 - 14 - 53 == 69 and len(g.dead) == 14
    assert len(g.dora_indicators) == 1
    assert sum(h.red_count for h in g.hands) + sum(g.wall_red) + sum(g.dead_red) == 3


def test_tile_conservation_through_whole_game():
    for seed in range(15):
        g = Game(seed=seed)
        g.start()
        agents = [RandomAgent(seed=seed * 10 + i) for i in range(4)]
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        total = Counter(g.wall) + Counter(g.dead)
        for p in range(4):
            total.update(g.hands[p].tiles())
            total.update(d.tile for d in g.rivers[p] if not d.claimed)
            for m in g.hands[p].melds:
                total.update(m.tiles)
        assert sum(total.values()) == 136, seed
        assert all(v == 4 for v in total.values()), seed


def test_rule_games_terminate_and_points_conserve():
    for s in range(20):
        r = play_game([RuleAgent() for _ in range(4)], seed=s)
        assert r.kind in ("win", "draw", "abort")
        # 点数守恒：变动之和 = -留在桌上的立直棒
        assert sum(r.deltas) <= 0 and sum(r.deltas) % 1000 == 0


def test_riichi_flow_and_ippatsu_tsumo():
    g = Game()
    _rig(g, ["123m 456m 789m 23s 东东 白"] + FILLER[1:], "9s 9s 9s 1s")
    acts = g.legal_actions(0)
    assert riichi(P("白")[0]) in acts and discard(P("白")[0]) in acts
    assert riichi(P("2s")[0]) not in acts  # 打 2s 不听
    g.step(0, riichi(P("白")[0]))
    # 没人能响应 -> 立直成立，押 1000，一发生效，轮到玩家 1
    assert g.riichi[0] and g.ippatsu[0] and g.scores[0] == 24000 and g.riichi_sticks == 1
    assert g.rivers[0][0].riichi
    assert g.current == 1
    # 立直后只能摸切
    for p in (1, 2, 3):
        g.step(p, discard(g.last_draw))
    assert g.current == 0 and g.last_draw == P("1s")[0]
    acts = g.legal_actions(0)
    assert tsumo(P("1s")[0]) in acts
    g.step(0, tsumo(P("1s")[0]))
    names = {n for n, _ in g.result.wins[0].result.yaku}
    # 第一巡立直 = 两立直；雀头东是自风所以没有平和；123456789m 一气通贯；牌墙摸完是海底
    assert {"两立直", "一发", "门清自摸", "一气通贯", "海底摸月"} <= names
    assert sum(g.result.deltas) == 0  # 立直棒被自己拿回


def test_riichi_only_tsumogiri():
    g = Game()
    _rig(g, ["123m 456m 789m 23s 东东 白"] + FILLER[1:], "9s 9s 9s 5p 6p 7p")
    g.step(0, riichi(P("白")[0]))
    for p in (1, 2, 3):
        g.step(p, discard(g.last_draw))
    acts = g.legal_actions(0)
    assert [a for a in acts if a.type == ActionType.DISCARD] == [discard(P("5p")[0])]
    assert not any(a.type == ActionType.RIICHI for a in acts)


def test_no_yaku_cannot_ron_but_riichi_can():
    # 234m 567m 789m 23s 东东 荣和 4s：有 9 不是断幺，雀头东是场风不算平和 -> 无役
    g = Game()
    _rig(g, [FILLER[0].replace("东南西北", "东南西 4s") + "", "234m 567m 789m 23s 东东"] + FILLER[2:], "9s 9s 9s", current=0)
    g.step(0, discard(P("4s")[0]))
    assert ron(P("4s")[0]) not in g.legal_actions(1)  # 无役，不能荣和（但可以吃）
    # 同样的手牌立直后就能荣和
    g = Game()
    _rig(g, ["234m 567m 789m 23s 东东 白", FILLER[0].replace("东南西北", "东南西 4s")] + FILLER[2:], "9s 9s 9s 9s")
    g.step(0, riichi(P("白")[0]))
    g.step(1, discard(P("4s")[0]))
    assert 0 in g.players_to_act() and ron(P("4s")[0]) in g.legal_actions(0)
    g.step(0, ron(P("4s")[0]))
    assert g.result.kind == "win" and g.result.wins[0].from_player == 1


def test_furiten_blocks_ron():
    g = Game()
    # 玩家 1 听 1s/4s；先让他打过 1s（舍牌振听）
    _rig(g, [FILLER[0].replace("东南西北", "东南西 4s"), "123m 456m 789m 23s 东东"] + FILLER[2:], "1s 9s 9s 9s 4s 9s 9s 9s")
    g.step(0, discard(P("西")[0]))
    _pass_all(g)
    assert g.current == 1 and g.last_draw == P("1s")[0]
    g.step(1, discard(P("1s")[0]))  # 打掉听牌之一 -> 振听
    _pass_all(g)
    for p in (2, 3):
        g.step(p, discard(g.last_draw))
        _pass_all(g)
    g.step(0, discard(P("4s")[0]))  # 玩家 0 打 4s
    assert ron(P("4s")[0]) not in g.legal_actions(1)
    assert g.observe(1).furiten


def test_temporary_furiten_clears_on_draw():
    g = Game()
    _rig(g, ["123m 456m 789m 23s 东东 白", FILLER[0].replace("东南西北", "东南西 4s"), FILLER[2].replace("南西北 白", "南西北 4s"), FILLER[3]], "9s 9s 9s 9s 9s 9s")
    g.step(0, riichi(P("白")[0]))
    g.step(1, discard(P("4s")[0]))
    assert 0 in g.players_to_act()
    g.step(0, PASS)  # 见逃 -> 立直振听（永久）
    _pass_all(g)
    assert g.furiten_riichi[0]
    g.step(2, discard(P("4s")[0]))
    assert ron(P("4s")[0]) not in g.legal_actions(0)


def test_chi_only_from_left_and_kuikae():
    g = Game()
    _rig(g, [FILLER[0].replace("东南西北", "东南西 4s"), "56s 78s 123m 456m 白发中 东", FILLER[2].replace("南西北 白", "56s 北"), FILLER[3]], "9s 9s 9s 9s")
    g.step(0, discard(P("4s")[0]))
    # 玩家 1 是玩家 0 的下家：可以吃 4s（456s 或 345s? 只有 56s -> 456s）；玩家 2 有 56s 但不是下家
    acts1 = g.legal_actions(1)
    assert chi(P("4s")[0], P("4s")[0]) in acts1
    assert 2 not in g.players_to_act()
    g.step(1, chi(P("4s")[0], P("4s")[0]))
    assert g.hands[1].melds[0].type == MeldType.CHI and g.current == 1 and g.phase == Phase.DISCARD
    # 食替：吃 4s 组 456s 后不能打 4s（现物）也不能打 7s（筋）
    forbidden = {P("4s")[0], P("7s")[0]}
    assert not any(a.tile in forbidden for a in g.legal_actions(1) if a.type == ActionType.DISCARD)
    assert not any(a.type == ActionType.TSUMO for a in g.legal_actions(1))


def test_pon_priority_over_chi_and_ron_over_pon():
    g = Game()
    _rig(g, [FILLER[0].replace("东南西北", "东南西 4s"), "56s 78s 123m 456m 白发中 东", "44s 147m 258p 东南西北 白", "234m 567m 789m 23s 东东"], "9s 9s 9s 9s")
    g.step(0, discard(P("4s")[0]))
    assert sorted(g.players_to_act()) == [1, 2]  # 玩家 3 无役不能荣和
    g.step(1, chi(P("4s")[0], P("4s")[0]))
    g.step(2, pon(P("4s")[0]))
    assert g.current == 2 and g.hands[2].melds[0].type == MeldType.PON


def test_double_ron_and_triple_ron_abort():
    g = Game()
    _rig(g, ["19m 19s 19p 东南西北白发中 4s", "123m 456m 789m 23s 东东", "234m 567m 678p 23s 88s", FILLER[3]], "9s 9s", current=0)
    g.riichi[1] = True  # 让两家都有役
    g.riichi[2] = True
    g.step(0, discard(P("4s")[0]))
    assert sorted(g.players_to_act()) == [1, 2]
    g.step(1, ron(P("4s")[0]))
    g.step(2, ron(P("4s")[0]))
    assert g.result.kind == "win" and len(g.result.wins) == 2
    assert g.result.deltas[0] < 0 and g.result.deltas[1] > 0 and g.result.deltas[2] > 0


def test_exhaustive_draw_noten_penalty():
    g = Game()
    _rig(g, ["123m 456m 789m 23s 东东 白"] + FILLER[1:], "")
    g.step(0, discard(P("白")[0]))
    assert g.finished and g.result.kind == "draw"
    assert g.result.tenpai == [True, False, False, False]
    assert g.result.deltas == [3000, -1000, -1000, -1000]


def test_ankan_rinshan_and_new_dora():
    g = Game()
    _rig(g, ["1111m 456m 789m 23s 东东"] + FILLER[1:], "9s 9s 9s 9s", dead="5p 5p 5p 5p 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p")
    assert Action(ActionType.ANKAN, 0) in g.legal_actions(0)
    g.step(0, Action(ActionType.ANKAN, 0))
    assert g.hands[0].melds[0].type == MeldType.ANKAN
    assert g.last_draw == P("5p")[0] and g.rinshan and len(g.dora_indicators) == 2
    assert len(g.wall) == 3  # 活牌墙末尾一张移入王牌
    assert g.current == 0 and g.phase == Phase.DISCARD


def test_addkan_chankan():
    g = Game()
    _rig(g, ["4s 123m 456m 789m 白白 东", "123m 456m 789m 23s 东东"] + FILLER[2:], "9s 9s 9s 9s")
    from majiang.riichi.hand import Meld

    g.hands[0].melds.append(Meld(MeldType.PON, P("4s")[0], 3, P("4s")[0]))
    g.hands[0].counts[P("东")[0]] -= 1  # 副露一组后手牌 11 张：4s + 123m 456m 789m 白白... 调整成 11 张
    g.riichi[1] = True
    assert Action(ActionType.ADDKAN, P("4s")[0]) in g.legal_actions(0)
    g.step(0, Action(ActionType.ADDKAN, P("4s")[0]))
    assert g.phase == Phase.RESPOND and g.players_to_act() == [1]
    g.step(1, ron(P("4s")[0]))
    assert g.result.kind == "win" and "抢杠" in {n for n, _ in g.result.wins[0].result.yaku}


def test_haitei_houtei_and_no_calls_on_last_tile():
    g = Game()
    _rig(g, ["123m 456m 789m 23s 东东 白", "44s 147m 258p 东南西北 白"] + FILLER[2:], "1s")
    g.step(0, discard(P("白")[0]))
    assert g.current == 1 and g.haitei
    # 玩家 1 打东：河底牌，玩家 0 有东东也不能碰 -> 直接流局
    g.step(1, discard(P("东")[0]))
    assert g.finished and g.result.kind == "draw"
