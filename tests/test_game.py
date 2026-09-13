from collections import Counter

from majiang.agents.random_agent import RandomAgent
from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Game, Phase
from majiang.engine.hand import MeldType
from majiang.engine.tile import counts_from_tiles, parse_hand
from majiang.run import play_game


def test_start_deals_correctly():
    g = Game(seed=1)
    g.start()
    assert g.hands[0].size == 14
    assert all(g.hands[p].size == 13 for p in (1, 2, 3))
    assert len(g.wall) == 136 - 14 - 39
    assert g.phase == Phase.DISCARD and g.players_to_act() == [0]


def test_tile_conservation_through_whole_game():
    """打完一局，136 张牌必须一张不多一张不少地分布在 牌墙+手牌+牌河+副露 里。"""
    for seed in range(20):
        g = Game(seed=seed)
        g.start()
        agents = [RandomAgent(seed=seed * 10 + i) for i in range(4)]
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        total = Counter(g.wall)
        for p in range(4):
            total.update(g.hands[p].tiles())
            total.update(g.rivers[p])
            for m in g.hands[p].melds:
                total[m.tile] += 3 if m.type == MeldType.PON else 4
        assert sum(total.values()) == 136, seed
        assert all(v == 4 for v in total.values()), seed


def test_random_games_terminate_and_have_result():
    results = [play_game([RandomAgent(seed=i) for i in range(4)], seed=s) for s in range(50)]
    assert all(r.winner is None or 0 <= r.winner < 4 for r in results)
    # 随机 bot 也应该偶尔胡牌
    assert any(r.winner is not None for r in results)
    # 分数守恒
    assert all(sum(r.scores) == 0 for r in results)


def _rig(g: Game, hands: list[str], wall: str) -> None:
    """人工摆牌：hands[0] 14 张（庄家已摸），其余 13 张；wall 按摸牌顺序给出。"""
    from majiang.engine.hand import Hand

    g.hands = [Hand(counts_from_tiles(parse_hand(h))) for h in hands]
    g.wall = list(reversed(parse_hand(wall)))  # pop() 从末尾取，所以反转
    g.rivers = [[] for _ in range(4)]
    g.current = 0
    g.phase = Phase.DISCARD
    g.last_draw = -1


def test_ron_flow():
    g = Game()
    _rig(
        g,
        [
            "123m 456m 789m 123s 东东 白",     # 庄家 14 张，打白
            "123m 456m 789m 23s 白白",         # 玩家1 听 1s 4s
            "123m 456m 789m 123s 白白白",       # 玩家2 有 白白白 -> 打出白时可以... 不，白是 3 张，可以杠
            "111m 222m 333m 444m 5m",
        ],
        "1s 2s 3s",
    )
    assert Action(ActionType.DISCARD, parse_hand("白")[0]) in g.legal_actions(0)
    g.step(0, Action(ActionType.DISCARD, parse_hand("白")[0]))
    # 玩家2 可以 KAN 白，玩家1 白白 可以 PON；两家都要回应
    assert g.phase == Phase.RESPOND
    assert sorted(g.players_to_act()) == [1, 2]
    acts1 = g.legal_actions(1)
    assert Action(ActionType.PON, 33) in acts1 and Action(ActionType.RON, 33) not in acts1
    acts2 = g.legal_actions(2)
    assert Action(ActionType.KAN, 33) in acts2
    # 玩家1 碰，玩家2 杠 -> 同优先级，下家(玩家1)优先
    g.step(1, Action(ActionType.PON, 33))
    g.step(2, Action(ActionType.KAN, 33))
    assert g.phase == Phase.DISCARD and g.current == 1
    assert g.hands[1].melds[0].type == MeldType.PON
    assert g.rivers[0] == []  # 白被拿走
    # 玩家1 碰后打 3s（把听牌打掉一部分也无所谓，这里测试流程）
    g.step(1, Action(ActionType.DISCARD, parse_hand("2s")[0]))
    # 没人能响应 2s? 玩家2 手里 123s 有 2s 一张 -> 不能碰；继续到玩家2 摸牌
    assert g.current == 2 and g.phase == Phase.DISCARD


def test_tsumo_and_scores():
    g = Game()
    _rig(g, ["123m 456m 789m 123s 东东", "1m 2m 3m 4m 5m 6m 7m 8m 9m 1s 2s 3s 4s", "东 南 西 北 中 发 白 东 南 西 北 中 发", "1p 2p 3p 4p 5p 6p 7p 8p 9p 1p 2p 3p 4p"], "5p")
    g.last_draw = 27
    acts = g.legal_actions(0)
    assert any(a.type == ActionType.TSUMO for a in acts)
    g.step(0, next(a for a in acts if a.type == ActionType.TSUMO))
    assert g.finished and g.result.winner == 0 and g.result.is_tsumo
    assert g.result.scores == [3, -1, -1, -1]


def test_ron_scores_and_priority_over_pon():
    g = Game()
    _rig(
        g,
        [
            "123m 456m 789m 123s 东东 1p",
            "123m 456m 789m 23s 白白",          # 听 1s/4s，对 1p 无反应
            "111m 222m 333m 44m 11p",          # 1p 可以碰
            "123m 456m 789m 1p 东东东",          # 1p 可以荣胡（单钓）
        ],
        "5p 5p 5p",
    )
    g.step(0, Action(ActionType.DISCARD, 18))
    assert sorted(g.players_to_act()) == [2, 3]
    g.step(2, Action(ActionType.PON, 18))
    g.step(3, Action(ActionType.RON, 18))
    assert g.finished and g.result.winner == 3 and g.result.loser == 0
    assert g.result.scores == [-1, 0, 0, 1]


def test_ankan_then_replacement_draw():
    g = Game()
    _rig(g, ["1111m 456m 789m 12s 东东", "1m 2m 3m 4m 5m 6m 7m 8m 9m 1s 2s 3s 4s", "东 南 西 北 中 发 白 东 南 西 北 中 发", "1p 2p 3p 4p 5p 6p 7p 8p 9p 1p 2p 3p 4p"], "9p 5p")
    g.step(0, Action(ActionType.ANKAN, 0))
    assert g.hands[0].melds[0].type == MeldType.ANKAN
    assert g.hands[0].size == 11  # 14 - 4 + 1
    # 补牌从牌墙另一头（这里是 5p）
    assert g.hands[0].counts[22] == 1
    assert g.current == 0 and g.phase == Phase.DISCARD


def test_draw_game_when_wall_empty():
    g = Game()
    _rig(g, ["147m 258s 369p 东南西北 中", "1m 2m 3m 4m 5m 6m 7m 8m 9m 1s 2s 3s 4s", "东 南 西 北 中 发 白 东 南 西 北 中 发", "1p 2p 3p 4p 5p 6p 7p 8p 9p 1p 2p 3p 4p"], "")
    g.step(0, Action(ActionType.DISCARD, 0))
    assert g.finished and g.result.winner is None
