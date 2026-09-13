from majiang.agents.rule_agent import RuleAgent
from majiang.engine.match import Match
from majiang.engine.tile import EAST, SOUTH


def _play(m: Match) -> None:
    agents = [RuleAgent() for _ in range(4)]
    while not m.ended:
        g = m.new_hand()
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        m.finish_hand(g.result)


def test_tonpuu_ends_and_points_conserve():
    m = Match(seed=1, length="tonpuu")
    _play(m)
    assert m.ended and m.hands_played >= 4 or m.end_reason == "击飞"
    assert sum(m.scores) + m.riichi_sticks * 1000 == 100000
    ranks = m.ranking()
    assert [r["rank"] for r in ranks] == [1, 2, 3, 4]
    assert abs(sum(r["final"] for r in ranks)) < 1e-9  # uma/oka 零和


def test_hanchan_reaches_south_round():
    m = Match(seed=3, length="hanchan")
    seen_south = False
    agents = [RuleAgent() for _ in range(4)]
    while not m.ended:
        seen_south = seen_south or m.round_wind == SOUTH
        g = m.new_hand()
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        m.finish_hand(g.result)
    assert seen_south or m.end_reason == "击飞"


def test_dealer_rotation_and_honba():
    from majiang.engine.game import Result

    m = Match(seed=0, length="hanchan")
    m.finish_hand(Result("win", [], [1000, -1000, 0, 0], dealer_continues=True))
    assert m.dealer == 0 and m.honba == 1 and m.hand_no == 1
    m.finish_hand(Result("win", [], [-1000, 1000, 0, 0], dealer_continues=False))
    assert m.dealer == 1 and m.honba == 0 and m.hand_no == 2
    m.finish_hand(Result("draw", [], [0, 0, 0, 0], [False] * 4, dealer_continues=False))
    assert m.dealer == 2 and m.honba == 1 and m.hand_no == 3
    for _ in range(2):
        m.finish_hand(Result("win", [], [0, 0, 0, 0], dealer_continues=False))
    assert m.round_wind == SOUTH and m.hand_no == 1 and m.dealer == 0
