from majiang.agents.random_agent import RandomAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import ActionType
from majiang.engine.game import Game, Observation, Phase
from majiang.engine.hand import Hand
from majiang.engine.tile import counts_from_tiles, parse_hand
from majiang.evaluate import evaluate


def _obs(hand: str, legal, melds=()):
    return Observation(
        player=0, phase=Phase.DISCARD, hand=counts_from_tiles(parse_hand(hand)), melds=list(melds),
        rivers=[[], [], [], []], all_melds=[list(melds), [], [], []], wall_remaining=50,
        current_player=0, last_discard=-1, last_draw=-1, legal_actions=legal,
    )


def test_discards_isolated_honor_first():
    from majiang.engine.actions import discard

    hand = "123m 456m 789m 23s 东东 白"
    legal = [discard(t) for t in set(parse_hand(hand))]
    a = RuleAgent().act(_obs(hand, legal))
    assert a.type == ActionType.DISCARD and a.tile == parse_hand("白")[0]


def test_pon_only_when_it_helps():
    from majiang.engine.actions import PASS, pon

    agent = RuleAgent()
    # 东东 是唯一的对子（雀头），碰了反而没雀头 -> 向听不减 -> 过
    obs = _obs("123m 456m 789m 23s 东东", [pon(27), PASS])
    obs.phase = Phase.RESPOND
    assert agent.act(obs) == PASS
    # 有额外对子 55s，碰东能进一步 -> 碰
    obs = _obs("123m 456m 79m 55s 东东", [pon(27), PASS])
    obs.phase = Phase.RESPOND
    assert agent.act(obs) == pon(27)


def test_rule_agent_beats_random_and_never_illegal():
    rep = evaluate(["rule", "random", "random", "random"], n=60, base_seed=123)
    s = rep["stats"]
    assert s["rule"]["wins"] / s["rule"]["games"] > 0.5
    assert rep["draws"] / rep["n"] < 0.5
