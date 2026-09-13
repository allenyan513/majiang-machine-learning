from majiang.agents.random_agent import RandomAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import ActionType, discard, riichi
from majiang.engine.game import Game, Phase
from majiang.engine.tile import parse_hand
from majiang.evaluate import evaluate
from tests.test_game import FILLER, _rig


def test_discards_isolated_honor_first():
    g = Game()
    _rig(g, ["123m 456m 78m 23s 55s 东 白"] + FILLER[1:], "9p 9p 9p 9p")
    a = RuleAgent().act(g.observe(0))
    assert a.type == ActionType.DISCARD and a.tile in parse_hand("东 白")


def test_riichi_when_tenpai_picks_more_waits():
    g = Game()
    # 打白 -> 听 1s4s（两面）；打 2s -> 不听
    _rig(g, ["123m 456m 789m 23s 东东 白"] + FILLER[1:], "9p 9p 9p 9p")
    a = RuleAgent().act(g.observe(0))
    assert a == riichi(parse_hand("白")[0])


def test_never_calls_and_passes():
    g = Game()
    _rig(g, [FILLER[0].replace("东南西北", "东南西 4s"), "56s 78s 123m 456m 白发中 东"] + FILLER[2:], "9p 9p 9p 9p")
    g.step(0, discard(parse_hand("4s")[0]))
    assert g.phase == Phase.RESPOND
    a = RuleAgent().act(g.observe(1))
    assert a.type == ActionType.PASS


def test_rule_agent_beats_random_and_never_illegal():
    rep = evaluate(["rule", "random", "random", "random"], n=40, base_seed=123)
    s = rep["stats"]
    assert s["rule"]["wins"] / s["rule"]["games"] > 0.4
