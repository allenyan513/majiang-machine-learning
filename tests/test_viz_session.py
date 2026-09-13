from majiang.engine.actions import ActionType
from web.server import Session


def test_god_view_and_stepping_through_a_match():
    s = Session(["rule", "rule", "rule", "rule"], model_path=None)
    s.new(seed=1, length="tonpuu")
    st = s.state()
    assert all(len(p["hand"]) == p["hand_size"] for p in st["players"])
    assert st["analysis"]["player"] == st["to_act"][0] and "rule" in st["analysis"]
    assert st["round_label"] == "东1局" and len(st["dora_indicators"]) == 1
    hands = 0
    while not s.match.ended and hands < 20:
        while s.step():
            pass
        assert s.state()["finished"] and s.state()["result"] is not None
        s.next_hand()
        hands += 1
    assert s.match.ended and s.state()["match"]["ranking"] is not None


def test_player_view_hides_others_and_human_acts():
    s = Session(["rule", "rule", "rule", "rule"], model_path=None)
    s.new(seed=3, human=2)
    for _ in range(200):
        if not s.step():
            if s.state()["phase"] == "RESPOND":
                s.act(2, "PASS", -1)  # 人在响应阶段：过，直到轮到打牌
                continue
            break
    st = s.state(view=2)
    assert st["waiting_human"] and st["to_act"] == [2] and st["phase"] == "DISCARD"
    assert len(st["players"][2]["hand"]) > 0
    assert all(st["players"][p]["hand"] == [] and st["players"][p]["hand_size"] > 0 for p in (0, 1, 3))
    a = st["analysis"]
    assert a["player"] == 2 and a["rule_pick"]["type"] in ("DISCARD", "RIICHI")
    s.act(2, a["rule_pick"]["type"], a["rule_pick"]["tile"])
    assert s.game.log[-1][0] == 2
    # 非法动作被忽略（不是人的座位）
    before = len(s.game.log)
    s.act(0, "DISCARD", 0)
    assert len(s.game.log) == before
