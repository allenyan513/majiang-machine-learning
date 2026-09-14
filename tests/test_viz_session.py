from web.server import Session


def test_god_view_and_stepping():
    s = Session(["rule", "rule", "rule", "rule"], model_path=None)
    s.new(seed=1)
    st = s.state()
    assert all(len(p["hand"]) == p["hand_size"] for p in st["players"])
    assert st["analysis"]["player"] == st["to_act"][0] and "rule" in st["analysis"]
    while s.step():
        pass
    assert s.state()["finished"]


def test_player_view_hides_others_and_waits_for_human():
    s = Session(["rule", "rule", "rule", "rule"], model_path=None)
    s.new(seed=3, human=2)
    for _ in range(200):
        if not s.step():
            break
    st = s.state(view=2)
    assert st["waiting_human"] and st["to_act"] == [2]
    assert len(st["players"][2]["hand"]) > 0
    assert all(st["players"][p]["hand"] == [] and st["players"][p]["hand_size"] > 0 for p in (0, 1, 3))
    assert st["analysis"]["player"] == 2 and st["analysis"]["rule_pick"] >= 0
    # 人打出规则建议的那张牌
    s.act(2, "DISCARD", st["analysis"]["rule_pick"])
    assert not s.waiting_human() or s.game.players_to_act() != [2] or s.game.phase.name != "DISCARD"
    # 非法动作被忽略（不是人的座位）
    before = len(s.game.log)
    s.act(0, "DISCARD", 0)
    assert len(s.game.log) == before
