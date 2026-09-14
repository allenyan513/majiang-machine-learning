from majiang.riichi import tile as T


def test_roundtrip_str():
    for t in range(34):
        assert T.from_str(T.to_str(t)) == t


def test_parse_hand_mixed():
    tiles = T.parse_hand("123m 55s 东东 白")
    assert sorted(tiles) == [0, 1, 2, 13, 13, 27, 27, 31]
    assert T.hand_to_str(tiles) == "123m 55s 东东白"


def test_suit_number_honor():
    assert T.suit(0) == 0 and T.number(0) == 1
    assert T.suit(17) == 1 and T.number(17) == 9
    assert T.suit(18) == 2 and T.number(18) == 1
    assert T.is_honor(27) and T.suit(27) == 3 and T.number(27) == 0
    assert T.is_terminal(8) and T.is_terminal(18) and not T.is_terminal(4)
    assert T.is_yaochu(T.WHITE) and T.is_simple(4) and not T.is_simple(T.EAST)


def test_dora_from_indicator():
    p = T.parse_hand
    assert T.dora_from_indicator(p("3m")[0]) == p("4m")[0]
    assert T.dora_from_indicator(p("9s")[0]) == p("1s")[0]
    assert T.dora_from_indicator(T.NORTH) == T.EAST
    assert T.dora_from_indicator(T.EAST) == T.SOUTH
    assert T.dora_from_indicator(T.RED) == T.WHITE
    assert T.dora_from_indicator(T.WHITE) == T.GREEN


def test_full_wall():
    wall = T.full_wall()
    assert len(wall) == 136
    assert all(T.counts_from_tiles(wall)[t] == 4 for t in range(34))
