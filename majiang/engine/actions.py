"""玩家动作。

所有动作都是不可变的小对象，方便放进集合、做比较、以后编码成神经网络输出。
"""

from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    DISCARD = auto()   # 打出一张牌
    RIICHI = auto()    # 立直宣言并打出 tile
    CHI = auto()       # 吃上家打的 tile，组成 extra 起始的顺子
    PON = auto()
    KAN = auto()       # 大明杠
    ANKAN = auto()     # 暗杠
    ADDKAN = auto()    # 加杠
    RON = auto()       # 荣和（含抢杠）
    TSUMO = auto()     # 自摸
    PASS = auto()      # 放弃响应


@dataclass(frozen=True)
class Action:
    type: ActionType
    tile: int = -1   # 打出 / 吃碰杠的那张 / 和的那张；PASS 为 -1
    extra: int = -1  # CHI：顺子起始牌

    def __repr__(self) -> str:
        from .tile import to_str

        if self.tile < 0:
            return self.type.name
        if self.type == ActionType.CHI:
            return f"CHI({to_str(self.tile)} in {''.join(to_str(self.extra + i) for i in range(3))})"
        return f"{self.type.name}({to_str(self.tile)})"


def discard(t: int) -> Action:
    return Action(ActionType.DISCARD, t)


def riichi(t: int) -> Action:
    return Action(ActionType.RIICHI, t)


def chi(t: int, start: int) -> Action:
    return Action(ActionType.CHI, t, start)


def pon(t: int) -> Action:
    return Action(ActionType.PON, t)


def kan(t: int) -> Action:
    return Action(ActionType.KAN, t)


def ankan(t: int) -> Action:
    return Action(ActionType.ANKAN, t)


def addkan(t: int) -> Action:
    return Action(ActionType.ADDKAN, t)


def ron(t: int) -> Action:
    return Action(ActionType.RON, t)


def tsumo(t: int) -> Action:
    return Action(ActionType.TSUMO, t)


PASS = Action(ActionType.PASS)

# 响应阶段的优先级：数字越大越优先
PRIORITY = {
    ActionType.RON: 3,
    ActionType.KAN: 2,
    ActionType.PON: 2,
    ActionType.CHI: 1,
    ActionType.PASS: 0,
}
