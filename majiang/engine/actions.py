"""玩家动作。

所有动作都是不可变的小对象，方便放进集合、做比较、以后编码成神经网络输出。
"""

from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    DISCARD = auto()   # 打出一张牌
    PON = auto()       # 碰（别人打出的牌）
    KAN = auto()       # 明杠（别人打出的牌，手里有 3 张）
    ANKAN = auto()     # 暗杠（自己摸到第 4 张）
    ADDKAN = auto()    # 加杠（碰过的刻子摸到第 4 张）
    RON = auto()       # 点炮胡（胡别人打出的牌）
    TSUMO = auto()     # 自摸胡
    PASS = auto()      # 放弃响应


@dataclass(frozen=True)
class Action:
    type: ActionType
    tile: int = -1  # 打出/碰/杠/胡的那张牌；PASS 时为 -1

    def __repr__(self) -> str:
        from .tile import to_str

        if self.tile < 0:
            return self.type.name
        return f"{self.type.name}({to_str(self.tile)})"


def discard(t: int) -> Action:
    return Action(ActionType.DISCARD, t)


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
    ActionType.PASS: 0,
}
