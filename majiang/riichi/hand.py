"""一家的手牌 + 副露。"""

from dataclasses import dataclass, field
from enum import Enum, auto

from .tile import NUM_TILE_TYPES, RED_FIVES, hand_to_str, tiles_from_counts


class MeldType(Enum):
    CHI = auto()     # 吃：tile = 顺子最小的那张
    PON = auto()
    KAN = auto()     # 大明杠
    ANKAN = auto()   # 暗杠
    ADDKAN = auto()  # 加杠


@dataclass(frozen=True)
class Meld:
    type: MeldType
    tile: int                # 碰/杠：那张牌；吃：顺子的起始牌
    from_player: int = -1    # 牌来自谁；暗杠为 -1
    called: int = -1         # 吃/碰/明杠时拿的那张牌（用于显示横放）；暗杠为 -1
    reds: int = 0            # 这组里有几张赤牌（算宝牌用）

    @property
    def is_open(self) -> bool:
        """是否破坏门清。暗杠不算。"""
        return self.type != MeldType.ANKAN

    @property
    def tiles(self) -> list[int]:
        if self.type == MeldType.CHI:
            return [self.tile, self.tile + 1, self.tile + 2]
        return [self.tile] * (3 if self.type == MeldType.PON else 4)

    def __repr__(self) -> str:
        from .tile import to_str

        return f"{self.type.name}[{''.join(to_str(t) for t in self.tiles)}]"


def meld_tiles(m: Meld) -> list[int]:
    return m.tiles


@dataclass
class Hand:
    counts: list[int] = field(default_factory=lambda: [0] * NUM_TILE_TYPES)
    melds: list[Meld] = field(default_factory=list)
    reds: list[bool] = field(default_factory=lambda: [False] * NUM_TILE_TYPES)  # 手里是否持有该种牌的赤牌

    def add(self, t: int, red: bool = False) -> None:
        self.counts[t] += 1
        if red:
            self.reds[t] = True

    def remove(self, t: int, red: bool | None = None) -> bool:
        """移走一张 t。red=None 时优先移普通牌、留赤牌；返回移走的是否是赤牌。"""
        if self.counts[t] <= 0:
            raise ValueError(f"手里没有 {t}")
        if red is None:
            red = self.reds[t] and self.counts[t] == 1
        if red and not self.reds[t]:
            raise ValueError(f"手里没有赤 {t}")
        self.counts[t] -= 1
        if red:
            self.reds[t] = False
        return red

    def has(self, t: int, n: int = 1) -> bool:
        return self.counts[t] >= n

    @property
    def size(self) -> int:
        return sum(self.counts)

    @property
    def is_menzen(self) -> bool:
        return all(not m.is_open for m in self.melds)

    @property
    def red_count(self) -> int:
        return sum(1 for t in RED_FIVES if self.reds[t])

    def tiles(self) -> list[int]:
        return tiles_from_counts(self.counts)

    def __str__(self) -> str:
        s = hand_to_str(self.tiles())
        if self.melds:
            s += "  副露: " + " ".join(repr(m) for m in self.melds)
        return s
