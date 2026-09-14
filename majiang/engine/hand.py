"""一家的手牌 + 副露。"""

from dataclasses import dataclass, field
from enum import Enum, auto

from .tile import NUM_TILE_TYPES, hand_to_str, tiles_from_counts


class MeldType(Enum):
    PON = auto()
    KAN = auto()     # 明杠
    ANKAN = auto()   # 暗杠
    ADDKAN = auto()  # 加杠


@dataclass(frozen=True)
class Meld:
    type: MeldType
    tile: int
    from_player: int = -1  # 明杠/碰的牌来自谁；暗杠为 -1

    def __repr__(self) -> str:
        from .tile import to_str

        return f"{self.type.name}[{to_str(self.tile)}]"


@dataclass
class Hand:
    counts: list[int] = field(default_factory=lambda: [0] * NUM_TILE_TYPES)
    melds: list[Meld] = field(default_factory=list)

    def add(self, t: int) -> None:
        self.counts[t] += 1

    def remove(self, t: int) -> None:
        if self.counts[t] <= 0:
            raise ValueError(f"手里没有 {t}")
        self.counts[t] -= 1

    def has(self, t: int, n: int = 1) -> bool:
        return self.counts[t] >= n

    @property
    def size(self) -> int:
        return sum(self.counts)

    def tiles(self) -> list[int]:
        return tiles_from_counts(self.counts)

    def __str__(self) -> str:
        s = hand_to_str(self.tiles())
        if self.melds:
            s += "  副露: " + " ".join(repr(m) for m in self.melds)
        return s
