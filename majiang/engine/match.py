"""一场比赛：把单局 (Game) 串起来。

    m = Match(seed=0, length="hanchan")   # 或 "tonpuu"（东风战）
    while not m.ended:
        g = m.new_hand()
        ... 打完 ...
        m.finish_hand(g.result)
    print(m.ranking())

规则：
    起始 25000；连庄（庄家和牌 / 流局时庄家听牌）本场 +1，否则庄家轮到下家、本场清零（流局本场也 +1）
    东风战 4 局、半庄 8 局；南 4 局（东风战为东 4）庄家第一名时可结束（和了止め），否则连庄继续
    有人跌破 0 立即结束（击飞）
    终局顺位 uma +20/+10/-10/-20（千点），返点 30000（oka 归第一名）
"""

from dataclasses import dataclass, field

from .game import Game, Result
from .rules import NUM_PLAYERS, START_POINTS
from .tile import EAST, SOUTH, WINDS

UMA = [20000, 10000, -10000, -20000]
OKA_RETURN = 30000


@dataclass
class Match:
    seed: int = 0
    length: str = "hanchan"           # "hanchan" 半庄 / "tonpuu" 东风战
    scores: list[int] = field(default_factory=lambda: [START_POINTS] * NUM_PLAYERS)
    dealer: int = 0
    round_wind: int = EAST
    hand_no: int = 1                  # 东 1 局 = 1
    honba: int = 0
    riichi_sticks: int = 0
    hands_played: int = 0
    ended: bool = False
    end_reason: str = ""
    history: list[str] = field(default_factory=list)

    @property
    def max_rounds(self) -> int:
        return 2 if self.length == "hanchan" else 1

    @property
    def round_label(self) -> str:
        return f"{'东南西北'[WINDS.index(self.round_wind)]}{self.hand_no}局"

    def new_hand(self) -> Game:
        assert not self.ended
        g = Game(
            seed=self.seed * 1000 + self.hands_played,
            dealer=self.dealer,
            round_wind=self.round_wind,
            scores=self.scores,
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
        )
        g.start()
        return g

    def finish_hand(self, result: Result) -> None:
        self.hands_played += 1
        self.history.append(f"{self.round_label} {self.honba}本场: {result}")
        self.scores = [self.scores[p] + result.deltas[p] for p in range(NUM_PLAYERS)]
        # 立直棒：和了时被拿走（deltas 里已含），流局时留在桌上（本局押的 + 之前留下的）
        self.riichi_sticks = 0 if result.kind == "win" else self._sticks_on_table(result)
        if any(s < 0 for s in self.scores):
            self._end("击飞")
            return
        last_hand = self.round_wind == WINDS[self.max_rounds - 1] and self.hand_no == 4
        if result.dealer_continues:
            self.honba += 1
            if last_hand and self._dealer_is_top():
                self._end("和了止め")
            return
        # 轮庄
        self.honba = self.honba + 1 if result.kind != "win" else 0
        if last_hand:
            self._end("终局")
            return
        self.dealer = (self.dealer + 1) % NUM_PLAYERS
        if self.hand_no == 4:
            self.hand_no = 1
            self.round_wind = SOUTH if self.round_wind == EAST else WINDS[WINDS.index(self.round_wind) + 1]
        else:
            self.hand_no += 1

    def _sticks_on_table(self, result: Result) -> int:
        # 流局：本局新押的立直棒 + 之前留下的。deltas 里负 1000 的整数倍是押金；这里用总变动推算
        return -sum(result.deltas) // 1000 + self.riichi_sticks

    def _dealer_is_top(self) -> bool:
        return self.scores[self.dealer] == max(self.scores) and self.scores.count(max(self.scores)) == 1

    def _end(self, reason: str) -> None:
        self.ended = True
        self.end_reason = reason

    def ranking(self) -> list[dict]:
        """终局顺位（含 uma/oka）。同分按座位靠近起家的优先。"""
        order = sorted(range(NUM_PLAYERS), key=lambda p: (-self.scores[p], p))
        out = []
        for rank, p in enumerate(order):
            final = self.scores[p] - OKA_RETURN + UMA[rank] + (self.riichi_sticks * 1000 if rank == 0 else 0)
            if rank == 0:
                final += (OKA_RETURN - START_POINTS) * NUM_PLAYERS
            out.append({"player": p, "rank": rank + 1, "points": self.scores[p], "final": final / 1000})
        return out
