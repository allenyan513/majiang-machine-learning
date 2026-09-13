"""一局麻将的状态机。

用法：
    g = Game(seed=0)
    g.start()
    while not g.finished:
        for p in g.players_to_act():
            obs = g.observe(p)
            action = agents[p].act(obs)
            g.step(p, action)
    print(g.result)

阶段 (Phase)：
    DISCARD  当前玩家手里 14 张（或 3k+2），要打牌 / 自摸 / 暗杠 / 加杠
    RESPOND  有人打出一张牌，其他玩家决定 碰 / 明杠 / 点炮胡 / 过
    FINISHED 胡牌或流局

简化：
    - 没有吃、没有花牌、没有抢杠、没有一炮多响（多家能胡时离放炮者最近的胡）
    - 没有王牌/岭上牌区，杠后从牌墙末尾补牌，牌墙摸完即流局
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto

from .actions import PASS, Action, ActionType, PRIORITY
from .hand import Hand, Meld, MeldType
from .rules import HAND_SIZE, NUM_PLAYERS, RON_POINTS, TSUMO_POINTS_EACH
from .tile import full_wall
from .win import is_win


class Phase(Enum):
    DISCARD = auto()
    RESPOND = auto()
    FINISHED = auto()


@dataclass
class Result:
    winner: int | None            # None = 流局
    loser: int | None = None      # 点炮者；自摸时为 None
    win_tile: int = -1
    is_tsumo: bool = False
    scores: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)

    def __str__(self) -> str:
        from .tile import to_str

        if self.winner is None:
            return "流局"
        how = "自摸" if self.is_tsumo else f"点炮(玩家{self.loser})"
        return f"玩家{self.winner} 胡 {to_str(self.win_tile)} {how}  分数 {self.scores}"


@dataclass
class Observation:
    """某个玩家在决策时能看到的全部信息。AI 只能基于它做决定。"""

    player: int
    phase: Phase
    hand: list[int]                  # 自己的手牌计数 (34)
    melds: list[Meld]                # 自己的副露
    rivers: list[list[int]]          # 四家牌河（打出的牌，被碰/杠走的已移除）
    all_melds: list[list[Meld]]      # 四家副露
    wall_remaining: int
    current_player: int              # 轮到谁（DISCARD 阶段=打牌者，RESPOND 阶段=放炮者）
    last_discard: int                # RESPOND 阶段：刚打出的牌；否则 -1
    last_draw: int                   # DISCARD 阶段：刚摸到的牌（碰后打牌时为 -1）
    legal_actions: list[Action]


class Game:
    def __init__(self, seed: int | None = None, dealer: int = 0):
        self.rng = random.Random(seed)
        self.dealer = dealer
        self.wall: list[int] = []
        self.hands: list[Hand] = [Hand() for _ in range(NUM_PLAYERS)]
        self.rivers: list[list[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.phase = Phase.FINISHED
        self.current = dealer
        self.last_discard = -1
        self.last_draw = -1
        self.pending: dict[int, Action | None] = {}  # RESPOND 阶段各家的回应
        self.result: Result | None = None
        self.log: list[tuple[int, Action]] = []      # (player, action) 完整动作序列

    # ------------------------------------------------------------ 开局
    def start(self) -> None:
        self.wall = full_wall()
        self.rng.shuffle(self.wall)
        for p in range(NUM_PLAYERS):
            for _ in range(HAND_SIZE):
                self.hands[p].add(self.wall.pop())
        self.current = self.dealer
        self._draw(self.dealer)

    @property
    def finished(self) -> bool:
        return self.phase == Phase.FINISHED

    # ------------------------------------------------------------ 查询接口
    def players_to_act(self) -> list[int]:
        if self.phase == Phase.DISCARD:
            return [self.current]
        if self.phase == Phase.RESPOND:
            return [p for p, a in self.pending.items() if a is None]
        return []

    def legal_actions(self, player: int) -> list[Action]:
        if self.phase == Phase.DISCARD:
            return self._legal_discard_phase(player)
        if self.phase == Phase.RESPOND:
            return self._legal_respond_phase(player)
        return []

    def observe(self, player: int) -> Observation:
        return Observation(
            player=player,
            phase=self.phase,
            hand=list(self.hands[player].counts),
            melds=list(self.hands[player].melds),
            rivers=[list(r) for r in self.rivers],
            all_melds=[list(h.melds) for h in self.hands],
            wall_remaining=len(self.wall),
            current_player=self.current,
            last_discard=self.last_discard if self.phase == Phase.RESPOND else -1,
            last_draw=self.last_draw if self.phase == Phase.DISCARD else -1,
            legal_actions=self.legal_actions(player),
        )

    # ------------------------------------------------------------ 合法动作
    def _legal_discard_phase(self, player: int) -> list[Action]:
        if player != self.current:
            return []
        hand = self.hands[player]
        acts: list[Action] = []
        if is_win(hand.counts):
            acts.append(Action(ActionType.TSUMO, self.last_draw))
        if self.wall:  # 杠需要补牌
            for t in range(34):
                if hand.counts[t] == 4:
                    acts.append(Action(ActionType.ANKAN, t))
                elif hand.counts[t] == 1 and any(
                    m.type == MeldType.PON and m.tile == t for m in hand.melds
                ):
                    acts.append(Action(ActionType.ADDKAN, t))
        acts.extend(Action(ActionType.DISCARD, t) for t in range(34) if hand.counts[t] > 0)
        return acts

    def _legal_respond_phase(self, player: int) -> list[Action]:
        if player not in self.pending or self.pending[player] is not None:
            return []
        return self._responses_for(player, self.last_discard)

    def _responses_for(self, player: int, t: int) -> list[Action]:
        """player 对刚打出的 t 可以做什么（不含 PASS 时返回空）。"""
        hand = self.hands[player]
        acts: list[Action] = []
        hand.counts[t] += 1
        if is_win(hand.counts):
            acts.append(Action(ActionType.RON, t))
        hand.counts[t] -= 1
        if hand.counts[t] >= 3 and self.wall:
            acts.append(Action(ActionType.KAN, t))
        if hand.counts[t] >= 2:
            acts.append(Action(ActionType.PON, t))
        if acts:
            acts.append(PASS)
        return acts

    # ------------------------------------------------------------ 推进
    def step(self, player: int, action: Action) -> None:
        if action not in self.legal_actions(player):
            raise ValueError(f"玩家{player} 非法动作 {action}（阶段 {self.phase.name}）")
        self.log.append((player, action))

        if self.phase == Phase.DISCARD:
            self._step_discard_phase(player, action)
        else:
            self._step_respond_phase(player, action)

    def _step_discard_phase(self, player: int, action: Action) -> None:
        hand = self.hands[player]
        t = action.tile
        if action.type == ActionType.TSUMO:
            self._finish_win(player, t, loser=None)
        elif action.type == ActionType.DISCARD:
            hand.remove(t)
            self.rivers[player].append(t)
            self.last_discard = t
            self.last_draw = -1
            self._open_respond_phase(player, t)
        elif action.type == ActionType.ANKAN:
            for _ in range(4):
                hand.remove(t)
            hand.melds.append(Meld(MeldType.ANKAN, t))
            self._draw(player, replacement=True)
        elif action.type == ActionType.ADDKAN:
            hand.remove(t)
            idx = next(i for i, m in enumerate(hand.melds) if m.type == MeldType.PON and m.tile == t)
            hand.melds[idx] = Meld(MeldType.ADDKAN, t, hand.melds[idx].from_player)
            self._draw(player, replacement=True)

    def _open_respond_phase(self, discarder: int, t: int) -> None:
        self.pending = {}
        for p in range(NUM_PLAYERS):
            if p != discarder and self._responses_for(p, t):
                self.pending[p] = None
        if self.pending:
            self.phase = Phase.RESPOND
        else:
            self._next_turn()

    def _step_respond_phase(self, player: int, action: Action) -> None:
        self.pending[player] = action
        if any(a is None for a in self.pending.values()):
            return  # 还有人没回应
        self._resolve_responses()

    def _resolve_responses(self) -> None:
        discarder = self.current
        t = self.last_discard
        # 按优先级、同优先级按离放炮者的顺序（下家最先）
        best: tuple[int, int, Action] | None = None
        for p, a in self.pending.items():
            assert a is not None
            key = (PRIORITY[a.type], -((p - discarder) % NUM_PLAYERS))
            if best is None or key > (best[0], best[1]):
                best = (key[0], key[1], a)
                best_player = p
        assert best is not None
        action = best[2]
        self.pending = {}

        if action.type == ActionType.PASS:
            self._next_turn()
            return

        hand = self.hands[best_player]
        self.rivers[discarder].pop()  # 牌被拿走
        if action.type == ActionType.RON:
            hand.add(t)
            self._finish_win(best_player, t, loser=discarder)
        elif action.type == ActionType.PON:
            hand.remove(t)
            hand.remove(t)
            hand.melds.append(Meld(MeldType.PON, t, discarder))
            self.current = best_player
            self.last_draw = -1
            self.phase = Phase.DISCARD
        elif action.type == ActionType.KAN:
            for _ in range(3):
                hand.remove(t)
            hand.melds.append(Meld(MeldType.KAN, t, discarder))
            self.current = best_player
            self._draw(best_player, replacement=True)

    def _next_turn(self) -> None:
        self.current = (self.current + 1) % NUM_PLAYERS
        self._draw(self.current)

    def _draw(self, player: int, replacement: bool = False) -> None:
        if not self.wall:
            self._finish_draw()
            return
        t = self.wall.pop(0) if replacement else self.wall.pop()
        self.hands[player].add(t)
        self.current = player
        self.last_draw = t
        self.phase = Phase.DISCARD

    # ------------------------------------------------------------ 结算
    def _finish_win(self, winner: int, tile: int, loser: int | None) -> None:
        scores = [0] * NUM_PLAYERS
        if loser is None:
            for p in range(NUM_PLAYERS):
                if p != winner:
                    scores[p] -= TSUMO_POINTS_EACH
                    scores[winner] += TSUMO_POINTS_EACH
        else:
            scores[loser] -= RON_POINTS
            scores[winner] += RON_POINTS
        self.result = Result(winner, loser, tile, loser is None, scores)
        self.phase = Phase.FINISHED

    def _finish_draw(self) -> None:
        self.result = Result(None)
        self.phase = Phase.FINISHED
