"""一局（一巡）日本立直麻将的状态机。

用法：
    g = Game(seed=0, dealer=0, round_wind=EAST, scores=[25000]*4)
    g.start()
    while not g.finished:
        for p in g.players_to_act():
            g.step(p, agents[p].act(g.observe(p)))
    print(g.result)

阶段 (Phase)：
    DISCARD  当前玩家要 打牌 / 立直 / 自摸 / 暗杠 / 加杠
    RESPOND  有人打出一张牌（或加杠），其他玩家决定 荣和 / 碰 / 杠 / 吃 / 过
    FINISHED 和牌或流局

实现的规则（天凤四人）：
    王牌 14 张、宝牌/里宝牌/杠宝牌、赤宝牌 3 张、岭上牌
    吃（限上家）、碰、大明杠、暗杠、加杠、抢杠
    立直（门清、牌墙 >= 4、有 1000 点）、一发、两立直、立直后摸切、立直后暗杠须不改听
    振听三种：舍牌、同巡、立直
    和牌必须有役（yaku.evaluate 不为 None）
    双响允许，三响流局；天和 / 地和；海底 / 河底
    荒牌流局：听牌费 3000；食替禁止（现物 + 筋）
未实现（Tier 2）：九种九牌、四风连打、四家立直、四杠散了、流局满贯、包牌、国士抢暗杠
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto

from .actions import PASS, Action, ActionType, PRIORITY
from .hand import Hand, Meld, MeldType
from .rules import (
    DEAD_WALL,
    HAND_SIZE,
    MAX_KANS,
    MIN_WALL_FOR_RIICHI,
    NOTEN_PENALTY,
    NUM_PLAYERS,
    RIICHI_BET,
    START_POINTS,
)
from .tile import EAST, RED_FIVES, WINDS, full_wall, is_honor, number, suit
from .win import is_win, waiting_tiles
from .yaku import WinContext, WinResult, evaluate


class Phase(Enum):
    DISCARD = auto()
    RESPOND = auto()
    FINISHED = auto()


@dataclass
class Discard:
    tile: int
    red: bool = False
    riichi: bool = False      # 立直宣言牌（横放）
    tsumogiri: bool = False   # 摸切
    claimed: bool = False     # 被吃/碰/杠走


@dataclass
class Win:
    player: int
    result: WinResult
    from_player: int | None   # 点炮者；自摸为 None
    tile: int


@dataclass
class Result:
    kind: str                                  # "win" / "draw" / "abort"
    wins: list[Win] = field(default_factory=list)
    deltas: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)
    tenpai: list[bool] = field(default_factory=lambda: [False] * NUM_PLAYERS)  # 流局时
    dealer_continues: bool = False
    reason: str = ""

    def __str__(self) -> str:
        from .tile import to_str

        if self.kind == "win":
            parts = []
            for w in self.wins:
                how = "自摸" if w.from_player is None else f"荣和(玩家{w.from_player})"
                parts.append(f"玩家{w.player} {how} {to_str(w.tile)} {w.result}")
            return "；".join(parts) + f"  点数变动 {self.deltas}"
        if self.kind == "draw":
            return f"荒牌流局 听牌 {[i for i, t in enumerate(self.tenpai) if t]} 点数变动 {self.deltas}"
        return f"途中流局（{self.reason}）"


@dataclass
class Observation:
    """某个玩家在决策时能看到的全部信息。AI 只能基于它做决定。"""

    player: int
    phase: Phase
    hand: list[int]                    # 自己的手牌计数 (34)
    reds: list[bool]                   # 自己手里的赤牌
    melds: list[Meld]
    rivers: list[list[int]]            # 四家牌河（未被拿走的牌，只有 tile id；ML 用）
    rivers_full: list[list[Discard]]   # 四家牌河完整信息（含被拿走的、立直牌、摸切）
    all_melds: list[list[Meld]]
    wall_remaining: int
    current_player: int
    last_discard: int                  # RESPOND 阶段：刚打出的牌；否则 -1
    last_draw: int                     # DISCARD 阶段：刚摸到的牌（吃碰后为 -1）
    legal_actions: list[Action]
    dora_indicators: list[int]
    riichi: list[bool]
    scores: list[int]
    honba: int
    riichi_sticks: int
    dealer: int
    round_wind: int
    seat_wind: int
    furiten: bool
    waits: list[int]                   # 自己当前听的牌（3k+1 张时；打牌前为空）
    chankan: bool = False              # RESPOND 阶段：是否在响应加杠


class Game:
    def __init__(
        self,
        seed: int | None = None,
        dealer: int = 0,
        round_wind: int = EAST,
        scores: list[int] | None = None,
        honba: int = 0,
        riichi_sticks: int = 0,
    ):
        self.rng = random.Random(seed)
        self.dealer = dealer
        self.round_wind = round_wind
        self.scores = list(scores) if scores else [START_POINTS] * NUM_PLAYERS
        self.start_scores = list(self.scores)
        self.honba = honba
        self.riichi_sticks = riichi_sticks

        self.wall: list[int] = []          # 活牌墙，pop(0) 摸牌
        self.wall_red: list[bool] = []
        self.dead: list[int] = []          # 王牌 14 张：[0:4] 岭上，[4:9] 宝牌指示，[9:14] 里宝牌
        self.dead_red: list[bool] = []
        self.rinshan_taken = 0
        self.dora_count = 1
        self.pending_dora = 0              # 明杠/加杠后延迟到打牌时翻的宝牌数
        self.kans = 0

        self.hands = [Hand() for _ in range(NUM_PLAYERS)]
        self.rivers: list[list[Discard]] = [[] for _ in range(NUM_PLAYERS)]
        self.riichi = [False] * NUM_PLAYERS
        self.double_riichi = [False] * NUM_PLAYERS
        self.ippatsu = [False] * NUM_PLAYERS
        self.furiten_riichi = [False] * NUM_PLAYERS   # 立直振听（永久）
        self.furiten_temp = [False] * NUM_PLAYERS     # 同巡振听
        self.draws = [0] * NUM_PLAYERS                # 各家摸牌次数（天和/地和/两立直）
        self.any_call = False

        self.phase = Phase.FINISHED
        self.current = dealer
        self.last_draw = -1
        self.last_draw_red = False
        self.last_discard = -1
        self.discarder = -1
        self.just_called = False           # 刚吃/碰，不能自摸/杠
        self.forbidden: set[int] = set()   # 食替禁止
        self.riichi_pending = -1           # 立直宣言牌正在等响应
        self.chankan_tile = -1             # 正在响应的加杠牌
        self.haitei = False                # 刚摸的是最后一张
        self.rinshan = False               # 刚摸的是岭上牌
        self.pending: dict[int, Action | None] = {}
        self.result: Result | None = None
        self.log: list[tuple[int, Action]] = []

    # ------------------------------------------------------------ 开局
    def start(self) -> None:
        tiles = full_wall()
        # 每种 5 里第一张是赤牌，洗牌时跟着走
        reds = [False] * len(tiles)
        for t in RED_FIVES:
            reds[tiles.index(t)] = True
        order = list(range(len(tiles)))
        self.rng.shuffle(order)
        tiles = [tiles[i] for i in order]
        reds = [reds[i] for i in order]
        self.dead, self.dead_red = tiles[-DEAD_WALL:], reds[-DEAD_WALL:]
        self.wall, self.wall_red = tiles[:-DEAD_WALL], reds[:-DEAD_WALL]
        for p in range(NUM_PLAYERS):
            for _ in range(HAND_SIZE):
                self.hands[p].add(self.wall[0], self.wall_red[0])
                self.wall.pop(0)
                self.wall_red.pop(0)
        self._draw(self.dealer)

    @property
    def finished(self) -> bool:
        return self.phase == Phase.FINISHED

    @property
    def dora_indicators(self) -> list[int]:
        return self.dead[4 : 4 + self.dora_count]

    @property
    def ura_indicators(self) -> list[int]:
        return self.dead[9 : 9 + self.dora_count]

    def seat_wind(self, p: int) -> int:
        return WINDS[(p - self.dealer) % NUM_PLAYERS]

    # ------------------------------------------------------------ 查询接口
    def players_to_act(self) -> list[int]:
        if self.phase == Phase.DISCARD:
            return [self.current]
        if self.phase == Phase.RESPOND:
            return [p for p, a in self.pending.items() if a is None]
        return []

    def legal_actions(self, player: int) -> list[Action]:
        if self.phase == Phase.DISCARD and player == self.current:
            return self._legal_discard_phase(player)
        if self.phase == Phase.RESPOND and self.pending.get(player, PASS) is None:
            return self._responses_for(player)
        return []

    def is_furiten(self, p: int) -> bool:
        if self.furiten_riichi[p] or self.furiten_temp[p]:
            return True
        waits = waiting_tiles(self.hands[p].counts)
        mine = {d.tile for d in self.rivers[p]}
        return any(w in mine for w in waits)

    def observe(self, player: int) -> Observation:
        h = self.hands[player]
        return Observation(
            player=player,
            phase=self.phase,
            hand=list(h.counts),
            reds=list(h.reds),
            melds=list(h.melds),
            rivers=[[d.tile for d in r if not d.claimed] for r in self.rivers],
            rivers_full=[list(r) for r in self.rivers],
            all_melds=[list(x.melds) for x in self.hands],
            wall_remaining=len(self.wall),
            current_player=self.current,
            last_discard=self.last_discard if self.phase == Phase.RESPOND else -1,
            last_draw=self.last_draw if self.phase == Phase.DISCARD else -1,
            legal_actions=self.legal_actions(player),
            dora_indicators=list(self.dora_indicators),
            riichi=list(self.riichi),
            scores=list(self.scores),
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            dealer=self.dealer,
            round_wind=self.round_wind,
            seat_wind=self.seat_wind(player),
            furiten=self.is_furiten(player),
            waits=waiting_tiles(h.counts) if h.size % 3 == 1 else [],
            chankan=self.chankan_tile >= 0,
        )

    # ------------------------------------------------------------ 役判断
    def _win_context(self, p: int, tile: int, is_tsumo: bool, chankan: bool = False) -> WinContext:
        h = self.hands[p]
        counts = list(h.counts)
        if not is_tsumo:
            counts[tile] += 1
        first_round = not self.any_call and self.draws[p] <= 1
        return WinContext(
            counts=counts,
            melds=list(h.melds),
            win_tile=tile,
            is_tsumo=is_tsumo,
            seat_wind=self.seat_wind(p),
            round_wind=self.round_wind,
            riichi=self.riichi[p] and not self.double_riichi[p],
            double_riichi=self.double_riichi[p],
            ippatsu=self.ippatsu[p],
            haitei=(self.haitei and is_tsumo) or (not is_tsumo and not self.wall and not chankan),
            rinshan=self.rinshan and is_tsumo,
            chankan=chankan,
            tenhou=is_tsumo and p == self.dealer and first_round and self.draws[p] == 1,
            chiihou=is_tsumo and p != self.dealer and first_round and self.draws[p] == 1,
            dora_indicators=list(self.dora_indicators),
            ura_indicators=list(self.ura_indicators) if self.riichi[p] else [],
            red_count=h.red_count + sum(m.reds for m in h.melds),
        )

    def _can_tsumo(self, p: int) -> WinResult | None:
        if not is_win(self.hands[p].counts):
            return None
        return evaluate(self._win_context(p, self.last_draw, True))

    def _can_ron(self, p: int, tile: int, chankan: bool = False) -> WinResult | None:
        h = self.hands[p]
        if h.counts[tile] >= 4:
            return None
        h.counts[tile] += 1
        ok = is_win(h.counts)
        h.counts[tile] -= 1
        if not ok or self.is_furiten(p):
            return None
        return evaluate(self._win_context(p, tile, False, chankan))

    # ------------------------------------------------------------ 合法动作：打牌阶段
    def _legal_discard_phase(self, p: int) -> list[Action]:
        h = self.hands[p]
        acts: list[Action] = []
        if not self.just_called and self._can_tsumo(p) is not None:
            acts.append(Action(ActionType.TSUMO, self.last_draw))
        can_kan = self.wall and self.kans < MAX_KANS and not self.just_called
        if can_kan:
            for t in range(34):
                if h.counts[t] == 4:
                    if self.riichi[p]:
                        # 立直后只能暗杠刚摸的牌，且不改变听牌
                        if t != self.last_draw:
                            continue
                        before = list(h.counts)
                        before[t] -= 1
                        after = list(h.counts)
                        after[t] -= 4
                        if waiting_tiles(before) != waiting_tiles(after):
                            continue
                    acts.append(Action(ActionType.ANKAN, t))
                elif h.counts[t] == 1 and not self.riichi[p] and any(
                    m.type == MeldType.PON and m.tile == t for m in h.melds
                ):
                    acts.append(Action(ActionType.ADDKAN, t))
        if self.riichi[p]:
            acts.append(Action(ActionType.DISCARD, self.last_draw))
            return acts
        can_riichi = (
            h.is_menzen and len(self.wall) >= MIN_WALL_FOR_RIICHI and self.scores[p] >= RIICHI_BET and not self.just_called
        )
        for t in range(34):
            if h.counts[t] == 0 or t in self.forbidden:
                continue
            acts.append(Action(ActionType.DISCARD, t))
            if can_riichi:
                h.counts[t] -= 1
                tenpai = bool(waiting_tiles(h.counts))
                h.counts[t] += 1
                if tenpai:
                    acts.append(Action(ActionType.RIICHI, t))
        return acts

    # ------------------------------------------------------------ 合法动作：响应阶段
    def _responses_for(self, p: int) -> list[Action]:
        t = self.chankan_tile if self.chankan_tile >= 0 else self.last_discard
        acts: list[Action] = []
        if self._can_ron(p, t, chankan=self.chankan_tile >= 0) is not None:
            acts.append(Action(ActionType.RON, t))
        if self.chankan_tile >= 0 or self.riichi[p] or not self.wall:
            return acts + [PASS] if acts else []  # 抢杠只能荣和；立直不能吃碰杠；海底牌不能吃碰杠
        h = self.hands[p]
        if h.counts[t] >= 3 and self.kans < MAX_KANS:
            acts.append(Action(ActionType.KAN, t))
        if h.counts[t] >= 2:
            acts.append(Action(ActionType.PON, t))
        if p == (self.discarder + 1) % NUM_PLAYERS and not is_honor(t):
            base = suit(t) * 9
            for start in (t - 2, t - 1, t):
                if start < base or start + 2 > base + 8:
                    continue
                others = [x for x in (start, start + 1, start + 2) if x != t]
                if all(h.counts[x] >= 1 for x in others):
                    acts.append(Action(ActionType.CHI, t, start))
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

    def _step_discard_phase(self, p: int, a: Action) -> None:
        h = self.hands[p]
        t = a.tile
        if a.type == ActionType.TSUMO:
            r = self._can_tsumo(p)
            assert r is not None
            self._finish_wins([Win(p, r, None, t)], loser=None)
            return
        if a.type in (ActionType.DISCARD, ActionType.RIICHI):
            is_tsumogiri = t == self.last_draw and not self.just_called
            red = h.remove(t, red=self.last_draw_red if is_tsumogiri else None)
            self.rivers[p].append(Discard(t, red, a.type == ActionType.RIICHI, is_tsumogiri))
            self.last_discard = t
            self.discarder = p
            self.last_draw = -1
            self.just_called = False
            self.forbidden = set()
            self.ippatsu[p] = False
            if self.pending_dora:
                self.dora_count = min(5, self.dora_count + self.pending_dora)
                self.pending_dora = 0
            if a.type == ActionType.RIICHI:
                self.riichi_pending = p
            self._open_respond_phase(p, t)
            return
        # 杠
        self.any_call = True
        for q in range(NUM_PLAYERS):
            self.ippatsu[q] = False
        if a.type == ActionType.ANKAN:
            reds = sum(h.remove(t, red=h.reds[t]) for _ in range(4))
            h.melds.append(Meld(MeldType.ANKAN, t, reds=reds))
            self.kans += 1
            self.dora_count = min(5, self.dora_count + 1)
            self._draw(p, rinshan=True)
        elif a.type == ActionType.ADDKAN:
            red = h.remove(t)
            idx = next(i for i, m in enumerate(h.melds) if m.type == MeldType.PON and m.tile == t)
            old = h.melds[idx]
            h.melds[idx] = Meld(MeldType.ADDKAN, t, old.from_player, old.called, old.reds + red)
            self.kans += 1
            self.pending_dora += 1
            # 抢杠：其他人可以荣和这张牌
            self.chankan_tile = t
            self.discarder = p
            self.pending = {q: None for q in range(NUM_PLAYERS) if q != p and self._responses_for(q)}
            if self.pending:
                self.phase = Phase.RESPOND
            else:
                self.chankan_tile = -1
                self._draw(p, rinshan=True)

    def _open_respond_phase(self, discarder: int, t: int) -> None:
        self.pending = {q: None for q in range(NUM_PLAYERS) if q != discarder and self._responses_for(q)}
        if self.pending:
            self.phase = Phase.RESPOND
        else:
            self._after_discard_passed()

    def _after_discard_passed(self) -> None:
        """没人响应（或都过）之后：立直成立、同巡振听、下一家摸牌。"""
        t = self.last_discard
        for q in range(NUM_PLAYERS):
            if q == self.discarder:
                continue
            if t in waiting_tiles(self.hands[q].counts):
                self.furiten_temp[q] = True
                if self.riichi[q]:
                    self.furiten_riichi[q] = True
        if self.riichi_pending >= 0:
            p = self.riichi_pending
            self._establish_riichi()
            self.ippatsu[p] = True
        self._next_turn()

    def _step_respond_phase(self, p: int, a: Action) -> None:
        self.pending[p] = a
        if any(x is None for x in self.pending.values()):
            return
        self._resolve_responses()

    def _resolve_responses(self) -> None:
        discarder = self.discarder
        chankan = self.chankan_tile >= 0
        t = self.chankan_tile if chankan else self.last_discard
        responses = {p: a for p, a in self.pending.items() if a is not None}
        self.pending = {}

        rons = [p for p, a in responses.items() if a.type == ActionType.RON]
        if len(rons) >= 3:
            self._finish_abort("三家和")
            return
        if rons:
            # 按离放炮者的顺序（下家优先）排，立直棒归最近者
            rons.sort(key=lambda p: (p - discarder) % NUM_PLAYERS)
            if chankan:
                # 加杠被抢：杠不成立，牌退回... 简化：牌归和牌者，杠子退回成碰
                h = self.hands[discarder]
                idx = next(i for i, m in enumerate(h.melds) if m.type == MeldType.ADDKAN and m.tile == t)
                old = h.melds[idx]
                h.melds[idx] = Meld(MeldType.PON, t, old.from_player, old.called, old.reds)
                self.kans -= 1
                self.pending_dora -= 1
            else:
                self.rivers[discarder][-1].claimed = True
            wins = []
            for p in rons:
                r = self._can_ron(p, t, chankan)
                assert r is not None
                wins.append(Win(p, r, discarder, t))
            self.riichi_pending = -1  # 立直宣言牌被荣和，立直不成立
            self._finish_wins(wins, loser=discarder)
            return

        if chankan:
            for q in responses:
                if t in waiting_tiles(self.hands[q].counts):
                    self.furiten_temp[q] = True
                    if self.riichi[q]:
                        self.furiten_riichi[q] = True
            self.chankan_tile = -1
            self._draw(discarder, rinshan=True)
            return

        best_p, best_a = max(
            responses.items(), key=lambda kv: (PRIORITY[kv[1].type], -((kv[0] - discarder) % NUM_PLAYERS))
        )
        if best_a.type == ActionType.PASS:
            self._after_discard_passed()
            return

        # 吃 / 碰 / 大明杠
        self.any_call = True
        for q in range(NUM_PLAYERS):
            self.ippatsu[q] = False
        if self.riichi_pending >= 0:  # 立直宣言牌被叫走：立直仍然成立
            self._establish_riichi()
        for q in range(NUM_PLAYERS):
            if q != discarder and t in waiting_tiles(self.hands[q].counts) and q != best_p:
                self.furiten_temp[q] = True
                if self.riichi[q]:
                    self.furiten_riichi[q] = True
        self.rivers[discarder][-1].claimed = True
        called_red = int(self.rivers[discarder][-1].red)
        h = self.hands[best_p]
        if best_a.type == ActionType.CHI:
            start = best_a.extra
            reds = called_red + sum(h.remove(x) for x in (start, start + 1, start + 2) if x != t)
            h.melds.append(Meld(MeldType.CHI, start, discarder, t, reds))
            # 食替禁止：现物 + 筋
            self.forbidden = {t}
            if t == start and number(start) <= 6:
                self.forbidden.add(start + 3)
            if t == start + 2 and number(start) >= 2:
                self.forbidden.add(start - 1)
            self._enter_discard_after_call(best_p)
        elif best_a.type == ActionType.PON:
            reds = called_red + h.remove(t) + h.remove(t)
            h.melds.append(Meld(MeldType.PON, t, discarder, t, reds))
            self.forbidden = {t}
            self._enter_discard_after_call(best_p)
        elif best_a.type == ActionType.KAN:
            reds = called_red + sum(h.remove(t, red=h.reds[t]) for _ in range(3))
            h.melds.append(Meld(MeldType.KAN, t, discarder, t, reds))
            self.kans += 1
            self.pending_dora += 1
            self.current = best_p
            self._draw(best_p, rinshan=True)

    def _establish_riichi(self) -> None:
        p = self.riichi_pending
        self.riichi[p] = True
        if not self.any_call and self.draws[p] == 1:
            self.double_riichi[p] = True
        self.scores[p] -= RIICHI_BET
        self.riichi_sticks += 1
        self.riichi_pending = -1

    def _enter_discard_after_call(self, p: int) -> None:
        self.current = p
        self.last_draw = -1
        self.just_called = True
        self.haitei = False
        self.rinshan = False
        self.phase = Phase.DISCARD

    def _next_turn(self) -> None:
        self._draw((self.current + 1) % NUM_PLAYERS)

    def _draw(self, p: int, rinshan: bool = False) -> None:
        if rinshan:
            if self.rinshan_taken >= 4 or not self.wall:
                self._finish_draw()
                return
            k = self.rinshan_taken
            t, red = self.dead[k], self.dead_red[k]
            # 活牌墙末尾一张补进王牌区（占岭上牌的位置）
            self.dead[k], self.dead_red[k] = self.wall.pop(), self.wall_red.pop()
            self.rinshan_taken += 1
        else:
            if not self.wall:
                self._finish_draw()
                return
            t, red = self.wall.pop(0), self.wall_red.pop(0)
        self.hands[p].add(t, red)
        self.draws[p] += 1
        self.furiten_temp[p] = False
        self.current = p
        self.last_draw = t
        self.last_draw_red = red
        self.just_called = False
        self.forbidden = set()
        self.haitei = not self.wall and not rinshan
        self.rinshan = rinshan
        self.phase = Phase.DISCARD

    # ------------------------------------------------------------ 结算
    def _finish_wins(self, wins: list[Win], loser: int | None) -> None:
        from .score import payment

        deltas = [0] * NUM_PLAYERS
        for i, w in enumerate(wins):
            pay = payment(w.result.base, w.player, self.dealer, loser is None, loser, self.honba)
            for q, amt in pay.losses.items():
                deltas[q] -= amt
            deltas[w.player] += pay.winner_gain
        # 立直棒归（离放炮者最近的）第一个和牌者
        deltas[wins[0].player] += self.riichi_sticks * RIICHI_BET
        self.riichi_sticks = 0
        for q in range(NUM_PLAYERS):
            self.scores[q] += deltas[q]
        self.result = Result("win", wins, self._deltas(), dealer_continues=any(w.player == self.dealer for w in wins))
        self.phase = Phase.FINISHED

    def _finish_draw(self) -> None:
        tenpai = [bool(waiting_tiles(self.hands[p].counts)) for p in range(NUM_PLAYERS)]
        n = sum(tenpai)
        deltas = [0] * NUM_PLAYERS
        if 0 < n < 4:
            for p in range(NUM_PLAYERS):
                deltas[p] = NOTEN_PENALTY // n if tenpai[p] else -NOTEN_PENALTY // (4 - n)
        for q in range(NUM_PLAYERS):
            self.scores[q] += deltas[q]
        self.result = Result("draw", [], self._deltas(), tenpai, dealer_continues=tenpai[self.dealer])
        self.phase = Phase.FINISHED

    def _finish_abort(self, reason: str) -> None:
        self.result = Result("abort", deltas=self._deltas(), reason=reason, dealer_continues=True)
        self.phase = Phase.FINISHED

    def _deltas(self) -> list[int]:
        """本局点数变动 = 现在 - 开局（包含立直押金、听牌费、立直棒）。"""
        return [self.scores[q] - self.start_scores[q] for q in range(NUM_PLAYERS)]
