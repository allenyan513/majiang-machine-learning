"""役判定与符计算（天凤四人规则）。

入口：evaluate(ctx) -> WinResult | None
    None 表示牌型上能和但**没有役**（不能和）。

做法：枚举所有拆法 × "和牌那张属于哪一组"，每种组合各算一次役和符，取点数最高的。
这一步枚举解决了：平和要求两面听、荣和的刻子不算暗刻、四暗刻单骑 vs 双碰 等所有依赖拆法的判断。

役满一旦成立就忽略普通役；宝牌（表/里/赤）不是役，只加番。
"""

from dataclasses import dataclass, field

from .hand import Meld, MeldType
from .score import base_points
from .tile import (
    DRAGONS,
    GREEN,
    HONOR_START,
    NUM_TILE_TYPES,
    WINDS,
    dora_from_indicator,
    is_honor,
    is_simple,
    is_terminal,
    is_yaochu,
    number,
    suit,
)
from .win import YAOCHU, decompose, is_kokushi, is_seven_pairs


@dataclass
class WinContext:
    counts: list[int]                     # 门前手牌计数（**含**和牌那张）
    melds: list[Meld]                     # 副露（含暗杠）
    win_tile: int
    is_tsumo: bool
    seat_wind: int                        # 自风 tile id（东=27 …）
    round_wind: int                       # 场风
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    haitei: bool = False                  # 最后一张：自摸=海底摸月，荣和=河底捞鱼
    rinshan: bool = False
    chankan: bool = False
    tenhou: bool = False                  # 天和 / 地和
    chiihou: bool = False
    dora_indicators: list[int] = field(default_factory=list)
    ura_indicators: list[int] = field(default_factory=list)   # 只在立直时传
    red_count: int = 0

    @property
    def is_menzen(self) -> bool:
        return all(not m.is_open for m in self.melds)

    def all_tile_counts(self) -> list[int]:
        """手牌 + 副露的全部牌（杠算 4 张）。"""
        c = list(self.counts)
        for m in self.melds:
            for t in m.tiles:
                c[t] += 1
        return c


@dataclass
class WinResult:
    yaku: list[tuple[str, int]]   # (役名, 番)；役满时番记为 13 的倍数
    han: int                      # 总番（含宝牌）；役满时 = 13 × 倍数
    fu: int
    yakuman: int                  # 役满倍数，0 = 不是役满
    base: int                     # 基本分
    limit: str                    # "满贯" / "跳满" / … / ""

    def __str__(self) -> str:
        y = " ".join(f"{n}({h})" for n, h in self.yaku)
        head = f"{self.limit} " if self.limit else ""
        return f"{head}{self.han}番{self.fu}符 [{y}]"


# 一组牌：kind "chi"/"pon"/"pair"，tile 起点或牌，concealed 暗，kan 是否杠
@dataclass(frozen=True)
class _Set:
    kind: str
    tile: int
    concealed: bool
    kan: bool = False

    @property
    def tiles(self) -> list[int]:
        if self.kind == "chi":
            return [self.tile, self.tile + 1, self.tile + 2]
        if self.kind == "pair":
            return [self.tile] * 2
        return [self.tile] * (4 if self.kan else 3)


def _meld_to_set(m: Meld) -> _Set:
    if m.type == MeldType.CHI:
        return _Set("chi", m.tile, False)
    if m.type == MeldType.PON:
        return _Set("pon", m.tile, False)
    if m.type == MeldType.ANKAN:
        return _Set("pon", m.tile, True, True)
    return _Set("pon", m.tile, False, True)  # 明杠 / 加杠


def _wait_type(win_set: _Set, win_tile: int) -> str:
    """tanki / kanchan / penchan / ryanmen / shanpon。"""
    if win_set.kind == "pair":
        return "tanki"
    if win_set.kind == "pon":
        return "shanpon"
    s = win_set.tile
    if win_tile == s + 1:
        return "kanchan"
    if (win_tile == s + 2 and number(s) == 1) or (win_tile == s and number(s) == 7):
        return "penchan"
    return "ryanmen"


def _dora_han(ctx: WinContext) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    all_c = ctx.all_tile_counts()
    n = sum(all_c[dora_from_indicator(i)] for i in ctx.dora_indicators)
    if n:
        out.append(("宝牌", n))
    if ctx.red_count:
        out.append(("赤宝牌", ctx.red_count))
    if ctx.riichi or ctx.double_riichi:
        u = sum(all_c[dora_from_indicator(i)] for i in ctx.ura_indicators)
        if u:
            out.append(("里宝牌", u))
    return out


def _situational(ctx: WinContext) -> list[tuple[str, int]]:
    """和牌型无关的役。"""
    y: list[tuple[str, int]] = []
    if ctx.double_riichi:
        y.append(("两立直", 2))
    elif ctx.riichi:
        y.append(("立直", 1))
    if ctx.ippatsu:
        y.append(("一发", 1))
    if ctx.is_tsumo and ctx.is_menzen:
        y.append(("门清自摸", 1))
    if ctx.haitei:
        y.append(("海底摸月", 1) if ctx.is_tsumo else ("河底捞鱼", 1))
    if ctx.rinshan:
        y.append(("岭上开花", 1))
    if ctx.chankan:
        y.append(("抢杠", 1))
    return y


def _flush(all_c: list[int]) -> tuple[bool, bool]:
    """(混一色, 清一色)。"""
    suits = {suit(t) for t in range(NUM_TILE_TYPES) if all_c[t]}
    number_suits = suits - {3}
    if len(number_suits) != 1:
        return False, False
    return (3 in suits), (3 not in suits)


# ---------------------------------------------------------------- 役满
def _yakuman_standard(ctx: WinContext, sets: list[_Set], pair: int, win_set: _Set) -> list[tuple[str, int]]:
    y: list[tuple[str, int]] = []
    all_c = ctx.all_tile_counts()
    pons = [s for s in sets if s.kind == "pon"]
    concealed_pons = [s for s in pons if s.concealed]
    if len(pons) == 4 and len(concealed_pons) == 4:
        y.append(("四暗刻单骑", 26) if win_set.kind == "pair" else ("四暗刻", 13))
    if sum(1 for s in pons if s.tile in DRAGONS) == 3:
        y.append(("大三元", 13))
    wind_pons = sum(1 for s in pons if s.tile in WINDS)
    if wind_pons == 4:
        y.append(("大四喜", 26))
    elif wind_pons == 3 and pair in WINDS:
        y.append(("小四喜", 13))
    if all(is_honor(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
        y.append(("字一色", 13))
    if all(is_terminal(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
        y.append(("清老头", 13))
    green = {10, 11, 12, 14, 16, GREEN}  # 2s 3s 4s 6s 8s 发
    if all(t in green for t in range(NUM_TILE_TYPES) if all_c[t]):
        y.append(("绿一色", 13))
    if sum(1 for s in sets if s.kan) == 4:
        y.append(("四杠子", 13))
    if ctx.is_menzen and not ctx.melds:
        _, chin = _flush(all_c)
        if chin:
            base = suit(ctx.win_tile) * 9
            seg = [ctx.counts[base + i] for i in range(9)]
            pattern = [3, 1, 1, 1, 1, 1, 1, 1, 3]
            extra = [seg[i] - pattern[i] for i in range(9)]
            if all(e >= 0 for e in extra) and sum(extra) == 1:
                pure = extra[number(ctx.win_tile) - 1] == 1  # 去掉和牌张后正好是 1112345678999
                y.append(("纯正九莲宝灯", 26) if pure else ("九莲宝灯", 13))
    if ctx.tenhou:
        y.append(("天和", 13))
    if ctx.chiihou:
        y.append(("地和", 13))
    return y


# ---------------------------------------------------------------- 普通役
def _yaku_standard(ctx: WinContext, sets: list[_Set], pair: int, win_set: _Set) -> list[tuple[str, int]]:
    y: list[tuple[str, int]] = []
    menzen = ctx.is_menzen
    all_c = ctx.all_tile_counts()
    chis = [s for s in sets if s.kind == "chi"]
    pons = [s for s in sets if s.kind == "pon"]
    yakuhai_tiles = set(DRAGONS) | {ctx.seat_wind, ctx.round_wind}

    if all(is_simple(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
        y.append(("断幺九", 1))
    for s in pons:
        if s.tile in DRAGONS:
            y.append((f"役牌 {['白', '发', '中'][s.tile - 31]}", 1))
        if s.tile == ctx.seat_wind:
            y.append(("自风", 1))
        if s.tile == ctx.round_wind:
            y.append(("场风", 1))
    if menzen and len(chis) == 4 and pair not in yakuhai_tiles and _wait_type(win_set, ctx.win_tile) == "ryanmen":
        y.append(("平和", 1))
    if menzen:
        chi_keys = sorted(s.tile for s in chis)
        dup = sum(1 for i in range(len(chi_keys) - 1) if chi_keys[i] == chi_keys[i + 1])
        # 两组相同 = 一杯口；两对相同（4 顺子两两相同）= 二杯口
        if len(chi_keys) == 4 and chi_keys[0] == chi_keys[1] and chi_keys[2] == chi_keys[3]:
            y.append(("二杯口", 3))
        elif dup >= 1:
            y.append(("一杯口", 1))
    # 三色同顺
    for n in range(1, 8):
        starts = {suit(s.tile) for s in chis if number(s.tile) == n}
        if starts == {0, 1, 2}:
            y.append(("三色同顺", 2 if menzen else 1))
            break
    # 三色同刻
    for n in range(1, 10):
        if {suit(s.tile) for s in pons if not is_honor(s.tile) and number(s.tile) == n} == {0, 1, 2}:
            y.append(("三色同刻", 2))
            break
    # 一气通贯
    for su in range(3):
        starts = {number(s.tile) for s in chis if suit(s.tile) == su}
        if {1, 4, 7} <= starts:
            y.append(("一气通贯", 2 if menzen else 1))
            break
    # 全带 / 纯全带 / 混老头
    all_sets = sets + [_Set("pair", pair, True)]
    every_has_yaochu = all(any(is_yaochu(t) for t in s.tiles) for s in all_sets)
    if every_has_yaochu:
        has_honor = any(is_honor(t) for t in range(HONOR_START, NUM_TILE_TYPES) if all_c[t])
        if not chis:
            y.append(("混老头", 2))  # 没有顺子：全是幺九刻子/对子（有字牌才叫混老头；无字牌是清老头役满）
        elif has_honor:
            y.append(("混全带幺九", 2 if menzen else 1))
        else:
            y.append(("纯全带幺九", 3 if menzen else 2))
    if len(pons) == 4:
        y.append(("对对和", 2))
    if sum(1 for s in pons if s.concealed) == 3:
        y.append(("三暗刻", 2))
    if sum(1 for s in sets if s.kan) == 3:
        y.append(("三杠子", 2))
    if sum(1 for s in pons if s.tile in DRAGONS) == 2 and pair in DRAGONS:
        y.append(("小三元", 2))
    hon, chin = _flush(all_c)
    if chin:
        y.append(("清一色", 6 if menzen else 5))
    elif hon:
        y.append(("混一色", 3 if menzen else 2))
    return y


def _fu_standard(ctx: WinContext, sets: list[_Set], pair: int, win_set: _Set, has_pinfu: bool) -> int:
    menzen = ctx.is_menzen
    if has_pinfu:
        return 20 if ctx.is_tsumo else 30
    fu = 20
    if menzen and not ctx.is_tsumo:
        fu += 10
    if ctx.is_tsumo:
        fu += 2
    for s in sets:
        if s.kind != "pon":
            continue
        v = 2
        if s.concealed:
            v *= 2
        if s.kan:
            v *= 4
        if is_yaochu(s.tile):
            v *= 2
        fu += v
    if pair in DRAGONS:
        fu += 2
    if pair == ctx.seat_wind:
        fu += 2
    if pair == ctx.round_wind:
        fu += 2
    if _wait_type(win_set, ctx.win_tile) in ("tanki", "kanchan", "penchan"):
        fu += 2
    if fu == 20 and not menzen:
        fu = 30  # 副露平和形固定 30 符
    return -(-fu // 10) * 10


# ---------------------------------------------------------------- 入口
def _finish(ctx: WinContext, yaku: list[tuple[str, int]], fu: int, yakuman: int) -> WinResult:
    if yakuman:
        han = 13 * yakuman
        base, limit = base_points(han, fu, yakuman)
        return WinResult(yaku, han, fu, yakuman, base, limit)
    dora = _dora_han(ctx)
    han = sum(h for _, h in yaku) + sum(h for _, h in dora)
    base, limit = base_points(han, fu)
    return WinResult(yaku + dora, han, fu, 0, base, limit)


def _better(a: WinResult | None, b: WinResult) -> WinResult:
    if a is None:
        return b
    return b if (b.base, b.han, b.fu) > (a.base, a.han, a.fu) else a


def evaluate(ctx: WinContext) -> WinResult | None:
    """返回最高分的役结果；没有役返回 None。"""
    best: WinResult | None = None
    situational = _situational(ctx)
    menzen_only = not ctx.melds

    # 国士无双（只可能是门清 14 张）
    if menzen_only and is_kokushi(ctx.counts):
        before = list(ctx.counts)
        before[ctx.win_tile] -= 1
        thirteen = all(before[t] == 1 for t in YAOCHU)
        y = [("国士无双十三面", 26)] if thirteen else [("国士无双", 13)]
        if ctx.tenhou:
            y.append(("天和", 13))
        if ctx.chiihou:
            y.append(("地和", 13))
        return _finish(ctx, y, 30, sum(h for _, h in y) // 13)

    # 七对子
    if menzen_only and is_seven_pairs(ctx.counts):
        all_c = ctx.counts
        if all(is_honor(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
            y = [("字一色", 13)] + [("天和", 13)] * ctx.tenhou + [("地和", 13)] * ctx.chiihou
            best = _better(best, _finish(ctx, y, 25, sum(h for _, h in y) // 13))
        else:
            y = list(situational) + [("七对子", 2)]
            if all(is_simple(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
                y.append(("断幺九", 1))
            if all(is_yaochu(t) for t in range(NUM_TILE_TYPES) if all_c[t]):
                y.append(("混老头", 2))
            hon, chin = _flush(all_c)
            if chin:
                y.append(("清一色", 6))
            elif hon:
                y.append(("混一色", 3))
            best = _better(best, _finish(ctx, y, 25, 0))

    # 标准型
    meld_sets = [_meld_to_set(m) for m in ctx.melds]
    for pair, hand_sets in decompose(ctx.counts):
        # 和牌那张可能属于：雀头、或手牌里某个包含它的面子
        candidates: list[int] = []
        if pair == ctx.win_tile:
            candidates.append(-1)
        for i, (kind, tile) in enumerate(hand_sets):
            tiles = [tile, tile + 1, tile + 2] if kind == "chi" else [tile]
            if ctx.win_tile in tiles:
                candidates.append(i)
        for ci in candidates:
            sets: list[_Set] = []
            for i, (kind, tile) in enumerate(hand_sets):
                concealed = True
                if kind == "pon" and i == ci and not ctx.is_tsumo:
                    concealed = False  # 荣和补成的刻子是明刻
                sets.append(_Set(kind, tile, concealed))
            win_set = _Set("pair", pair, True) if ci == -1 else sets[ci]
            sets_all = sets + meld_sets

            ym = _yakuman_standard(ctx, sets_all, pair, win_set)
            if ym:
                cand = _finish(ctx, ym, 0, sum(h for _, h in ym) // 13)
                best = _better(best, cand)
                continue
            y = list(situational) + _yaku_standard(ctx, sets_all, pair, win_set)
            if not y:
                continue
            has_pinfu = any(n == "平和" for n, _ in y)
            fu = _fu_standard(ctx, sets_all, pair, win_set, has_pinfu)
            best = _better(best, _finish(ctx, y, fu, 0))
    return best


def has_yaku(ctx: WinContext) -> bool:
    return evaluate(ctx) is not None
