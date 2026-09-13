"""防守：估计每个对手的听牌概率，以及每张牌被对手胡的概率。

两张概率表由 ml/calibrate_defense.py 从规则自对局里统计得到（见 defense_tables.py）。
    threat(obs, q)      对手 q 听牌的概率            P_TENPAI[副露数][已打张数桶]
    danger_map(obs)     34 张牌各自的危险度 = Σ_q threat(q) × P_WAIT[类别][可见张数][q 是否打过]
    value(...)          打出某张牌后所处状态的平均终局分数  VALUE[向听][进张桶][牌墙桶][威胁桶]
"危险度"有明确含义：这张牌打出去被人胡的概率（近似）。
"""

from majiang.engine.game import Observation
from majiang.engine.hand import MeldType
from majiang.engine.tile import NUM_TILE_TYPES, is_honor, number

from . import defense_tables as T
from .defense_tables import BUCKETS, P_TENPAI, P_WAIT

# 价值表由 calibrate_defense.py 第二步生成；第一次校准时还没有，用简单回退
VALUE = getattr(T, "VALUE", None)


def category(t: int) -> int:
    """0 字牌 1 幺九 2 二八 3 中张。"""
    if is_honor(t):
        return 0
    n = number(t)
    return 1 if n in (1, 9) else 2 if n in (2, 8) else 3


def bucket(n_discards: int) -> int:
    return max(i for i, b in enumerate(BUCKETS) if n_discards >= b)


def threat(obs: Observation, q: int) -> float:
    return P_TENPAI[len(obs.all_melds[q])][bucket(len(obs.rivers[q]))]


def visible_counts(obs: Observation) -> list[int]:
    """防守方能看到的每种牌张数：四家牌河 + 四家副露 + 自己手牌（与校准时一致）。"""
    v = list(obs.hand)
    for river in obs.rivers:
        for t in river:
            v[t] += 1
    for melds in obs.all_melds:
        for m in melds:
            v[m.tile] += 3 if m.type == MeldType.PON else 4
    return v


def danger_map(obs: Observation) -> list[float]:
    vis = visible_counts(obs)
    danger = [0.0] * NUM_TILE_TYPES
    for q in range(4):
        if q == obs.player:
            continue
        th = threat(obs, q)
        if th < 0.02:
            continue
        river = set(obs.rivers[q])
        for t in range(NUM_TILE_TYPES):
            danger[t] += th * P_WAIT[category(t)][min(vis[t], 3)][int(t in river)]
    return danger


def max_threat(obs: Observation) -> float:
    return max(threat(obs, q) for q in range(4) if q != obs.player)


def _b(x: float, edges: list) -> int:
    return max(i for i, e in enumerate(edges) if x >= e)


def _centers(edges: list[int]) -> list[float]:
    return [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])] + [edges[-1] + 5]


def value(shanten_after: int, effective: int, wall: int, threat_max: float) -> float:
    """当前规则 bot 处于该状态时的平均终局分数（策略评估表）。

    进张数维度做线性插值：否则同一桶内 V 相同，EV 只剩危险度，
    会为了 0.1% 的安全白白放弃好几张进张。
    """
    if VALUE is None:
        return -0.2 * min(max(shanten_after, 0), 4)
    row = VALUE[min(max(shanten_after, 0), 4)]
    w, th = _b(wall, T.WALL_BUCKETS), _b(threat_max, T.THREAT_BUCKETS)
    centers = _centers(T.EFF_BUCKETS)
    e = _b(effective, T.EFF_BUCKETS)
    v_e = row[e][w][th]
    if effective >= centers[e] and e + 1 < len(centers):
        v2, c1, c2 = row[e + 1][w][th], centers[e], centers[e + 1]
    elif effective < centers[e] and e > 0:
        v2, c1, c2 = row[e - 1][w][th], centers[e], centers[e - 1]
    else:
        return v_e
    return v_e + (v2 - v_e) * (effective - c1) / (c2 - c1)
