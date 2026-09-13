"""规则 bot：牌效率 + 简单防守。

决策逻辑（按优先级）：
    1. 能胡就胡（自摸 / 点炮）
    2. 碰 / 杠：只在向听数减少（杠：不增加）时才做
    3. 打牌：向听数最低 > 进张最多 > 危险度最低
       "危险度"只有一条启发式：对手副露 >= DANGER_MELDS 时视为可能听牌，
       他打过的牌相对安全（这是弱假设——中国规则没有振听，但作为第一版够用）

它有两个用途：
    - 作为"打得像样"的基线，让对局有意义（流局率降下来）
    - 阶段 3 用它自对局生成监督学习数据
"""

from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Observation
from majiang.engine.hand import MeldType
from majiang.engine.shanten import discard_options, shanten
from majiang.engine.tile import NUM_TILE_TYPES, is_honor, is_terminal

from .base import Agent

DANGER_MELDS = 3  # 对手副露达到这么多组，认为他可能听牌


class RuleAgent(Agent):
    def act(self, obs: Observation) -> Action:
        legal = obs.legal_actions
        by_type = {a.type: a for a in legal}

        if ActionType.TSUMO in by_type:
            return by_type[ActionType.TSUMO]
        if ActionType.RON in by_type:
            return by_type[ActionType.RON]

        n_melds = len(obs.melds)
        hand = obs.hand

        # ---------------- 响应阶段：碰 / 明杠 / 过
        if ActionType.PON in by_type or ActionType.KAN in by_type:
            cur = shanten(hand, n_melds)
            if ActionType.KAN in by_type:
                t = by_type[ActionType.KAN].tile
                after = self._with(hand, t, -3)
                if shanten(after, n_melds + 1) <= cur:
                    return by_type[ActionType.KAN]
            if ActionType.PON in by_type:
                t = by_type[ActionType.PON].tile
                after = self._with(hand, t, -2)  # 碰完 3k+2 张，shanten 会自动取最优打法
                if shanten(after, n_melds + 1) < cur:
                    return by_type[ActionType.PON]
            return by_type[ActionType.PASS]

        # ---------------- 打牌阶段：暗杠 / 加杠 / 打牌
        visible = self._visible(obs)
        options = discard_options(hand, n_melds, visible)
        best_shanten = options[0][1]

        for kt, k in ((ActionType.ANKAN, 4), (ActionType.ADDKAN, 1)):
            if kt in by_type:
                t = by_type[kt].tile
                after = self._with(hand, t, -k)
                if shanten(after, n_melds + 1) <= best_shanten:
                    return by_type[kt]

        danger = self._danger_map(obs)
        push = best_shanten <= 0  # 自己已听牌就不防守

        def key(opt: tuple[int, int, int]) -> tuple:
            d, s, n = opt
            iso = 2 if is_honor(d) else 1 if is_terminal(d) else 0  # 同分时先扔字牌/幺九
            if push:
                return (s, -n, -iso)
            return (s, danger[d], -n, -iso)

        best = min(options, key=key)
        return Action(ActionType.DISCARD, best[0])

    # ---------------- 工具
    @staticmethod
    def _with(counts: list[int], t: int, delta: int) -> list[int]:
        c = list(counts)
        c[t] += delta
        return c

    @staticmethod
    def _visible(obs: Observation) -> list[int]:
        """牌桌上除自己手牌外能看见的牌：四家牌河 + 四家副露。"""
        v = [0] * NUM_TILE_TYPES
        for river in obs.rivers:
            for t in river:
                v[t] += 1
        for melds in obs.all_melds:
            for m in melds:
                v[m.tile] += 3 if m.type == MeldType.PON else 4
        return v

    @staticmethod
    def _danger_map(obs: Observation) -> list[int]:
        """每张牌的危险度 = 有多少个"疑似听牌"的对手没打过它。"""
        danger = [0] * NUM_TILE_TYPES
        for p in range(4):
            if p == obs.player or len(obs.all_melds[p]) < DANGER_MELDS:
                continue
            seen = set(obs.rivers[p])
            for t in range(NUM_TILE_TYPES):
                if t not in seen:
                    danger[t] += 1
        return danger
