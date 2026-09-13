"""规则 bot（日麻版，第一版）：门清速攻。

决策：
    1. 能和就和（自摸 / 荣和）
    2. 响应阶段一律过——不吃不碰不明杠，保持门清（这样永远有立直这个役，省去役判断）
    3. 打牌阶段：
         可以立直 -> 立直，选听牌张数最多的打法
         可以暗杠 -> 杠（立直后引擎只允许不改听的暗杠）
         否则按牌效率：向听数最低 > 进张最多 > 先扔字牌/幺九
    不防守。它是阶段 3 监督学习的老师和所有评估的基线，强度目标是"像样"，不是"强"。
"""

from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Observation, Phase
from majiang.engine.shanten import discard_options
from majiang.engine.tile import NUM_TILE_TYPES, is_honor, is_terminal
from majiang.engine.win import waiting_tiles

from .base import Agent


class RuleAgent(Agent):
    def act(self, obs: Observation) -> Action:
        legal = obs.legal_actions
        by_type: dict[ActionType, list[Action]] = {}
        for a in legal:
            by_type.setdefault(a.type, []).append(a)

        if ActionType.TSUMO in by_type:
            return by_type[ActionType.TSUMO][0]
        if ActionType.RON in by_type:
            return by_type[ActionType.RON][0]
        if obs.phase == Phase.RESPOND:
            return by_type[ActionType.PASS][0]

        if ActionType.ANKAN in by_type:
            return by_type[ActionType.ANKAN][0]
        visible = self.visible(obs)
        if ActionType.RIICHI in by_type:
            # 选听牌张数（牌池里还剩的）最多的立直打法
            def waits_left(a: Action) -> int:
                h = list(obs.hand)
                h[a.tile] -= 1
                return sum(4 - h[t] - visible[t] for t in waiting_tiles(h))

            return max(by_type[ActionType.RIICHI], key=waits_left)

        discards = by_type.get(ActionType.DISCARD, [])
        if len(discards) == 1:
            return discards[0]
        allowed = {a.tile for a in discards}
        options = [o for o in discard_options(obs.hand, len(obs.melds), visible) if o[0] in allowed]

        def key(opt: tuple[int, int, int]) -> tuple:
            d, s, n = opt
            iso = 2 if is_honor(d) else 1 if is_terminal(d) else 0
            return (s, -n, -iso)

        best = min(options, key=key)
        return Action(ActionType.DISCARD, best[0])

    @staticmethod
    def visible(obs: Observation) -> list[int]:
        """牌桌上除自己手牌外能看见的牌：四家牌河 + 四家副露 + 宝牌指示牌。"""
        v = [0] * NUM_TILE_TYPES
        for river in obs.rivers:
            for t in river:
                v[t] += 1
        for melds in obs.all_melds:
            for m in melds:
                for t in m.tiles:
                    v[t] += 1
        for t in obs.dora_indicators:
            v[t] += 1
        return v
