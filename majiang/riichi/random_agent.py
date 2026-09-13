import random

from majiang.riichi.actions import Action, ActionType
from majiang.riichi.game import Observation

from majiang.agents.base import Agent


class RandomAgent(Agent):
    """随机打牌。唯一的"智能"：能和就和——否则一局几乎永远打不完。"""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        for a in obs.legal_actions:
            if a.type in (ActionType.TSUMO, ActionType.RON):
                return a
        return self.rng.choice(obs.legal_actions)
