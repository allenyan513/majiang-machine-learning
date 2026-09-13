"""神经网络 agent：打牌由模型决定，其余决策（和、立直、杠、吃碰）沿用 RuleAgent。

"混合"是刻意的：阶段 3 只训练了"打哪张"（数据里最多、最有意思的部分）。
立直/吃碰/杠的决策数据少且规则逻辑够用，以后再一起学。
"""

import torch

from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Observation, Phase
from majiang.ml.features import encode
from majiang.ml.model import ActorCritic, DiscardNet, load

from .base import Agent
from .rule_agent import RuleAgent


def legal_discard_mask(obs: Observation) -> torch.Tensor:
    """34 维 bool：哪些牌现在可以打（立直后只能摸切、食替禁止都体现在 legal_actions 里）。"""
    m = torch.zeros(34, dtype=torch.bool)
    for a in obs.legal_actions:
        if a.type == ActionType.DISCARD:
            m[a.tile] = True
    return m


class NNAgent(Agent):
    def __init__(self, model: DiscardNet | ActorCritic | str, temperature: float = 0.0, seed: int | None = None):
        self.model = load(model) if isinstance(model, str) else model
        self.model.eval()
        self.temperature = temperature  # 0 = 取最大概率；>0 = 按概率采样（RL 需要）
        self.rule = RuleAgent()
        self.gen = torch.Generator().manual_seed(seed or 0)

    def discard_probs(self, obs: Observation) -> torch.Tensor:
        """34 维概率，不能打的牌为 0。可视化直接用它。"""
        with torch.no_grad():
            return self.model.probs(encode(obs), legal_discard_mask(obs))[0]

    def act(self, obs: Observation) -> Action:
        rule_action = self.rule.act(obs)
        # 非打牌决策（和、立直、杠、响应）交给规则；只有普通打牌用网络
        if obs.phase != Phase.DISCARD or rule_action.type != ActionType.DISCARD:
            return rule_action
        legal = [a for a in obs.legal_actions if a.type == ActionType.DISCARD]
        if len(legal) == 1:
            return legal[0]
        probs = self.discard_probs(obs)
        if self.temperature <= 0:
            t = int(probs.argmax())
        else:
            p = probs ** (1.0 / self.temperature)
            t = int(torch.multinomial(p / p.sum(), 1, generator=self.gen))
        return Action(ActionType.DISCARD, t)
