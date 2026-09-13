"""神经网络 agent：打牌由模型决定，碰/杠/胡沿用 RuleAgent 的逻辑。

这种"混合"是刻意的：阶段 3 只训练了打牌决策（数据里最多、最有意思的部分），
响应动作数据少且规则逻辑已经够好。阶段 4 RL 再把它们一起学。
"""

import torch

from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Observation, Phase
from majiang.ml.features import encode
from majiang.ml.model import DiscardNet, load

from .base import Agent
from .rule_agent import RuleAgent


class NNAgent(Agent):
    def __init__(self, model: DiscardNet | str, temperature: float = 0.0, seed: int | None = None):
        self.model = load(model) if isinstance(model, str) else model
        self.model.eval()
        self.temperature = temperature  # 0 = 取最大概率；>0 = 按概率采样（RL 需要）
        self.rule = RuleAgent()
        self.gen = torch.Generator().manual_seed(seed or 0)

    def discard_probs(self, obs: Observation) -> torch.Tensor:
        """34 维概率，不在手里的牌为 0。可视化直接用它。"""
        with torch.no_grad():
            return self.model.probs(encode(obs))[0]

    def act(self, obs: Observation) -> Action:
        by_type = {a.type: a for a in obs.legal_actions}
        # 胡牌、响应阶段、杠：交给规则
        if obs.phase != Phase.DISCARD or ActionType.TSUMO in by_type:
            return self.rule.act(obs)
        rule_action = self.rule.act(obs)
        if rule_action.type in (ActionType.ANKAN, ActionType.ADDKAN):
            return rule_action

        probs = self.discard_probs(obs)
        if self.temperature <= 0:
            t = int(probs.argmax())
        else:
            p = probs ** (1.0 / self.temperature)
            t = int(torch.multinomial(p / p.sum(), 1, generator=self.gen))
        return Action(ActionType.DISCARD, t)
