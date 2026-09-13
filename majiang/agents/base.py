from abc import ABC, abstractmethod

from majiang.engine.actions import Action
from majiang.engine.game import Observation


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """必须从 obs.legal_actions 里选一个。"""
