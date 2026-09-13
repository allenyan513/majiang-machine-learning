"""打牌模型：输入 [B, 22, 34] 平面，输出 34 维 logits。

结构：几层一维卷积（沿牌的顺序，kernel=3 刚好看到一个顺子的范围）+ 全连接。
字牌之间没有"相邻"关系，但卷积也不会因此出错，只是那部分权重学到的东西不同。
"""

import torch
from torch import nn

from .features import NUM_PLANES


class DiscardNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(NUM_PLANES, channels, 3, padding=1), nn.ReLU()]
        for _ in range(blocks - 1):
            layers += [nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 34, 256),
            nn.ReLU(),
            nn.Linear(256, 34),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))

    @staticmethod
    def mask_logits(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """手里没有的牌不能打：平面 0 是 hand>=1。"""
        in_hand = x[:, 0] > 0
        return logits.masked_fill(~in_hand, float("-inf"))

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mask_logits(self(x), x), dim=-1)


class ActorCritic(nn.Module):
    """阶段 4 用：在 DiscardNet 的卷积干上加一个价值头（critic）。

    policy 直接复用监督学习的权重；value_head 从零开始，估计"当前局面最终能得几分"。
    对外接口和 DiscardNet 一样有 probs()，所以 NNAgent / 可视化不用改。
    """

    def __init__(self, policy: DiscardNet):
        super().__init__()
        self.policy = policy
        ch = policy.conv[0].out_channels
        self.value_head = nn.Sequential(nn.Flatten(), nn.Linear(ch * 34, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (未 mask 的 logits, value)。"""
        h = self.policy.conv(x)
        return self.policy.head(h), self.value_head(h).squeeze(-1)

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return torch.softmax(DiscardNet.mask_logits(logits, x), dim=-1)


def _arch(policy: DiscardNet) -> dict:
    return {"channels": policy.conv[0].out_channels, "blocks": (len(policy.conv) + 1) // 2}


def save(model: DiscardNet | ActorCritic, path: str) -> None:
    if isinstance(model, ActorCritic):
        torch.save({"kind": "ac", "state": model.state_dict(), **_arch(model.policy)}, path)
    else:
        torch.save({"kind": "policy", "state": model.state_dict(), **_arch(model)}, path)


def load(path: str) -> DiscardNet | ActorCritic:
    ck = torch.load(path, map_location="cpu")
    policy = DiscardNet(ck["channels"], ck["blocks"])
    m: DiscardNet | ActorCritic = ActorCritic(policy) if ck.get("kind") == "ac" else policy
    m.load_state_dict(ck["state"])
    m.eval()
    return m
