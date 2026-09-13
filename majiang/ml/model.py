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


def save(model: DiscardNet, path: str) -> None:
    torch.save({"state": model.state_dict(), "channels": model.conv[0].out_channels,
                "blocks": (len(model.conv) + 1) // 2}, path)


def load(path: str) -> DiscardNet:
    ck = torch.load(path, map_location="cpu")
    m = DiscardNet(ck["channels"], ck["blocks"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m
