"""把 Observation 编码成神经网络输入。

两步走：
    encode_compact(obs) -> uint8[308]      存盘用，小
    expand(batch)       -> float32[B, C, 34] 训练/推理用，展开成 0/1 平面

紧凑格式布局（每段 34 个计数，视角相对自己：0=自己 1=下家 2=对家 3=上家）：
    [0:34]     自己手牌计数
    [34:68]    自己副露里的牌计数（碰=3，杠=4）
    [68:204]   四家牌河计数 4×34
    [204:306]  另外三家副露计数 3×34
    [306]      刚摸到的牌 (0–33)，没有则 255
    [307]      牌墙剩余张数

展开后的平面 (C = 22)：
    0–3    手牌 >=1,2,3,4
    4      自己副露里有这张牌
    5–12   四家牌河 >=1, >=2（每家 2 个平面）
    13–15  三家副露里有这张牌
    16–19  牌池里还没见过的张数 >=1,2,3,4（= 4 - 手牌 - 所有副露 - 所有牌河）
    20     刚摸到的牌
    21     牌墙剩余 / 100（整平面同一个值）
"""

import numpy as np
import torch

from majiang.engine.game import Observation

COMPACT_SIZE = 308
NUM_PLANES = 22
NO_TILE = 255


def _meld_counts(melds) -> np.ndarray:
    c = np.zeros(34, dtype=np.uint8)
    for m in melds:
        for t in m.tiles:
            c[t] += 1
    return c


def encode_compact(obs: Observation) -> np.ndarray:
    me = obs.player
    out = np.zeros(COMPACT_SIZE, dtype=np.uint8)
    out[0:34] = obs.hand
    out[34:68] = _meld_counts(obs.melds)
    for rel in range(4):
        p = (me + rel) % 4
        river = np.bincount(obs.rivers[p], minlength=34)[:34]
        out[68 + rel * 34 : 68 + (rel + 1) * 34] = river
    for rel in range(1, 4):
        p = (me + rel) % 4
        out[204 + (rel - 1) * 34 : 204 + rel * 34] = _meld_counts(obs.all_melds[p])
    out[306] = obs.last_draw if obs.last_draw >= 0 else NO_TILE
    out[307] = obs.wall_remaining
    return out


def expand(batch: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8[B, 308] -> float32[B, 22, 34]。全部向量化，训练时每个 batch 调一次。"""
    x = torch.as_tensor(batch, dtype=torch.int64)
    B = x.shape[0]
    hand = x[:, 0:34]
    own_meld = x[:, 34:68]
    rivers = x[:, 68:204].reshape(B, 4, 34)
    others_meld = x[:, 204:306].reshape(B, 3, 34)
    last_draw = x[:, 306]
    wall = x[:, 307]

    planes = torch.zeros(B, NUM_PLANES, 34, dtype=torch.float32)
    for k in range(4):
        planes[:, k] = (hand >= k + 1).float()
    planes[:, 4] = (own_meld > 0).float()
    for p in range(4):
        planes[:, 5 + p * 2] = (rivers[:, p] >= 1).float()
        planes[:, 6 + p * 2] = (rivers[:, p] >= 2).float()
    for p in range(3):
        planes[:, 13 + p] = (others_meld[:, p] > 0).float()
    unseen = 4 - hand - own_meld - rivers.sum(1) - others_meld.sum(1)
    for k in range(4):
        planes[:, 16 + k] = (unseen >= k + 1).float()
    has_draw = last_draw != NO_TILE
    idx = torch.arange(B)[has_draw]
    planes[idx, 20, last_draw[has_draw]] = 1.0
    planes[:, 21] = (wall.float() / 100.0).unsqueeze(1)
    return planes


def encode(obs: Observation) -> torch.Tensor:
    """单个 Observation 直接到 [1, 22, 34]，推理用。"""
    return expand(encode_compact(obs)[None, :])
