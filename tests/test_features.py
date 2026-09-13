import numpy as np
import torch

from majiang.engine.game import Game
from majiang.ml.features import COMPACT_SIZE, NUM_PLANES, encode, encode_compact, expand


def test_shapes_and_unseen_consistency():
    g = Game(seed=5)
    g.start()
    obs = g.observe(0)
    c = encode_compact(obs)
    assert c.shape == (COMPACT_SIZE,) and c.dtype == np.uint8
    x = encode(obs)
    assert x.shape == (1, NUM_PLANES, 34)
    # 开局时自己 14 张，其他都没见过：unseen>=1 应该对所有牌成立
    assert x[0, 16].sum() == 34
    # 手牌平面：>=1 的张数 = 手牌种类数
    assert x[0, 0].sum() == sum(1 for n in obs.hand if n > 0)
    # 刚摸的牌 one-hot
    assert x[0, 20].sum() == 1 and x[0, 20, obs.last_draw] == 1
    assert torch.isclose(x[0, 21, 0], torch.tensor(obs.wall_remaining / 100.0))


def test_relative_perspective():
    g = Game(seed=5)
    g.start()
    g.rivers[1] = [3, 3]  # 玩家1 打过两张 4m
    obs0 = g.observe(0)
    obs2 = g.observe(2)
    c0, c2 = encode_compact(obs0), encode_compact(obs2)
    # 对玩家0 来说玩家1 是下家 (rel=1)；对玩家2 来说是上家 (rel=3)
    assert c0[68 + 34 + 3] == 2 and c2[68 + 3 * 34 + 3] == 2


def test_expand_batch_matches_single():
    g = Game(seed=9)
    g.start()
    rows = np.stack([encode_compact(g.observe(p)) for p in range(4)])
    batch = expand(rows)
    for p in range(4):
        assert torch.equal(batch[p], encode(g.observe(p))[0])
