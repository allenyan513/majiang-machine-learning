"""端到端：生成小数据 -> 训练 1 轮 -> NN agent 能合法打完对局。"""

import numpy as np

from majiang.agents.nn_agent import NNAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.ml.generate import generate
from majiang.ml.model import DiscardNet, load, save
from majiang.ml.train import train
from majiang.run import play_game


def test_generate_train_play(tmp_path):
    x, y = generate(games=12, seed=0, workers=2)
    assert x.shape[1] == 308 and len(x) == len(y) > 100
    assert y.max() < 34
    data = tmp_path / "d.npz"
    np.savez(data, x=x, y=y)
    out = tmp_path / "m.pt"
    model = train(str(data), epochs=1, out=str(out), bs=64, channels=8, blocks=2)
    assert out.exists()
    agent = NNAgent(load(str(out)), seed=1)
    for s in range(3):
        r = play_game([agent, RuleAgent(), RuleAgent(), agent], seed=s)
        assert sum(r.scores) == 0


def test_probs_only_on_tiles_in_hand():
    from majiang.engine.game import Game

    g = Game(seed=3)
    g.start()
    agent = NNAgent(DiscardNet(8, 2))
    obs = g.observe(0)
    p = agent.discard_probs(obs)
    assert abs(float(p.sum()) - 1) < 1e-5
    assert all(float(p[t]) == 0 for t in range(34) if obs.hand[t] == 0)
    assert agent.act(obs) in obs.legal_actions
