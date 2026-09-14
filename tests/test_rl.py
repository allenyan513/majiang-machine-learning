import numpy as np
import torch

from majiang.agents.nn_agent import NNAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.game import Game
from majiang.ml.model import ActorCritic, DiscardNet, load, save
from majiang.ml.rl import gae, masked, train
from majiang.run import play_game


def test_gae_single_step_episode():
    # 一步就结束：优势 = r - v
    adv, ret = gae(np.array([1.0]), np.array([0.3]), np.array([True]), gamma=1.0, lam=0.95)
    assert np.isclose(adv[0], 0.7) and np.isclose(ret[0], 1.0)


def test_gae_two_episodes_do_not_leak():
    r = np.array([0.0, 1.0, 0.0, -1.0])
    v = np.array([0.0, 0.0, 0.0, 0.0])
    done = np.array([False, True, False, True])
    adv, _ = gae(r, v, done, gamma=1.0, lam=1.0)
    assert np.allclose(adv, [1.0, 1.0, -1.0, -1.0])


def test_masked_keeps_finite_and_actor_critic_roundtrip(tmp_path):
    g = Game(seed=2)
    g.start()
    from majiang.ml.features import encode

    x = encode(g.observe(0))
    ac = ActorCritic(DiscardNet(8, 2)).eval()
    with torch.no_grad():
        logits, v = ac(x)
    assert logits.shape == (1, 34) and v.shape == (1,)
    m = masked(logits, x)
    assert torch.isfinite(m).all()
    with torch.no_grad():
        p = ac.probs(x)
    assert abs(float(p.sum()) - 1) < 1e-5
    path = tmp_path / "ac.pt"
    save(ac, str(path))
    loaded = load(str(path))
    assert isinstance(loaded, ActorCritic)
    with torch.no_grad():
        assert torch.allclose(loaded.probs(x), p)


def test_nn_agent_accepts_actor_critic():
    agent = NNAgent(ActorCritic(DiscardNet(8, 2)), seed=0)
    r = play_game([agent, RuleAgent(), RuleAgent(), RuleAgent()], seed=4)
    assert sum(r.scores) == 0


def test_rl_smoke(tmp_path):
    init = tmp_path / "init.pt"
    save(DiscardNet(8, 2), str(init))
    out = tmp_path / "rl.pt"
    train(str(init), str(out), iters=2, games=8, eval_every=2, eval_games=8, workers=2, value_warmup=1,
          log_csv=str(tmp_path / "log.csv"))
    assert out.exists() and (tmp_path / "rl_last.pt").exists()
    rows = open(tmp_path / "log.csv").read().strip().splitlines()
    assert len(rows) == 4  # header + iter 0（起点评估）+ 2 iters
    assert isinstance(load(str(out)), ActorCritic)


def test_summarize_probes():
    from majiang.ml.rl import summarize_probes

    probes = [
        {"turn": 8, "danger": 0.05, "dangerous": True, "threat": 0.6, "high_threat": True, "fold": True, "shanten": 2, "entropy": 0.5},
        {"turn": 9, "danger": 0.0, "dangerous": False, "threat": 0.6, "high_threat": True, "fold": False, "shanten": 1, "entropy": 0.3},
        {"turn": 8, "danger": 0.01, "dangerous": False, "threat": 0.1, "high_threat": False, "fold": False, "shanten": 0, "entropy": 0.1},
    ]
    s = summarize_probes(probes)
    assert s["danger_rate"] == 1 / 3 and s["fold_rate"] == 0.5 and s["shanten8"] == 1.0
    assert summarize_probes([])["fold_rate"] == 0.0
