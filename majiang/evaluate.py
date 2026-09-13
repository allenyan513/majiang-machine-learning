"""让不同 agent 混战 N 局，统计每类 agent 的胜率 / 放炮率 / 平均得分。

    uv run python -m majiang.evaluate --agents rule,random,random,random --n 1000

座位每局轮转，避免庄家位置带来的偏差。
"""

import argparse
import time
from collections import defaultdict

from majiang.agents.base import Agent
from majiang.agents.random_agent import RandomAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.run import play_game

AGENT_FACTORIES = {
    "random": lambda seed: RandomAgent(seed=seed),
    "rule": lambda seed: RuleAgent(),
    "rule_v1": lambda seed: RuleAgent(defense="v1"),
}

_MODEL_CACHE: dict[str, object] = {}


def make_agent(name: str, seed: int) -> Agent:
    """支持 'random' / 'rule' / 'nn:models/discard.pt'。"""
    if name.startswith("nn:"):
        from majiang.agents.nn_agent import NNAgent

        path = name[3:]
        if path not in _MODEL_CACHE:
            from majiang.ml.model import load

            _MODEL_CACHE[path] = load(path)
        return NNAgent(_MODEL_CACHE[path], seed=seed)  # type: ignore[arg-type]
    return AGENT_FACTORIES[name](seed)


def evaluate(names: list[str], n: int, base_seed: int = 0) -> dict:
    stats: dict[str, dict] = defaultdict(lambda: {"games": 0, "wins": 0, "tsumo": 0, "deal_in": 0, "score": 0})
    draws = 0
    t0 = time.time()
    for g in range(n):
        shift = g % 4
        seat_names = names[shift:] + names[:shift]  # 轮转座位
        agents = [make_agent(nm, base_seed + g * 4 + i) for i, nm in enumerate(seat_names)]
        r = play_game(agents, seed=base_seed + g, dealer=g % 4)
        if r.winner is None:
            draws += 1
        for seat, nm in enumerate(seat_names):
            s = stats[nm]
            s["games"] += 1
            s["score"] += r.scores[seat]
            if r.winner == seat:
                s["wins"] += 1
                s["tsumo"] += r.is_tsumo
            if r.loser == seat:
                s["deal_in"] += 1
    dt = time.time() - t0
    return {"stats": dict(stats), "draws": draws, "n": n, "seconds": dt}


def print_report(rep: dict) -> None:
    n = rep["n"]
    print(f"{n} 局，用时 {rep['seconds']:.1f}s，流局 {rep['draws']} ({rep['draws'] / n:.1%})")
    print(f"{'agent':8} {'局数':>6} {'胜率':>7} {'自摸率':>7} {'放炮率':>7} {'均分':>7}")
    for nm, s in rep["stats"].items():
        g = s["games"]
        print(
            f"{nm:8} {g:6d} {s['wins'] / g:7.1%} {s['tsumo'] / g:7.1%} "
            f"{s['deal_in'] / g:7.1%} {s['score'] / g:+7.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="rule,random,random,random")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = args.agents.split(",")
    assert len(names) == 4
    print_report(evaluate(names, args.n, args.seed))


if __name__ == "__main__":
    main()
