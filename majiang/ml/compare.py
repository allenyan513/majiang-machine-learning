"""配对评估：多个候选 agent 用完全相同的牌局（seed + 座位）各打 N 局，比差值。

    uv run python -m majiang.ml.compare --candidates rule,nn:models/discard.pt,nn:models/rl.pt --n 2000

每个候选坐同一个座位对 3 个规则 bot。因为牌一样，候选之间的分差方差比各自独立评估小得多，
2000 局就能分辨 ±0.03 左右的差距（独立评估要 1 万局以上）。
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from majiang.agents.base import Agent
from majiang.agents.rule_agent import RuleAgent
from majiang.evaluate import make_agent
from majiang.run import play_game


def _worker(args: tuple[str, list[int]]) -> np.ndarray:
    name, seeds = args
    out = np.zeros((len(seeds), 3))  # score, win, deal_in
    for i, s in enumerate(seeds):
        seat = s % 4
        agents: list[Agent] = [make_agent(name, s) if p == seat else RuleAgent() for p in range(4)]
        r = play_game(agents, seed=s, dealer=(s // 4) % 4)
        out[i] = (r.scores[seat], r.winner == seat, r.loser == seat)
    return out


def compare(candidates: list[str], n: int, seed: int = 0, workers: int | None = None) -> dict[str, np.ndarray]:
    workers = workers or os.cpu_count() or 1
    seeds = list(range(seed, seed + n))
    chunks = [seeds[i::workers] for i in range(workers)]
    results: dict[str, np.ndarray] = {}
    with Pool(workers) as pool:
        for name in candidates:
            parts = pool.map(_worker, [(name, c) for c in chunks if c])
            # 各 chunk 是交错切的，按 seed 顺序拼回去
            rows = np.zeros((n, 3))
            for c, part in zip([c for c in chunks if c], parts):
                rows[[s - seed for s in c]] = part
            results[name] = rows
    return results


def report(results: dict[str, np.ndarray]) -> None:
    names = list(results)
    base = names[0]
    n = len(results[base])
    print(f"{n} 局配对评估（每个候选坐同一座位 vs 3 规则 bot）\n")
    print(f"{'candidate':28} {'均分':>8} {'胜率':>7} {'放炮':>7}   {'vs ' + base:>20}")
    for nm in names:
        r = results[nm]
        line = f"{nm:28} {r[:, 0].mean():+8.3f} {r[:, 1].mean():7.1%} {r[:, 2].mean():7.1%}"
        if nm != base:
            d = r[:, 0] - results[base][:, 0]
            se = d.std(ddof=1) / np.sqrt(n)
            line += f"   {d.mean():+.3f} ± {1.96 * se:.3f} (95%)"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="rule,nn:models/discard.pt,nn:models/rl.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5_000_000)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    report(compare(a.candidates.split(","), a.n, a.seed, a.workers))


if __name__ == "__main__":
    main()
