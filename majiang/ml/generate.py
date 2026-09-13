"""规则 bot 自对局，记录每次打牌决策，生成监督学习数据。

    uv run python -m majiang.ml.generate --games 5000 --out data/rule_5k.npz

输出 npz：
    x: uint8[N, 308]  紧凑特征（见 features.py）
    y: uint8[N]       打出的牌 0–33
只记录 DISCARD 阶段的打牌决策（含立直宣言打的那张；自摸/杠不记）。
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from majiang.agents.base import Agent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Observation, Phase
from majiang.ml.features import encode_compact
from majiang.run import play_game


class Recorder(Agent):
    """包住任意 agent，把它的打牌决策录下来。"""

    def __init__(self, inner: Agent, xs: list, ys: list):
        self.inner, self.xs, self.ys = inner, xs, ys

    def act(self, obs: Observation) -> Action:
        a = self.inner.act(obs)
        if obs.phase == Phase.DISCARD and a.type in (ActionType.DISCARD, ActionType.RIICHI):
            self.xs.append(encode_compact(obs))
            self.ys.append(a.tile)
        return a


def _worker(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, n = args
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for s in range(start, start + n):
        agents = [Recorder(RuleAgent(), xs, ys) for _ in range(4)]
        play_game(agents, seed=s, dealer=s % 4)
    return np.stack(xs), np.array(ys, dtype=np.uint8)


def generate(games: int, seed: int = 0, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    workers = workers or os.cpu_count() or 1
    chunk = max(1, games // (workers * 4))  # 每个任务几百局，负载均衡
    tasks = [(seed + i, min(chunk, games - i)) for i in range(0, games, chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, tasks)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/rule_selfplay.npz")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    x, y = generate(args.games, args.seed, args.workers)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y)
    dt = time.time() - t0
    print(f"{args.games} 局 -> {len(y)} 条样本，用时 {dt:.0f}s，"
          f"文件 {os.path.getsize(args.out) / 1e6:.0f}MB -> {args.out}")


if __name__ == "__main__":
    main()
