"""终端观战 / 批量统计。

    uv run python -m majiang.cli            # 看一局
    uv run python -m majiang.cli --n 1000   # 跑 1000 局统计
"""

import argparse
import time
from collections import Counter

from majiang.agents.random_agent import RandomAgent
from majiang.run import play_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="对局数；1 时打印过程")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n == 1:
        agents = [RandomAgent(seed=args.seed * 4 + i) for i in range(4)]
        play_game(agents, seed=args.seed, verbose=True)
        return

    t0 = time.time()
    wins = Counter()
    draws = 0
    tsumo = 0
    for s in range(args.n):
        agents = [RandomAgent(seed=s * 4 + i) for i in range(4)]
        r = play_game(agents, seed=s, dealer=s % 4)
        if r.winner is None:
            draws += 1
        else:
            wins[r.winner] += 1
            tsumo += r.is_tsumo
    dt = time.time() - t0
    print(f"{args.n} 局用时 {dt:.2f}s ({args.n / dt:.0f} 局/秒)")
    print(f"流局 {draws} ({draws / args.n:.1%})   胡牌 {args.n - draws}  其中自摸 {tsumo}")
    print("各家胜局:", dict(sorted(wins.items())))


if __name__ == "__main__":
    main()
