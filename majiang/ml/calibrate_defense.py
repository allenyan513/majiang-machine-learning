"""用上帝视角的规则自对局，统计防守用的两张概率表，生成 agents/defense_tables.py。

    uv run python -m majiang.ml.calibrate_defense --games 2000

表 1  P_TENPAI[melds][discard_bucket]        对手听牌的概率
表 2  P_WAIT[category][visible][genbutsu]    对手已听牌时，某张牌在其听牌集合里的概率
      category: 0 字牌 1 幺九 2 二八 3 中张(3–7)
      visible:  防守方视角能看到的该牌张数（四家牌河 + 四家副露 + 自己手牌），封顶 3
      genbutsu: 该对手是否打过这张牌
表 3  VALUE[shanten][eff_bucket][wall_bucket][threat_bucket]
      打出一张牌之后处于该状态时，自己的平均终局分数（策略评估；用当前规则 bot 自对局统计）
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from majiang.agents import defense
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import ActionType
from majiang.engine.game import Game, Phase
from majiang.engine.hand import MeldType
from majiang.engine.shanten import discard_options, shanten
from majiang.engine.tile import is_honor, number
from majiang.engine.win import waiting_tiles

BUCKETS = [0, 4, 8, 12, 16]  # 已打张数分桶下界
EFF_BUCKETS = [0, 1, 2, 3, 4, 6, 8, 12, 20, 30]  # 进张数分桶下界
WALL_BUCKETS = [0, 11, 31, 61]                   # 牌墙剩余分桶下界
THREAT_BUCKETS = [0.0, 0.2, 0.5]                 # 最大威胁分桶下界
MIN_CELL = 30                                    # 样本少于它的格子回退到边际均值


def category(t: int) -> int:
    if is_honor(t):
        return 0
    n = number(t)
    return 1 if n in (1, 9) else 2 if n in (2, 8) else 3


def bucket(n_discards: int) -> int:
    return max(i for i, b in enumerate(BUCKETS) if n_discards >= b)


def _b(x: float, edges: list) -> int:
    return max(i for i, e in enumerate(edges) if x >= e)


def _worker(seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tenpai = np.zeros((5, len(BUCKETS), 2))  # [melds][bucket][is_tenpai]
    wait = np.zeros((4, 4, 2, 2))            # [cat][visible][genbutsu][in_wait]
    vshape = (5, len(EFF_BUCKETS), len(WALL_BUCKETS), len(THREAT_BUCKETS))
    vsum, vcnt = np.zeros(vshape), np.zeros(vshape)
    for s in seeds:
        agents = [RuleAgent() for _ in range(4)]
        g = Game(seed=s, dealer=s % 4)
        g.start()
        pending: list[tuple[int, tuple[int, int, int, int]]] = []  # (player, 状态格子)
        while not g.finished:
            if g.phase == Phase.DISCARD:
                me = g.current
                # 防守方视角的可见张数
                vis = np.zeros(34, dtype=int)
                for r in g.rivers:
                    for t in r:
                        vis[t] += 1
                for h in g.hands:
                    for m in h.melds:
                        vis[m.tile] += 3 if m.type == MeldType.PON else 4
                vis += np.array(g.hands[me].counts)
                for q in range(4):
                    if q == me:
                        continue
                    hq = g.hands[q]
                    is_tp = shanten(hq.counts, len(hq.melds)) == 0
                    tenpai[len(hq.melds), bucket(len(g.rivers[q])), int(is_tp)] += 1
                    if is_tp:
                        waits = set(waiting_tiles(hq.counts))
                        river = set(g.rivers[q])
                        for t in range(34):
                            wait[category(t), min(int(vis[t]), 3), int(t in river), int(t in waits)] += 1
                # 表 3：记录打牌后的状态，终局回填分数
                obs = g.observe(me)
                a = agents[me].act(obs)
                if a.type == ActionType.DISCARD:
                    opts = {d: (sh, n) for d, sh, n in discard_options(obs.hand, len(obs.melds), RuleAgent._visible(obs))}
                    sh, n = opts[a.tile]
                    cell = (min(max(sh, 0), 4), _b(n, EFF_BUCKETS), _b(obs.wall_remaining, WALL_BUCKETS),
                            _b(defense.max_threat(obs), THREAT_BUCKETS))
                    pending.append((me, cell))
                g.step(me, a)
                continue
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        assert g.result is not None
        for p, cell in pending:
            vsum[cell] += g.result.scores[p]
            vcnt[cell] += 1
    return tenpai, wait, vsum, vcnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", default="majiang/agents/defense_tables.py")
    a = ap.parse_args()
    workers = os.cpu_count() or 1
    seeds = list(range(3_000_000, 3_000_000 + a.games))
    with Pool(workers) as pool:
        parts = pool.map(_worker, [seeds[i::workers] for i in range(workers)])
    tenpai = sum(p[0] for p in parts)
    wait = sum(p[1] for p in parts)
    vsum = sum(p[2] for p in parts)
    vcnt = sum(p[3] for p in parts)

    p_tenpai = (tenpai[..., 1] + 1) / (tenpai.sum(-1) + 2)  # 加 1 平滑
    p_tenpai[4, :] = 1.0  # 4 副露手里只剩 1 张，必然单钓听牌；数据里几乎没有，手动设定
    p_wait = (wait[..., 1] + 1) / (wait.sum(-1) + 2)

    print("P(听牌 | 副露数, 已打张数桶)   桶下界:", BUCKETS)
    for m in range(5):
        print(f"  {m} 副露: " + "  ".join(f"{p_tenpai[m, b]:.2f}({int(tenpai[m, b].sum())})" for b in range(len(BUCKETS))))
    print("\nP(在听 | 已听牌, 类别, 可见张数, 是否打过)   行=类别 字/幺九/二八/中张，列=可见 0..3；括号里是打过的")
    for c, nm in enumerate(["字牌", "幺九", "二八", "中张"]):
        print(f"  {nm}: " + "  ".join(f"{p_wait[c, v, 0]:.3f}({p_wait[c, v, 1]:.3f})" for v in range(4)))

    # 表 3：样本少的格子逐级回退（去掉威胁维 -> 去掉牌墙维 -> 只看向听）
    value = np.zeros(vsum.shape)
    for idx in np.ndindex(*vsum.shape):
        sh, e, w, th = idx
        for sl in ((sh, e, w, th), (sh, e, w, slice(None)), (sh, e, slice(None), slice(None)),
                   (sh, slice(None), slice(None), slice(None))):
            c = vcnt[sl].sum()
            if c >= MIN_CELL:
                value[idx] = vsum[sl].sum() / c
                break
    print("\nV(向听, 进张桶) 在牌墙/威胁上取平均：  进张桶下界", EFF_BUCKETS)
    for sh in range(5):
        row = []
        for e in range(len(EFF_BUCKETS)):
            c = vcnt[sh, e].sum()
            row.append(f"{vsum[sh, e].sum() / c:+.2f}" if c >= MIN_CELL else "  -  ")
        print(f"  {sh} 向听: " + " ".join(row))

    with open(a.out, "w") as f:
        f.write('"""由 ml/calibrate_defense.py 自动生成（%d 局规则自对局），不要手改。"""\n\n' % a.games)
        f.write(f"BUCKETS = {BUCKETS}\n\n")
        f.write("# P_TENPAI[melds][bucket]\nP_TENPAI = [\n")
        for m in range(5):
            f.write("    [" + ", ".join(f"{x:.4f}" for x in p_tenpai[m]) + "],\n")
        f.write("]\n\n# P_WAIT[category][visible][genbutsu]\nP_WAIT = [\n")
        for c in range(4):
            f.write("    [" + ", ".join("[" + ", ".join(f"{x:.4f}" for x in p_wait[c, v]) + "]" for v in range(4)) + "],\n")
        f.write("]\n\n")
        f.write(f"EFF_BUCKETS = {EFF_BUCKETS}\nWALL_BUCKETS = {WALL_BUCKETS}\nTHREAT_BUCKETS = {THREAT_BUCKETS}\n\n")
        f.write("# VALUE[shanten][eff_bucket][wall_bucket][threat_bucket]\nVALUE = [\n")
        for sh in range(5):
            f.write("    [\n")
            for e in range(len(EFF_BUCKETS)):
                f.write("        [" + ", ".join("[" + ", ".join(f"{x:+.4f}" for x in value[sh, e, w]) + "]"
                                                for w in range(len(WALL_BUCKETS))) + "],\n")
            f.write("    ],\n")
        f.write("]\n")
    print("->", a.out)


if __name__ == "__main__":
    main()
