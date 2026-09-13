"""把 RL 训练日志画成曲线。

    uv run python -m majiang.ml.plot models/rl_log.csv --out models/rl_curve.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.log)))
    it = [int(r["iter"]) for r in rows]
    ev = [(int(r["iter"]), float(r["eval_score"]), float(r["eval_win"])) for r in rows if r["eval_score"]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(it, [float(r["train_score"]) for r in rows], color="#999", label="采样策略均分")
    if ev:
        ax.plot([e[0] for e in ev], [e[1] for e in ev], "o-", color="#e74c3c", label="贪心评估均分（vs 3 规则 bot）")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("均分（>0 = 超过规则 bot）"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    if ev:
        ax.plot([e[0] for e in ev], [e[2] for e in ev], "o-", color="#2980b9")
    ax.axhline(0.25, color="k", lw=0.8, ls="--"); ax.set_title("评估胜率（0.25 = 持平）")
    ax = axes[1, 0]
    ax.plot(it, [float(r["entropy"]) for r in rows], label="策略熵")
    ax.plot(it, [float(r["kl_teacher"]) for r in rows], label="对老师的 KL")
    ax.set_title("探索程度 / 偏离老师的程度"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(it, [float(r["value_loss"]) for r in rows], label="价值损失")
    ax.plot(it, [float(r["policy_loss"]) for r in rows], label="策略损失")
    ax.set_title("损失"); ax.legend(fontsize=8)
    for row in axes:
        for x in row:
            x.set_xlabel("iteration"); x.grid(alpha=0.3)
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
    fig.tight_layout()
    out = a.out or a.log.replace(".csv", ".png")
    fig.savefig(out, dpi=120)
    print("->", out)


if __name__ == "__main__":
    main()
