"""阶段 4：PPO 微调打牌策略，目标是超过规则 bot。

    uv run python -m majiang.ml.rl --init models/discard.pt --iters 100 --out models/rl.pt

每轮 (iteration)：
    1. 多进程并行打 N 局：每局随机 2 个座位是学习者（按概率采样打牌），其余是规则 bot
       记录学习者每次打牌的 (特征, 动作, log 概率, 价值估计, 奖励)
    2. 奖励 = 终局分数 + 势函数塑形（向听数减少 -> 小正奖励；Ng et al. 1999，不改变最优策略）
    3. GAE 算优势，PPO 剪切目标更新策略 + 价值头，外加对老师（监督模型）的 KL 惩罚
    4. 每隔几轮用贪心策略对 3 个规则 bot 评估，记录到 CSV（画曲线用）。评估时顺手记**行为探针**：
       危险牌打出率、弃听率、第 8 巡向听、决策熵——分数曲线是结果，这些是原因
    5. 每次评估存一个 checkpoint（<out>_ckpt/iter_NNN.pt），并录一局带学习信号的轨迹
       （每步的 V、塑形/终局奖励、优势、π(a)，给 web/train 的回放页用）

只学打牌；碰/杠/胡仍由规则决定（和 NNAgent 一致）。
"""

import argparse
import csv
import os
import random
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

import json

from majiang.agents import defense
from majiang.agents.base import Agent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Game, Observation, Phase
from majiang.engine.shanten import shanten
from majiang.ml.features import encode_compact, expand
from majiang.ml.model import ActorCritic, DiscardNet, load, save

NEG = -1e9  # RL 里用大负数而不是 -inf 做 mask，这样 KL / 熵里 0 * (-1e9) 仍是 0，不会出 nan


def masked(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~(x[:, 0] > 0), NEG)


# ============================================================ 采样（子进程）
_W: dict = {}  # 子进程全局缓存：模型 / 权重版本


def _init_worker() -> None:
    torch.set_num_threads(1)


def _get_model(path: str, version: int) -> ActorCritic:
    if _W.get("version") != version:
        _W["model"] = load(path)
        _W["version"] = version
    return _W["model"]


class Trajectory:
    def __init__(self) -> None:
        self.x: list[np.ndarray] = []
        self.a: list[int] = []
        self.logp: list[float] = []
        self.v: list[float] = []
        self.r: list[float] = []


class LearnerAgent(Agent):
    """采样版 NN agent：按概率打牌并记录轨迹。"""

    DANGER_THRESHOLD = 0.04   # 危险度超过它算"危险牌"（第 13 章：这一档实际放炮率 7%）
    THREAT_THRESHOLD = 0.5    # 对手听牌概率超过它算"高威胁局面"

    def __init__(self, model: ActorCritic, traj: Trajectory, rng: random.Random, shaping: float, greedy: bool = False,
                 probes: list | None = None, snapshots: list | None = None):
        self.model, self.traj, self.rng, self.shaping, self.greedy = model, traj, rng, shaping, greedy
        self.rule = RuleAgent()
        self.prev_phi: float | None = None
        self.probes = probes        # 评估时：每次打牌记一条行为探针
        self.snapshots = snapshots  # 录轨迹时：每次打牌记概率分布等

    def act(self, obs: Observation) -> Action:
        by_type = {a.type: a for a in obs.legal_actions}
        if obs.phase != Phase.DISCARD or ActionType.TSUMO in by_type:
            return self.rule.act(obs)
        rule_action = self.rule.act(obs)
        if rule_action.type in (ActionType.ANKAN, ActionType.ADDKAN):
            return rule_action

        x = encode_compact(obs)
        with torch.no_grad():
            logits, v = self.model(expand(x[None, :]))
            logp_all = torch.log_softmax(masked(logits, torch.as_tensor(expand(x[None, :]))), -1)[0]
        if self.greedy:
            a = int(logp_all.argmax())
        else:
            a = int(torch.multinomial(logp_all.exp(), 1))
        # 势函数：打完这张牌后的 -向听数
        counts = list(obs.hand)
        counts[a] -= 1
        s_after = shanten(counts, len(obs.melds))
        phi = -s_after
        r = 0.0 if self.prev_phi is None else self.shaping * (phi - self.prev_phi)
        self.prev_phi = phi
        t = self.traj
        t.x.append(x)
        t.a.append(a)
        t.logp.append(float(logp_all[a]))
        t.v.append(float(v[0]))
        t.r.append(r)
        if self.probes is not None or self.snapshots is not None:
            probs = logp_all.exp()
            ent = float(-(probs * logp_all.clamp(min=-50)).sum())
            danger = defense.danger_map(obs)
            threat = defense.max_threat(obs)
            s_best = shanten(obs.hand, len(obs.melds))  # 14 张的向听 = 最优打法后的向听
            probe = {
                "turn": len(t.a), "danger": danger[a], "dangerous": danger[a] >= self.DANGER_THRESHOLD,
                "threat": threat, "high_threat": threat >= self.THREAT_THRESHOLD,
                "fold": s_after > s_best, "shanten": s_after, "entropy": ent,
            }
            if self.probes is not None:
                self.probes.append(probe)
            if self.snapshots is not None:
                self.snapshots.append({**probe, "action": a, "v": float(v[0]), "r_shaping": r,
                                       "p_action": float(probs[a]),
                                       "probs": {int(i): round(float(probs[i]), 4) for i in range(34) if probs[i] > 1e-4},
                                       "danger_map": [round(d, 4) for d in danger]})
        return Action(ActionType.DISCARD, a)


def _rollout(args: tuple) -> dict:
    """打一批局，返回拼好的轨迹数组。"""
    path, version, seeds, n_learners, shaping = args
    model = _get_model(path, version)
    X, A, LOGP, V, R, DONE = [], [], [], [], [], []
    scores: list[int] = []
    for seed in seeds:
        rng = random.Random(seed)
        learner_seats = rng.sample(range(4), n_learners)
        trajs = {p: Trajectory() for p in learner_seats}
        agents: list[Agent] = [
            LearnerAgent(model, trajs[p], rng, shaping) if p in learner_seats else RuleAgent() for p in range(4)
        ]
        g = Game(seed=seed, dealer=seed % 4)
        g.start()
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        assert g.result is not None
        for p, t in trajs.items():
            if not t.a:
                continue
            t.r[-1] += g.result.scores[p]  # 终局奖励挂在最后一次决策上
            scores.append(g.result.scores[p])
            X.extend(t.x); A.extend(t.a); LOGP.extend(t.logp); V.extend(t.v); R.extend(t.r)
            DONE.extend([False] * (len(t.a) - 1) + [True])
    return {
        "x": np.stack(X), "a": np.array(A), "logp": np.array(LOGP, dtype=np.float32),
        "v": np.array(V, dtype=np.float32), "r": np.array(R, dtype=np.float32), "done": np.array(DONE),
        "scores": np.array(scores),
    }


def _eval_games(args: tuple) -> dict:
    """贪心学习者坐 1 个座位对 3 个规则 bot，同时记行为探针。"""
    path, version, seeds = args
    model = _get_model(path, version)
    scores, wins, deal_ins = [], 0, 0
    probes: list[dict] = []
    for seed in seeds:
        seat = seed % 4
        agents: list[Agent] = [
            LearnerAgent(model, Trajectory(), random.Random(seed), 0.0, greedy=True, probes=probes)
            if p == seat else RuleAgent()
            for p in range(4)
        ]
        g = Game(seed=seed, dealer=(seed // 4) % 4)
        g.start()
        while not g.finished:
            for p in g.players_to_act():
                g.step(p, agents[p].act(g.observe(p)))
        assert g.result is not None
        scores.append(g.result.scores[seat])
        wins += g.result.winner == seat
        deal_ins += g.result.loser == seat
    return {"scores": np.array(scores), "wins": wins, "deal_ins": deal_ins, "probes": probes}


def summarize_probes(probes: list[dict]) -> dict:
    """把每次打牌的探针聚合成几条行为指标。"""
    if not probes:
        return {"danger_rate": 0.0, "mean_danger": 0.0, "fold_rate": 0.0, "shanten8": 0.0, "entropy": 0.0}
    high = [p for p in probes if p["high_threat"]]
    t8 = [p for p in probes if p["turn"] == 8]
    return {
        "danger_rate": sum(p["dangerous"] for p in probes) / len(probes),
        "mean_danger": sum(p["danger"] for p in probes) / len(probes),
        "fold_rate": (sum(p["fold"] for p in high) / len(high)) if high else 0.0,
        "shanten8": (sum(p["shanten"] for p in t8) / len(t8)) if t8 else 0.0,
        "entropy": sum(p["entropy"] for p in probes) / len(probes),
    }


def _trace_game(args: tuple) -> dict:
    """录一局：贪心学习者坐座位 0，每次打牌存上帝视角快照 + 学习信号。终局后补 GAE 优势。"""
    path, version, seed, shaping, gamma, lam = args
    model = _get_model(path, version)
    traj, snaps = Trajectory(), []
    agents: list[Agent] = [LearnerAgent(model, traj, random.Random(seed), shaping, greedy=True, snapshots=snaps)]
    agents += [RuleAgent() for _ in range(3)]
    g = Game(seed=seed, dealer=0)
    g.start()
    steps: list[dict] = []
    while not g.finished:
        for p in g.players_to_act():
            if p == 0 and g.phase == Phase.DISCARD:
                board = {
                    "hands": [h.tiles() for h in g.hands],
                    "melds": [[{"type": m.type.name, "tile": m.tile} for m in h.melds] for h in g.hands],
                    "rivers": [list(r) for r in g.rivers],
                    "wall": len(g.wall), "last_draw": g.last_draw,
                }
                n_before = len(snaps)
                a = agents[p].act(g.observe(p))
                if len(snaps) > n_before:
                    steps.append({"board": board, **snaps[-1]})
                g.step(p, a)
            else:
                g.step(p, agents[p].act(g.observe(p)))
    assert g.result is not None
    if traj.a:
        traj.r[-1] += g.result.scores[0] / 1.0
        r = np.array(traj.r, dtype=np.float32)
        v = np.array(traj.v, dtype=np.float32)
        done = np.array([False] * (len(traj.a) - 1) + [True])
        adv, ret = gae(r, v, done, gamma, lam)
        for i, st in enumerate(steps):
            st["r_terminal"] = float(g.result.scores[0]) if i == len(steps) - 1 else 0.0
            st["adv"] = float(adv[i])
            st["ret"] = float(ret[i])
    return {"seed": seed, "result": str(g.result), "scores": g.result.scores, "winner": g.result.winner,
            "loser": g.result.loser, "steps": steps}


# ============================================================ 训练（主进程）
def gae(r: np.ndarray, v: np.ndarray, done: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(r)
    last = 0.0
    for t in range(len(r) - 1, -1, -1):
        next_v = 0.0 if done[t] else v[t + 1]
        delta = r[t] + gamma * next_v - v[t]
        last = delta + gamma * lam * (0.0 if done[t] else last)
        adv[t] = last
    return adv, adv + v


def ppo_update(model: ActorCritic, teacher: DiscardNet, opt: torch.optim.Optimizer, batch: dict, device: torch.device,
               epochs: int, bs: int, clip: float, vf_coef: float, ent_coef: float, kl_coef: float,
               policy_coef: float) -> dict:
    x_all = batch["x"]
    a_all = torch.as_tensor(batch["a"], dtype=torch.int64)
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.as_tensor(batch["ret"])
    n = len(a_all)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl_teacher": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = expand(x_all[idx.numpy()]).to(device)
            logits, v = model(xb)
            logits = masked(logits, xb)
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, a_all[idx].to(device)[:, None])[:, 0]
            ratio = torch.exp(logp - old_logp[idx].to(device))
            a_b = adv[idx].to(device)
            pg = -torch.min(ratio * a_b, torch.clamp(ratio, 1 - clip, 1 + clip) * a_b).mean()
            vl = ((v - ret[idx].to(device)) ** 2).mean()
            p = logp_all.exp()
            ent = -(p * logp_all).sum(-1).mean()
            with torch.no_grad():
                t_logp = torch.log_softmax(masked(teacher(xb), xb), -1)
            kl_t = (p * (logp_all - t_logp)).sum(-1).mean()
            loss = policy_coef * (pg - ent_coef * ent + kl_coef * kl_t) + vf_coef * vl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"] += pg.item(); stats["value_loss"] += vl.item(); stats["entropy"] += ent.item()
                stats["kl_teacher"] += kl_t.item()
                stats["approx_kl"] += (old_logp[idx].to(device) - logp).mean().item()
                stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
            steps += 1
    return {k: v / max(steps, 1) for k, v in stats.items()}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(init: str, out: str, iters: int, games: int, n_learners: int = 2, shaping: float = 0.1,
          lr: float = 1e-4, epochs: int = 3, bs: int = 1024, clip: float = 0.2, vf_coef: float = 0.5,
          ent_coef: float = 0.01, kl_coef: float = 0.05, gamma: float = 1.0, lam: float = 0.95,
          value_warmup: int = 2, eval_every: int = 5, eval_games: int = 200, workers: int | None = None,
          seed: int = 0, log_csv: str | None = None) -> None:
    workers = workers or os.cpu_count() or 1
    device = pick_device()
    init_model = load(init)
    if isinstance(init_model, ActorCritic):
        model = init_model
        teacher = load(init).policy  # type: ignore[union-attr]
    else:
        model = ActorCritic(init_model)
        teacher = load(init)  # type: ignore[assignment]
    model.to(device).train()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tmpdir = tempfile.mkdtemp(prefix="majiang_rl_")
    weights_path = os.path.join(tmpdir, "w.pt")
    log_csv = log_csv or os.path.splitext(out)[0] + "_log.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fields = ["iter", "samples", "train_score", "eval_score", "eval_win", "eval_deal_in",
              "eval_danger_rate", "eval_mean_danger", "eval_fold_rate", "eval_shanten8", "eval_entropy",
              "policy_loss", "value_loss", "entropy", "kl_teacher", "approx_kl", "clipfrac", "seconds"]
    ckpt_dir = os.path.splitext(out)[0] + "_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    def publish(version: int) -> None:
        model.cpu()
        save(model, weights_path)
        model.to(device)

    def evaluate(version: int, pool: Pool) -> dict:
        seeds = list(range(10_000_000 + seed, 10_000_000 + seed + eval_games))
        chunks = [seeds[i::workers] for i in range(workers)]
        parts = pool.map(_eval_games, [(weights_path, version, c) for c in chunks if c])
        sc = np.concatenate([p["scores"] for p in parts])
        probes = [q for p in parts for q in p["probes"]]
        return {"score": float(sc.mean()), "win": sum(p["wins"] for p in parts) / len(sc),
                "deal_in": sum(p["deal_ins"] for p in parts) / len(sc), **summarize_probes(probes)}

    def checkpoint(it: int, pool: Pool) -> None:
        """存权重 + 录一局轨迹（固定 seed，好跨版本对比）。"""
        model.cpu(); save(model, os.path.join(ckpt_dir, f"iter_{it:03d}.pt")); model.to(device)
        trace = pool.apply(_trace_game, ((weights_path, it, 20_000_000 + seed, shaping, gamma, lam),))
        with open(os.path.join(ckpt_dir, f"trace_{it:03d}.json"), "w") as f:
            json.dump(trace, f)

    best = -1e9
    game_seed = seed * 1_000_000
    print(f"设备 {device}，{workers} 进程，每轮 {games} 局 × {n_learners} 学习者座位；日志 -> {log_csv}  checkpoint -> {ckpt_dir}")
    with Pool(workers, initializer=_init_worker) as pool:
        publish(0)
        ev0 = evaluate(0, pool)
        checkpoint(0, pool)
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(
                {"iter": 0, "samples": 0, "train_score": "", "eval_score": ev0["score"], "eval_win": ev0["win"],
                 "eval_deal_in": ev0["deal_in"], "eval_danger_rate": ev0["danger_rate"], "eval_mean_danger": ev0["mean_danger"],
                 "eval_fold_rate": ev0["fold_rate"], "eval_shanten8": ev0["shanten8"], "eval_entropy": ev0["entropy"]})
        print(f"iter   0  起点（监督模型）评估 均分 {ev0['score']:+.3f} 胜率 {ev0['win']:.1%} 放炮 {ev0['deal_in']:.1%} "
              f"危险牌 {ev0['danger_rate']:.1%} 弃听 {ev0['fold_rate']:.1%}", flush=True)
        for it in range(1, iters + 1):
            t0 = time.time()
            publish(it)
            seeds = list(range(game_seed, game_seed + games))
            game_seed += games
            chunks = [seeds[i::workers] for i in range(workers)]
            parts = pool.map(_rollout, [(weights_path, it, c, n_learners, shaping) for c in chunks if c])
            batch = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "a", "logp", "v", "r", "done", "scores")}
            batch["adv"], batch["ret"] = gae(batch["r"], batch["v"], batch["done"], gamma, lam)
            stats = ppo_update(model, teacher, opt, batch, device, epochs, bs, clip, vf_coef, ent_coef, kl_coef,
                               policy_coef=0.0 if it <= value_warmup else 1.0)
            row = {"iter": it, "samples": len(batch["a"]), "train_score": float(batch["scores"].mean()),
                   "eval_score": "", "eval_win": "", "eval_deal_in": "", "eval_danger_rate": "", "eval_mean_danger": "",
                   "eval_fold_rate": "", "eval_shanten8": "", "eval_entropy": "", **stats}
            msg = (f"iter {it:3d}  样本 {row['samples']:6d}  采样均分 {row['train_score']:+.3f}  "
                   f"pg {stats['policy_loss']:+.3f} vf {stats['value_loss']:.3f} ent {stats['entropy']:.2f} "
                   f"kl_t {stats['kl_teacher']:.3f}")
            if it % eval_every == 0 or it == iters:
                publish(it + 0)  # 评估用更新后的权重
                ev = evaluate(it, pool)
                checkpoint(it, pool)
                row.update(eval_score=ev["score"], eval_win=ev["win"], eval_deal_in=ev["deal_in"],
                           eval_danger_rate=ev["danger_rate"], eval_mean_danger=ev["mean_danger"],
                           eval_fold_rate=ev["fold_rate"], eval_shanten8=ev["shanten8"], eval_entropy=ev["entropy"])
                msg += (f"  | 评估 均分 {ev['score']:+.3f} 胜率 {ev['win']:.1%} 放炮 {ev['deal_in']:.1%} "
                        f"危险牌 {ev['danger_rate']:.1%} 弃听 {ev['fold_rate']:.1%} 向听@8 {ev['shanten8']:.2f}")
                if ev["score"] > best:
                    best = ev["score"]
                    model.cpu(); save(model, out); model.to(device)
                    msg += "  *best*"
            row["seconds"] = round(time.time() - t0, 1)
            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(msg + f"  ({row['seconds']}s)", flush=True)
    model.cpu()
    save(model, os.path.splitext(out)[0] + "_last.pt")
    print(f"完成。最佳评估均分 {best:+.3f} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/discard.pt")
    ap.add_argument("--out", default="models/rl.pt")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--learners", type=int, default=2)
    ap.add_argument("--shaping", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.init, a.out, a.iters, a.games, a.learners, a.shaping, a.lr, kl_coef=a.kl, ent_coef=a.ent,
          eval_every=a.eval_every, eval_games=a.eval_games, workers=a.workers, seed=a.seed)


if __name__ == "__main__":
    main()
