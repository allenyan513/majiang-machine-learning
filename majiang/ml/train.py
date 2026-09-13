"""训练打牌模型（模仿规则 bot）。

    uv run python -m majiang.ml.train --data data/rule_10k.npz --epochs 5 --out models/discard.pt

指标：val_acc = 模型最高概率的牌 == 规则 bot 打的牌 的比例（"和老师一致率"）。
"""

import argparse
import os
import time

import numpy as np
import torch
from torch import nn

from .features import expand
from .model import DiscardNet, save


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: DiscardNet, x: np.ndarray, y: np.ndarray, device: torch.device, bs: int = 4096) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = expand(x[i : i + bs]).to(device)
            yb = torch.as_tensor(y[i : i + bs], dtype=torch.int64, device=device)
            logits = model.mask_logits(model(xb), xb)
            total_loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(-1) == yb).sum().item()
    return total_loss / len(y), correct / len(y)


def train(data: str, epochs: int, out: str, bs: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
          channels: int = 64, blocks: int = 3, seed: int = 0) -> DiscardNet:
    d = np.load(data)
    x, y = d["x"], d["y"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr, y_tr, x_val, y_val = x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]
    print(f"样本 {len(y)}：训练 {len(tr_idx)}，验证 {n_val}")

    device = pick_device()
    torch.manual_seed(seed)
    model = DiscardNet(channels, blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备 {device}，参数量 {n_params / 1e3:.0f}K")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(y_tr))
        run_loss, run_correct, seen = 0.0, 0, 0
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            xb = expand(x_tr[idx]).to(device)
            yb = torch.as_tensor(y_tr[idx], dtype=torch.int64, device=device)
            logits = model.mask_logits(model(xb), xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(-1) == yb).sum().item()
            seen += len(idx)
        sched.step()
        val_loss, val_acc = evaluate(model, x_val, y_val, device)
        print(f"epoch {ep}/{epochs}  train loss {run_loss / seen:.3f} acc {run_correct / seen:.1%}  "
              f"| val loss {val_loss:.3f} acc {val_acc:.1%}  ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.cpu()
    save(model, out)
    print(f"已保存 -> {out}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rule_selfplay.npz")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="models/discard.pt")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    args = ap.parse_args()
    train(args.data, args.epochs, args.out, args.bs, args.lr, channels=args.channels, blocks=args.blocks)


if __name__ == "__main__":
    main()
