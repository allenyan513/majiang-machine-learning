"""番符 -> 点数（天凤规则）。

基本分 = 符 × 2^(番+2)，封顶：
    满贯 2000（5 番，或基本分已达 2000）、跳满 3000（6–7）、倍满 4000（8–10）、
    三倍满 6000（11–12）、役满 8000（13+，或役满役 × 倍数）
支付（向上取整到 100）：
    荣和：闲家 4×，庄家 6×，点炮者付；每本场 +300
    自摸：闲家和 = 庄付 2×、闲各 1×；庄家和 = 三家各 2×；每本场每家 +100
"""

from dataclasses import dataclass


def _ceil100(x: int) -> int:
    return -(-x // 100) * 100


def base_points(han: int, fu: int, yakuman: int = 0) -> tuple[int, str]:
    """返回 (基本分, 档位名)。yakuman > 0 时 han 忽略。"""
    if yakuman > 0:
        return 8000 * yakuman, ("役满" if yakuman == 1 else f"{yakuman}倍役满")
    if han >= 13:
        return 8000, "累计役满"
    if han >= 11:
        return 6000, "三倍满"
    if han >= 8:
        return 4000, "倍满"
    if han >= 6:
        return 3000, "跳满"
    base = fu * (2 ** (han + 2))
    if han >= 5 or base >= 2000:
        return 2000, "满贯"
    return base, ""


@dataclass
class Payment:
    winner_gain: int            # 和牌者拿到的总点数（含本场，不含立直棒）
    losses: dict[int, int]      # 每个付钱者的点数（正数）


def payment(base: int, winner: int, dealer: int, is_tsumo: bool, loser: int | None, honba: int) -> Payment:
    """按座位算谁付多少。loser 是点炮者（自摸为 None）。"""
    is_dealer = winner == dealer
    losses: dict[int, int] = {}
    if is_tsumo:
        for p in range(4):
            if p == winner:
                continue
            mult = 2 if (is_dealer or p == dealer) else 1
            losses[p] = _ceil100(base * mult) + 100 * honba
    else:
        assert loser is not None
        losses[loser] = _ceil100(base * (6 if is_dealer else 4)) + 300 * honba
    return Payment(sum(losses.values()), losses)
