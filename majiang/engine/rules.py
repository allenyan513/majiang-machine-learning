"""规则常量（简化推倒胡）。以后加番型时只改这里和 game.py 的结算部分。"""

NUM_PLAYERS = 4
HAND_SIZE = 13

# 计分：先用最简单的固定分，只为让 AI 有"胡 > 不胡、自摸 > 点炮"的信号
RON_POINTS = 1          # 点炮：放炮者付给胡家
TSUMO_POINTS_EACH = 1   # 自摸：其他三家各付给胡家
