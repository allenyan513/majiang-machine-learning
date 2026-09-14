"""让 4 个 agent 打一局日麻，返回结果。"""

from majiang.agents.base import Agent
from majiang.riichi.game import Game, Result


def play_game(agents: list[Agent], seed: int | None = None, dealer: int = 0, verbose: bool = False, **kw) -> Result:
    g = Game(seed=seed, dealer=dealer, **kw)
    g.start()
    while not g.finished:
        for p in g.players_to_act():
            obs = g.observe(p)
            action = agents[p].act(obs)
            if verbose:
                print(f"[{g.phase.name:7}] 玩家{p} 手牌 {g.hands[p]}  -> {action}")
            g.step(p, action)
    assert g.result is not None
    if verbose:
        print(g.result)
    return g.result
