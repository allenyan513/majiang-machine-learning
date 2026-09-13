"""网站服务器（标准库，无额外依赖）：教程 + 单机游戏 + AI 分析 API，一个进程全包。

    uv run python -m web.server                                   # 全规则 bot
    uv run python -m web.server --model models/discard.pt --agents nn,nn,rule,rule
    然后打开 http://localhost:8000

路由：
    /            首页（web/index.html）
    /docs/       教程阅读器（web/docs/）
    /play/       单机游戏 / 观战（web/play/）
    /api/...     引擎 + 模型

API（都返回 JSON）：
    POST /api/new?seed=N&human=P&length=hanchan|tonpuu   新开一场；human=P 表示座位 P 由人操作（-1 = 全 AI）
    POST /api/step?n=K             让 AI 执行最多 K 步；轮到人时停下
    POST /api/next                 本局结束后开下一局
    POST /api/act?player=P&type=DISCARD&tile=T&extra=E   人操作的动作（CHI 用 extra 指定顺子起点）
    GET  /api/state?view=P         状态；view=P 时只暴露 P 的手牌（玩家视角），省略 = 上帝视角

部署：监听地址和端口可用 --host/--port 或环境变量 HOST/PORT 指定（Cloud Run 注入 PORT）。
"""

import argparse
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from majiang.agents.base import Agent
from majiang.agents.nn_agent import NNAgent
from majiang.agents.random_agent import RandomAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Game, Phase
from majiang.engine.match import Match
from majiang.engine.shanten import discard_options, shanten
from majiang.engine.tile import WINDS, dora_from_indicator
from majiang.ml.model import load

WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Session:
    def __init__(self, agent_names: list[str], model_path: str | None):
        self.model = load(model_path) if model_path and os.path.exists(model_path) else None
        self.agent_names = list(agent_names)
        self.agents: list[Agent] = [self._make(n, i) for i, n in enumerate(agent_names)]
        self.match: Match | None = None
        self.game: Game | None = None
        self.last_action: dict | None = None
        self.human: int | None = None
        self.lock = threading.Lock()

    def _make(self, name: str, seed: int) -> Agent:
        if name == "nn":
            if self.model is None:
                print("警告：没有模型文件，nn 座位降级为 rule（先训练或用 --model 指定）")
                self.agent_names[seed] = "rule"
                return RuleAgent()
            return NNAgent(self.model, seed=seed)
        if name == "rule":
            return RuleAgent()
        return RandomAgent(seed=seed)

    # ---------------- 流程
    def new(self, seed: int, human: int | None = None, length: str = "hanchan") -> None:
        self.match = Match(seed=seed, length=length)
        self.human = human
        self.last_action = None
        self.hand_label = self.match.round_label
        self.game = self.match.new_hand()

    def next_hand(self) -> None:
        if self.match is None or self.game is None or not self.game.finished or self.match.ended:
            return
        self.last_action = None
        self.hand_label = self.match.round_label
        self.game = self.match.new_hand()

    def waiting_human(self) -> bool:
        g = self.game
        return g is not None and not g.finished and self.human is not None and g.players_to_act() == [self.human]

    def step(self) -> bool:
        """让一个 AI 玩家行动。返回 False 表示没动（结束了 / 在等人）。"""
        g = self.game
        if g is None or g.finished:
            return False
        ai_players = [p for p in g.players_to_act() if p != self.human]
        if not ai_players:
            return False
        p = ai_players[0]
        self._apply(p, self.agents[p].act(g.observe(p)))
        return True

    def act(self, player: int, type_name: str, tile: int, extra: int = -1) -> None:
        g = self.game
        if g is None or player != self.human or player not in g.players_to_act():
            return
        wanted = Action(ActionType[type_name], tile, extra)
        for a in g.legal_actions(player):
            if a == wanted or (a.type == wanted.type and a.type in (ActionType.PASS, ActionType.TSUMO, ActionType.RON)):
                self._apply(player, a)
                return

    def _apply(self, p: int, a: Action) -> None:
        assert self.game is not None and self.match is not None
        self.game.step(p, a)
        self.last_action = {"player": p, "type": a.type.name, "tile": a.tile, "extra": a.extra}
        if self.game.finished:
            self.match.finish_hand(self.game.result)

    # ---------------- 序列化
    def analysis(self, p: int) -> dict:
        g = self.game
        assert g is not None
        obs = g.observe(p)
        out: dict = {
            "player": p,
            "phase": g.phase.name,
            "legal": [{"type": a.type.name, "tile": a.tile, "extra": a.extra} for a in obs.legal_actions],
            "waits": obs.waits,
            "furiten": obs.furiten,
        }
        n_melds = len(obs.melds)
        out["shanten"] = shanten(obs.hand, n_melds)
        if g.phase == Phase.DISCARD:
            visible = RuleAgent.visible(obs)
            legal_discards = {a.tile for a in obs.legal_actions if a.type == ActionType.DISCARD}
            out["rule"] = [
                {"tile": d, "shanten": s, "effective": n}
                for d, s, n in discard_options(obs.hand, n_melds, visible)
                if d in legal_discards
            ]
            rule_action = RuleAgent().act(obs)
            out["rule_pick"] = {"type": rule_action.type.name, "tile": rule_action.tile}
            if self.model is not None and len(legal_discards) > 1:
                probs = NNAgent(self.model).discard_probs(obs)
                out["nn"] = sorted(
                    ({"tile": t, "p": float(probs[t])} for t in legal_discards), key=lambda r: -r["p"]
                )
        return out

    def state(self, view: int | None = None) -> dict:
        g, m = self.game, self.match
        if g is None or m is None:
            return {"started": False}
        players = []
        for p in range(4):
            h = g.hands[p]
            visible = view is None or p == view or g.finished
            players.append(
                {
                    "agent": "human" if p == self.human else self.agent_names[p],
                    "hand": h.tiles() if visible else [],
                    "reds": [t for t in range(34) if h.reds[t]] if visible else [],
                    "hand_size": h.size,
                    "melds": [
                        {"type": mm.type.name, "tiles": mm.tiles, "called": mm.called, "from": mm.from_player, "reds": mm.reds}
                        for mm in h.melds
                    ],
                    "river": [
                        {"tile": d.tile, "red": d.red, "riichi": d.riichi, "tsumogiri": d.tsumogiri, "claimed": d.claimed}
                        for d in g.rivers[p]
                    ],
                    "riichi": g.riichi[p],
                    "score": g.scores[p],
                    "wind": "东南西北"[WINDS.index(g.seat_wind(p))],
                    "shanten": shanten(h.counts, len(h.melds)) if visible else None,
                }
            )
        st: dict = {
            "started": True,
            "phase": g.phase.name,
            "current": g.current,
            "dealer": g.dealer,
            "wall": len(g.wall),
            "dora_indicators": list(g.dora_indicators),
            "dora": [dora_from_indicator(t) for t in g.dora_indicators],
            "round_label": self.hand_label,
            "honba": g.honba,
            "riichi_sticks": g.riichi_sticks,
            "last_draw": g.last_draw if g.phase == Phase.DISCARD else -1,
            "last_discard": g.last_discard if g.phase == Phase.RESPOND else -1,
            "chankan": g.chankan_tile >= 0,
            "players": players,
            "to_act": g.players_to_act(),
            "last_action": self.last_action,
            "finished": g.finished,
            "human": self.human,
            "waiting_human": self.waiting_human(),
            "match": {
                "length": m.length,
                "ended": m.ended,
                "end_reason": m.end_reason,
                "hands_played": m.hands_played,
                "ranking": m.ranking() if m.ended else None,
            },
            "result": None,
            "analysis": None,
        }
        if g.finished and g.result is not None:
            r = g.result
            st["result"] = {
                "kind": r.kind,
                "reason": r.reason,
                "deltas": r.deltas,
                "tenpai": r.tenpai,
                "ura_indicators": list(g.ura_indicators) if any(g.riichi[w.player] for w in r.wins) else [],
                "wins": [
                    {
                        "player": w.player,
                        "from": w.from_player,
                        "tile": w.tile,
                        "yaku": w.result.yaku,
                        "han": w.result.han,
                        "fu": w.result.fu,
                        "limit": w.result.limit,
                        "base": w.result.base,
                        "hand": g.hands[w.player].tiles(),
                        "melds": [{"type": mm.type.name, "tiles": mm.tiles, "called": mm.called} for mm in g.hands[w.player].melds],
                    }
                    for w in r.wins
                ],
                "text": str(r),
            }
        elif g.players_to_act():
            focus = view if view is not None else g.players_to_act()[0]
            if focus in g.players_to_act():
                st["analysis"] = self.analysis(focus)
        return st


def make_handler(session: Session):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=WEB_DIR, **kw)

        def log_message(self, *a):  # 安静
            pass

        def _json(self, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/state":
                q = parse_qs(u.query)
                view = int(q["view"][0]) if "view" in q and q["view"][0] != "" else None
                with session.lock:
                    return self._json(session.state(view))
            return super().do_GET()  # /、/docs/、/play/ 都由各自目录的 index.html 提供

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            view = int(q["view"][0]) if "view" in q and q["view"][0] != "" else None
            with session.lock:
                if u.path == "/api/new":
                    human = int(q.get("human", ["-1"])[0])
                    session.new(int(q.get("seed", ["0"])[0]), None if human < 0 else human, q.get("length", ["hanchan"])[0])
                elif u.path == "/api/step":
                    n = int(q.get("n", ["1"])[0])
                    for _ in range(n):
                        if not session.step():
                            break
                elif u.path == "/api/next":
                    session.next_hand()
                elif u.path == "/api/act":
                    session.act(int(q["player"][0]), q["type"][0], int(q.get("tile", ["-1"])[0]), int(q.get("extra", ["-1"])[0]))
                else:
                    self.send_error(404)
                    return
                self._json(session.state(view))

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="打牌模型路径；不传则不显示神经网络分析")
    ap.add_argument("--agents", default="rule,rule,rule,rule")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    session = Session(args.agents.split(","), args.model)
    session.new(0)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"教程 http://localhost:{args.port}/docs/   游戏 http://localhost:{args.port}/play/   "
          f"agents={','.join(session.agent_names)}  model={'有' if session.model else '无'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
