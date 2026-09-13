"""对局可视化服务器（标准库，无额外依赖）。

    uv run python -m majiang.viz.server --model models/discard.pt --agents nn,nn,rule,rule
    然后打开 http://localhost:8000

API（都返回 JSON）：
    POST /api/new?seed=N      新开一局
    POST /api/step            执行当前待决策玩家的动作
    GET  /api/state           当前状态 + 待决策玩家的 AI 分析
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
from majiang.engine.actions import ActionType
from majiang.engine.game import Game, Phase
from majiang.engine.hand import MeldType
from majiang.engine.shanten import discard_options, shanten
from majiang.engine.tile import to_str
from majiang.ml.model import load

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


class Session:
    def __init__(self, agent_names: list[str], model_path: str | None):
        self.model = load(model_path) if model_path and os.path.exists(model_path) else None
        self.agent_names = agent_names
        self.agents: list[Agent] = [self._make(n, i) for i, n in enumerate(agent_names)]
        self.game: Game | None = None
        self.last_action: dict | None = None
        self.lock = threading.Lock()

    def _make(self, name: str, seed: int) -> Agent:
        if name == "nn":
            if self.model is None:
                raise SystemExit("--agents 里有 nn 但没有可用的模型文件，请先训练或用 --model 指定")
            return NNAgent(self.model, seed=seed)
        if name == "rule":
            return RuleAgent()
        return RandomAgent(seed=seed)

    def new(self, seed: int) -> None:
        self.game = Game(seed=seed, dealer=seed % 4)
        self.game.start()
        self.last_action = None

    def step(self) -> None:
        g = self.game
        if g is None or g.finished:
            return
        p = g.players_to_act()[0]
        obs = g.observe(p)
        a = self.agents[p].act(obs)
        g.step(p, a)
        self.last_action = {"player": p, "type": a.type.name, "tile": a.tile}

    # ---------------- 序列化
    def analysis(self, p: int) -> dict:
        g = self.game
        assert g is not None
        obs = g.observe(p)
        out: dict = {"player": p, "phase": g.phase.name, "legal": [a.type.name for a in obs.legal_actions]}
        n_melds = len(obs.melds)
        out["shanten"] = shanten(obs.hand, n_melds)
        if g.phase == Phase.DISCARD:
            visible = RuleAgent._visible(obs)
            out["rule"] = [
                {"tile": d, "shanten": s, "effective": n} for d, s, n in discard_options(obs.hand, n_melds, visible)
            ]
            if self.model is not None:
                probs = NNAgent(self.model).discard_probs(obs)
                out["nn"] = [{"tile": t, "p": float(probs[t])} for t in range(34) if obs.hand[t] > 0]
                out["nn"].sort(key=lambda r: -r["p"])
        return out

    def state(self) -> dict:
        g = self.game
        if g is None:
            return {"started": False}
        players = []
        for p in range(4):
            h = g.hands[p]
            players.append(
                {
                    "agent": self.agent_names[p],
                    "hand": h.tiles(),
                    "melds": [{"type": m.type.name, "tile": m.tile} for m in h.melds],
                    "river": list(g.rivers[p]),
                    "shanten": shanten(h.counts, len(h.melds)),
                }
            )
        st: dict = {
            "started": True,
            "phase": g.phase.name,
            "current": g.current,
            "dealer": g.dealer,
            "wall": len(g.wall),
            "last_draw": g.last_draw if g.phase == Phase.DISCARD else -1,
            "last_discard": g.last_discard if g.phase == Phase.RESPOND else -1,
            "players": players,
            "to_act": g.players_to_act(),
            "last_action": self.last_action,
            "finished": g.finished,
            "result": None,
            "analysis": None,
        }
        if g.finished and g.result is not None:
            r = g.result
            st["result"] = {"winner": r.winner, "loser": r.loser, "tile": r.win_tile, "tsumo": r.is_tsumo,
                            "scores": r.scores, "text": str(r)}
        elif g.players_to_act():
            st["analysis"] = self.analysis(g.players_to_act()[0])
        return st


def make_handler(session: Session):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=STATIC_DIR, **kw)

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
                with session.lock:
                    return self._json(session.state())
            if u.path == "/":
                self.path = "/index.html"
            return super().do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            with session.lock:
                if u.path == "/api/new":
                    session.new(int(q.get("seed", ["0"])[0]))
                elif u.path == "/api/step":
                    n = int(q.get("n", ["1"])[0])
                    for _ in range(n):
                        session.step()
                        if session.game and session.game.finished:
                            break
                else:
                    self.send_error(404)
                    return
                self._json(session.state())

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/discard.pt")
    ap.add_argument("--agents", default="nn,nn,rule,rule")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    session = Session(args.agents.split(","), args.model)
    session.new(0)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"打开 http://localhost:{args.port}   agents={args.agents}  model={'有' if session.model else '无'}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
