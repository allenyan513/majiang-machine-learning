"""网站服务器（标准库，无额外依赖）：教程 + 单机游戏 + AI 分析 API，一个进程全包。

    uv run python -m web.server --model models/discard.pt --agents nn,nn,rule,rule
    然后打开 http://localhost:8000

路由：
    /            首页（web/index.html）
    /docs/       教程阅读器（web/docs/）
    /play/       单机游戏 / 观战（web/play/）
    /train/      看 RL 训练：曲线 + 行为探针 + 带学习信号的对局回放（web/train/）
    /api/...     引擎 + 模型

部署：监听地址和端口可用 --host/--port 或环境变量 HOST/PORT 指定（Cloud Run 注入 PORT）。

API（都返回 JSON）：
    POST /api/new?seed=N&human=P   新开一局；human=P 表示座位 P 由人操作（-1 = 全 AI）
    POST /api/step?n=K             让 AI 执行最多 K 步；轮到人时停下
    POST /api/act?player=P&type=DISCARD&tile=T   人操作的动作
    GET  /api/state?view=P         状态；view=P 时只暴露 P 的手牌（玩家视角），省略 = 上帝视角
    GET  /api/train?run=rl         训练日志（models/<run>_log.csv）+ checkpoint 列表
    GET  /api/train/trace?run=rl&iter=N   第 N 轮录的那一局（models/<run>_ckpt/trace_N.json）

分析面板：上帝视角显示当前决策者的分析；玩家视角只显示该座位的分析。
"""

import argparse
import csv
import json
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from majiang.agents.base import Agent
from majiang.agents.nn_agent import NNAgent
from majiang.agents.random_agent import RandomAgent
from majiang.agents.rule_agent import RuleAgent
from majiang.engine.actions import Action, ActionType
from majiang.engine.game import Game, Phase
from majiang.engine.hand import MeldType
from majiang.engine.shanten import discard_options, shanten
from majiang.engine.tile import to_str
from majiang.ml.model import load

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(WEB_DIR), "models")


def _safe_run(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("bad run name")
    return name


def train_log(run: str) -> dict:
    """读 models/<run>_log.csv 和 checkpoint 目录，给曲线页用。"""
    run = _safe_run(run)
    log = os.path.join(MODELS_DIR, f"{run}_log.csv")
    ckpt_dir = os.path.join(MODELS_DIR, f"{run}_ckpt")
    if not os.path.exists(log):
        return {"found": False, "run": run}
    with open(log, newline="") as f:
        rows = [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]
    traces = sorted(int(m.group(1)) for fn in os.listdir(ckpt_dir) if (m := re.fullmatch(r"trace_(\d+)\.json", fn))) if os.path.isdir(ckpt_dir) else []
    return {"found": True, "run": run, "rows": rows, "traces": traces, "mtime": os.path.getmtime(log)}


def train_trace(run: str, it: int) -> dict:
    path = os.path.join(MODELS_DIR, f"{_safe_run(run)}_ckpt", f"trace_{it:03d}.json")
    if not os.path.exists(path):
        return {"found": False}
    with open(path) as f:
        return {"found": True, **json.load(f)}


class Session:
    def __init__(self, agent_names: list[str], model_path: str | None):
        self.model = load(model_path) if model_path and os.path.exists(model_path) else None
        self.agent_names = agent_names
        self.agents: list[Agent] = [self._make(n, i) for i, n in enumerate(agent_names)]
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

    def new(self, seed: int, human: int | None = None) -> None:
        self.game = Game(seed=seed, dealer=seed % 4)
        self.game.start()
        self.last_action = None
        self.human = human

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

    def act(self, player: int, type_name: str, tile: int) -> None:
        g = self.game
        if g is None or player != self.human or player not in g.players_to_act():
            return
        wanted = Action(ActionType[type_name], tile)
        for a in g.legal_actions(player):
            if a == wanted or (a.type == wanted.type and a.type in (ActionType.PASS, ActionType.TSUMO, ActionType.RON)):
                self._apply(player, a)
                return

    def _apply(self, p: int, a: Action) -> None:
        assert self.game is not None
        self.game.step(p, a)
        self.last_action = {"player": p, "type": a.type.name, "tile": a.tile}

    # ---------------- 序列化
    def analysis(self, p: int) -> dict:
        g = self.game
        assert g is not None
        obs = g.observe(p)
        out: dict = {
            "player": p,
            "phase": g.phase.name,
            "legal": [{"type": a.type.name, "tile": a.tile} for a in obs.legal_actions],
        }
        n_melds = len(obs.melds)
        out["shanten"] = shanten(obs.hand, n_melds)
        if g.phase == Phase.DISCARD:
            visible = RuleAgent._visible(obs)
            out["rule"] = [
                {"tile": d, "shanten": s, "effective": n} for d, s, n in discard_options(obs.hand, n_melds, visible)
            ]
            rule_action = RuleAgent().act(obs)  # 规则 bot 真正会怎么做（含同分时的取舍）
            out["rule_pick"] = rule_action.tile if rule_action.type == ActionType.DISCARD else -1
            if self.model is not None:
                probs = NNAgent(self.model).discard_probs(obs)
                out["nn"] = [{"tile": t, "p": float(probs[t])} for t in range(34) if obs.hand[t] > 0]
                out["nn"].sort(key=lambda r: -r["p"])
        return out

    def state(self, view: int | None = None) -> dict:
        g = self.game
        if g is None:
            return {"started": False}
        players = []
        for p in range(4):
            h = g.hands[p]
            visible = view is None or p == view or g.finished
            players.append(
                {
                    "agent": "human" if p == self.human else self.agent_names[p],
                    "hand": h.tiles() if visible else [],
                    "hand_size": h.size,
                    "melds": [{"type": m.type.name, "tile": m.tile} for m in h.melds],
                    "river": list(g.rivers[p]),
                    "shanten": shanten(h.counts, len(h.melds)) if visible else None,
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
            "human": self.human,
            "waiting_human": self.waiting_human(),
            "result": None,
            "analysis": None,
        }
        if g.finished and g.result is not None:
            r = g.result
            st["result"] = {"winner": r.winner, "loser": r.loser, "tile": r.win_tile, "tsumo": r.is_tsumo,
                            "scores": r.scores, "text": str(r)}
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
            q = parse_qs(u.query)
            if u.path == "/api/state":
                view = int(q["view"][0]) if "view" in q and q["view"][0] != "" else None
                with session.lock:
                    return self._json(session.state(view))
            if u.path == "/api/train":
                return self._json(train_log(q.get("run", ["rl"])[0]))
            if u.path == "/api/train/trace":
                return self._json(train_trace(q.get("run", ["rl"])[0], int(q.get("iter", ["0"])[0])))
            return super().do_GET()  # /、/docs/、/play/ 都由各自目录的 index.html 提供

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            view = int(q["view"][0]) if "view" in q and q["view"][0] != "" else None
            with session.lock:
                if u.path == "/api/new":
                    human = int(q.get("human", ["-1"])[0])
                    session.new(int(q.get("seed", ["0"])[0]), None if human < 0 else human)
                elif u.path == "/api/step":
                    n = int(q.get("n", ["1"])[0])
                    for _ in range(n):
                        if not session.step():
                            break
                elif u.path == "/api/act":
                    session.act(int(q["player"][0]), q["type"][0], int(q.get("tile", ["-1"])[0]))
                else:
                    self.send_error(404)
                    return
                self._json(session.state(view))

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/discard.pt")
    ap.add_argument("--agents", default="nn,nn,rule,rule")
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
