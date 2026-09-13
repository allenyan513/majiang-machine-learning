# 第 11 章 可视化与人机对战

> 对应代码：`web/server.py`、`web/play/index.html`、`tests/test_viz_session.py`

训练出来的模型是一个 2.4 MB 的二进制文件和一串指标。想知道它到底学会了什么，最有效的办法是**看它打牌**——而且要能看到它每一步在想什么。

```bash
uv run python -m web.server --model models/discard.pt --agents nn,nn,rule,rule
# 打开 http://localhost:8000
```

## 为什么要花时间做可视化

因为指标会骗人，眼睛不会。

举几个只能靠看才能发现的问题：

- 模型在特定局面下反复打同一张牌（策略塌缩）
- 听牌之后不胡（动作掩码有 bug）
- 副露之后打法突然变得莫名其妙（特征里副露那部分编码错了）
- 手里有安全牌却打危险牌（防守没学到）

这些在"均分 −0.07"这个数字里全部看不出来。

另一个价值是**人机对战**：你亲自和它打 20 局，对它的水平会有一个数字给不了的直觉。

## 服务端：标准库就够

整个服务器 227 行，零额外依赖：

```python
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
```

不用 Flask、不用 FastAPI。理由：这是个本地工具，只有你一个用户，加一个 Web 框架依赖不值得。`SimpleHTTPRequestHandler` 顺便还帮我们托管了静态文件。

四个 API：

```
POST /api/new?seed=N&human=P   新开一局；human=P 表示座位 P 由人操作（-1 = 全 AI）
POST /api/step?n=K             让 AI 执行最多 K 步；轮到人时停下
POST /api/act?player=P&type=DISCARD&tile=T   人操作的动作
GET  /api/state?view=P         状态；view=P 时只暴露 P 的手牌
```

### Session：把 Game 包一层

```python
class Session:
    def __init__(self, agent_names: list[str], model_path: str | None):
        self.model = load(model_path) if model_path and os.path.exists(model_path) else None
        self.agents = [self._make(n, i) for i, n in enumerate(agent_names)]
        self.game: Game | None = None
        self.human: int | None = None
        self.lock = threading.Lock()
```

`ThreadingHTTPServer` 会并发处理请求，而 `Game` 不是线程安全的（浏览器可能同时发 step 和 state）。一把大锁解决：

```python
with session.lock:
    return self._json(session.state(view))
```

粗粒度锁在单用户工具里完全够用，别过度设计。

### 步进：AI 走一步就停

```python
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
```

第 4 章那个"引擎被动、调用方驱动"的设计在这里发挥了作用：**可以随时暂停**。如果引擎是回调式的（agent 注册进去，引擎跑完整局），就没法在中间停下来等人点击。

### 信息隔离：玩家视角

```python
def state(self, view: int | None = None) -> dict:
    for p in range(4):
        h = g.hands[p]
        visible = view is None or p == view or g.finished
        players.append({
            "hand": h.tiles() if visible else [],
            "hand_size": h.size,
            ...
        })
```

`view=None` 是上帝视角（四家手牌都可见，看 AI 对战用）；`view=P` 只暴露 P 的手牌。

注意 `hand_size` 单独给出——前端需要知道盖着的牌有几张才能画出来。**隔离信息的时候，要区分"内容"和"元数据"**：手牌内容是秘密，张数不是（真实牌桌上你也数得出来）。

对局结束后 `g.finished` 让所有手牌可见——这是真实麻将的"亮牌"环节，也是复盘的关键。

### 分析面板：AI 在想什么

这是整个可视化最有价值的部分：

```python
def analysis(self, p: int) -> dict:
    obs = g.observe(p)
    out = {"player": p, "phase": g.phase.name, "legal": [...]}
    out["shanten"] = shanten(obs.hand, n_melds)
    if g.phase == Phase.DISCARD:
        visible = RuleAgent._visible(obs)
        out["rule"] = [
            {"tile": d, "shanten": s, "effective": n}
            for d, s, n in discard_options(obs.hand, n_melds, visible)
        ]
        rule_action = RuleAgent().act(obs)   # 规则 bot 真正会怎么做
        out["rule_pick"] = rule_action.tile if rule_action.type == ActionType.DISCARD else -1
        if self.model is not None:
            probs = NNAgent(self.model).discard_probs(obs)
            out["nn"] = [{"tile": t, "p": float(probs[t])} for t in range(34) if obs.hand[t] > 0]
            out["nn"].sort(key=lambda r: -r["p"])
    return out
```

同时给出三样东西：

1. **向听数**：离胡多远
2. **规则 bot 的分析表**：每种打法的向听数和进张数，加上它实际的选择
3. **神经网络的概率分布**：每张牌被打出的概率

**把两个策略并排显示**，分歧一目了然。看到网络给某张牌 80% 概率而规则表显示它明显是坏选择，就找到了一个具体的失败案例，可以拿去复现、调试。

这比看聚合指标高效得多。

## 前端：一个 HTML 文件

234 行，无框架、无构建步骤、无 npm。用 `fetch` 调 API，用 DOM 操作渲染。

布局仿 QQ 麻将：

```
        对家（上）
  上家              下家
  （左）            （右）
        自己（下）
```

中间是剩余牌数和风位，四家的牌河摆在各自面前，副露放手牌旁边。

CSS 里用渐变和阴影把牌做出立体感：

```css
.tile {
  width:30px; height:42px; border-radius:4px;
  background:linear-gradient(#fffdf7,#efe9dc);
  border:1px solid #b9b2a3; border-bottom:4px solid #2e8b57;
}
.tile.m { color:#c0392b; }  /* 万：红 */
.tile.s { color:#1e8449; }  /* 条：绿 */
.tile.p { color:#1f5fa8; }  /* 筒：蓝 */
```

花色用颜色区分，扫一眼就能看出手牌结构——这在调试时很重要，你要快速判断"这手牌合理吗"。

**为调试工具花时间做视觉设计是值得的**，因为你会盯着它看几百局。清晰的配色能让你更快发现异常。

## 两种模式

**上帝视角**：四家手牌全可见，轮到谁决策就显示谁的分析。适合看 AI 之间对战，观察不同 agent 的风格差异。

**玩家视角**：选一个座位坐下，其他三家的牌盖着。勾上"我来打"后由你操作——轮到你时点手牌打出，按钮碰/杠/胡/过。

玩家视角里右侧的分析面板变成了你的**辅助器**：它显示你当前手牌的向听数、每种打法的进张数、以及神经网络推荐打什么。这反过来是个学麻将的好工具。

## 测试

GUI 难测，但**业务逻辑可以从 GUI 里剥出来测**：

```python
def test_god_view_and_stepping():
    s = Session(["rule", "rule", "rule", "rule"], model_path=None)
    s.new(seed=1)
    st = s.state()
    assert all(len(p["hand"]) == p["hand_size"] for p in st["players"])
    assert st["analysis"]["player"] == st["to_act"][0] and "rule" in st["analysis"]
    while s.step():
        pass
    assert s.state()["finished"]
```

`Session` 类不依赖 HTTP，可以直接实例化测试。HTTP handler 只是薄薄一层参数解析。**把逻辑和传输层分开**，前者能测，后者不用测。

## 坑

**模型不存在时不要崩。** 

```python
self.model = load(model_path) if model_path and os.path.exists(model_path) else None
```

你会在还没训练模型时就想看看牌桌。允许 `--agents rule,rule,rule,rule` 无模型运行。但如果 agent 列表里有 `nn` 却没有模型，就要明确报错：

```python
if name == "nn":
    if self.model is None:
        raise SystemExit("--agents 里有 nn 但没有可用的模型文件，请先训练或用 --model 指定")
```

**人的动作要校验。** 前端可能发来任何东西：

```python
def act(self, player: int, type_name: str, tile: int) -> None:
    if g is None or player != self.human or player not in g.players_to_act():
        return
    wanted = Action(ActionType[type_name], tile)
    for a in g.legal_actions(player):
        if a == wanted or (a.type == wanted.type and a.type in (PASS, TSUMO, RON)):
            self._apply(player, a)
            return
```

三层检查：是不是人的座位、是不是轮到他、动作是否合法。第 4 章引擎里的合法性检查是最后一道防线，但在这里提前拦截能给出更好的用户体验（点错了没反应，而不是服务器 500）。

`PASS/TSUMO/RON` 那个特判是因为前端发的 tile 可能和引擎期望的不一致（比如 PASS 的 tile 是 -1），只匹配类型即可。

## 练习

1. 加"复盘"：记录整局的动作序列，可以前后拖动查看任意时刻的局面和当时的分析
2. 并排显示两个模型（监督 vs RL）对同一局面的概率分布，直接看它们的分歧在哪
3. 加一个"AI 建议"按钮：你打牌之前先让 AI 给出它的选择和理由，练完一局统计你和 AI 的一致率

下一章：[扩展路线](12-next.md)
