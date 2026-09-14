# web/ — 教程 + 单机游戏 + AI，一次部署

```
Dockerfile        仓库根：一个镜像跑全站（gcloud run deploy --source 要求它在根目录）
web/
  server.py       统一服务：/docs 教程、/play 游戏、/api 引擎 + 模型（标准库，无框架）
  docs/           教程 Markdown + 离线阅读器 index.html
  play/index.html 单机游戏 / 观战前端
  train/index.html RL 训练面板：曲线、行为探针、带学习信号的对局回放（读 models/<run>_log.csv 和 <run>_ckpt/）
```

## 本地

```bash
uv run python -m web.server                       # http://localhost:8000/docs/  和  /play/
uv run python -m web.server --agents rule,rule,rule,rule   # 没有模型时
```

## Docker

```bash
docker build -t majiang-web .
docker run --rm -p 8000:8000 majiang-web
```

## Google Cloud Run

```bash
gcloud run deploy majiang --source . --region us-east1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 1
```

`--source .` 用仓库根的 Dockerfile 在 Cloud Build 里构建。两个细节：

- gcloud 上传源码时按 `.gcloudignore` 过滤；没有这个文件它会拿 `.gitignore` 顶替，而 `.gitignore` 排除了整个 `models/`，`COPY models` 就会失败。仓库里的 `.gcloudignore` 放行了 `models/discard.pt`。
- `--max-instances 1`：服务器只有一个全局对局（`Session`），扩到多实例后各实例状态不一致，/play 会错乱。同理多人同时打开 /play 会互相干扰——它是单机演示，不是多用户服务。教程 /docs 是纯静态，不受影响。

Cloud Run 注入 `PORT` 环境变量，server 会读它。冷启动加载 torch 约 2–3 秒。/train 面板读 `models/<run>_log.csv` 和 checkpoint 目录，这些不在镜像里，线上那一页会显示"没找到"。

## Cloudflare

Cloudflare Pages 只能托管静态内容。教程可以单独发（`web/docs/` 整个目录就是一个静态站），游戏需要 Python 后端，用 Cloud Run 或 Cloudflare Containers。
