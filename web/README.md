# web/ — 教程 + 单机游戏 + AI，一次部署

```
web/
  server.py       统一服务：/docs 教程、/play 游戏、/api 引擎 + 模型（标准库，无框架）
  docs/           教程 Markdown + 离线阅读器 index.html
  play/index.html 单机游戏 / 观战前端
  Dockerfile      一个镜像跑全站
```

## 本地

```bash
uv run python -m web.server                       # http://localhost:8000/docs/  和  /play/
uv run python -m web.server --agents rule,rule,rule,rule   # 没有模型时
```

## Docker

```bash
docker build -f web/Dockerfile -t majiang-web .
docker run --rm -p 8000:8000 majiang-web
```

## Google Cloud Run

```bash
gcloud run deploy majiang --source . --region asia-east1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 0
```

`--source .` 会用根目录的 Dockerfile；仓库根没有，所以先 `cp web/Dockerfile Dockerfile`，或者：

```bash
docker build -f web/Dockerfile -t asia-east1-docker.pkg.dev/<PROJECT>/majiang/web .
docker push asia-east1-docker.pkg.dev/<PROJECT>/majiang/web
gcloud run deploy majiang --image asia-east1-docker.pkg.dev/<PROJECT>/majiang/web --region asia-east1 --allow-unauthenticated
```

Cloud Run 注入 `PORT` 环境变量，server 会读它。冷启动加载 torch 约 2–3 秒。

注意：服务器只有一个全局对局（`Session`），多人同时打开 /play 会互相干扰——它是单机演示，不是多用户服务。教程 /docs 是纯静态，不受影响。

## Cloudflare

Cloudflare Pages 只能托管静态内容。教程可以单独发（`web/docs/` 整个目录就是一个静态站），游戏需要 Python 后端，用 Cloud Run 或 Cloudflare Containers。
