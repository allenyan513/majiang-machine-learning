# 一个镜像跑全站：教程 + 单机游戏 + AI 分析 API
#   docker build -f web/Dockerfile -t majiang-web .
#   docker run -p 8000:8000 majiang-web
# 构建前先训练好 models/discard.pt（没有也能跑，nn 座位会降级成规则 bot）
FROM python:3.11-slim

WORKDIR /app
# CPU 版 torch（默认源的 torch 带 CUDA，镜像会大 2GB）
RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu

COPY majiang ./majiang
COPY web ./web
COPY models ./models

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "-m", "web.server", "--agents", "rule,rule,rule,rule"]
