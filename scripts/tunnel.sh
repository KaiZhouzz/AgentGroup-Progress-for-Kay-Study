#!/bin/bash
# 外网隧道开关：./scripts/tunnel.sh start|stop|status|url
# 需要三个服务已启动：后端 8000、主对话 8501、知识库管理 8502；start 后打印三个外网地址
CF="$HOME/bin/cloudflared"
BE_LOG=/tmp/cf-backend.log
FE_LOG=/tmp/cf-frontend.log
ADMIN_LOG=/tmp/cf-admin.log

case "$1" in
  start)
    if pgrep -f '[c]loudflared tunnel' > /dev/null; then
      echo "隧道已在运行"; exec "$0" url
    fi
    nohup "$CF" tunnel --url http://localhost:8000 --no-autoupdate > "$BE_LOG" 2>&1 &
    sleep 10
    nohup "$CF" tunnel --url http://localhost:8501 --no-autoupdate > "$FE_LOG" 2>&1 &
    sleep 10
    nohup "$CF" tunnel --url http://localhost:8502 --no-autoupdate > "$ADMIN_LOG" 2>&1 &
    sleep 12
    exec "$0" url
    ;;
  stop)
    pkill -f '[c]loudflared tunnel' && echo "隧道已关闭" || echo "隧道本来就没开"
    ;;
  status)
    if pgrep -f '[c]loudflared tunnel' > /dev/null; then echo "运行中"; else echo "未运行"; fi
    ;;
  url)
    echo "主对话界面: $(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$FE_LOG" | head -1)"
    echo "知识库管理: $(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$ADMIN_LOG" | head -1)"
    echo "后端 API: $(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$BE_LOG" | head -1)"
    echo "（每次 start 地址都会变，用 url 子命令重新查看）"
    ;;
  *)
    echo "用法: $0 start|stop|status|url"; exit 1
    ;;
esac
