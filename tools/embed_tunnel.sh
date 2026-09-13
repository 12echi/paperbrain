#!/bin/bash
# 保持到远端 LM Studio 的 SSH 隧道 (本机 5010 -> 远端 127.0.0.1:5000)
# 用法: bash tools/embed_tunnel.sh start|stop|status
PORT="${EMBED_TUNNEL_PORT:-5010}"
TARGET="liaosheng@100.68.193.47"
KEY="$HOME/.ssh/id_ed25519"
case "$1" in
  start)
    pkill -f "L ${PORT}:127.0.0.1:5000" 2>/dev/null; sleep 1
    nohup ssh -i "$KEY" -p 3032 -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -L ${PORT}:127.0.0.1:5000 "$TARGET" \
      >/tmp/pb_embed_tunnel.log 2>&1 &
    sleep 2
    if pgrep -f "L ${PORT}:" >/dev/null; then
      pgrep -fl "L ${PORT}:" | head -1
    else
      echo "start FAILED (见 /tmp/pb_embed_tunnel.log)"; exit 1
    fi
    ;;
  stop) pkill -f "L ${PORT}:127.0.0.1:5000" && echo stopped ;;
  status)
    pgrep -f "L ${PORT}:" >/dev/null && echo "tunnel UP (127.0.0.1:${PORT} -> remote:5000)" || echo "tunnel DOWN"
    KEY="${PAPERBRAIN_EMBED_API_KEY:-$(python3 -c "import json,os
p=os.path.expanduser('~/.config/paperbrain/env.json')
print(json.load(open(p)).get('PAPERBRAIN_EMBED_API_KEY','') if os.path.exists(p) else '')" 2>/dev/null)}"
    # Key 走 stdin 配置, 不进 argv (防 ps 泄露)
    printf 'header = "Authorization: Bearer %s"\n' "$KEY" \
      | curl -s -m 5 -K - "http://127.0.0.1:${PORT}/v1/models" | head -c 160; echo ;;
  *) echo "usage: $0 start|stop|status" ;;
esac
