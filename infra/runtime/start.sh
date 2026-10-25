#!/bin/sh
set -eu
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --workers 1 &
server=$!
ready=0
i=0
while [ "$i" -lt 30 ]; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"; then
    ready=1
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "command center did not become ready" >&2
  exit 1
fi
python scripts/post_vision.py --base-url http://127.0.0.1:8000 --interval 0.2 &
vision=$!
wait "$server" "$vision"
