#!/bin/sh
set -eu
case "${1:-}" in red|yellow|green|blue) ;; *) echo 'color must be red, yellow, green, or blue' >&2; exit 2;; esac
curl --noproxy '*' --fail-with-body --silent --show-error --max-time 1800 \
  -H 'Content-Type: application/json' \
  -d "{\"color\":\"$1\",\"source\":\"camera\",\"mode\":\"execute\"}" \
  http://127.0.0.1:8765/v1/tasks
