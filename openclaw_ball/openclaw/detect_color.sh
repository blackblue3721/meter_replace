#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  red|yellow|green|blue) color="$1" ;;
  *) echo 'color must be red, yellow, green, or blue' >&2; exit 2 ;;
esac

curl --noproxy '*' --fail --silent --show-error \
  -X POST http://127.0.0.1:8765/v1/tasks \
  -H 'Content-Type: application/json' \
  -d "{\"color\":\"$color\",\"mode\":\"detect\",\"source\":\"camera\"}"
