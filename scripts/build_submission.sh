#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$repo/scripts/validate_submission.py" "$repo/main.py"
tar -C "$repo" -czf "$repo/submission.tar.gz" main.py
gzip -t "$repo/submission.tar.gz"
test "$(tar -tzf "$repo/submission.tar.gz")" = "main.py"
echo "submission archive: $repo/submission.tar.gz"
