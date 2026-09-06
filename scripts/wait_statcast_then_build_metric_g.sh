#!/usr/bin/env bash
# Wait for pybaseball shard download to finish, then rebuild Metric Folder Full Data Set (G features, BIP-only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/data/statcast_pybaseball/download_manifest.json"
LOG="$ROOT/data/statcast_pybaseball/metric_g_build.log"
echo "[wait_statcast_then_build_metric_g] watching for $MANIFEST" | tee -a "$LOG"
while [[ ! -f "$MANIFEST" ]]; do
  sleep 120
done
echo "[wait_statcast_then_build_metric_g] manifest found; building Full Data Set.parquet" | tee -a "$LOG"
cd "$ROOT"
python3 "Metric Folder/build_full_g_dataset.py" \
  --statcast-chunks-dir "$ROOT/data/statcast_pybaseball/chunks" \
  --require-download-manifest \
  2>&1 | tee -a "$LOG"
echo "[wait_statcast_then_build_metric_g] done" | tee -a "$LOG"
