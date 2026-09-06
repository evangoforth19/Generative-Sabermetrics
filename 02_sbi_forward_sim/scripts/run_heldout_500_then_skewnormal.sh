#!/usr/bin/env bash
# Full held-out forward pipeline: N admissible decoder draws per event (default N=500), then
# per-event skew-normal EV refit vs observed_EV (MAE, RMSE, NLL, COV50/80/90, CRPS).
#
# Usage (from anywhere):
#   bash scripts/run_heldout_500_then_skewnormal.sh
#   TARGET_ADMISSIBLE=250 bash scripts/run_heldout_500_then_skewnormal.sh
#
# Optional env overrides:
#   TARGET_ADMISSIBLE=500   admissible draws per event (--target-admissible-per-event)
#   DEVICE=cuda|cpu         (default: cuda)
#   BATCH=2000              batch size per sampling round (--n-samples)
#   U_RUN, Z_RUN            stage-u / stage-z run directories (defaults below)
#   LOG=path                tee pipeline stdout/stderr here
#   EXTRA_PIPELINE_FLAGS='--max-events 5'   smoke / subset only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

U_RUN="${U_RUN:-$ROOT/outputs/p_u_given_g/20260407_193337Z}"
Z_RUN="${Z_RUN:-$ROOT/outputs/p_z_given_u_g/20260407_221024Z}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-2000}"
TARGET_ADMISSIBLE="${TARGET_ADMISSIBLE:-500}"
LOG="${LOG:-$ROOT/outputs/heldout_pipeline_test/last_adm_skewnormal_run_${TARGET_ADMISSIBLE}.log}"
EXTRA_PIPELINE_FLAGS="${EXTRA_PIPELINE_FLAGS:-}"

SK_SCRIPT="$ROOT/reports/ev_skewnormal_refit_20260408_003612Z/build_ev_skewnormal_refit_evaluation.py"

if [[ ! -r "$U_RUN/checkpoint.pt" ]]; then
  echo "Missing: $U_RUN/checkpoint.pt" >&2
  exit 1
fi
if [[ ! -r "$Z_RUN/checkpoint.pt" ]]; then
  echo "Missing: $Z_RUN/checkpoint.pt" >&2
  exit 1
fi
if [[ ! -f "$SK_SCRIPT" ]]; then
  echo "Missing skew-normal script: $SK_SCRIPT" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

export PYTHONUNBUFFERED=1

echo "Target admissible draws per event: $TARGET_ADMISSIBLE"
echo "Logging pipeline stdout/stderr to: $LOG"
# shellcheck disable=SC2086
python3 "$ROOT/scripts/run_heldout_pipeline_test.py" \
  --u-run-dir "$U_RUN" \
  --z-run-dir "$Z_RUN" \
  --target-admissible-per-event "$TARGET_ADMISSIBLE" \
  --n-samples "$BATCH" \
  --device "$DEVICE" \
  $EXTRA_PIPELINE_FLAGS 2>&1 | tee "$LOG"

OUT="$(grep '^Done\.' "$LOG" | tail -n1 || true)"
OUT="${OUT#Done. }"

if [[ -z "${OUT}" ]] || [[ ! -d "${OUT}" ]]; then
  echo "Could not parse output directory from pipeline (expected a line: Done. <path>). See $LOG" >&2
  exit 1
fi

echo "Pipeline output directory: $OUT"

python3 "$SK_SCRIPT" --heldout-run "$OUT"

echo "Empirical draw benchmark (no parametric fit) + EV table vs NGBoost..."
python3 "$ROOT/scripts/build_empirical_admissible_draw_benchmark_report.py" --heldout-run "$OUT"
python3 "$ROOT/scripts/build_ev_diagnostics_empirical_skew_ngboost_report.py" --heldout-run "$OUT"

echo ""
echo "Done."
echo "  Run artifacts:     $OUT"
echo "  Skew-normal JSON:  $OUT/ev_skewnormal_refit_summary.json"
echo "  Skew-normal report: $OUT/ev_skewnormal_refit_report.md"
echo "  Empirical benchmark: $OUT/empirical_admissible_draw_benchmark_report.md"
echo "  Empirical vs skew vs NGBoost: $OUT/ev_diagnostics_empirical_skew_ngboost.md"
