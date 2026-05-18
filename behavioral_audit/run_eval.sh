#!/bin/bash
# Run SWE-bench evaluation on a predictions file
# Usage: ./run_eval.sh <predictions.jsonl> <run_id>
set -e
PRED="$1"
RUNID="$2"
[ -z "$PRED" ] || [ -z "$RUNID" ] && { echo "Usage: $0 <predictions.jsonl> <run_id>"; exit 1; }
[ ! -f "$PRED" ] && { echo "ERROR: $PRED not found"; exit 1; }

python -m swebench.harness.run_evaluation \
    --predictions_path "$PRED" \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --max_workers 4 \
    --run_id "$RUNID" \
    --cache_level env \
    --clean True
