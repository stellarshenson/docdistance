#!/usr/bin/env bash
# Re-execute experiment notebooks in place, one after another, in the project environment.
# usage: reports/E13-campaign/reexec.sh <status.tsv> <notebook>...
# One status line per notebook: name <TAB> exit code <TAB> error outputs <TAB> seconds.
# A notebook already recorded with exit code 0 and 0 error outputs is skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1
status=$1; shift
for nb in "$@"; do
  name=$(basename "$nb" .ipynb)
  if grep -q "^${name}	0	0	" "$status" 2>/dev/null; then continue; fi
  start=$(date +%s)
  CI=true .venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    "$nb" > "logs/reexec-${name}.log" 2>&1
  rc=$?
  errs=$(.venv/bin/python -c "import json,sys; nb=json.load(open(sys.argv[1])); print(sum(o.get('output_type')=='error' for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[])))" "$nb")
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$errs" "$(( $(date +%s) - start ))" >> "$status"
done
printf 'LANE-DONE\t%s\n' "$(date -u +%FT%TZ)" >> "$status"
