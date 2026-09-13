#!/usr/bin/env bash
# Preflight every config. Train configs must all pass before launching.
# Infer configs are EXPECTED to fail with PATH-INPUT-001 until that arm has a model.pth:
# modelpath is validated as an input file in inference mode.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
cd "$AICAE_ROOT"

pass=0; fail=0
echo "== train configs (--check --strict)"
for arm in $ARMS_ALL; do
  out=$("$PYBIN" -B AI_CAE4ALL_main.py --config "$(config_path "$arm" train)" --check --strict 2>&1)
  line=$(echo "$out" | grep -E '^Preflight:' || echo 'Preflight: NO OUTPUT')
  printf '  %-4s %s\n' "$arm" "$line"
  if echo "$line" | grep -q PASSED; then pass=$((pass+1)); else
    fail=$((fail+1)); echo "$out" | grep -E '^\s+\[' | sed 's/^/       /'
  fi
done
echo "  -> $pass passed, $fail failed"

echo
echo "== infer configs (--check; PATH-INPUT-001 on modelpath is expected pre-training)"
for arm in $ARMS_ALL; do
  out=$("$PYBIN" -B AI_CAE4ALL_main.py --config "$(config_path "$arm" infer)" --check 2>&1)
  line=$(echo "$out" | grep -E '^Preflight:' || echo 'Preflight: NO OUTPUT')
  other=$(echo "$out" | grep -E '^\s+\[' | grep -v 'PATH-INPUT-001' || true)
  printf '  %-4s %s\n' "$arm" "$line"
  [ -n "$other" ] && echo "$other" | sed 's/^/       /'
done

[ "$fail" -eq 0 ] || { echo; echo "train configs failed -- do not launch"; exit 1; }
echo
echo "next: bash configs/campaigns/$CAMPAIGN/train_all.sh"
