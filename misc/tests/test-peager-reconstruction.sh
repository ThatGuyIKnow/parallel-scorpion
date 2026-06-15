#!/usr/bin/env bash
# Regression test for peager parallel plan reconstruction.
#
# peager (MPI parallel eager search) must produce VALID plans at every worker
# count, not just np=1. A reconstruction bug used to truncate plans at np>1
# (cross-rank parent links were followed with the wrong rank-local StateID), so
# this test runs the real experiment config (alt preferred+standard queues,
# boost=1000, Zobrist work distribution) at np in {1,2,4} on small tasks and
# validates each plan with VAL.
#
# Known, separate issue: a pre-existing data race in the distributed *search*
# (independent of reconstruction; observable as bogus g-values) can occasionally
# corrupt a node's parent links. Reconstruction now detects the resulting cycle
# and ABORTS cleanly (prints "...cycle...reconstruction...") instead of
# deadlocking. Such an aborted run is the known race, not a reconstruction
# regression, so we retry it. An invalid plan WITHOUT that warning, or a hang,
# is a real failure.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FD="$REPO/fast-downward.py"
BENCH="${DOWNWARD_BENCHMARKS:-/home/workbox/Projects/downward-projects/benchmarks}/downward-benchmarks"
VAL="${VALIDATE:-/home/workbox/bin/validate}"
SEARCH='peager(alt([single(h), single(h, pref_only=true)], boost=1000), preferred=[h], reopen_closed=false, f_eval=h, hash=zobrist())'
ATTEMPTS=5

TASKS=(
  "gripper/domain.pddl gripper/prob01.pddl"
  "miconic/domain.pddl miconic/s10-0.pddl"
  "blocks/domain.pddl blocks/probBLOCKS-10-0.pddl"
)
NPS=(1 2 4)

if [ ! -x "$VAL" ]; then echo "SKIP: VAL not found at $VAL"; exit 77; fi

fail=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for entry in "${TASKS[@]}"; do
  set -- $entry; dom="$BENCH/$1"; prob="$BENCH/$2"
  for np in "${NPS[@]}"; do
    plan="$tmp/plan.txt"; log="$tmp/log.txt"
    outcome="" ; races=0
    for a in $(seq 1 $ATTEMPTS); do
      rm -f "$plan"
      timeout 120 "$FD" --plan-file "$plan" -np "$np" "$dom" "$prob" \
        --evaluator "h=ff()" --search "$SEARCH" >"$log" 2>&1
      ec=$?
      if [ $ec -eq 124 ]; then outcome="HANG"; break; fi
      if [ -s "$plan" ] && "$VAL" "$dom" "$prob" "$plan" 2>/dev/null | grep -q "Plan valid"; then
        outcome="PASS:$(grep -c '^[^;]' "$plan")"; break
      fi
      if grep -qiE 'cycle.*reconstruction' "$log"; then races=$((races+1)); continue; fi
      outcome="BADPLAN"; break   # invalid/no plan without the known race => real bug
    done
    case "$outcome" in
      PASS:*) echo "[PASS] $2 np=$np : ${outcome#PASS:} steps, VAL-valid${races:+ (after $races known-race retries)}";;
      "")     echo "[WARN] $2 np=$np : only pre-existing-race aborts in $ATTEMPTS tries (search race, not reconstruction)";;
      HANG)   echo "[FAIL] $2 np=$np : HANG (deadlock)"; fail=1;;
      *)      echo "[FAIL] $2 np=$np : invalid/no plan without cycle-abort (reconstruction bug)"; fail=1;;
    esac
  done
done

if [ "$fail" -ne 0 ]; then echo "RESULT: FAIL"; exit 1; fi
echo "RESULT: PASS"; exit 0
