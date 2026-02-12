#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 /path/to/root_folder [np] [--force]"
  exit 1
}

ROOT="${1:-}"
if [[ -z "$ROOT" ]]; then usage; fi
if [[ ! -d "$ROOT" ]]; then echo "Directory not found: $ROOT" >&2; exit 2; fi

NP=12
FORCE=false

for arg in "${@:2}"; do
  if [[ "$arg" == "--force" ]]; then
    FORCE=true
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    NP="$arg"
  fi
done

while IFS= read -r -d '' ctl; do
  dir=$(dirname "$ctl")
  base=$(basename "$ctl" .ctl)
  out="$dir/${base}_out.out"

  if [[ -f "$out" && "$FORCE" != "true" ]]; then
    echo "Skipping (output exists): $ctl"
    continue
  fi

  echo "--------------------------------------------------------"
  echo "Running: $ctl"
  echo "--------------------------------------------------------"

  tmp_log=$(mktemp)

  # Run mpirun in subshell and write into temp file
  (
    cd "$dir" || exit 1
    echo "[MPB START] $(date)"
    echo "mpirun -np $NP mpb-mpi ${base}.ctl"
    echo
    set +e
    setsid mpirun -np "$NP" mpb-mpi "./${base}.ctl" < /dev/null
    rc=$?
    echo
    echo "[MPB END] exit code: $rc"
    echo "[END TIME] $(date)"
    exit $rc
  ) > "$tmp_log" 2>&1

  rc=$?   # capture mpirun exit code correctly

  # Now safely tee into final log
  cat "$tmp_log" | tee "$out"

  rm "$tmp_log"

  echo "Finished: $ctl (exit code $rc)"

done < <(find "$ROOT" -type f -name '*.ctl' -print0)

echo "========================================================"
echo "All CTL files processed."
echo "========================================================"
