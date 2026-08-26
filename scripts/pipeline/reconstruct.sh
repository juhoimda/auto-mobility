#!/usr/bin/env bash
# V2 reconstruction thin launcher (next.md #10).
# ROS env != Open3D env isolation: same sanitization strategy as compare.sh.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=6

_sanitize_ld() {
    local var="$1" out="" IFS=':'
    for entry in ${!var-}; do
        [[ "$entry" == /opt/ros* ]] && continue
        out+="${out:+:}$entry"
    done
    export "$var=$out"
}
_sanitize_ld LD_LIBRARY_PATH
_sanitize_ld PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

exec python3 -m auto_mobility.reconstruction.cli "$@"
