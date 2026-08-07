#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash 2>/dev/null || true

if [ -f "$PROJECT_DIR/install/setup.bash" ]; then
    source "$PROJECT_DIR/install/setup.bash"
fi

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/launch:$PYTHONPATH"

echo "=========================================================="
echo "🧪 Running Unit & Integration Tests (Coverage Report)"
echo "=========================================================="

python3 -m coverage run -m unittest discover -s "$PROJECT_DIR/tests"
echo ""
python3 -m coverage report --include='src/auto_mobility/*,launch/*' -m
