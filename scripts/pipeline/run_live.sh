#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DB_NAME=${1:-live_$(date +%Y%m%d_%H%M%S)}
DB_PATH="$DB_DIR/$DB_NAME.db"

ros2 launch auto_mobility rtab_live.launch.py database_path:="$DB_PATH"
