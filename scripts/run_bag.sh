#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

USE_COMPRESSED=false
DB_NAME=""

for arg in "$@"; do
    case $arg in
        --compressed)
            USE_COMPRESSED=true
            ;;
        *)
            if [ -z "$DB_NAME" ]; then
                DB_NAME="$arg"
            fi
            ;;
    esac
done

DB_NAME=${DB_NAME:-bag_$(date +%Y%m%d_%H%M%S)}
DB_PATH="$DB_DIR/$DB_NAME.db"

ros2 launch auto_mobility rtab_bag.launch.py database_path:="$DB_PATH" use_compressed:="$USE_COMPRESSED"
