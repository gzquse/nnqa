#!/bin/bash
# Helper script to run cloud batch operations with correct Python version

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [submit|status|retrieve] [options...]"
    echo ""
    echo "Examples:"
    echo "  $0 submit --backend ibm_boston --trials 10"
    echo "  $0 status"
    echo "  $0 retrieve"
    exit 1
fi

COMMAND="$1"
shift

$PYTHON "$SCRIPT_DIR/submit_cloud_batch.py" "$COMMAND" "$@"

