#!/bin/bash
# View logs of the Face Recognition WebRTC Stream containers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default: follow logs. Use -n to show last N lines
LINES=${1:-100}

if [[ "$1" == "-f" ]] || [[ -z "$1" ]]; then
    echo "Following logs (Ctrl+C to exit)..."
    docker compose logs -f --tail=100
else
    echo "Showing last $LINES lines..."
    docker compose logs --tail="$LINES"
fi
