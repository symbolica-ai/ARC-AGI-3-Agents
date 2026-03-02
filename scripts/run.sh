#!/usr/bin/env bash
#
# Run the arcgentica agent against one or more ARC-AGI-3 games.
#
# Starts the session manager automatically, runs the agent, then shuts
# the server down when finished.
#
# Usage:
#   ./scripts/run.sh                    # all games
#   ./scripts/run.sh ls20               # single game (prefix match)
#   ./scripts/run.sh ls20,vc33,ft09     # multiple games
#
# Optional env vars:
#   AGENT              - Agent name (default: arcgentica)
#   SM_PORT            - Session manager port (default: 2345)
#   AGENTICA_SERVER_DIR - Path to agentica-server checkout
#   TAGS               - Comma-separated scorecard tags
#   NO_SERVER          - Set to 1 to skip starting the session manager
#                        (if you already have one running)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

AGENT="${AGENT:-arcgentica}"
SM_PORT="${SM_PORT:-2345}"
GAME_FILTER="${1:-}"
TAGS="${TAGS:-}"
NO_SERVER="${NO_SERVER:-0}"

SM_PID=""

cleanup() {
    if [[ -n "$SM_PID" ]]; then
        echo ""
        echo "Stopping session manager (pid $SM_PID) ..."
        kill "$SM_PID" 2>/dev/null || true
        wait "$SM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local port="$1"
    local max_wait=90
    local waited=0
    echo -n "Waiting for session manager on port $port "
    while ! nc -z localhost "$port" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        echo -n "."
        if [[ $waited -ge $max_wait ]]; then
            echo " TIMEOUT"
            echo "ERROR: Session manager did not start within ${max_wait}s" >&2
            exit 1
        fi
    done
    echo " ready"
}

# --- Start session manager ---
if [[ "$NO_SERVER" != "1" ]]; then
    echo "=== Starting session manager ==="
    "$SCRIPT_DIR/server.sh" &
    SM_PID=$!
    wait_for_server "$SM_PORT"
    echo ""
fi

# --- Run agent ---
echo "=== Running agent: $AGENT ==="

GAME_ARG=""
if [[ -n "$GAME_FILTER" ]]; then
    GAME_ARG="--game=$GAME_FILTER"
    echo "  games: $GAME_FILTER"
else
    echo "  games: all"
fi

TAG_ARG=""
if [[ -n "$TAGS" ]]; then
    TAG_ARG="--tags=$TAGS"
fi

cd "$PROJECT_DIR"
export S_M_BASE_URL="http://localhost:${SM_PORT}"
unset AGENTICA_BASE_URL AGENTICA_API_KEY 2>/dev/null || true
# shellcheck disable=SC2086
uv run main.py --agent="$AGENT" $GAME_ARG $TAG_ARG

echo ""
echo "=== Done ==="
