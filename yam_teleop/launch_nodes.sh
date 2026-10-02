#!/usr/bin/env bash
# Launch the YAM teleop stack in one tmux window — one pane per step.
#
# Each pane is PRE-LOADED with its command but NOT executed. Attach, then press
# Enter in each pane, in order (1 -> 5), to start it. Start order matters:
#   1 cameras -> 2 followers -> 3 leaders -> 4 broker, then 5 collect.
# The step number is shown in each pane's border.
#
# Usage:
#   ./launch_nodes.sh                  # YAM leaders, real USB cameras (default)
#   ./launch_nodes.sh --leader gello   # GELLO (Dynamixel) leaders instead
#   ./launch_nodes.sh --camera-mock    # synthetic frames (no USB cameras)
#   ./launch_nodes.sh --robot-config <yaml>  # override the follower config
#   ./launch_nodes.sh --no-attach      # build the session but don't attach
#
# --leader picks the leader node, its config, the follower config, and the
# collection script:
#   yam   -> yam_leader_node + leader.yaml, robot.yaml,       collect_yam
#   gello -> gello_node      + gello.yaml,  robot_gello.yaml, collect_data
#            (needs `uv sync --extra gello`)
#
# Kill everything:  tmux kill-session -t teleop

set -euo pipefail

SESSION="teleop"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # repo root — uv + the relative paths below resolve here

usage() {
    sed -n '2,24p' "$0" | sed 's/^#\{1,2\} \{0,1\}//'
}

# --- options ---
CAMERA_CONFIG="yam_teleop/configs/camera.yaml"
LEADER="yam"
ROBOT_CONFIG=""
ATTACH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera-mock)  CAMERA_CONFIG="yam_teleop/configs/camera_mock.yaml" ;;
        --leader)       LEADER="${2:-}"; shift ;;
        --robot-config) ROBOT_CONFIG="${2:-}"; shift ;;
        --no-attach)    ATTACH=0 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
    shift
done

case "$LEADER" in
    yam)
        RUN="uv run"
        DEFAULT_ROBOT_CONFIG="yam_teleop/configs/robot.yaml"
        LEADER_STEP="[3] leaders: YAM teaching handles|${RUN} python -m yam_teleop.nodes.yam_leader_node --config yam_teleop/configs/leader.yaml"
        COLLECT="yam_teleop.scripts.collect_yam" ;;
    gello)
        RUN="uv run --extra gello"
        DEFAULT_ROBOT_CONFIG="yam_teleop/configs/robot_gello.yaml"
        LEADER_STEP="[3] leaders: GELLO|${RUN} python -m yam_teleop.nodes.gello_node --config yam_teleop/configs/gello.yaml"
        COLLECT="yam_teleop.scripts.collect_data" ;;
    *) echo "Unknown --leader '${LEADER}' (expected yam or gello)" >&2; exit 1 ;;
esac
ROBOT_CONFIG="${ROBOT_CONFIG:-$DEFAULT_ROBOT_CONFIG}"

# "pane-border title | command to pre-type" (| never appears in a command)
STEPS=(
  "[1] cameras: ${CAMERA_CONFIG}|${RUN} python -m yam_teleop.nodes.camera_node --config ${CAMERA_CONFIG}"
  "[2] followers: ${ROBOT_CONFIG}  (grippers auto-calibrate on startup -- keep them CLEAR)|${RUN} python -m yam_teleop.nodes.robot_node --config ${ROBOT_CONFIG}"
  "${LEADER_STEP}"
  "[4] sync broker|${RUN} python -m yam_teleop.nodes.sync_broker --config yam_teleop/configs/broker.yaml"
  "[5] collect  (start AFTER 1-4 are up; edit <task_name>)|${RUN} python -m ${COLLECT} --env-config yam_teleop/configs/env.yaml --output-dir data/<task_name>"
)

# --- fresh session, sized to the current terminal ---
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    -x "$(tput cols 2>/dev/null || echo 200)" \
    -y "$(tput lines 2>/dev/null || echo 50)"

# Label each pane in its border (best-effort; ignored on very old tmux).
tmux set-window-option -t "$SESSION" pane-border-status top 2>/dev/null || true
tmux set-window-option -t "$SESSION" pane-border-format " #{pane_index}: #{pane_title} " 2>/dev/null || true

first_pane=""
for step in "${STEPS[@]}"; do
    title="${step%%|*}"
    cmd="${step#*|}"
    if [[ -z "$first_pane" ]]; then
        pane="$(tmux display-message -p -t "$SESSION" '#{pane_id}')"
        first_pane="$pane"
    else
        pane="$(tmux split-window -t "$SESSION" -c "$ROOT" -P -F '#{pane_id}')"
        tmux select-layout -t "$SESSION" tiled >/dev/null
    fi
    tmux select-pane -t "$pane" -T "$title"
    # -l = send literally, and NO trailing Enter: the command sits at the prompt,
    # pre-typed and ready. Press Enter in the pane to run it.
    tmux send-keys -t "$pane" -l "$cmd"
done

tmux select-layout -t "$SESSION" tiled >/dev/null
tmux select-pane -t "$first_pane"

# --- attach (or switch if already inside tmux) ---
if [[ "$ATTACH" -eq 0 ]]; then
    echo "Session '$SESSION' ready (detached). Attach with: tmux attach -t $SESSION"
elif [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach-session -t "$SESSION"
fi
