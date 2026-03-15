#!/usr/bin/env bash
# Launch all 4 teleop nodes in a single tmux window with 4 panes.
# Usage: ./launch_nodes.sh
#
# Layout (2x2 grid):
#   ┌──────────────┬──────────────┐
#   │   camera     │    gello     │
#   ├──────────────┼──────────────┤
#   │     yam      │    broker    │
#   └──────────────┴──────────────┘
#
# To kill all nodes: tmux kill-session -t teleop

SESSION="teleop"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Create session with first pane (camera)
tmux new-session -d -s "$SESSION" -c "$DIR" \
    -x "$(tput cols)" -y "$(tput lines)"
tmux send-keys -t "$SESSION" \
    "conda activate gello && python -m yam_teleop.nodes.camera_node --config configs/camera.yaml" Enter

# Pane 2: gello (split right)
tmux split-window -h -t "$SESSION" -c "$DIR"
tmux send-keys -t "$SESSION" \
    "conda activate gello && python -m yam_teleop.nodes.gello_node --config configs/gello.yaml" Enter

# Pane 3: yam (split bottom-left)
tmux select-pane -t "$SESSION":0.0
tmux split-window -v -t "$SESSION" -c "$DIR"
tmux send-keys -t "$SESSION" \
    "conda activate gello && python -m yam_teleop.nodes.robot_node --config configs/robot.yaml" Enter

# Pane 4: broker (split bottom-right)
tmux select-pane -t "$SESSION":0.1
tmux split-window -v -t "$SESSION" -c "$DIR"
tmux send-keys -t "$SESSION" \
    "conda activate gello && python -m yam_teleop.nodes.sync_broker --config configs/broker.yaml" Enter

# Attach to the session
tmux attach-session -t "$SESSION"
