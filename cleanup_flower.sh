#!/bin/bash
# Flower Federated Learning Cleanup Script
# Terminates all processes, tmux sessions, and frees up ports

echo "🧹 Starting Flower cleanup..."

# Kill all Flower processes
echo "🔧 Terminating Flower processes..."
pkill -f "flower-superlink" 2>/dev/null || true
pkill -f "flwr-serverapp" 2>/dev/null || true
pkill -f "flower-supernode" 2>/dev/null || true
pkill -f "flwr-clientapp" 2>/dev/null || true

# Kill SSH tunnels
echo "🔗 Terminating SSH tunnels..."
pkill -f "ssh.*-L.*:909[2-9]" 2>/dev/null || true

# Kill tmux sessions
echo "📺 Terminating tmux sessions..."
tmux kill-session -t flower_server 2>/dev/null || true
tmux kill-session -t flower_serverapp 2>/dev/null || true
tmux kill-session -t flower_supernode_0 2>/dev/null || true
tmux kill-session -t flower_supernode_1 2>/dev/null || true
tmux kill-session -t flower_supernode_2 2>/dev/null || true
tmux kill-session -t flower_clientapp_0 2>/dev/null || true
tmux kill-session -t flower_clientapp_1 2>/dev/null || true
tmux kill-session -t flower_clientapp_2 2>/dev/null || true

# Force kill processes on specific ports
echo "🔌 Freeing up ports..."
fuser -k 9092/tcp 2>/dev/null || true  # Fleet API
fuser -k 9093/tcp 2>/dev/null || true  # REST API
fuser -k 9094/tcp 2>/dev/null || true  # Client 0
fuser -k 9095/tcp 2>/dev/null || true  # Client 1
fuser -k 9096/tcp 2>/dev/null || true  # Client 2

# Wait a moment for processes to terminate
sleep 3

# Check if ports are now free
echo "🔍 Checking port status..."
echo "Port 9092 (Fleet API):"
lsof -i :9092 || echo "  ✅ Port 9092 is free"
echo "Port 9093 (REST API):"
lsof -i :9093 || echo "  ✅ Port 9093 is free"
echo "Port 9094 (Client 0):"
lsof -i :9094 || echo "  ✅ Port 9094 is free"

# Check remaining tmux sessions
echo "📺 Remaining tmux sessions:"
tmux list-sessions 2>/dev/null || echo "  ✅ No tmux sessions running"

echo "✅ Flower cleanup completed!"
echo "💡 You can now restart your federated learning setup"
