#!/bin/bash
# Launch script for Swarm Stigmergy GUI
# This script sources ROS 2 Humble and starts the control GUI

# Exit on error
set -e

# Change to script directory
cd "$(dirname "$0")"

# Source ROS 2 environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "ROS 2 Humble environment sourced"
else
    echo "Error: ROS 2 Humble not found at /opt/ros/humble/setup.bash"
    exit 1
fi

# Add system Python packages to PYTHONPATH so ROS can find numpy
export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

# Verify numpy is available
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "Error: numpy not found in Python environment"
    echo "Please install: sudo apt-get install python3-numpy"
    exit 1
fi
echo "numpy found: OK"

# Launch the GUI
echo "Launching Swarm Control GUI..."
exec python3 scripts/python_sim/swarm_control_gui.py
