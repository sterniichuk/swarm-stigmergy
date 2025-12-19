# Swarm Stigmergy Lab

A ROS 2-based multi-agent coordination framework using **digital stigmergy** (pheromones). This project provides a high-performance simulation environment for large-scale drone swarms to perform collective exploration, target search, and navigation in complex environments with static and dynamic obstacles.

## 🚀 Key Features

-   **Digital Stigmergy**: Decentralized coordination via a shared, sparse pheromone grid.
-   **Fast Simulation**: Time-accelerated multi-agent simulation capable of handling dozens of drones simultaneously.
-   **Intelligent Path Planning**: Hybrid planning using Ant Colony Optimization (ACO) for exploration and A* for deterministic navigation.
-   **Building & Terrain Awareness**: 3D obstacle avoidance using spatial indexing of buildings loaded from Gazebo SDF worlds.
-   **Danger Map Management**: Support for static and dynamic "no-fly zones" with real-time updates and persistence.
-   **Energy & Mission Logic**: Drones manage energy levels, automatically returning to base for recharge and mission handovers.
-   **ROS 2 Ecosystem**: Built on ROS 2 (Humble/Foxy), with full RViz integration for 3D visualization of pheromones, paths, and drone states.

## 🛠 Prerequisites

-   **ROS 2** (Humble or newer recommended)
-   **Python 3.10+**
-   **Dependencies**: Listed in `requirements.txt`

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/swarm-stigmergy.git
    cd swarm-stigmergy
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Source your ROS 2 environment**:
    ```bash
    source /opt/ros/humble/setup.bash
    ```

## 🎮 Usage

The project consists of a central control GUI and an RViz visualization window.

1.  **Launch the Swarm Control GUI**:
    ```bash
    python3 scripts/python_sim/swarm_control_gui.py
    ```
    This script initializes the ROS 2 executor, the fast simulation node, and the Qt management interface.

2.  **Launch RViz**:
    In a separate terminal, run:
    ```bash
    rviz2 -d config/rviz-config.rviz
    ```

3.  **Start Simulation**:
    - In the GUI, click **"Start Python Drones"**.
    - Use the **Speed slider** to accelerate time (up to 100x).
    - Toggle **"Target Add Mode"** and use the **"Publish Point"** tool in RViz to place targets on the map.

## 🏗 Architecture

-   **`scripts/python_sim/`**: Core simulation logic, including `PythonFastSim` and agent behavior (ACO/A* planning).
-   **`scripts/danger/`**: Management of threat zones and dynamic obstacles via `DangerMapManager`.
-   **`scripts/publishers/`**: ROS 2 nodes for environment visualization (buildings, ground plane, markers).
-   **`data/`**: Mission data, including `city_map.osm` for building footprints and persistence files for pheromone states.
-   **`worlds/`**: SDF world definitions used to synchronize simulation obstacles with RViz.