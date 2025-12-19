#!/usr/bin/env python3
"""
Danger Map Manager

Main script that manages the danger map overlay on the pheromone map.
- Receives grid parameters from pheromone map dynamically
- Displays danger map as marker array (white cells with opacity)
- Handles adding/deleting dangers from other scripts
- Moves dynamic dangers along their paths
- Saves to JSON every 5 seconds

Usage:
    python3 scripts/danger/danger_map_manager.py --ros-args \
        -p data_file:=data/danger_map.json \
        -p z_position:=0.15 \
        -p frame_id:=world
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Float64MultiArray, String
import json
import os
import time
import threading
import uuid
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DangerMapManager(Node):
    """Manages danger map overlay on pheromone map."""
    
    # Encoding for /danger/add_static (PointStamped):
    # - point.z historically encoded only radius (int, in cells).
    # - We now also encode height (meters) in the fractional part to keep backward compatibility:
    #     z = radius_cells + (height_m * 0.001)
    #   where height_m is clamped to [0..999] to fit in the fractional part.
    _STATIC_Z_HEIGHT_SCALE = 0.001
    _DEFAULT_THREAT_HEIGHT_M = 50.0

    def __init__(self):
        super().__init__('danger_map_manager')
        # Ensure timers run on wall-time even if /use_sim_time was set globally elsewhere.
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        except Exception:
            pass
        
        # Parameters
        self.declare_parameter('data_file', 'data/danger_map.json')
        self.declare_parameter('z_position', 0.15)  # Slightly above pheromone map
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('save_interval', 5.0)  # seconds
        # Blinking for dynamic-danger creation preview:
        # - ON duration is `blink_interval`
        # - OFF duration is `blink_interval * blink_off_ratio`
        self.declare_parameter('blink_interval', 2.0)  # seconds ON
        self.declare_parameter('blink_off_ratio', 0.5)  # OFF = ON * ratio (default: OFF is half of ON)
        # Publish the non-danger background/grid at a slow rate so RViz restarts
        # still show the grid even when there are no dangers.
        self.declare_parameter('safe_refresh_interval', 10.0)  # seconds
        self.declare_parameter('default_dynamic_speed', 4.0)  # seconds per cell (default)
        # Visualization controls
        self.declare_parameter('dynamic_path_viz_enabled', True)
        # If True, close the LINE_STRIP (only makes sense for looping paths).
        self.declare_parameter('dynamic_path_viz_close_loop', True)
        # Dynamic danger radius visualization (separate topic; does not affect any logic).
        self.declare_parameter('dynamic_radius_viz_enabled', False)
        # Cap per-threat radius used for viz (cells) to keep RViz responsive.
        self.declare_parameter('dynamic_radius_viz_max_cells', 35)
        self.declare_parameter('dynamic_radius_viz_alpha', 0.18)
        
        # Get parameters
        data_file = self.get_parameter('data_file').value
        self.z_position = float(self.get_parameter('z_position').value)
        self.frame_id = self.get_parameter('frame_id').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        save_interval = float(self.get_parameter('save_interval').value)
        self.blink_interval = float(self.get_parameter('blink_interval').value)
        self.blink_off_ratio = float(self.get_parameter('blink_off_ratio').value)
        self.safe_refresh_interval = float(self.get_parameter('safe_refresh_interval').value)
        self.default_dynamic_speed = float(self.get_parameter('default_dynamic_speed').value)
        self.dynamic_path_viz_enabled = bool(self.get_parameter('dynamic_path_viz_enabled').value)
        self.dynamic_path_viz_close_loop = bool(self.get_parameter('dynamic_path_viz_close_loop').value)
        self.dynamic_radius_viz_enabled = bool(self.get_parameter('dynamic_radius_viz_enabled').value)
        self.dynamic_radius_viz_max_cells = int(self.get_parameter('dynamic_radius_viz_max_cells').value)
        self.dynamic_radius_viz_alpha = float(self.get_parameter('dynamic_radius_viz_alpha').value)

        # Drone pointer params (used to match pillar height/scale with drones in RViz).
        # Source: /swarm/cmd/drone_pointer_params = [enabled(0/1), z, scale, alpha]
        self._ptr_enabled: bool = True
        self._ptr_z: float = 8.0
        self._ptr_scale: float = 1.0
        self._ptr_alpha: float = 1.0
        
        # Resolve data file path
        workspace_root = Path(__file__).parent.parent.parent
        self.data_file_path = workspace_root / data_file
        self.data_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Grid parameters (received from pheromone map)
        self.grid_size = None
        self.cell_size = None
        self.grid_cells = None
        self.grid_half = None
        self.grid_params_received = False
        self.grid_params_lock = threading.Lock()
        
        # Danger data structures
        # Store radius+height per danger source cell (center cell), but publish only the center cell on /danger_map.
        # Radius is used by consumers (e.g., python_fast_sim) to paint pheromones around discovered dangers.
        self.static_dangers = {}  # {(cell_x, cell_y): {"radius": int, "height_m": float}}
        self.dynamic_dangers = {}  # {danger_id: {type, speed, radius, height_m, arrayOfCells, current_index, last_update_time}}
        self.creation_mode_dangers = {}  # {danger_id: {points: [(cell_x, cell_y, ...)], blink_state}}
        self.creation_mode_lock = threading.Lock()
        self._blink_next_toggle_time = time.time() + max(self.blink_interval, 0.01)  # start ON
        
        # Track changes for auto-save
        self.data_changed = False

        # Stable marker IDs for per-danger labels (avoid flicker / stale labels in RViz).
        self._dyn_label_base_id = 20000
        
        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/danger_map', 10)
        self.ack_pub = self.create_publisher(String, '/danger/ack', 10)
        # Metadata publisher for consumers: JSON list of danger source cells + radius + kind.
        self.meta_pub = self.create_publisher(String, '/danger_map_cells', 10)
        # Separate visualization topic for dynamic danger radius (orange disk on ground).
        self.dynamic_radius_pub = self.create_publisher(Marker, '/danger_map_dynamic_radius', 10)

        # Subscribe to pointer params so danger "pillar" matches drone pillar height.
        self.sub_pointer_params = self.create_subscription(
            Float64MultiArray,
            "/swarm/cmd/drone_pointer_params",
            self._on_pointer_params,
            10,
        )
        
        # Subscribers for grid parameters
        grid_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.grid_params_sub = self.create_subscription(
            Float64MultiArray,
            '/pheromone_grid_params',
            self.grid_params_callback,
            grid_qos
        )
        
        # Subscribers for danger commands
        self.add_static_sub = self.create_subscription(
            PointStamped,
            '/danger/add_static',
            self.add_static_callback,
            10
        )
        
        self.add_dynamic_point_sub = self.create_subscription(
            PointStamped,
            '/danger/add_dynamic_point',
            self.add_dynamic_point_callback,
            10
        )
        
        self.finish_dynamic_sub = self.create_subscription(
            String,
            '/danger/finish_dynamic',
            self.finish_dynamic_callback,
            10
        )
        
        # Timers
        # - Danger cells: keep fast updates
        self.create_timer(1.0 / publish_rate, self.publish_danger_map)
        # - Non-danger background/grid: refresh slowly (RViz restarts)
        self.create_timer(max(self.safe_refresh_interval, 0.5), self.publish_safe_layer)
        self.create_timer(save_interval, self.auto_save)
        self.create_timer(0.1, self.update_dynamic_dangers)  # Update dynamic dangers frequently
        self.create_timer(0.1, self.update_blink_state)  # Blink creation mode dangers (asymmetric duty cycle)
        
        # Load persisted data
        self.load_data()
        
        self.get_logger().info(
            f'Danger Map Manager initialized:\n'
            f'  Data file: {self.data_file_path}\n'
            f'  Z position: {self.z_position}m\n'
            f'  Frame: {self.frame_id}\n'
            f'  Publish rate: {publish_rate}Hz\n'
            f'  Save interval: {save_interval}s\n'
            f'  Waiting for grid parameters from pheromone map...'
        )

    def _on_pointer_params(self, msg: Float64MultiArray):
        """Keep local copy of drone pointer params (height/scale/alpha).

        Format: [enabled(0/1), z, scale, alpha]
        """
        try:
            if msg is None or len(msg.data) < 4:
                return
            enabled = bool(int(msg.data[0]) != 0)
            z = float(msg.data[1])
            scale = float(msg.data[2])
            alpha = float(msg.data[3])

            # Clamp to sane values (match python_fast_sim behavior).
            z = max(-50.0, min(200.0, z))
            scale = max(0.1, min(10.0, scale))
            alpha = max(0.0, min(1.0, alpha))

            self._ptr_enabled = bool(enabled)
            self._ptr_z = float(z)
            self._ptr_scale = float(scale)
            self._ptr_alpha = float(alpha)
        except Exception:
            return
    
    def grid_params_callback(self, msg: Float64MultiArray):
        """Handle grid parameters from pheromone map."""
        if len(msg.data) < 2:
            return
        
        new_grid_size = float(msg.data[0])
        new_cell_size = float(msg.data[1])
        
        with self.grid_params_lock:
            if not self.grid_params_received:
                self.grid_size = new_grid_size
                self.cell_size = new_cell_size
                self.grid_cells = int(self.grid_size / self.cell_size)
                self.grid_half = self.grid_size / 2.0
                self.grid_params_received = True
                
                self.get_logger().info(
                    f'✓ Received grid parameters:\n'
                    f'  Grid size: {self.grid_size}m\n'
                    f'  Cell size: {self.cell_size}m\n'
                    f'  Grid cells: {self.grid_cells} x {self.grid_cells}'
                )
            elif abs(self.grid_size - new_grid_size) > 0.1 or abs(self.cell_size - new_cell_size) > 0.001:
                self.get_logger().warn(
                    f'⚠ Grid parameters changed:\n'
                    f'  Old: grid_size={self.grid_size}m, cell_size={self.cell_size}m\n'
                    f'  New: grid_size={new_grid_size}m, cell_size={new_cell_size}m'
                )
                self.grid_size = new_grid_size
                self.cell_size = new_cell_size
                self.grid_cells = int(self.grid_size / self.cell_size)
                self.grid_half = self.grid_size / 2.0
    
    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to cell coordinates."""
        with self.grid_params_lock:
            if not self.grid_params_received:
                return None
            cell_x = int((x + self.grid_half) / self.cell_size)
            cell_y = int((y + self.grid_half) / self.cell_size)
            return (cell_x, cell_y)
    
    def cell_to_world(self, cell_x: int, cell_y: int) -> Optional[Tuple[float, float]]:
        """Convert cell coordinates to world coordinates (cell center)."""
        with self.grid_params_lock:
            if not self.grid_params_received:
                return None
            x = (cell_x * self.cell_size) - self.grid_half + (self.cell_size / 2.0)
            y = (cell_y * self.cell_size) - self.grid_half + (self.cell_size / 2.0)
            return (x, y)

    @staticmethod
    def _stable_id(s: str) -> int:
        """Deterministic small int ID for marker ids from a string."""
        try:
            return int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFF)
        except Exception:
            return 0

    @staticmethod
    def _bresenham_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Discrete line between two grid cells (inclusive) using Bresenham."""
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        out: List[Tuple[int, int]] = []
        while True:
            out.append((int(x0), int(y0)))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return out
    
    def add_static_callback(self, msg: PointStamped):
        """Handle adding static danger."""
        cell = self.world_to_cell(msg.point.x, msg.point.y)
        if cell is None:
            self.get_logger().warn('Cannot add static danger: grid parameters not received yet')
            return
        
        # Radius (cells) is encoded in the integer part of msg.point.z (backward compatible).
        # Height (meters) is encoded in the fractional part (see _STATIC_Z_HEIGHT_SCALE).
        z = float(msg.point.z)
        radius = int(z) if z > 0 else 0
        height_m = float(self._DEFAULT_THREAT_HEIGHT_M)
        try:
            frac = float(z) - float(int(z))
            # Only treat as encoded height if there is a meaningful fractional part.
            if abs(frac) >= (0.5 * float(self._STATIC_Z_HEIGHT_SCALE)):
                height_m = float(frac) / float(self._STATIC_Z_HEIGHT_SCALE)
        except Exception:
            height_m = float(self._DEFAULT_THREAT_HEIGHT_M)
        radius = int(max(0, radius))
        height_m = float(max(0.0, min(999.0, float(height_m))))
        self.static_dangers[cell] = {"radius": int(radius), "height_m": float(height_m)}
        self.data_changed = True
        
        self.get_logger().info(
            f'✓ Added static danger at cell {cell} (world: {msg.point.x:.2f}, {msg.point.y:.2f}) '
            f'r={radius} height={height_m:.1f}m'
        )
    
    def add_dynamic_point_callback(self, msg: PointStamped):
        """Handle adding point to dynamic danger in creation mode."""
        # Extract danger_id from frame_id (format: "danger_id:<uuid>")
        if not msg.header.frame_id.startswith('danger_id:'):
            self.get_logger().warn(f'Invalid frame_id format: {msg.header.frame_id}')
            return
        
        danger_id = msg.header.frame_id.replace('danger_id:', '')
        cell = self.world_to_cell(msg.point.x, msg.point.y)
        if cell is None:
            self.get_logger().warn('Cannot add dynamic point: grid parameters not received yet')
            return
        
        with self.creation_mode_lock:
            if danger_id not in self.creation_mode_dangers:
                self.creation_mode_dangers[danger_id] = {
                    'points': [],
                    'blink_state': True
                }

            pts: List[Tuple[int, int]] = self.creation_mode_dangers[danger_id]['points']
            # Interpolate missing cells between the last point and the new point (realistic continuous motion).
            if pts:
                last = pts[-1]
                seg = self._bresenham_cells(last, cell)
                # Skip the first cell (already in pts), append the rest in order.
                for cc in seg[1:]:
                    if cc not in pts:
                        pts.append(cc)
            else:
                if cell not in pts:
                    pts.append(cell)

            self.get_logger().info(
                f'✓ Added point to dynamic danger {danger_id[:8]}... at cell {cell} '
                f'(world: {msg.point.x:.2f}, {msg.point.y:.2f})'
            )
    
    def finish_dynamic_callback(self, msg: String):
        """Handle finishing dynamic danger creation."""
        # Parse message: can be just ID or JSON with ID and speed
        try:
            data = json.loads(msg.data)
            danger_id = data.get('id', '')
            speed_per_cell = float(data.get('speed', self.default_dynamic_speed))
            radius = int(data.get('radius', 0))
            height_m = float(data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M))
        except:
            # Fallback: treat as plain ID
            danger_id = msg.data.strip()
            speed_per_cell = self.default_dynamic_speed
            radius = 0
            height_m = float(self._DEFAULT_THREAT_HEIGHT_M)
        
        with self.creation_mode_lock:
            if danger_id not in self.creation_mode_dangers:
                self.get_logger().warn(f'Cannot finish dynamic danger {danger_id[:8]}...: not in creation mode')
                return
            
            points = self.creation_mode_dangers[danger_id]['points']
            if len(points) < 2:
                self.get_logger().warn(
                    f'Cannot finish dynamic danger {danger_id[:8]}...: need at least 2 points, got {len(points)}'
                )
                return

            # Forbid "dynamic" dangers that don't actually move (all points collapse to one cell).
            uniq = list(dict.fromkeys(points))  # preserve order, remove duplicates
            if len(uniq) < 2:
                self.get_logger().warn(
                    f'Cannot finish dynamic danger {danger_id[:8]}...: path has no movement (unique cells={len(uniq)})'
                )
                return
            points = uniq
            
            # Create dynamic danger
            self.dynamic_dangers[danger_id] = {
                'type': 'dynamic',
                'speed': speed_per_cell,  # seconds per cell
                'radius': int(max(0, radius)),
                'height_m': float(max(0.0, min(999.0, float(height_m)))),
                'arrayOfCells': points,
                'current_index': 0,
                'last_update_time': time.time()
            }
            
            # Remove from creation mode
            del self.creation_mode_dangers[danger_id]
            
            self.data_changed = True
            
            self.get_logger().info(
                f'✓ Finished dynamic danger {danger_id[:8]}... with {len(points)} points, '
                f'speed: {speed_per_cell}s per cell, height: {float(height_m):.1f}m'
            )
            
            # Send ACK
            ack_msg = String()
            ack_msg.data = danger_id
            self.ack_pub.publish(ack_msg)
    
    def update_blink_state(self):
        """Blink creation-mode dangers with OFF time shorter than ON time."""
        now = time.time()
        if now < self._blink_next_toggle_time:
            return

        # Clamp ratios to sane values
        off_ratio = self.blink_off_ratio
        if off_ratio <= 0.0:
            off_ratio = 0.5
        if off_ratio > 10.0:
            off_ratio = 10.0

        with self.creation_mode_lock:
            # Toggle all creation-mode dangers together
            any_creation = bool(self.creation_mode_dangers)
            if not any_creation:
                # Nothing to blink; check again later
                self._blink_next_toggle_time = now + 0.5
                return

            # Determine current state from first entry (they are kept in sync)
            first_state = next(iter(self.creation_mode_dangers.values()))['blink_state']
            new_state = not first_state

            for danger in self.creation_mode_dangers.values():
                danger['blink_state'] = new_state

        # Schedule next toggle based on new state (ON longer, OFF shorter)
        on_s = max(self.blink_interval, 0.01)
        off_s = max(on_s * off_ratio, 0.01)
        self._blink_next_toggle_time = now + (on_s if new_state else off_s)
    
    def update_dynamic_dangers(self):
        """Update positions of dynamic dangers along their paths."""
        current_time = time.time()
        
        for danger_id, danger_data in list(self.dynamic_dangers.items()):
            cells = danger_data['arrayOfCells']
            if len(cells) == 0:
                continue
            
            current_index = danger_data['current_index']
            speed = danger_data['speed']  # seconds per cell
            last_update = danger_data['last_update_time']
            
            # Check if enough time has passed to move to next cell
            time_elapsed = current_time - last_update
            if time_elapsed >= speed:
                # Move to next cell
                new_index = (current_index + 1) % len(cells)
                danger_data['current_index'] = new_index
                danger_data['last_update_time'] = current_time

    def publish_safe_layer(self):
        """Publish non-danger cells/grid (slow refresh).

        This is intentionally not published at high rate to avoid RViz lag on large grids.
        """
        if not self.grid_params_received:
            return

        with self.grid_params_lock:
            grid_size = float(self.grid_size)
            grid_half = float(self.grid_half)
            cell_size = float(self.cell_size)
            grid_cells = int(self.grid_cells)

        sheet_thickness = 0.1
        stamp = self.get_clock().now().to_msg()

        # Base overlay (safe area) — low alpha
        base = Marker()
        base.header.frame_id = self.frame_id
        base.header.stamp = stamp
        base.ns = 'danger_map'
        base.id = 0
        base.type = Marker.CUBE
        base.action = Marker.ADD
        base.pose.orientation.w = 1.0
        base.pose.position.x = 0.0
        base.pose.position.y = 0.0
        base.pose.position.z = float(self.z_position)
        base.scale.x = grid_size
        base.scale.y = grid_size
        base.scale.z = float(sheet_thickness)
        base.color.r = 1.0
        base.color.g = 1.0
        base.color.b = 1.0
        base.color.a = 0.05
        base.lifetime.sec = 0

        # Grid lines — gives you a visible grid without publishing millions of cubes
        grid = Marker()
        grid.header.frame_id = self.frame_id
        grid.header.stamp = stamp
        grid.ns = 'danger_map'
        grid.id = 2
        grid.type = Marker.LINE_LIST
        grid.action = Marker.ADD
        grid.scale.x = 0.06  # line width
        grid.color.r = 1.0
        grid.color.g = 1.0
        grid.color.b = 1.0
        grid.color.a = 0.05
        grid.points = []
        grid.lifetime.sec = 0

        z = float(self.z_position + 0.001)

        # Vertical lines
        for i in range(grid_cells + 1):
            x = -grid_half + i * cell_size
            p1 = Point()
            p1.x = float(x)
            p1.y = float(-grid_half)
            p1.z = z
            p2 = Point()
            p2.x = float(x)
            p2.y = float(grid_half)
            p2.z = z
            grid.points.append(p1)
            grid.points.append(p2)

        # Horizontal lines
        for j in range(grid_cells + 1):
            y = -grid_half + j * cell_size
            p1 = Point()
            p1.x = float(-grid_half)
            p1.y = float(y)
            p1.z = z
            p2 = Point()
            p2.x = float(grid_half)
            p2.y = float(y)
            p2.z = z
            grid.points.append(p1)
            grid.points.append(p2)

        self.marker_pub.publish(base)
        self.marker_pub.publish(grid)
    
    def publish_danger_map(self):
        """Publish only the danger cells (fast refresh)."""
        if not self.grid_params_received:
            return

        # Snapshot grid params
        with self.grid_params_lock:
            grid_half = float(self.grid_half)
            cell_size = float(self.cell_size)

        sheet_thickness = 0.1

        # Danger cells: cube list only for cells that are dangerous right now
        danger = Marker()
        danger.header.frame_id = self.frame_id
        danger.header.stamp = self.get_clock().now().to_msg()
        danger.ns = 'danger_map'
        danger.id = 1
        danger.type = Marker.CUBE_LIST
        danger.action = Marker.ADD
        danger.scale.x = cell_size
        danger.scale.y = cell_size
        danger.scale.z = float(sheet_thickness)
        danger.points = []
        danger.colors = []
        danger.lifetime.sec = 0

        # Collect all danger *source* cells (center cells only)
        danger_cells = set()
        danger_cells.update(self.static_dangers.keys())

        # Dynamic dangers (current position)
        for danger_data in self.dynamic_dangers.values():
            cells = danger_data['arrayOfCells']
            if cells:
                danger_cells.add(cells[danger_data['current_index']])

        # Creation mode dangers (blinking)
        with self.creation_mode_lock:
            for creation in self.creation_mode_dangers.values():
                if creation['blink_state']:
                    danger_cells.update(creation['points'])

        # Convert danger cells to points
        for (cell_x, cell_y) in danger_cells:
            # Center of cell in world coordinates
            x = (cell_x * cell_size) - grid_half + (cell_size / 2.0)
            y = (cell_y * cell_size) - grid_half + (cell_size / 2.0)
            p = Point()
            p.x = float(x)
            p.y = float(y)
            # Static threats: show the square at the danger altitude (height_m), not on the ground sheet.
            # Dynamic threats (and creation mode) remain on the ground sheet.
            z_here = float(self.z_position)
            try:
                info = self.static_dangers.get((int(cell_x), int(cell_y)))
                if isinstance(info, dict):
                    h = float(info.get("height_m", self._DEFAULT_THREAT_HEIGHT_M))
                    h = float(max(0.2, min(999.0, h)))
                    z_here = float(h)
            except Exception:
                z_here = float(self.z_position)
            p.z = float(z_here)
            danger.points.append(p)

            c = ColorRGBA()
            c.r = 1.0
            c.g = 1.0
            c.b = 1.0
            c.a = 0.1
            danger.colors.append(c)

        self.marker_pub.publish(danger)

        # Static danger pillar(s) (cyan): from ground to their configured height.
        static_pillar = Marker()
        static_pillar.header.frame_id = self.frame_id
        static_pillar.header.stamp = danger.header.stamp
        static_pillar.ns = "danger_map"
        static_pillar.id = 4
        static_pillar.type = Marker.LINE_LIST
        static_pillar.lifetime.sec = 0

        try:
            pts: List[Point] = []
            for (cell_x, cell_y), info in self.static_dangers.items():
                try:
                    r = int(info.get("radius", 0) or 0)
                    _ = r  # radius isn't used for pillars (only metadata), but keep parse symmetry
                except Exception:
                    pass
                try:
                    h = float(info.get("height_m", self._DEFAULT_THREAT_HEIGHT_M))
                except Exception:
                    h = float(self._DEFAULT_THREAT_HEIGHT_M)
                h = float(max(0.2, min(999.0, float(h))))
                x = (float(cell_x) * cell_size) - grid_half + (cell_size / 2.0)
                y = (float(cell_y) * cell_size) - grid_half + (cell_size / 2.0)
                p0 = Point()
                p0.x = float(x)
                p0.y = float(y)
                p0.z = 0.05
                p1 = Point()
                p1.x = float(x)
                p1.y = float(y)
                p1.z = float(h)
                pts.append(p0)
                pts.append(p1)

            if not pts:
                static_pillar.action = Marker.DELETE
                self.marker_pub.publish(static_pillar)
            else:
                static_pillar.action = Marker.ADD
                static_pillar.scale.x = float(max(0.01, 0.35 * float(self._ptr_scale)))
                static_pillar.color.r = 0.2
                static_pillar.color.g = 1.0
                static_pillar.color.b = 1.0
                static_pillar.color.a = float(max(0.0, min(1.0, 0.9 * float(self._ptr_alpha))))
                static_pillar.points = pts
                self.marker_pub.publish(static_pillar)
        except Exception:
            # Don't break the main marker publishing if pillar creation fails.
            try:
                static_pillar.action = Marker.DELETE
                self.marker_pub.publish(static_pillar)
            except Exception:
                pass

        # Dynamic danger current-position marker(s): red ball at the current cell + threat height.
        # Note: publish on the same /danger_map topic but with a different id.
        dyn_ball = Marker()
        dyn_ball.header.frame_id = self.frame_id
        dyn_ball.header.stamp = danger.header.stamp
        dyn_ball.ns = "danger_map"
        dyn_ball.id = 3
        dyn_ball.type = Marker.SPHERE_LIST
        dyn_ball.lifetime.sec = 0
        dyn_ball.pose.orientation.w = 1.0

        # If the drone pointer viz is disabled, remove the pillar too (keeps UI semantics consistent).
        if not bool(self._ptr_enabled):
            dyn_ball.action = Marker.DELETE
            self.marker_pub.publish(dyn_ball)
        else:
            pts: List[Point] = []
            for danger_data in self.dynamic_dangers.values():
                cells = danger_data.get("arrayOfCells", [])
                if not cells:
                    continue
                idx = int(danger_data.get("current_index", 0)) % len(cells)
                cell_x, cell_y = cells[idx]
                x = (float(cell_x) * cell_size) - grid_half + (cell_size / 2.0)
                y = (float(cell_y) * cell_size) - grid_half + (cell_size / 2.0)

                # Dynamic threats have their own height_m (default 50m). Fall back to pointer z if missing.
                try:
                    h = float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                except Exception:
                    h = float(self._DEFAULT_THREAT_HEIGHT_M)
                if h <= 0.0:
                    h = float(self._DEFAULT_THREAT_HEIGHT_M)
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = float(max(0.2, float(h)))
                pts.append(p)

            if not pts:
                dyn_ball.action = Marker.DELETE
                self.marker_pub.publish(dyn_ball)
            else:
                dyn_ball.action = Marker.ADD
                # Diameter relative to cell size (scaled with pointer scale for UI consistency).
                d = float(max(0.2, 0.65 * float(cell_size) * float(self._ptr_scale)))
                dyn_ball.scale.x = float(d)
                dyn_ball.scale.y = float(d)
                dyn_ball.scale.z = float(d)
                dyn_ball.color.r = 1.0
                dyn_ball.color.g = 0.0
                dyn_ball.color.b = 0.0
                dyn_ball.color.a = float(max(0.0, min(1.0, float(self._ptr_alpha))))
                dyn_ball.points = pts
                self.marker_pub.publish(dyn_ball)

        # Dynamic danger speed labels (TEXT_VIEW_FACING) at the same "pointer" height convention.
        try:
            stamp = danger.header.stamp
            if bool(self._ptr_enabled) and self.dynamic_dangers:
                for danger_id, danger_data in self.dynamic_dangers.items():
                    cells = danger_data.get("arrayOfCells", [])
                    if not cells:
                        continue
                    idx = int(danger_data.get("current_index", 0)) % len(cells)
                    cell_x, cell_y = cells[idx]
                    x = (float(cell_x) * cell_size) - grid_half + (cell_size / 2.0)
                    y = (float(cell_y) * cell_size) - grid_half + (cell_size / 2.0)
                    # Keep labels glued to the red ball (same XY and same altitude baseline).
                    try:
                        h = float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                    except Exception:
                        h = float(self._DEFAULT_THREAT_HEIGHT_M)
                    if h <= 0.0:
                        h = float(self._DEFAULT_THREAT_HEIGHT_M)
                    h = float(max(0.2, min(999.0, float(h))))
                    # Match ball diameter logic (so we can place text just above it).
                    ball_d = float(max(0.2, 0.65 * float(cell_size) * float(self._ptr_scale)))
                    sp_s = float(danger_data.get("speed", self.default_dynamic_speed) or self.default_dynamic_speed)
                    sp_s = max(0.01, sp_s)
                    # Display both sec/cell and approx m/s for readability.
                    mps = float(cell_size) / float(sp_s)

                    t = Marker()
                    t.header.frame_id = self.frame_id
                    t.header.stamp = stamp
                    t.ns = "danger_map_dyn_speed"
                    t.id = int(self._dyn_label_base_id + self._stable_id(str(danger_id)))
                    t.type = Marker.TEXT_VIEW_FACING
                    t.action = Marker.ADD
                    t.pose.position.x = float(x)
                    t.pose.position.y = float(y)
                    # Put the text right above the red ball (prevents parallax "lag" in RViz).
                    t.pose.position.z = float(h + 0.5 * float(ball_d) + max(0.25, 0.35 * float(self._ptr_scale)))
                    t.pose.orientation.w = 1.0
                    t.scale.z = float(max(0.2, 0.9 * float(self._ptr_scale)))
                    t.color.r = 1.0
                    t.color.g = 0.35
                    t.color.b = 0.0
                    t.color.a = float(max(0.0, min(1.0, float(self._ptr_alpha))))
                    t.text = f"{sp_s:.2f}s/cell  ({mps:.2f}m/s)"
                    # Short lifetime prevents stale labels if a danger is removed.
                    t.lifetime.sec = 1
                    self.marker_pub.publish(t)
        except Exception:
            pass

        # Dynamic danger trajectories (red LINE_STRIP) at threat altitude.
        # Published on /danger_map (same topic as other danger overlays).
        try:
            if bool(self._ptr_enabled) and bool(self.dynamic_path_viz_enabled) and self.dynamic_dangers:
                stamp = danger.header.stamp
                close_loop = bool(self.dynamic_path_viz_close_loop)
                for danger_id, danger_data in self.dynamic_dangers.items():
                    try:
                        cells = list(danger_data.get("arrayOfCells", []) or [])
                        if len(cells) < 2:
                            continue
                        # Threat height (meters)
                        try:
                            h = float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                        except Exception:
                            h = float(self._DEFAULT_THREAT_HEIGHT_M)
                        h = float(max(0.2, min(999.0, float(h))))

                        tr = Marker()
                        tr.header.frame_id = self.frame_id
                        tr.header.stamp = stamp
                        tr.ns = "danger_map_dyn_path"
                        # Use a larger stable ID space to avoid collisions across many dangers.
                        tr.id = int(10000 + (zlib.crc32(str(danger_id).encode("utf-8")) & 0x7FFFFFFF) % 1000000)
                        tr.type = Marker.LINE_STRIP
                        tr.action = Marker.ADD
                        tr.pose.orientation.w = 1.0
                        tr.scale.x = float(max(0.03, 0.18 * float(self._ptr_scale)))
                        tr.color.r = 1.0
                        tr.color.g = 0.0
                        tr.color.b = 0.0
                        tr.color.a = float(max(0.0, min(1.0, 0.7 * float(self._ptr_alpha))))
                        # Short lifetime prevents stale path if danger removed.
                        tr.lifetime.sec = 1
                        tr.points = []

                        for (cell_x, cell_y) in cells:
                            x = (float(cell_x) * cell_size) - grid_half + (cell_size / 2.0)
                            y = (float(cell_y) * cell_size) - grid_half + (cell_size / 2.0)
                            p = Point()
                            p.x = float(x)
                            p.y = float(y)
                            p.z = float(h)
                            tr.points.append(p)
                        if close_loop and len(tr.points) >= 2:
                            tr.points.append(tr.points[0])
                        self.marker_pub.publish(tr)
                    except Exception:
                        continue
        except Exception:
            pass

        # Dynamic danger radius visualization (orange disk on ground) on separate topic.
        # This is *visual-only*; the radius itself is used by consumers via /danger_map_cells metadata.
        try:
            radm = Marker()
            radm.header.frame_id = self.frame_id
            radm.header.stamp = danger.header.stamp
            radm.ns = "danger_map_dyn_radius"
            radm.id = 0
            radm.type = Marker.CUBE_LIST
            radm.pose.orientation.w = 1.0
            radm.scale.x = float(cell_size)
            radm.scale.y = float(cell_size)
            radm.scale.z = 0.05
            radm.lifetime.sec = 1

            if (not bool(self._ptr_enabled)) or (not bool(self.dynamic_radius_viz_enabled)):
                radm.action = Marker.DELETE
                self.dynamic_radius_pub.publish(radm)
            else:
                max_r = int(max(0, min(80, int(self.dynamic_radius_viz_max_cells))))
                alpha = float(max(0.0, min(1.0, float(self.dynamic_radius_viz_alpha) * float(self._ptr_alpha))))
                pts: List[Point] = []
                cols: List[ColorRGBA] = []
                z0 = float(self.z_position + 0.025)
                for _, danger_data in (self.dynamic_dangers or {}).items():
                    cells = danger_data.get("arrayOfCells", [])
                    if not cells:
                        continue
                    r = int(danger_data.get("radius", 0) or 0)
                    r = int(max(0, min(max_r, r)))
                    if r <= 0:
                        continue
                    idx = int(danger_data.get("current_index", 0)) % len(cells)
                    cx0, cy0 = int(cells[idx][0]), int(cells[idx][1])
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if (dx * dx + dy * dy) > (r * r):
                                continue
                            ccx = int(cx0 + dx)
                            ccy = int(cy0 + dy)
                            x = (float(ccx) * cell_size) - grid_half + (cell_size / 2.0)
                            y = (float(ccy) * cell_size) - grid_half + (cell_size / 2.0)
                            p = Point()
                            p.x = float(x)
                            p.y = float(y)
                            p.z = float(z0)
                            pts.append(p)
                            c = ColorRGBA()
                            c.r = 1.0
                            c.g = 0.45
                            c.b = 0.0
                            c.a = float(alpha)
                            cols.append(c)
                if not pts:
                    radm.action = Marker.DELETE
                    self.dynamic_radius_pub.publish(radm)
                else:
                    radm.action = Marker.ADD
                    radm.points = pts
                    radm.colors = cols
                    self.dynamic_radius_pub.publish(radm)
        except Exception:
            pass

        # Publish metadata for consumers (radius etc.)
        try:
            meta = []
            for (cx, cy), info in self.static_dangers.items():
                r = int(info.get("radius", 0) or 0) if isinstance(info, dict) else int(info or 0)
                hm = float(info.get("height_m", self._DEFAULT_THREAT_HEIGHT_M)) if isinstance(info, dict) else float(self._DEFAULT_THREAT_HEIGHT_M)
                meta.append({"cell_x": int(cx), "cell_y": int(cy), "radius": int(r), "height_m": float(hm), "kind": "static"})
            dyn_full = []
            for danger_id, danger_data in self.dynamic_dangers.items():
                cells = danger_data.get("arrayOfCells", [])
                if not cells:
                    continue
                cx, cy = cells[int(danger_data.get("current_index", 0)) % len(cells)]
                sp = float(danger_data.get("speed", self.default_dynamic_speed) or self.default_dynamic_speed)
                hm = float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                meta.append(
                    {
                        "cell_x": int(cx),
                        "cell_y": int(cy),
                        "radius": int(danger_data.get("radius", 0)),
                        "id": str(danger_id),
                        "kind": "dynamic",
                        "speed": float(sp),
                        "height_m": float(hm),
                    }
                )
                # Rich dynamic metadata (path + current index) for consumers that want trajectory/risk.
                dyn_full.append(
                    {
                        "id": str(danger_id),
                        "type": "dynamic",
                        "speed": float(sp),
                        "radius": int(danger_data.get("radius", 0)),
                        "height_m": float(hm),
                        "current_index": int(danger_data.get("current_index", 0)),
                        "arrayOfCells": [{"cell_x": int(x), "cell_y": int(y)} for (x, y) in cells],
                    }
                )
            m = String()
            m.data = json.dumps({"time": time.time(), "cells": meta, "dynamic": dyn_full})
            self.meta_pub.publish(m)
        except Exception:
            pass
    
    def load_data(self):
        """Load danger map data from JSON file."""
        if not self.data_file_path.exists():
            self.get_logger().info(f'No existing danger map data found at {self.data_file_path}, starting fresh')
            return
        
        try:
            with open(self.data_file_path, 'r') as f:
                data = json.load(f)
            
            # Load static dangers
            if 'static' in data:
                for cell_data in data['static']:
                    cell = (cell_data['cell_x'], cell_data['cell_y'])
                    r = int(cell_data.get('danger_radius', cell_data.get('radius', 0)) or 0)
                    hm = float(cell_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                    self.static_dangers[cell] = {"radius": int(max(0, r)), "height_m": float(max(0.0, min(999.0, hm)))}
            
            # Load dynamic dangers
            if 'dynamic' in data:
                for danger_data in data['dynamic']:
                    danger_id = danger_data.get('id', str(uuid.uuid4()))
                    cells = [(c['cell_x'], c['cell_y']) for c in danger_data.get('arrayOfCells', [])]
                    hm = float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M) or self._DEFAULT_THREAT_HEIGHT_M)
                    # Forbid degenerate "dynamic" paths with no movement.
                    uniq = list(dict.fromkeys(cells))
                    if len(uniq) > 1:
                        self.dynamic_dangers[danger_id] = {
                            'type': 'dynamic',
                            'speed': danger_data.get('speed', 4.0),
                            'radius': int(danger_data.get('danger_radius', danger_data.get('radius', 0)) or 0),
                            'height_m': float(max(0.0, min(999.0, hm))),
                            'arrayOfCells': uniq,
                            'current_index': 0,
                            'last_update_time': time.time()
                        }
                    elif len(uniq) == 1:
                        self.get_logger().warn(f"Skipping dynamic danger {str(danger_id)[:8]}...: no movement (1 unique cell)")
            
            static_count = len(self.static_dangers)
            dynamic_count = len(self.dynamic_dangers)
            
            self.get_logger().info(
                f'Loaded danger map data:\n'
                f'  Static dangers: {static_count}\n'
                f'  Dynamic dangers: {dynamic_count}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load danger map data: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def save_data(self):
        """Save danger map data to JSON file."""
        if not self.data_changed:
            return
        
        try:
            # Convert to JSON format
            data = {
                'static': [
                    {
                        'cell_x': cell_x,
                        'cell_y': cell_y,
                        'danger_radius': int((self.static_dangers[(cell_x, cell_y)] or {}).get("radius", 0)),
                        'height_m': float((self.static_dangers[(cell_x, cell_y)] or {}).get("height_m", self._DEFAULT_THREAT_HEIGHT_M)),
                    }
                    for (cell_x, cell_y) in self.static_dangers.keys()
                ],
                'dynamic': [
                    {
                        'id': danger_id,
                        'type': 'dynamic',
                        'speed': danger_data['speed'],
                        'danger_radius': int(danger_data.get('radius', 0)),
                        'height_m': float(danger_data.get("height_m", self._DEFAULT_THREAT_HEIGHT_M)),
                        'arrayOfCells': [
                            {'cell_x': cell_x, 'cell_y': cell_y}
                            for (cell_x, cell_y) in danger_data['arrayOfCells']
                        ]
                    }
                    for danger_id, danger_data in self.dynamic_dangers.items()
                ]
            }
            
            # Write atomically
            temp_file = str(self.data_file_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.replace(temp_file, self.data_file_path)
            
            static_count = len(self.static_dangers)
            dynamic_count = len(self.dynamic_dangers)
            
            self.get_logger().info(
                f'Saved danger map data:\n'
                f'  Static dangers: {static_count}\n'
                f'  Dynamic dangers: {dynamic_count}\n'
                f'  File: {self.data_file_path}'
            )
            
            self.data_changed = False
        except Exception as e:
            self.get_logger().error(f'Failed to save danger map data: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def auto_save(self):
        """Auto-save data periodically if changed."""
        self.save_data()


def main(args=None):
    rclpy.init(args=args)
    node = DangerMapManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Save data before shutdown
        node.save_data()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

