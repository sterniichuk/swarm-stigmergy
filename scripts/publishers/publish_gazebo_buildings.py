#!/usr/bin/env python3
"""
Publish Gazebo buildings as visualization markers in RViz2.

Reads the city_map.sdf file and publishes building models as MarkerArray
so they can be visualized in RViz2.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32
import xml.etree.ElementTree as ET
import os
import math


class GazeboBuildingsPublisher(Node):
    """Publish Gazebo buildings as RViz2 markers."""
    
    def __init__(self):
        super().__init__('gazebo_buildings_publisher')
        # Ensure timer uses wall-time even if /use_sim_time was set elsewhere.
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        except Exception:
            pass
        
        # Declare parameters
        self.declare_parameter('world_file', 'worlds/city_map.sdf')
        # Publishing buildings frequently is expensive in RViz (reprocessing thousands of markers).
        # With TRANSIENT_LOCAL QoS (latched), it's safe to publish rarely.
        self.declare_parameter('publish_rate', 0.1)  # Hz (default: every 10s)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('marker_lifetime', 0)  # 0 = infinite
        self.declare_parameter('mirror_x', False)  # Mirror along X axis (flip Y)
        self.declare_parameter('mirror_y', False)  # Mirror along Y axis (flip X)
        self.declare_parameter('rotate_90', 0)  # Rotate 90 degrees (0, 1, 2, or 3 times)
        self.declare_parameter('offset_x', 0.0)  # Offset X coordinate
        self.declare_parameter('offset_y', 0.0)  # Offset Y coordinate
        self.declare_parameter('debug', False)  # Print coordinates for debugging
        self.declare_parameter('building_alpha', 1.0)  # 1.0 = opaque (recommended to avoid seeing maps through buildings)
        # Live updates (for GUI): publish alpha to this topic to change opacity without restarting the node.
        self.declare_parameter('alpha_topic', '/swarm/cmd/building_alpha')
        self.declare_parameter('use_sdf_material_colors', True)  # approximate Gazebo building colors in RViz (diffuse/ambient)
        
        # Get parameters
        world_file = self.get_parameter('world_file').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.frame_id = self.get_parameter('frame_id').value
        marker_lifetime = int(self.get_parameter('marker_lifetime').value)
        self.mirror_x = bool(self.get_parameter('mirror_x').value)
        self.mirror_y = bool(self.get_parameter('mirror_y').value)
        self.rotate_90 = int(self.get_parameter('rotate_90').value)
        self.offset_x = float(self.get_parameter('offset_x').value)
        self.offset_y = float(self.get_parameter('offset_y').value)
        self.debug = bool(self.get_parameter('debug').value)
        self.building_alpha = float(self.get_parameter('building_alpha').value)
        self.alpha_topic = self.get_parameter('alpha_topic').value
        self.use_sdf_material_colors = bool(self.get_parameter('use_sdf_material_colors').value)
        
        # Resolve world file path (repo-root relative by default).
        # Repo layout: <repo>/scripts/publishers/this_file.py → repo_root = parents[2]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        world_path = os.path.join(repo_root, world_file)
        
        if not os.path.exists(world_path):
            self.get_logger().error(f'World file not found: {world_path}')
            self.get_logger().info('Please provide correct path to world file')
            raise FileNotFoundError(f'World file not found: {world_path}')
        
        # Parse SDF file
        self.get_logger().info(f'Loading buildings from: {world_path}')
        buildings = self._parse_buildings(world_path)
        self.get_logger().info(f'Found {len(buildings)} buildings')
        
        # Debug: Print first few building coordinates
        if self.debug and buildings:
            self.get_logger().info('=== DEBUG: First 5 building coordinates (original) ===')
            for i, building in enumerate(buildings[:5]):
                self.get_logger().info(
                    f'Building {i+1}: {building["name"]} '
                    f'at ({building["x"]:.2f}, {building["y"]:.2f}, {building["z"]:.2f})'
                )
        
        # Publisher
        # Use TRANSIENT_LOCAL so late subscribers (RViz) receive the last MarkerArray without waiting.
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/gazebo_buildings',
            qos
        )

        # Live alpha updates
        self.alpha_sub = self.create_subscription(
            Float32,
            self.alpha_topic,
            self._alpha_callback,
            10
        )
        
        # Create markers
        self.marker_array = self._create_building_markers(buildings, marker_lifetime)
        
        # Timer to publish markers
        self.create_timer(1.0 / publish_rate, self.publish_markers)
        # Publish once immediately so RViz sees buildings right away.
        # With TRANSIENT_LOCAL QoS, this also behaves like a latched publish.
        try:
            self.publish_markers()
        except Exception:
            pass
        
        transform_info = []
        if self.mirror_x:
            transform_info.append('mirror X (flip Y)')
        if self.mirror_y:
            transform_info.append('mirror Y (flip X)')
        if self.rotate_90 > 0:
            transform_info.append(f'rotate {self.rotate_90 * 90}°')
        if self.offset_x != 0.0 or self.offset_y != 0.0:
            transform_info.append(f'offset ({self.offset_x}, {self.offset_y})')
        
        transform_str = f'  Transforms: {", ".join(transform_info)}\n' if transform_info else ''
        
        self.get_logger().info(
            f'✓ Publishing {len(buildings)} buildings as markers\n'
            f'  Topic: /gazebo_buildings\n'
            f'  Frame: {self.frame_id}\n'
            f'  Rate: {publish_rate}Hz\n'
            f'  Alpha updates: {self.alpha_topic}\n'
            f'{transform_str}'
        )

    def _alpha_callback(self, msg: Float32):
        """Update building opacity live (no restart required)."""
        self.building_alpha = max(0.0, min(1.0, float(msg.data)))
        # Update cached marker colors immediately
        try:
            for marker in self.marker_array.markers:
                if marker.action != Marker.DELETEALL:
                    marker.color.a = self.building_alpha
        except Exception:
            pass
        # Republish immediately so RViz updates without waiting for the slow publish timer.
        try:
            self.publish_markers()
        except Exception:
            pass
    
    def _parse_buildings(self, sdf_path):
        """Parse SDF file and extract building models."""
        buildings = []
        
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            
            # Find all model elements
            # SDF format: <world><model name="building_...">...</model></world>
            for model in root.findall('.//model'):
                name = model.get('name', '')
                
                # Skip non-building models (drones, ground plane, etc.)
                if not name.startswith('building_'):
                    continue
                
                # Get pose
                pose_elem = model.find('pose')
                if pose_elem is None:
                    continue
                
                pose_str = pose_elem.text.strip()
                pose_parts = pose_str.split()
                
                if len(pose_parts) < 6:
                    continue
                
                x = float(pose_parts[0])
                y = float(pose_parts[1])
                z = float(pose_parts[2])
                roll = float(pose_parts[3])
                pitch = float(pose_parts[4])
                yaw = float(pose_parts[5])
                
                # Get size from link/collision/geometry/box/size
                size_x = 10.0  # Default size
                size_y = 10.0
                size_z = 20.0
                
                # Try to find box size
                size_elem = model.find('.//box/size')
                if size_elem is not None:
                    size_parts = size_elem.text.strip().split()
                    if len(size_parts) >= 3:
                        size_x = float(size_parts[0])
                        size_y = float(size_parts[1])
                        size_z = float(size_parts[2])
                
                # Try to find scale in visual
                scale_elem = model.find('.//visual/geometry/box/size')
                if scale_elem is not None:
                    scale_parts = scale_elem.text.strip().split()
                    if len(scale_parts) >= 3:
                        size_x = float(scale_parts[0])
                        size_y = float(scale_parts[1])
                        size_z = float(scale_parts[2])
                
                buildings.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'size_x': size_x,
                    'size_y': size_y,
                    'size_z': size_z,
                    'color_rgba': self._extract_model_color_rgba(model),
                })
        
        except Exception as e:
            self.get_logger().error(f'Error parsing SDF file: {e}')
        
        return buildings

    def _extract_model_color_rgba(self, model_elem):
        """
        Best-effort extraction of Gazebo/SDF visual material color.
        Returns (r,g,b,a) or None.
        """
        if not getattr(self, 'use_sdf_material_colors', False):
            return None
        try:
            # Prefer diffuse, fallback to ambient
            diffuse = model_elem.find('.//visual//material/diffuse')
            ambient = model_elem.find('.//visual//material/ambient')
            txt = None
            if diffuse is not None and diffuse.text:
                txt = diffuse.text.strip()
            elif ambient is not None and ambient.text:
                txt = ambient.text.strip()
            if not txt:
                return None
            parts = [p for p in txt.split() if p]
            if len(parts) < 3:
                return None
            r = float(parts[0])
            g = float(parts[1])
            b = float(parts[2])
            a = float(parts[3]) if len(parts) >= 4 else 1.0
            # clamp
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            a = max(0.0, min(1.0, a))
            return (r, g, b, a)
        except Exception:
            return None
    
    def _create_building_markers(self, buildings, lifetime):
        """Create MarkerArray from building data."""
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Create marker for each building
        for i, building in enumerate(buildings):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'buildings'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Get original position
            # Note: building['z'] already represents the center of the building
            # (set to height/2 in the SDF file), so we don't need to add half height again
            x = building['x']
            y = building['y']
            z = building['z']
            
            # Apply transforms
            x_orig, y_orig = x, y
            x, y = self._transform_coordinates(x, y)
            
            # Position (center of building)
            marker.pose.position.x = x + self.offset_x
            marker.pose.position.y = y + self.offset_y
            marker.pose.position.z = z
            
            # Debug: Print transformed coordinates for first few buildings
            if self.debug and i < 5:
                self.get_logger().info(
                    f'Building {i+1} transformed: '
                    f'({x_orig:.2f}, {y_orig:.2f}) → '
                    f'({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f})'
                )
            
            # Orientation (convert roll/pitch/yaw to quaternion, then apply rotation)
            yaw = building['yaw'] + (self.rotate_90 * math.pi / 2.0)  # Add rotation
            qx, qy, qz, qw = self._euler_to_quaternion(
                building['roll'],
                building['pitch'],
                yaw
            )
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            
            # Size (swap X/Y if rotated 90/270 degrees)
            size_x = building['size_x']
            size_y = building['size_y']
            if self.rotate_90 % 2 == 1:  # 90° or 270° rotation
                size_x, size_y = size_y, size_x
            
            marker.scale.x = size_x
            marker.scale.y = size_y
            marker.scale.z = building['size_z']
            
            # Color (gray buildings)
            rgba = building.get('color_rgba')
            if rgba:
                marker.color.r = float(rgba[0])
                marker.color.g = float(rgba[1])
                marker.color.b = float(rgba[2])
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
            marker.color.a = max(0.0, min(1.0, float(self.building_alpha)))
            
            # Lifetime
            if lifetime > 0:
                marker.lifetime.sec = lifetime
            else:
                marker.lifetime.sec = 0  # Infinite
            
            marker_array.markers.append(marker)
        
        return marker_array
    
    def _transform_coordinates(self, x, y):
        """Apply mirror and rotation transforms to coordinates."""
        # Apply rotation first (around origin)
        for _ in range(self.rotate_90):
            # Rotate 90° counterclockwise: (x, y) -> (-y, x)
            x, y = -y, x
        
        # Apply mirrors
        if self.mirror_x:
            y = -y  # Mirror along X axis (flip Y)
        if self.mirror_y:
            x = -x  # Mirror along Y axis (flip X)
        
        return x, y
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        # Abbreviations for the various angular functions
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def publish_markers(self):
        """Publish building markers."""
        # Update timestamp
        now = self.get_clock().now().to_msg()
        for marker in self.marker_array.markers:
            if marker.action != Marker.DELETEALL:
                marker.header.stamp = now
                marker.color.a = max(0.0, min(1.0, float(self.building_alpha)))
        
        self.marker_pub.publish(self.marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboBuildingsPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

