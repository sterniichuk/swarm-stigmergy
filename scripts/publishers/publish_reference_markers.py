#!/usr/bin/env python3
"""
Publish reference markers to help align RViz2 with Gazebo.

This script publishes visual markers at known locations (origin, axes, buildings)
to help you determine what transform parameters are needed for alignment.

Usage:
    # Show origin and axes
    python3 scripts/publishers/publish_reference_markers.py
    
    # Show first N buildings as reference points
    python3 scripts/publishers/publish_reference_markers.py --ros-args \
        -p show_buildings:=true \
        -p num_buildings:=5
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import xml.etree.ElementTree as ET
import os
import math


class ReferenceMarkersPublisher(Node):
    """Publish reference markers for alignment."""
    
    def __init__(self):
        super().__init__('reference_markers_publisher')
        
        # Parameters
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('show_origin', True)
        self.declare_parameter('show_axes', True)
        self.declare_parameter('show_buildings', False)
        self.declare_parameter('num_buildings', 5)
        self.declare_parameter('world_file', 'worlds/city_map.sdf')
        self.declare_parameter('publish_rate', 1.0)
        
        # Get parameters
        self.frame_id = self.get_parameter('frame_id').value
        show_origin = bool(self.get_parameter('show_origin').value)
        show_axes = bool(self.get_parameter('show_axes').value)
        show_buildings = bool(self.get_parameter('show_buildings').value)
        num_buildings = int(self.get_parameter('num_buildings').value)
        world_file = self.get_parameter('world_file').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        
        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/reference_markers',
            10
        )
        
        # Parse buildings if needed
        buildings = []
        if show_buildings:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(script_dir))
            world_path = os.path.join(repo_root, world_file)
            
            if os.path.exists(world_path):
                buildings = self._parse_buildings(world_path, num_buildings)
                self.get_logger().info(f'Loaded {len(buildings)} reference buildings')
            else:
                self.get_logger().warn(f'World file not found: {world_path}')
        
        # Create markers
        self.marker_array = self._create_reference_markers(
            show_origin, show_axes, buildings
        )
        
        # Timer
        self.create_timer(1.0 / publish_rate, self.publish_markers)
        
        info = []
        if show_origin:
            info.append('origin')
        if show_axes:
            info.append('axes')
        if show_buildings:
            info.append(f'{len(buildings)} buildings')
        
        self.get_logger().info(
            f'✓ Publishing reference markers\n'
            f'  Topic: /reference_markers\n'
            f'  Frame: {self.frame_id}\n'
            f'  Showing: {", ".join(info)}'
        )
    
    def _parse_buildings(self, sdf_path, max_count):
        """Parse SDF file and extract first N buildings."""
        buildings = []
        
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            
            for model in root.findall('.//model'):
                name = model.get('name', '')
                
                if not name.startswith('building_'):
                    continue
                
                if len(buildings) >= max_count:
                    break
                
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
                
                buildings.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z
                })
        
        except Exception as e:
            self.get_logger().error(f'Error parsing SDF file: {e}')
        
        return buildings
    
    def _create_reference_markers(self, show_origin, show_axes, buildings):
        """Create reference markers."""
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        marker_id = 0
        
        # Origin marker (red sphere)
        if show_origin:
            origin = Marker()
            origin.header.frame_id = self.frame_id
            origin.header.stamp = self.get_clock().now().to_msg()
            origin.ns = 'reference'
            origin.id = marker_id
            origin.type = Marker.SPHERE
            origin.action = Marker.ADD
            origin.pose.position.x = 0.0
            origin.pose.position.y = 0.0
            origin.pose.position.z = 1.0  # Slightly above ground
            origin.scale.x = 2.0
            origin.scale.y = 2.0
            origin.scale.z = 2.0
            origin.color.r = 1.0  # Red
            origin.color.g = 0.0
            origin.color.b = 0.0
            origin.color.a = 1.0
            origin.lifetime.sec = 0
            marker_array.markers.append(origin)
            marker_id += 1
        
        # Axes (X=red, Y=green, Z=blue arrows)
        if show_axes:
            # X axis (red arrow)
            x_axis = Marker()
            x_axis.header.frame_id = self.frame_id
            x_axis.header.stamp = self.get_clock().now().to_msg()
            x_axis.ns = 'reference'
            x_axis.id = marker_id
            x_axis.type = Marker.ARROW
            x_axis.action = Marker.ADD
            x_axis.pose.position.x = 0.0
            x_axis.pose.position.y = 0.0
            x_axis.pose.position.z = 1.0
            x_axis.pose.orientation.w = 1.0
            x_axis.scale.x = 10.0  # Length
            x_axis.scale.y = 0.5   # Width
            x_axis.scale.z = 0.5   # Height
            x_axis.color.r = 1.0   # Red
            x_axis.color.g = 0.0
            x_axis.color.b = 0.0
            x_axis.color.a = 1.0
            x_axis.lifetime.sec = 0
            marker_array.markers.append(x_axis)
            marker_id += 1
            
            # Y axis (green arrow) - rotate 90° around Z axis
            y_axis = Marker()
            y_axis.header.frame_id = self.frame_id
            y_axis.header.stamp = self.get_clock().now().to_msg()
            y_axis.ns = 'reference'
            y_axis.id = marker_id
            y_axis.type = Marker.ARROW
            y_axis.action = Marker.ADD
            y_axis.pose.position.x = 0.0
            y_axis.pose.position.y = 0.0
            y_axis.pose.position.z = 1.0
            # Rotate 90° around Z axis: (x, y, z, w) = (0, 0, sin(45°), cos(45°))
            y_axis.pose.orientation.z = math.sin(math.pi / 4.0)
            y_axis.pose.orientation.w = math.cos(math.pi / 4.0)
            y_axis.scale.x = 10.0  # Length
            y_axis.scale.y = 0.5   # Width
            y_axis.scale.z = 0.5   # Height
            y_axis.color.r = 0.0
            y_axis.color.g = 1.0   # Green
            y_axis.color.b = 0.0
            y_axis.color.a = 1.0
            y_axis.lifetime.sec = 0
            marker_array.markers.append(y_axis)
            marker_id += 1
            
            # Z axis (blue arrow) - rotate -90° around Y axis
            z_axis = Marker()
            z_axis.header.frame_id = self.frame_id
            z_axis.header.stamp = self.get_clock().now().to_msg()
            z_axis.ns = 'reference'
            z_axis.id = marker_id
            z_axis.type = Marker.ARROW
            z_axis.action = Marker.ADD
            z_axis.pose.position.x = 0.0
            z_axis.pose.position.y = 0.0
            z_axis.pose.position.z = 1.0
            # Rotate -90° around Y axis: (x, y, z, w) = (0, -sin(45°), 0, cos(45°))
            z_axis.pose.orientation.y = -math.sin(math.pi / 4.0)
            z_axis.pose.orientation.w = math.cos(math.pi / 4.0)
            z_axis.scale.x = 10.0  # Length
            z_axis.scale.y = 0.5   # Width
            z_axis.scale.z = 0.5   # Height
            z_axis.color.r = 0.0
            z_axis.color.g = 0.0
            z_axis.color.b = 1.0   # Blue
            z_axis.color.a = 1.0
            z_axis.lifetime.sec = 0
            marker_array.markers.append(z_axis)
            marker_id += 1
        
        # Building reference points (yellow spheres)
        for i, building in enumerate(buildings):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'reference'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = building['x']
            marker.pose.position.y = building['y']
            marker.pose.position.z = building['z'] + 5.0  # Above building
            marker.scale.x = 3.0
            marker.scale.y = 3.0
            marker.scale.z = 3.0
            marker.color.r = 1.0   # Yellow
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
            
            # Print coordinates for debugging
            self.get_logger().info(
                f'Reference building {i+1}: {building["name"]} '
                f'at ({building["x"]:.2f}, {building["y"]:.2f}, {building["z"]:.2f})'
            )
            
            marker_id += 1
        
        return marker_array
    
    def publish_markers(self):
        """Publish reference markers."""
        # Update timestamp
        now = self.get_clock().now().to_msg()
        for marker in self.marker_array.markers:
            if marker.action != Marker.DELETEALL:
                marker.header.stamp = now
        
        self.marker_pub.publish(self.marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceMarkersPublisher()
    
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

