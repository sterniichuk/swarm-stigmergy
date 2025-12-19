#!/usr/bin/env python3
"""
Manually publish a single marker to RViz2 at specified coordinates.

Usage:
    # Publish a red sphere at (10, 20, 5)
    python3 scripts/publishers/publish_marker.py --ros-args \
        -p x:=10.0 -p y:=20.0 -p z:=5.0

    # Publish a blue cube with custom size
    python3 scripts/publishers/publish_marker.py --ros-args \
        -p x:=5.0 -p y:=5.0 -p z:=3.0 \
        -p marker_type:=cube \
        -p color_r:=0.0 -p color_g:=0.0 -p color_b:=1.0 \
        -p scale_x:=2.0 -p scale_y:=2.0 -p scale_z:=2.0

    # Publish an arrow (shows orientation)
    python3 scripts/publishers/publish_marker.py --ros-args \
        -p x:=0.0 -p y:=0.0 -p z:=2.0 \
        -p marker_type:=arrow \
        -p yaw:=45.0

Marker Types:
    - sphere: Sphere marker
    - cube: Cube marker
    - arrow: Arrow marker (shows direction)
    - cylinder: Cylinder marker
    - text_view_facing: Text marker
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import math


class MarkerPublisher(Node):
    """Publish a single marker at specified coordinates."""
    
    def __init__(self):
        super().__init__('manual_marker_publisher')
        
        # Position parameters
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 1.0)
        
        # Orientation (yaw in degrees)
        self.declare_parameter('yaw', 0.0)
        
        # Marker properties
        self.declare_parameter('marker_type', 'sphere')  # sphere, cube, arrow, cylinder, text_view_facing
        self.declare_parameter('marker_id', 999)
        self.declare_parameter('namespace', 'manual')
        self.declare_parameter('frame_id', 'world')
        
        # Size
        self.declare_parameter('scale_x', 1.0)
        self.declare_parameter('scale_y', 1.0)
        self.declare_parameter('scale_z', 1.0)
        
        # Color (RGB 0.0-1.0)
        self.declare_parameter('color_r', 1.0)
        self.declare_parameter('color_g', 0.0)
        self.declare_parameter('color_b', 0.0)
        self.declare_parameter('color_a', 1.0)
        
        # Text (for text_view_facing type)
        self.declare_parameter('text', 'Marker')
        
        # Lifetime (0 = infinite)
        self.declare_parameter('lifetime', 0)
        
        # Get parameters
        x = float(self.get_parameter('x').value)
        y = float(self.get_parameter('y').value)
        z = float(self.get_parameter('z').value)
        yaw_deg = float(self.get_parameter('yaw').value)
        marker_type_str = str(self.get_parameter('marker_type').value)
        marker_id = int(self.get_parameter('marker_id').value)
        namespace = str(self.get_parameter('namespace').value)
        frame_id = str(self.get_parameter('frame_id').value)
        scale_x = float(self.get_parameter('scale_x').value)
        scale_y = float(self.get_parameter('scale_y').value)
        scale_z = float(self.get_parameter('scale_z').value)
        color_r = float(self.get_parameter('color_r').value)
        color_g = float(self.get_parameter('color_g').value)
        color_b = float(self.get_parameter('color_b').value)
        color_a = float(self.get_parameter('color_a').value)
        text = str(self.get_parameter('text').value)
        lifetime = int(self.get_parameter('lifetime').value)
        
        # Convert yaw to quaternion
        yaw_rad = math.radians(yaw_deg)
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw_rad / 2.0)
        qw = math.cos(yaw_rad / 2.0)
        
        # Map marker type string to Marker constant
        marker_type_map = {
            'sphere': Marker.SPHERE,
            'cube': Marker.CUBE,
            'arrow': Marker.ARROW,
            'cylinder': Marker.CYLINDER,
            'text_view_facing': Marker.TEXT_VIEW_FACING,
        }
        
        marker_type = marker_type_map.get(marker_type_str.lower(), Marker.SPHERE)
        
        # Create publisher
        self.marker_pub = self.create_publisher(Marker, '/manual_marker', 10)
        
        # Create marker
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        
        # Orientation
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        
        # Size
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        
        # Color
        marker.color.r = color_r
        marker.color.g = color_g
        marker.color.b = color_b
        marker.color.a = color_a
        
        # Text (for text markers)
        if marker_type == Marker.TEXT_VIEW_FACING:
            marker.text = text
        
        # Lifetime
        if lifetime > 0:
            marker.lifetime.sec = lifetime
        else:
            marker.lifetime.sec = 0  # Infinite
        
        # Publish marker
        self.marker_pub.publish(marker)
        
        self.get_logger().info(
            f'✓ Published marker:\n'
            f'  Type: {marker_type_str}\n'
            f'  Position: ({x:.2f}, {y:.2f}, {z:.2f})\n'
            f'  Yaw: {yaw_deg:.1f}°\n'
            f'  Color: RGB({color_r:.2f}, {color_g:.2f}, {color_b:.2f})\n'
            f'  Size: ({scale_x:.2f}, {scale_y:.2f}, {scale_z:.2f})\n'
            f'  Topic: /manual_marker\n'
            f'  Frame: {frame_id}'
        )
        
        if lifetime > 0:
            self.get_logger().info(f'  Lifetime: {lifetime} seconds')
        else:
            self.get_logger().info('  Lifetime: infinite')
        
        # Keep node alive briefly to ensure message is sent
        import time
        time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerPublisher()
    
    # Spin briefly to publish
    rclpy.spin_once(node, timeout_sec=0.2)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




















