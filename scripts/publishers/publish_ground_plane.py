#!/usr/bin/env python3
"""
Publish a large ground plane marker for RViz2 "Publish Point" tool.

This creates an invisible (or semi-transparent) ground plane that covers a large area,
ensuring that clicks anywhere in RViz2 will intersect with something and register properly.

Usage:
    python3 scripts/publishers/publish_ground_plane.py --ros-args \
        -p size:=1000.0 \
        -p z_position:=0.0 \
        -p alpha:=0.01
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.parameter import Parameter


class GroundPlanePublisher(Node):
    """Publish a large ground plane marker for clickable area."""
    
    def __init__(self):
        super().__init__('ground_plane_publisher')
        # Ensure timer uses wall-time even if /use_sim_time was set elsewhere.
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        except Exception:
            pass
        
        # Parameters
        self.declare_parameter('size', 1000.0)  # Size of the plane (meters)
        self.declare_parameter('z_position', 0.0)  # Z position of the plane
        self.declare_parameter('alpha', 0.01)  # Transparency (0.0 = invisible, 1.0 = opaque)
        self.declare_parameter('frame_id', 'world')  # Frame ID
        self.declare_parameter('color_r', 0.5)  # Red component
        self.declare_parameter('color_g', 0.5)  # Green component
        self.declare_parameter('color_b', 0.5)  # Blue component
        self.declare_parameter('publish_rate', 1.0)  # Publish rate (Hz)
        
        # Get parameters
        size = float(self.get_parameter('size').value)
        z_pos = float(self.get_parameter('z_position').value)
        alpha = float(self.get_parameter('alpha').value)
        frame_id = self.get_parameter('frame_id').value
        color_r = float(self.get_parameter('color_r').value)
        color_g = float(self.get_parameter('color_g').value)
        color_b = float(self.get_parameter('color_b').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        
        # Publisher
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.marker_pub = self.create_publisher(Marker, '/ground_plane', qos)
        
        # Create marker
        self.marker = Marker()
        self.marker.header.frame_id = frame_id
        self.marker.ns = 'ground_plane'
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        
        # Position (centered at origin, at specified Z height)
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = z_pos
        self.marker.pose.orientation.w = 1.0
        
        # Size (large flat plane)
        self.marker.scale.x = size  # X dimension
        self.marker.scale.y = size  # Y dimension
        self.marker.scale.z = 0.1   # Thin plane (10cm thick)
        
        # Color (semi-transparent gray, or nearly invisible)
        self.marker.color.r = color_r
        self.marker.color.g = color_g
        self.marker.color.b = color_b
        self.marker.color.a = alpha
        
        # Lifetime (0 = infinite)
        self.marker.lifetime.sec = 0
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_marker)
        # Publish once immediately so RViz sees it even if it starts late.
        try:
            self.publish_marker()
        except Exception:
            pass
        
        self.get_logger().info(
            f'Ground plane publisher initialized:\n'
            f'  Size: {size}m x {size}m\n'
            f'  Z position: {z_pos}m\n'
            f'  Alpha (transparency): {alpha}\n'
            f'  Frame: {frame_id}\n'
            f'  Publish rate: {publish_rate}Hz\n'
            f'  Topic: /ground_plane'
        )
        
        if alpha < 0.1:
            self.get_logger().info('  Note: Plane is nearly invisible (good for clickable area)')
        else:
            self.get_logger().info('  Note: Plane is visible (you can see it in RViz2)')
    
    def publish_marker(self):
        """Publish the ground plane marker."""
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)
    node = GroundPlanePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

