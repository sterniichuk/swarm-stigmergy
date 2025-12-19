#!/usr/bin/env python3
"""
Publish static TF transform for 'map' frame.

This creates a static transform from 'world' to 'map' so RViz2 can use 'map' as fixed frame.
Since Gazebo uses 'world' frame, we'll make 'map' = 'world' (identity transform).
"""

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class MapTfPublisher(Node):
    """Publish static transform from world to map."""
    
    def __init__(self):
        super().__init__('map_tf_publisher')
        
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Create static transform: world -> map (identity transform)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'world'  # Parent frame (Gazebo's frame)
        transform.child_frame_id = 'map'     # Child frame (RViz2's frame)
        
        # Identity transform (no translation, no rotation)
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        # Publish transform
        self.tf_broadcaster.sendTransform(transform)
        
        self.get_logger().info(
            '✓ Published static transform: world -> map\n'
            '  RViz2 can now use "map" as fixed frame'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapTfPublisher()
    
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

