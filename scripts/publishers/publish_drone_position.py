#!/usr/bin/env python3
"""
Publish drone position from ArduPilot to RViz2.

Reads drone position via MAVLink and publishes as:
- geometry_msgs/PoseStamped (for TF/visualization)
- visualization_msgs/Marker (for visual marker)

Usage:
    # Publish all three drones (default)
    python3 scripts/publishers/publish_drone_position.py
    
    # Publish single drone
    python3 scripts/publishers/publish_drone_position.py --ros-args \
        -p drone_id:=1 \
        -p mavlink_port:=14550
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from pymavlink import mavutil
import threading
import time
import math


class DronePositionPublisher(Node):
    """Publish drone position from ArduPilot to RViz2."""
    
    def __init__(self, drone_id=None, mavlink_port=None):
        """
        Initialize drone position publisher.
        
        Args:
            drone_id: Drone ID (1-3). If None, will read from parameter.
            mavlink_port: MAVLink port. If None, will read from parameter.
        """
        super().__init__(f'drone_position_publisher_{drone_id if drone_id else "default"}')
        
        # Parameters
        self.declare_parameter('drone_id', 1)
        self.declare_parameter('mavlink_port', 14550)
        self.declare_parameter('connection_type', 'udp')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('use_gps', False)  # Use GPS or local position
        self.declare_parameter('home_lat', 52.52)  # Home latitude (for GPS conversion)
        self.declare_parameter('home_lon', 13.405)  # Home longitude
        self.declare_parameter('home_alt', 0.0)  # Home altitude
        self.declare_parameter('marker_scale_x', 2.0)  # Marker length (arrow)
        self.declare_parameter('marker_scale_y', 0.5)  # Marker width
        self.declare_parameter('marker_scale_z', 0.5)  # Marker height
        
        # Get parameters (use provided values or read from parameters)
        if drone_id is not None:
            self.drone_id = int(drone_id)
        else:
            self.drone_id = int(self.get_parameter('drone_id').value)
        
        if mavlink_port is not None:
            self.mavlink_port = int(mavlink_port)
        else:
            self.mavlink_port = int(self.get_parameter('mavlink_port').value)
        self.connection_type = self.get_parameter('connection_type').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.use_gps = bool(self.get_parameter('use_gps').value)
        self.home_lat = float(self.get_parameter('home_lat').value)
        self.home_lon = float(self.get_parameter('home_lon').value)
        self.home_alt = float(self.get_parameter('home_alt').value)
        self.marker_scale_x = float(self.get_parameter('marker_scale_x').value)
        self.marker_scale_y = float(self.get_parameter('marker_scale_y').value)
        self.marker_scale_z = float(self.get_parameter('marker_scale_z').value)
        
        # MAVLink connection
        self.mavlink_conn = None
        self.mavlink_connected = False
        self.position_lock = threading.Lock()
        self.current_position = None  # (x, y, z, heading)
        self.current_yaw = None  # Track yaw separately from ATTITUDE message
        self.last_published_position = None  # Track last published position
        
        # Connect to MAVLink
        self._connect_mavlink(self.mavlink_port, self.connection_type)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, f'/drone{self.drone_id}/pose', 10)
        self.marker_pub = self.create_publisher(Marker, f'/drone{self.drone_id}/marker', 10)
        
        # Timer for reading position and publishing
        self.create_timer(1.0 / self.publish_rate, self.update_position)
        
        # Track if we've received first position update
        self.first_position_received = False
        
        self.get_logger().info(
            f'Drone Position Publisher initialized:\n'
            f'  Drone ID: {self.drone_id}\n'
            f'  MAVLink port: {self.mavlink_port}\n'
            f'  Frame: {self.frame_id}\n'
            f'  Publish rate: {self.publish_rate}Hz\n'
            f'  Use GPS: {self.use_gps}\n'
            f'  Topics:\n'
            f'    Pose: /drone{self.drone_id}/pose\n'
            f'    Marker: /drone{self.drone_id}/marker'
        )
        
        if self.mavlink_connected:
            self.get_logger().info('✓ MAVLink connected - publishing drone position')
        else:
            self.get_logger().warn('⚠ MAVLink not connected - will retry')
    
    def _connect_mavlink(self, port, connection_type='udp'):
        """Connect to ArduPilot via MAVLink."""
        connection_types = [connection_type]
        if connection_type == 'udpin':
            connection_types.append('udp')
        elif connection_type == 'udp':
            connection_types.append('udpin')
        
        for conn_type in connection_types:
            try:
                connection_str = f'{conn_type}:127.0.0.1:{port}'
                self.get_logger().info(f'Connecting to MAVLink at {connection_str}...')
                
                self.mavlink_conn = mavutil.mavlink_connection(connection_str)
                
                heartbeat = self.mavlink_conn.wait_heartbeat(timeout=5)
                
                if heartbeat:
                    self.mavlink_connected = True
                    self.get_logger().info(f'✓ MAVLink connected via {conn_type}!')
                    return
                    
            except Exception as e:
                self.get_logger().debug(f'Connection attempt with {conn_type} failed: {e}')
                continue
        
        self.get_logger().warn(f'⚠ Could not connect to MAVLink on port {port}')
        self.mavlink_connected = False
    
    def _gps_to_local(self, lat, lon, alt):
        """Convert GPS coordinates to local NED coordinates."""
        # Convert lat/lon to meters relative to home
        meters_per_deg_lat = 111000.0
        meters_per_deg_lon = 111000.0 * math.cos(math.radians(self.home_lat))
        
        x = (lat - self.home_lat) * meters_per_deg_lat
        y = (lon - self.home_lon) * meters_per_deg_lon
        z = alt - self.home_alt
        
        return x, y, z
    
    def update_position(self):
        """Read position from MAVLink and publish."""
        if not self.mavlink_connected or self.mavlink_conn is None:
            # Try to reconnect
            if not hasattr(self, '_last_reconnect_attempt'):
                self._last_reconnect_attempt = 0
            
            current_time = time.time()
            if current_time - self._last_reconnect_attempt > 5.0:
                self._last_reconnect_attempt = current_time
                self._connect_mavlink(self.mavlink_port, self.connection_type)
            return
        
        try:
            # Read messages
            msg = None
            for _ in range(10):  # Read up to 10 messages
                msg = self.mavlink_conn.recv_match(blocking=False)
                if msg is None:
                    break
                
                # Handle ATTITUDE message (best source for yaw)
                if msg.get_type() == 'ATTITUDE':
                    # ATTITUDE yaw is in radians, NED frame (0 = North, increases clockwise)
                    # Convert NED yaw to ENU yaw:
                    # NED: 0 = North, increases clockwise
                    # ENU: 0 = East, increases counterclockwise
                    # Conversion: ENU_yaw = π/2 - NED_yaw
                    ned_yaw = msg.yaw
                    enu_yaw = math.pi / 2.0 - ned_yaw
                    
                    with self.position_lock:
                        self.current_yaw = enu_yaw
                
                # Handle GPS position
                elif msg.get_type() == 'GLOBAL_POSITION_INT':
                    if self.use_gps:
                        lat = msg.lat / 1e7
                        lon = msg.lon / 1e7
                        alt = msg.alt / 1000.0  # mm to meters
                        x, y, z = self._gps_to_local(lat, lon, alt)
                        heading = msg.hdg / 100.0  # centidegrees to degrees
                        
                        with self.position_lock:
                            self.current_position = (x, y, z, math.radians(heading))
                
                # Handle local position (NED)
                elif msg.get_type() == 'LOCAL_POSITION_NED':
                    if not self.use_gps:
                        # Convert NED to ENU (Gazebo/RViz2 coordinate system)
                        # NED: x=North, y=East, z=Down
                        # ENU: x=East, y=North, z=Up
                        x = msg.y  # East (NED y -> ENU x)
                        y = msg.x  # North (NED x -> ENU y)
                        z = -msg.z  # Up (NED Down -> ENU Up)
                        
                        # Use ATTITUDE yaw if available, otherwise fall back to velocity-based heading
                        with self.position_lock:
                            if self.current_yaw is not None:
                                heading = self.current_yaw
                            else:
                                # Fallback: calculate from velocity (less accurate)
                                # Velocity in NED: vx=North, vy=East
                                # For ENU heading: atan2(East_vel, North_vel) = atan2(vy, vx)
                                heading = math.atan2(msg.vy, msg.vx) if (msg.vx != 0 or msg.vy != 0) else 0.0
                            
                            self.current_position = (x, y, z, heading)
            
            # Publish current position
            with self.position_lock:
                if self.current_position is not None:
                    x, y, z, heading = self.current_position
                    self.last_published_position = (x, y, z, heading)
                    self._publish_position(x, y, z, heading)
                    
                    # Log first position received (one-time)
                    if not self.first_position_received:
                        self.first_position_received = True
                        self.get_logger().info(
                            f'[Drone {self.drone_id}] ✓ Position tracking active - '
                            f'Publishing updates at {self.publish_rate}Hz'
                        )
        
        except Exception as e:
            self.get_logger().error(f'Error reading position: {e}')
    
    def _publish_position(self, x, y, z, heading):
        """Publish drone position as PoseStamped and Marker."""
        now = self.get_clock().now()
        
        # Publish PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = now.to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        # Convert heading to quaternion
        qx, qy, qz, qw = self._heading_to_quaternion(heading)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        self.pose_pub.publish(pose)
        
        # Publish Marker (visual representation)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = now.to_msg()
        marker.ns = 'drones'
        marker.id = self.drone_id
        marker.type = Marker.ARROW  # Arrow shows heading
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        
        # Size (arrow)
        marker.scale.x = self.marker_scale_x  # Length
        marker.scale.y = self.marker_scale_y  # Width
        marker.scale.z = self.marker_scale_z  # Height
        
        # Color (blue for drone)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0  # Infinite
        
        self.marker_pub.publish(marker)
    
    def _heading_to_quaternion(self, heading):
        """Convert heading (yaw) to quaternion."""
        # Heading is rotation around Z axis
        qx = 0.0
        qy = 0.0
        qz = math.sin(heading / 2.0)
        qw = math.cos(heading / 2.0)
        return qx, qy, qz, qw
    


def main(args=None):
    rclpy.init(args=args)
    
    # Check if parameters are at defaults (no arguments specified)
    # If both drone_id and mavlink_port are defaults, publish all three drones
    temp_node = Node('temp_param_checker')
    temp_node.declare_parameter('drone_id', 1)
    temp_node.declare_parameter('mavlink_port', 14550)
    
    drone_id_param = temp_node.get_parameter('drone_id').value
    mavlink_port_param = temp_node.get_parameter('mavlink_port').value
    temp_node.destroy_node()
    
    # If both parameters are defaults, publish all three drones
    # Otherwise, publish just the specified drone
    should_publish_all = (drone_id_param == 1 and mavlink_port_param == 14550)
    
    if should_publish_all:
        # Publish all three drones (default behavior)
        base_port = 14550
        nodes = []
        
        for drone_id in range(1, 4):
            port = base_port + (drone_id - 1)
            node = DronePositionPublisher(drone_id=drone_id, mavlink_port=port)
            nodes.append(node)
            node.get_logger().info(f'Created publisher for drone {drone_id} on port {port}')
        
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            for node in nodes:
                node.destroy_node()
            try:
                rclpy.shutdown()
            except:
                pass
    else:
        # Publish single drone (explicit parameters provided)
        node = DronePositionPublisher()
        
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

