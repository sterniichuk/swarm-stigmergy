from __future__ import annotations
import math
import random
from typing import List, Tuple, Union
from geometry_msgs.msg import Quaternion
from scripts.python_sim.dto.sim_types import DroneType

def make_drone_uid(drone_type: DroneType, seq: int) -> str:
    """
    Generate a unique identifier for a drone.

    Implementation: Combines the drone's role/type with its sequential index using a hyphen.
    Behavior: Always returns a string like 'EXPLORER-0' or 'RELAY-5', providing a persistent
    callsign for communication and tracking.

    Args:
        drone_type (DroneType): The type/role of the drone.
        seq (int): Sequential index of the drone.

    Returns:
        str: A string UID in the format 'type-seq'.
    """
    return f"{drone_type}-{seq}"

def yaw_to_quat(yaw: float) -> Quaternion:
    """
    Convert a yaw angle to a ROS Quaternion message.

    Implementation: Uses standard trigonometry to map a single Z-axis rotation into 
    the w and z components of a unit quaternion.
    Behavior: Assuming a flat-earth/2D orientation (no roll or pitch), this creates
    a standard rotation message that ROS-compatible systems (like RViz or flight controllers)
    can use to orient drone models.

    Args:
        yaw (float): Rotation angle around the Z axis in radians.

    Returns:
        Quaternion: A ROS Quaternion message representing the rotation.
    """
    q = Quaternion()
    q.w = float(math.cos(yaw / 2.0))
    q.z = float(math.sin(yaw / 2.0))
    q.x = 0.0
    q.y = 0.0
    return q

def clamp(v: float, lo: float, hi: float) -> float:
    """
    Clamp a value between a lower and upper bound.

    Implementation: Uses nested min/max functions to force a value into a range.
    Behavior: Acts as a safety limiter. If the value exceeds 'hi', it returns 'hi';
    if it's below 'lo', it returns 'lo'. Useful for capping physical limits like speed or battery.

    Args:
        v (float): The value to clamp.
        lo (float): Minimum allowed value.
        hi (float): Maximum allowed value.

    Returns:
        float: The clamped value.
    """
    return max(lo, min(hi, v))

# Map bounds can be specified as:
# - float: symmetric square bounds (-b..b, -b..b)
# - (minx, maxx, miny, maxy): rectangular bounds
MapBounds = Union[float, Tuple[float, float, float, float]]

def bounds_minmax(bounds: MapBounds) -> Tuple[float, float, float, float]:
    """
    Convert flexible MapBounds into a explicit (minx, maxx, miny, maxy) tuple.

    Implementation: Checks if input is a single number (square bounds) or a 4-tuple.
    Behavior: Standardizes geofence coordinates. A single number '50' becomes 
    (-50, 50, -50, 50), defining a 100x100m square centered at the origin.

    Args:
        bounds (MapBounds): Either a float for symmetric bounds or a 4-tuple of bounds.

    Returns:
        Tuple[float, float, float, float]: (min_x, max_x, min_y, max_y).
    """
    if isinstance(bounds, (int, float)):
        b = float(bounds)
        return -b, b, -b, b
    # assume (minx, maxx, miny, maxy)
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])

def out_of_bounds(x: float, y: float, bounds: MapBounds) -> bool:
    """
    Check if a point (x, y) is outside the specified map bounds.

    Implementation: Compares coordinates against the geofence limits defined in MapBounds.
    Behavior: Returns True if the drone has breached the mission area boundaries (geofence).

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        bounds (MapBounds): The map bounds to check against.

    Returns:
        bool: True if the point is outside the bounds, False otherwise.
    """
    minx, maxx, miny, maxy = bounds_minmax(bounds)
    return (x < minx) or (x > maxx) or (y < miny) or (y > maxy)

def clamp_xy(x: float, y: float, bounds: MapBounds) -> Tuple[float, float]:
    """
    Clamp a point (x, y) to be within the specified map bounds.

    Implementation: Applies the 'clamp' utility to both X and Y world coordinates.
    Behavior: Forces a target coordinate to stay within the mission geofence,
    preventing navigation commands that would lead the drone out of bounds.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        bounds (MapBounds): The map bounds to clamp to.

    Returns:
        Tuple[float, float]: The clamped (x, y) coordinates.
    """
    minx, maxx, miny, maxy = bounds_minmax(bounds)
    return clamp(float(x), minx, maxx), clamp(float(y), miny, maxy)

def softmax_sample(items: List[Tuple[float, object]], temperature: float, rng: random.Random):
    """
    Sample an item from a list based on softmax probabilities of their scores.

    Implementation: Converts scores to probabilities using an exponential function,
    then performs a weighted random selection.
    Behavior: This is a stochastic decision-making tool. At high temperatures, the
    drone chooses somewhat randomly among options. At temperature ~0, it becomes
    deterministic (greedy), always picking the option with the highest score.

    Args:
        items (List[Tuple[float, object]]): List of (score, payload) pairs.
        temperature (float): Softmax temperature. High temperature means more random,
            low temperature (near 0) means greedy (highest score always wins).
        rng (random.Random): Random number generator instance.

    Returns:
        object: The payload of the sampled item, or None if the list is empty.
    """
    # items: [(score, payload), ...]
    if not items:
        return None
    if temperature <= 1e-6:
        return max(items, key=lambda t: t[0])[1]
    mx = max(s for s, _ in items)
    exps = [math.exp((s - mx) / temperature) for s, _ in items]
    total = sum(exps)
    if total <= 0:
        return max(items, key=lambda t: t[0])[1]
    r = rng.random() * total
    acc = 0.0
    for (s, payload), e in zip(items, exps):
        acc += e
        if acc >= r:
            return payload
    return items[-1][1]
