from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scripts.python_sim.dto.sim_types import Building

def load_buildings_from_sdf(world_path: Path) -> List[Building]:
    """
    Parse a Gazebo SDF world file and extract building models.
    Buildings are identified by names starting with 'building_'.

    Implementation: Reads the XML structure of an SDF file, looking for models
    named 'building_*'. It extracts their 'pose' (location) and 'box size' (dimensions).
    Behavior: Converts static map objects from the simulation environment into 
    mathematical Building objects that drones can use for collision avoidance.

    Args:
        world_path (Path): Path to the .sdf or .world file.

    Returns:
        List[Building]: A list of Building DTOs found in the world.
    """
    if not world_path.exists():
        return []
    try:
        tree = ET.parse(str(world_path))
        root = tree.getroot()
        out: List[Building] = []
        for model in root.findall(".//model"):
            name = model.get("name", "")
            if not name.startswith("building_"):
                continue
            pose_elem = model.find("pose")
            if pose_elem is None or not pose_elem.text:
                continue
            parts = pose_elem.text.strip().split()
            if len(parts) < 6:
                continue
            x, y, zc = float(parts[0]), float(parts[1]), float(parts[2])
            # sizes (default)
            sx, sy, sz = 10.0, 10.0, 20.0
            size_elem = model.find(".//box/size")
            if size_elem is not None and size_elem.text:
                sp = size_elem.text.strip().split()
                if len(sp) >= 3:
                    sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
            out.append(Building(x=x, y=y, z_center=zc, size_x=sx, size_y=sy, size_z=sz))
        return out
    except Exception:
        return []

class BuildingIndex:
    """
    Spatial index for buildings to allow fast collision and height queries.
    Uses a grid-based spatial hash (buckets).

    High-level Overview: Instead of checking every building on the map for every 
    drone move, this index "divides" the map into large squares (buckets). 
    A drone only checks buildings in its current and adjacent buckets.
    """
    def __init__(self, buildings: List[Building], margin_m: float, bucket_size_m: float = 50.0):
        """
        Initialize the spatial index.

        Implementation: Sets up the bucket structure and triggers the initial
        indexing of all provided buildings.

        Args:
            buildings (List[Building]): List of buildings to index.
            margin_m (float): Safety margin to add around each building footprint.
            bucket_size_m (float, optional): Size of each spatial hash bucket in meters.
        """
        self.buildings = buildings
        self.margin_m = float(margin_m)
        self.bucket_size_m = float(max(5.0, bucket_size_m))
        # Spatial hash: bucket -> list of building indices
        self._buckets: Dict[Tuple[int, int], List[int]] = {}
        self._build_spatial_index()

    def _build_spatial_index(self):
        """
        Build the internal grid-based spatial hash.

        Implementation: Iterates through all buildings, calculates which 
        grid buckets they overlap (considering the safety margin), and 
        records the building index in those buckets.
        Behavior: Populates the internal lookup table used by all query methods.
        """
        self._buckets.clear()
        bs = float(self.bucket_size_m)
        if bs <= 0:
            return
        for idx, b in enumerate(self.buildings):
            minx, maxx, miny, maxy = b.bbox_xy(self.margin_m)
            bx0 = int(math.floor(minx / bs))
            bx1 = int(math.floor(maxx / bs))
            by0 = int(math.floor(miny / bs))
            by1 = int(math.floor(maxy / bs))
            for bx in range(bx0, bx1 + 1):
                for by in range(by0, by1 + 1):
                    key = (bx, by)
                    lst = self._buckets.get(key)
                    if lst is None:
                        self._buckets[key] = [idx]
                    else:
                        lst.append(idx)

    def _bucket_for_xy(self, x: float, y: float) -> Tuple[int, int]:
        """
        Get the bucket coordinates for a world (x, y) position.

        Implementation: Performs a simple division of coordinates by the 
        bucket size and rounds down.
        """
        bs = float(self.bucket_size_m)
        return (int(math.floor(x / bs)), int(math.floor(y / bs)))

    def is_obstacle_xy(self, x: float, y: float, z_drone: float, safety_margin_z: float) -> bool:
        """
        Check if a position is obstructed by a building at the given altitude.

        Implementation: Finds the local bucket, iterates through buildings in 
        that bucket, checks if (x,y) is inside the building's footprint, 
        and then checks if the drone's altitude is below the building's top.
        Behavior: Returns True if the drone would "hit" a building at this 
        specific location and altitude.

        Args:
            x (float): X world coordinate.
            y (float): Y world coordinate.
            z_drone (float): Drone altitude.
            safety_margin_z (float): Vertical safety margin above buildings.

        Returns:
            bool: True if obstructed, False otherwise.
        """
        # Height rule: obstacle only if drone is at/below building top + safety margin
        key = self._bucket_for_xy(x, y)
        cand = self._buckets.get(key)
        if not cand:
            return False
        for idx in cand:
            b = self.buildings[idx]
            if b.contains_xy(x, y, self.margin_m):
                if z_drone <= b.top_z + safety_margin_z:
                    return True
        return False

    def is_footprint_xy(self, x: float, y: float) -> bool:
        """
        True if (x,y) lies within any building footprint (ignores altitude).

        Implementation: Similar to is_obstacle_xy, but skips the altitude check.
        Behavior: Identifies if a ground location is "under" a building, 
        regardless of how high the drone is flying.

        Args:
            x (float): X world coordinate.
            y (float): Y world coordinate.

        Returns:
            bool: True if the point is inside any building footprint.
        """
        key = self._bucket_for_xy(x, y)
        cand = self._buckets.get(key)
        if not cand:
            return False
        for idx in cand:
            b = self.buildings[idx]
            if b.contains_xy(x, y, self.margin_m):
                return True
        return False

    def top_z_at_xy(self, x: float, y: float) -> Optional[float]:
        """
        Return max top_z for buildings whose footprint contains (x,y).

        Implementation: Queries the local bucket and returns the altitude 
        of the tallest building at that point.
        Behavior: Useful for calculating "safe flight altitude" to clear 
        all obstacles at a specific (x,y) coordinate.

        Args:
            x (float): X world coordinate.
            y (float): Y world coordinate.

        Returns:
            Optional[float]: The highest building top altitude at this point, or None.
        """
        key = self._bucket_for_xy(x, y)
        cand = self._buckets.get(key)
        if not cand:
            return None
        best = None
        for idx in cand:
            b = self.buildings[idx]
            if b.contains_xy(x, y, self.margin_m):
                tz = float(b.top_z)
                best = tz if best is None else max(float(best), tz)
        return best

    def max_top_z_near(self, x: float, y: float, radius_m: float) -> Optional[float]:
        """
        Return maximum building top_z within radius of (x,y), using bbox distance.

        Implementation: Checks all buckets within the given radius and 
        finds the tallest building in that neighborhood.
        Behavior: Helps a drone determine a safe altitude for an entire 
        local area, not just a single point.

        Args:
            x (float): X world coordinate.
            y (float): Y world coordinate.
            radius_m (float): Search radius in meters.

        Returns:
            Optional[float]: Maximum building altitude found in the radius, or None.
        """
        r = max(0.0, float(radius_m))
        bs = float(self.bucket_size_m)
        if r <= 1e-6 or bs <= 0:
            return None
        bx, by = self._bucket_for_xy(x, y)
        br = int(math.ceil(r / bs))
        best = None
        for ix in range(bx - br, bx + br + 1):
            for iy in range(by - br, by + br + 1):
                cand = self._buckets.get((ix, iy))
                if not cand:
                    continue
                for idx in cand:
                    b = self.buildings[idx]
                    # bbox distance in XY
                    minx, maxx, miny, maxy = b.bbox_xy(self.margin_m)
                    dx = 0.0
                    if x < minx:
                        dx = minx - x
                    elif x > maxx:
                        dx = x - maxx
                    dy = 0.0
                    if y < miny:
                        dy = miny - y
                    elif y > maxy:
                        dy = y - maxy
                    if (dx * dx + dy * dy) <= (r * r + 1e-9):
                        tz = float(b.top_z)
                        best = tz if best is None else max(float(best), tz)
        return best
