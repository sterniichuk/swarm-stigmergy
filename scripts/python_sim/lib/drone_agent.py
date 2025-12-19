from __future__ import annotations
import math
import random
import heapq
from typing import Dict, List, Optional, Tuple, Set, Union
from collections import deque

from scripts.python_sim.dto.pheromones import PheromoneMap
from scripts.python_sim.dto.sim_types import (
    DroneState, 
    EnergyModel, 
    ExploreVector, 
    GridSpec, 
    Target, 
    TargetKnowledge, 
    DynamicInspectStatus
)
from scripts.python_sim.lib.helpers import clamp, out_of_bounds, clamp_xy, softmax_sample, MapBounds
from scripts.python_sim.lib.environment import BuildingIndex

class PythonDrone:
    def __init__(
        self,
        state: DroneState,
        grid: GridSpec,
        pher: PheromoneMap,
        energy: EnergyModel,
        rng: random.Random,
    ):
        """
        Initialize a PythonDrone agent.

        High-level Overview: This is the 'brain' of an individual drone. It stores
        the drone's current flight state, its local map of the world (what it has 
        seen with its own sensors), and links to the collective 'stigmergy' map 
        (pheromones).

        Implementation: Sets up data structures for sparse occupancy mapping 
        (known_free/known_occ), path planning buffers, and state variables for 
        various flight modes. It also initializes 'personality' traits like 
        exploration bias to ensure the swarm doesn't move in lockstep.

        Args:
            state (DroneState): Initial state of the drone (position, yaw, etc.).
            grid (GridSpec): Grid specification for the environment.
            pher (PheromoneMap): Shared or local pheromone map.
            energy (EnergyModel): Model for energy consumption.
            rng (random.Random): Random number generator for stochastic behaviors.
        """
        self.s = state
        self.grid = grid
        self.pher = pher
        self.energy = energy
        self.rng = rng
        # knowledge: unknown/free/occupied (stored sparsely; unknown default)
        self.known_free: Dict[Tuple[int, int], bool] = {}
        self.known_occ: Dict[Tuple[int, int], bool] = {}
        # current steering
        self._last_yaw = state.yaw
        # Debug: store last planned path (world xy points) for visualization
        self.last_plan_world: List[Tuple[float, float]] = []
        self.last_plan_t: float = -1e9
        # Debug: how the last motion direction was chosen ("ACO" vs "A*").
        self.last_move_source: str = ""
        self.last_aco_choice_world: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)
        self.last_aco_choice_t: float = -1e9
        # Candidate headings from the last ACO scoring pass: list of (score, (nx,ny,yaw)).
        self.last_aco_candidates: List[Tuple[float, Tuple[float, float, float]]] = []
        # Committed local plan (prevents oscillation between similar A* solutions).
        self._active_plan_world: List[Tuple[float, float]] = []
        self._active_plan_idx: int = 0
        self._active_plan_until_t: float = -1e9
        # Per-drone "crab side" preference (+1 left, -1 right)
        self.crab_side = -1 if (self.rng.random() < 0.5) else 1
        # Per-drone exploration "personality" to avoid lockstep behavior.
        self._explore_bias_yaw = float(self.rng.uniform(-math.pi, math.pi))
        # ACO commitment: stick to chosen next cell until reached unless perception changes.
        self._aco_commit_cell: Optional[Tuple[int, int]] = None
        self._aco_commit_rev: int = 0
        self._aco_commit_until_t: float = -1e9
        # Dynamic danger kernel knowledge cache:
        # id -> {"cell": (cx,cy), "t": sim_time, "speed_s": seconds-per-cell, "src": drone_uid}
        self._dyn_kernel_latest: Dict[str, dict] = {}
        # Kernel path reconstruction state per danger id.
        # id -> {"seq": [cell,...], "seen": set(cells), "last": cell, "last_t": t,
        #        "first": cell, "loop_hits": int, "loop_seen_count": int, "complete": bool}
        self._dyn_kernel_path: Dict[str, dict] = {}
        # Real-time (LiDAR) sightings of dynamic danger kernel (local only, short-lived).
        # id -> {"cell": (cx,cy), "t": sim_time, "speed_s": Optional[float]}
        self._dyn_kernel_realtime: Dict[str, dict] = {}
        # Inspector debug logging (print every N new cells while inspecting)
        self._insp_dbg_last_cell: Optional[Tuple[int, int]] = None
        self._insp_dbg_cell_steps: int = 0
        # Dynamic danger "no-fly" footprints derived from dynamic threat metadata.
        # Stored as list of (cx, cy, rad2_cells) so checks are O(#danger_ids) and we avoid O(r^2) rasterization.
        self._dyn_nofly_footprints: List[Tuple[int, int, int]] = []

        # Planning/scoring tuning injected from PythonFastSim (defaults kept for safety).
        # - altitude penalty: additional danger cost when below a static danger altitude
        # - A* altitude step: quantization for altitude-aware local A* (cost evaluation)
        # - vertical cost: uses the same model as overfly decision (mult + descend factor)
        self.static_danger_altitude_violation_weight: float = 6.0
        self.local_a_star_altitude_step_m: float = 0.5
        self.planning_vertical_cost_mult: float = 3.0
        self.planning_descend_cost_factor: float = 0.7

    def _dyn_cell_blocked(self, c: Tuple[int, int]) -> bool:
        """
        Return True if cell intersects any known dynamic threat footprint.

        High-level Overview: Performs a quick safety check against moving hazards
        (like other drones or moving ground vehicles) that have been recently detected.

        Implementation: Checks the requested cell against a list of 'no-fly zones'
        centered on the last known positions of dynamic threats. Uses squared distance 
        checks for performance.

        Args:
            c (Tuple[int, int]): Grid cell coordinates (cx, cy).

        Returns:
            bool: True if the cell is blocked by a dynamic threat.
        """
        try:
            fps = getattr(self, "_dyn_nofly_footprints", None) or []
            cx, cy = int(c[0]), int(c[1])
            for fx, fy, rad2 in fps:
                dx = cx - int(fx)
                dy = cy - int(fy)
                if (dx * dx + dy * dy) <= int(rad2):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _bresenham_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Discrete grid line (inclusive) using Bresenham's algorithm.

        High-level Overview: Converts a straight flight path between two points 
        into a list of specific grid cells that the drone would cross.

        Implementation: Standard Bresenham's line algorithm. It efficiently 
        determines which integer grid coordinates are closest to the ideal 
        mathematical line.

        Args:
            a (Tuple[int, int]): Start cell (x0, y0).
            b (Tuple[int, int]): End cell (x1, y1).

        Returns:
            List[Tuple[int, int]]: List of grid cells on the line.
        """
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

    def _record_dyn_kernel_obs(self, danger_id: str, cell: Tuple[int, int], t: float, speed_s: Optional[float], src: str):
        """
        Update kernel latest + reconstruct a contiguous kernel path sequence (with inferred in-between cells).

        High-level Overview: Tracks the movement of a dynamic threat over time.
        If the drone sees a threat at Point A and then later at Point B, this 
        method 'fills in the blanks' to estimate the path taken between them.

        Implementation: Updates a cache of the latest sighting. It uses 
        Bresenham's algorithm to connect consecutive sightings into a 
        continuous trajectory sequence. It also includes a heuristic to detect 
        if a threat is moving in a loop (closed circuit).

        Args:
            danger_id (str): ID of the dynamic threat.
            cell (Tuple[int, int]): Observed cell.
            t (float): Simulation time.
            speed_s (Optional[float]): Speed in seconds per cell.
            src (str): Source drone UID.
        """
        did = str(danger_id or "")
        if not did:
            return
        c = (int(cell[0]), int(cell[1]))
        t = float(t)
        # Update latest cache
        prev = self._dyn_kernel_latest.get(did)
        if prev is None or t >= float(prev.get("t", -1e9)) - 1e-6:
            self._dyn_kernel_latest[did] = {"cell": c, "t": float(t), "speed_s": (float(speed_s) if speed_s is not None else None), "src": str(src)}

        st = self._dyn_kernel_path.get(did)
        if st is None:
            st = {"seq": [c], "seen": {c}, "last": c, "last_t": t, "first": c, "loop_hits": 0, "loop_seen_count": 1, "complete": False}
            self._dyn_kernel_path[did] = st
            return
        if bool(st.get("complete", False)):
            return
        last = tuple(st.get("last", c))
        last_t = float(st.get("last_t", -1e9))
        if t <= last_t + 1e-6 and c == last:
            return
        # Fill holes in the sequence (continuous path) by inferring intermediate cells.
        seg = self._bresenham_cells((int(last[0]), int(last[1])), c)
        for cc in seg[1:]:
            st["seq"].append(cc)
            st["seen"].add(cc)
        st["last"] = c
        st["last_t"] = t
        # Completion heuristic:
        # If we hit the first cell twice and the unique set size didn't change between hits, call it complete.
        if c == tuple(st.get("first", c)):
            st["loop_hits"] = int(st.get("loop_hits", 0)) + 1
            cur_seen = int(len(st.get("seen", set())))
            prev_seen = int(st.get("loop_seen_count", cur_seen))
            if int(st["loop_hits"]) >= 2 and cur_seen == prev_seen:
                st["complete"] = True
            st["loop_seen_count"] = cur_seen

        # If we just became complete and we are (or were) the inspector, mark kernels as fully inspected.
        try:
            if bool(st.get("complete", False)) and str(getattr(self.s, "dynamic_inspect_active_id", "") or "") == did:
                self._mark_dyn_kernel_done(did, float(t))
                self.s.dynamic_inspect_active_id = ""
                self.s.dynamic_inspect_skip_ids.add(str(did))
        except Exception:
            pass

    def _mark_dyn_kernel_done(self, danger_id: str, t: float):
        """
        Mark all known kernel cells for this dynamic danger as fully inspected (render as white).

        High-level Overview: Updates the collective map to signal that a 
        specific dynamic threat trajectory has been fully mapped and 
        no longer needs active 'inspection'.

        Implementation: Iterates through all cells in the reconstructed 
        trajectory and updates their metadata kind to 'danger_dyn_kernel_done'. 
        This change in 'kind' usually triggers a visual change in the UI 
        (e.g., changing color to white).

        Args:
            danger_id (str): ID of the dynamic threat.
            t (float): Simulation time.
        """
        did = str(danger_id or "")
        if not did:
            return
        st = self._dyn_kernel_path.get(did)
        if not st:
            return
        # Use the set of observed cells; this is our reconstructed kernel trajectory.
        cells = list(st.get("seen", []) or [])
        for cc in cells:
            try:
                meta = self.pher.danger.meta.get(tuple(cc))
                if meta is None:
                    continue
                k = str(getattr(meta, "kind", "") or "")
                if not k.startswith("danger_dyn_kernel"):
                    continue
                # Preserve value; only change kind to a "done" marker (id preserved).
                v = float(self.pher.danger.get(tuple(cc)))
                if v <= 1e-9:
                    continue
                meta.kind = f"danger_dyn_kernel_done:{did}"
                meta.t = float(max(float(meta.t), float(t)))
                self.pher.danger.set(tuple(cc), v, meta)
            except Exception:
                continue

    def distance_to(self, x: float, y: float) -> float:
        """
        Euclidean distance to a world point.

        High-level Overview: Simple 'as-the-crow-flies' distance measurement 
        from the drone's current position to a target.

        Implementation: Uses the Pythagorean theorem (math.hypot) on world 
        coordinates.

        Args:
            x (float): World X.
            y (float): World Y.

        Returns:
            float: Distance in meters.
        """
        return math.hypot(self.s.x - x, self.s.y - y)

    def _ray_clearance_m(
        self,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
        yaw: float,
        max_dist_m: float,
        step_m: float,
    ) -> float:
        """
        Approximate free-space clearance along a heading.

        High-level Overview: This is like a virtual range-finder. It tells the 
        drone how far it can fly in a straight line at its current altitude 
        before hitting a building or leaving the mission area.

        Implementation: It 'marches' a point forward from the drone's position 
        along the specified heading. At each step, it checks the building 
        index for collisions.

        Args:
            building_index (BuildingIndex): Index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Map boundaries.
            yaw (float): Heading in radians.
            max_dist_m (float): Maximum search distance.
            step_m (float): Step size for ray casting.

        Returns:
            float: Clearance distance in meters.
        """
        step_m = max(0.25, float(step_m))
        max_steps = int(max_dist_m / step_m)
        if max_steps <= 0:
            return 0.0
        cx, cy = float(self.s.x), float(self.s.y)
        dx, dy = math.cos(yaw), math.sin(yaw)
        # Start with a small forward probe so "grazing" walls is detected early.
        for i in range(1, max_steps + 1):
            d = i * step_m
            px = cx + dx * d
            py = cy + dy * d
            if out_of_bounds(px, py, map_bounds_m):
                return float(d)
            if building_index.is_obstacle_xy(px, py, self.s.z, safety_margin_z):
                return float(d)
        return float(max_dist_m)

    def _cell_blocked_for_planning(
        self,
        c: Tuple[int, int],
        inflate_cells: int,
        building_index: Optional[BuildingIndex] = None,
        safety_margin_z: float = 0.0,
    ) -> bool:
        """
        Check if a cell should be considered blocked for A* path planning.

        High-level Overview: A safety check for the path planner. It determines 
        if a grid cell is a 'no-go' zone based on buildings, dynamic threats, 
        and recently discovered obstacles.

        Implementation: Combines several checks:
        1. Is it out of bounds?
        2. Is there a dynamic threat there?
        3. Is there a wall or obstacle detected by LiDAR?
        4. Is it inside a known building footprint?
        Includes 'inflation' logic to keep the drone a safe distance from walls.

        Args:
            c (Tuple[int, int]): Cell coordinates.
            inflate_cells (int): Radius for obstacle inflation.
            building_index (Optional[BuildingIndex]): Building index for altitude-aware checks.
            safety_margin_z (float): Vertical safety margin.

        Returns:
            bool: True if blocked.
        """
        if not self.grid.in_bounds_cell(c[0], c[1]):
            return True
        # Dynamic danger: never plan into the *current* dynamic danger footprint.
        if self._dyn_cell_blocked(c):
            return True
        # Navigation danger (walls/obstacles inferred by lidar): treat as blocked below the recorded blocking altitude.
        # This allows "overfly" to plan through it when high enough, but avoids aiming into walls at low altitude.
        try:
            mk = self.pher.danger.meta.get(c)
            if mk is not None:
                k = str(getattr(mk, "kind", "") or "")
                if k == "nav_danger" or k.startswith("nav_danger"):
                    alt = getattr(mk, "alt_m", None)
                    if alt is None:
                        return True
                    if float(self.s.z) < float(alt) - 1e-6:
                        return True
        except Exception:
            pass
        if c in self.known_occ:
            # Altitude-aware override: a cell marked occupied at low altitude should become traversable
            # once the drone is high enough that the building is no longer an obstacle at (x,y,z).
            if building_index is not None:
                wx, wy = self.grid.cell_to_world(c[0], c[1])
                if not building_index.is_obstacle_xy(float(wx), float(wy), float(self.s.z), float(safety_margin_z)):
                    return False
            return True
        if inflate_cells > 0:
            cx, cy = c
            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    cc = (cx + dx, cy + dy)
                    if cc in self.known_occ:
                        if building_index is not None and self.grid.in_bounds_cell(cc[0], cc[1]):
                            wx, wy = self.grid.cell_to_world(cc[0], cc[1])
                            if not building_index.is_obstacle_xy(float(wx), float(wy), float(self.s.z), float(safety_margin_z)):
                                continue
                        return True
        return False

    def _cell_cost_for_planning(
        self,
        c: Tuple[int, int],
        mission_phase: str,
        unknown_penalty: float,
        recent_penalty: float,
        z_eff_override: Optional[float] = None,
    ) -> float:
        """
        Calculate traversal cost for a cell, incorporating pheromones, penalties, and altitude.

        High-level Overview: Assigns a 'difficulty score' to a grid cell. The 
        path planner tries to find the path with the lowest total score. 
        Danger areas have high scores, while areas with 'navigation' 
        pheromones have lower scores.

        Implementation: Combines weights for navigation tau (attraction) and 
        danger tau (repulsion). It handles 'overfly' logic: if the drone is 
        flying higher than a static threat, the cost is reduced or removed. 
        It also adds penalties for visiting the same cells repeatedly 
        (to avoid loops).

        Args:
            c (Tuple[int, int]): Cell coordinates.
            mission_phase (str): Current mission phase (e.g., 'EXPLOIT').
            unknown_penalty (float): Penalty for unvisited cells.
            recent_penalty (float): Penalty for recently visited cells (to avoid loops).
            z_eff_override (Optional[float]): Override for altitude in cost calculation.

        Returns:
            float: Total traversal cost.
        """
        nav_tau = self.pher.nav.get(c)
        danger_tau = self.pher.danger.get(c)
        # Altitude-aware danger:
        # - Static threats (danger_static) can be "overflown": if our planned altitude is above the
        #   danger altitude, treat them as non-blocking (no extra safety margin like buildings).
        # - Dynamic threats are treated like other drones: altitude doesn't "solve" them; keep cost.
        try:
            if danger_tau > 1e-9:
                mk = self.pher.danger.meta.get(c)
                if mk is not None:
                    k = str(getattr(mk, "kind", "") or "")
                    # Effective altitude for evaluating "can overfly" semantics.
                    # If caller provides an override (altitude-aware A*), trust it.
                    z_eff = float(self.s.z) if z_eff_override is None else float(z_eff_override)
                    if z_eff_override is None:
                        # When overflying/hopping we plan for the target altitude (not the instantaneous z).
                        if bool(getattr(self.s, "overfly_active", False)) or bool(getattr(self.s, "hop_active", False)):
                            try:
                                z_eff = max(z_eff, float(getattr(self.s, "z_target", z_eff)))
                            except Exception:
                                pass
                    # Dynamic danger pheromone trails:
                    # - default (dynamic-aware): informational only; ignore in planning cost
                    # - ablation (treat dynamic as static): make A* route around the trail by *increasing* its cost
                    if k.startswith("danger_dyn_"):
                        treat_static = bool(getattr(self, "_dyn_trail_static_for_planning", False))
                        if not treat_static:
                            danger_tau = 0.0
                        else:
                            # "Temporary static overlay": amplify high (red) values more than low (green) ones.
                            # overlay = strength * (danger_tau ^ gamma)
                            strength = float(getattr(self, "_dyn_trail_overlay_strength", 3.0))
                            gamma = float(getattr(self, "_dyn_trail_overlay_gamma", 1.8))
                            strength = max(0.0, strength)
                            gamma = max(0.5, gamma)
                            overlay = float(strength) * (float(max(0.0, danger_tau)) ** float(gamma))
                            danger_tau = float(danger_tau) + float(overlay)
                            # When we explicitly "treat dynamic danger as static", allow overfly behavior:
                            # if our effective planned altitude is above the stored threat altitude, do not penalize.
                            alt_dyn = getattr(mk, "alt_m", None)
                            if alt_dyn is not None:
                                if float(z_eff) >= float(alt_dyn) - 1e-6:
                                    danger_tau = 0.0
                                else:
                                    # Penalize altitude violation proportionally to how far below the required altitude we are.
                                    dz = max(0.0, float(alt_dyn) - float(z_eff))
                                    frac = float(dz) / max(1.0, float(alt_dyn))
                                    w = float(getattr(self, "static_danger_altitude_violation_weight", 0.0))
                                    if w > 1e-9 and frac > 1e-9:
                                        danger_tau = float(danger_tau) + float(w) * float(frac)
                    if k.startswith("danger_static") and (getattr(mk, "alt_m", None) is not None):
                        if float(z_eff) >= float(getattr(mk, "alt_m")) - 1e-6:
                            danger_tau = 0.0
                        else:
                            # Penalize altitude violation proportionally to deficit.
                            alt_req = float(getattr(mk, "alt_m"))
                            dz = max(0.0, alt_req - float(z_eff))
                            frac = float(dz) / max(1.0, alt_req)
                            w = float(getattr(self, "static_danger_altitude_violation_weight", 0.0))
                            if w > 1e-9 and frac > 1e-9:
                                danger_tau = float(danger_tau) + float(w) * float(frac)
        except Exception:
            pass
        is_unknown = (c not in self.known_free) and (c not in self.known_occ)
        unk = float(unknown_penalty) if is_unknown else 0.0
        # Penalize loops in RETURN too (helps when boxed by danger fields).
        rec = float(recent_penalty) if c in self.s.recent_cells else 0.0
        if self.s.mode == "RETURN":
            # In RETURN, danger should influence but must not prevent passing the only corridor.
            # Soften danger weight and also soften unknown penalty.
            return (2.0 * danger_tau) + (0.5 * unk) + (1.5 * rec)
        if mission_phase == "EXPLOIT":
            return (4.0 * danger_tau) + unk + rec - (0.5 * nav_tau)
        return (5.0 * danger_tau) + unk + rec

    def _local_a_star_next_cell(
        self,
        goal_xy: Tuple[float, float],
        mission_phase: str,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
        plan_radius_m: float,
        inflate_cells: int,
        max_nodes: int,
        unknown_penalty: float,
        recent_penalty: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Local A* search on the drone's *known* occupancy (from mock lidar) within a radius.

        High-level Overview: A short-range pathfinding algorithm. It looks 
        at the known local obstacles and pheromones to find the single best 
        next grid cell to move into to reach a goal.

        Implementation: Standard A* search algorithm restricted to a local 
        circular window around the drone. It uses '_cell_blocked_for_planning' 
        for safety and '_cell_cost_for_planning' to weight different paths.

        Args:
            goal_xy (Tuple[float, float]): Goal world position.
            mission_phase (str): Current mission phase.
            building_index (BuildingIndex): Building index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Map boundaries.
            plan_radius_m (float): Radius of the local planning window.
            inflate_cells (int): Obstacle inflation radius in cells.
            max_nodes (int): Maximum nodes to expand in A*.
            unknown_penalty (float): Penalty for unvisited cells.
            recent_penalty (float): Penalty for recently visited cells.

        Returns:
            Optional[Tuple[int, int]]: The next grid cell to move into, or None if no path found.
        """
        start = self.grid.world_to_cell(self.s.x, self.s.y)
        goal = self.grid.world_to_cell(goal_xy[0], goal_xy[1])
        if not self.grid.in_bounds_cell(start[0], start[1]):
            return None

        r_cells = max(3, int(max(10.0, float(plan_radius_m)) / self.grid.cell_size_m))
        sx, sy = start
        min_cx = max(0, sx - r_cells)
        max_cx = min(self.grid.cells - 1, sx + r_cells)
        min_cy = max(0, sy - r_cells)
        max_cy = min(self.grid.cells - 1, sy + r_cells)

        def in_window(c: Tuple[int, int]) -> bool:
            return (min_cx <= c[0] <= max_cx) and (min_cy <= c[1] <= max_cy)

        def neighbors(c: Tuple[int, int]) -> List[Tuple[int, int]]:
            cx, cy = c
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nc = (cx + dx, cy + dy)
                    if in_window(nc):
                        out.append(nc)
            return out

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if not in_window(goal):
            goal = (int(clamp(goal[0], min_cx, max_cx)), int(clamp(goal[1], min_cy, max_cy)))

        if self._cell_blocked_for_planning(goal, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z):
            found = None
            for rr in range(1, 8):
                for dx in range(-rr, rr + 1):
                    for dy in range(-rr, rr + 1):
                        if abs(dx) != rr and abs(dy) != rr:
                            continue
                        cand = (goal[0] + dx, goal[1] + dy)
                        if not in_window(cand):
                            continue
                        if not self._cell_blocked_for_planning(cand, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z):
                            found = cand
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                return None
            goal = found

        # Cache per-cell checks: these are the hotspots inside the neighbor expansion loop.
        blocked_cache: Dict[Tuple[int, int], bool] = {}
        geom_blocked_cache: Dict[Tuple[int, int], bool] = {}
        cell_cost_cache: Dict[Tuple[int, int], float] = {}

        def _geom_blocked(c: Tuple[int, int]) -> bool:
            if c in geom_blocked_cache:
                return bool(geom_blocked_cache[c])
            wx, wy = self.grid.cell_to_world(c[0], c[1])
            v = bool(out_of_bounds(float(wx), float(wy), map_bounds_m)) or bool(
                building_index.is_obstacle_xy(float(wx), float(wy), float(self.s.z), float(safety_margin_z))
            )
            geom_blocked_cache[c] = v
            return v

        def _blocked(c: Tuple[int, int]) -> bool:
            if c in blocked_cache:
                return bool(blocked_cache[c])
            v = bool(
                self._cell_blocked_for_planning(
                    c, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z
                )
            ) or _geom_blocked(c)
            blocked_cache[c] = v
            return v

        def _cell_cost(c: Tuple[int, int]) -> float:
            if c in cell_cost_cache:
                return float(cell_cost_cache[c])
            v = float(
                self._cell_cost_for_planning(
                    c,
                    mission_phase=mission_phase,
                    unknown_penalty=unknown_penalty,
                    recent_penalty=recent_penalty,
                )
            )
            cell_cost_cache[c] = v
            return v

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g: Dict[Tuple[int, int], float] = {start: 0.0}
        closed: Set[Tuple[int, int]] = set()
        nodes = 0

        while open_heap and nodes < max(200, int(max_nodes)):
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            nodes += 1
            if cur == goal:
                break
            for nb in neighbors(cur):
                if nb in closed:
                    continue
                if _blocked(nb):
                    continue

                step_cost = math.hypot(nb[0] - cur[0], nb[1] - cur[1])
                cell_cost = _cell_cost(nb)
                tentative = g[cur] + step_cost + cell_cost
                if tentative < g.get(nb, 1e18):
                    came_from[nb] = cur
                    g[nb] = tentative
                    f = tentative + heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, nb))

        if goal != start and goal not in came_from:
            return None

        cur = goal
        path = [cur]
        while cur != start and cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        if len(path) < 2:
            return None
        return path[1]

    def _local_a_star_path_world(
        self,
        goal_xy: Tuple[float, float],
        mission_phase: str,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
        plan_radius_m: float,
        inflate_cells: int,
        max_nodes: int,
        unknown_penalty: float,
        recent_penalty: float,
        max_points: int = 300,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Similar to _local_a_star_next_cell, but returns the full path as world xy points.

        High-level Overview: Finds a complete path from the drone's position to 
         a goal, considering known obstacles. This is used for visualization 
         and long-term planning rather than just the immediate next step.

        Implementation: Identical to '_local_a_star_next_cell' in its search logic, 
         but instead of just returning the first step, it reconstructs the 
         entire sequence of cells from start to goal. It then downsamples these 
         cells into a set of world coordinates (waypoints).

        Args:
            goal_xy (Tuple[float, float]): Goal world position.
            mission_phase (str): Current mission phase.
            building_index (BuildingIndex): Building index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Map boundaries.
            plan_radius_m (float): Radius of the local planning window.
            inflate_cells (int): Obstacle inflation radius in cells.
            max_nodes (int): Maximum nodes to expand in A*.
            unknown_penalty (float): Penalty for unvisited cells.
            recent_penalty (float): Penalty for recently visited cells.
            max_points (int, optional): Maximum number of points in the returned path.

        Returns:
            Optional[List[Tuple[float, float]]]: List of (x, y) world points for the path, or None.
        """
        start = self.grid.world_to_cell(self.s.x, self.s.y)
        goal = self.grid.world_to_cell(goal_xy[0], goal_xy[1])
        if not self.grid.in_bounds_cell(start[0], start[1]):
            return None

        r_cells = max(3, int(max(10.0, float(plan_radius_m)) / self.grid.cell_size_m))
        sx, sy = start
        min_cx = max(0, sx - r_cells)
        max_cx = min(self.grid.cells - 1, sx + r_cells)
        min_cy = max(0, sy - r_cells)
        max_cy = min(self.grid.cells - 1, sy + r_cells)

        def in_window(c: Tuple[int, int]) -> bool:
            return (min_cx <= c[0] <= max_cx) and (min_cy <= c[1] <= max_cy)

        def neighbors(c: Tuple[int, int]) -> List[Tuple[int, int]]:
            cx, cy = c
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nc = (cx + dx, cy + dy)
                    if in_window(nc):
                        out.append(nc)
            return out

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if not in_window(goal):
            goal = (int(clamp(goal[0], min_cx, max_cx)), int(clamp(goal[1], min_cy, max_cy)))

        if self._cell_blocked_for_planning(goal, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z):
            found = None
            for rr in range(1, 8):
                for dx in range(-rr, rr + 1):
                    for dy in range(-rr, rr + 1):
                        if abs(dx) != rr and abs(dy) != rr:
                            continue
                        cand = (goal[0] + dx, goal[1] + dy)
                        if not in_window(cand):
                            continue
                        if not self._cell_blocked_for_planning(cand, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z):
                            found = cand
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                return None
            goal = found

        # Cache per-cell checks: this A* is called frequently and these checks dominate runtime.
        blocked_cache: Dict[Tuple[int, int], bool] = {}
        geom_blocked_cache: Dict[Tuple[int, int], bool] = {}
        cell_cost_cache: Dict[Tuple[int, int], float] = {}

        def _geom_blocked(c: Tuple[int, int]) -> bool:
            if c in geom_blocked_cache:
                return bool(geom_blocked_cache[c])
            wx, wy = self.grid.cell_to_world(c[0], c[1])
            v = bool(out_of_bounds(float(wx), float(wy), map_bounds_m)) or bool(
                building_index.is_obstacle_xy(float(wx), float(wy), float(self.s.z), float(safety_margin_z))
            )
            geom_blocked_cache[c] = v
            return v

        def _blocked(c: Tuple[int, int]) -> bool:
            if c in blocked_cache:
                return bool(blocked_cache[c])
            v = bool(
                self._cell_blocked_for_planning(
                    c, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z
                )
            ) or _geom_blocked(c)
            blocked_cache[c] = v
            return v

        def _cell_cost(c: Tuple[int, int]) -> float:
            if c in cell_cost_cache:
                return float(cell_cost_cache[c])
            v = float(
                self._cell_cost_for_planning(
                    c,
                    mission_phase=mission_phase,
                    unknown_penalty=unknown_penalty,
                    recent_penalty=recent_penalty,
                )
            )
            cell_cost_cache[c] = v
            return v

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g: Dict[Tuple[int, int], float] = {start: 0.0}
        closed: Set[Tuple[int, int]] = set()
        nodes = 0

        while open_heap and nodes < max(200, int(max_nodes)):
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            nodes += 1
            if cur == goal:
                break
            for nb in neighbors(cur):
                if nb in closed:
                    continue
                if _blocked(nb):
                    continue

                step_cost = math.hypot(nb[0] - cur[0], nb[1] - cur[1])
                cell_cost = _cell_cost(nb)
                tentative = g[cur] + step_cost + cell_cost
                if tentative < g.get(nb, 1e18):
                    came_from[nb] = cur
                    g[nb] = tentative
                    f = tentative + heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, nb))

        if goal != start and goal not in came_from:
            return None

        cur = goal
        cells = [cur]
        while cur != start and cur in came_from:
            cur = came_from[cur]
            cells.append(cur)
            if len(cells) > 2000:
                break
        cells.reverse()
        if len(cells) < 2:
            return None

        # Downsample to avoid huge markers
        stride = max(1, int(len(cells) / max(2, int(max_points))))
        out: List[Tuple[float, float]] = []
        for i in range(0, len(cells), stride):
            wx, wy = self.grid.cell_to_world(cells[i][0], cells[i][1])
            out.append((float(wx), float(wy)))
        # Ensure goal included
        if out and out[-1] != out[-1]:
            pass
        if out:
            wx, wy = self.grid.cell_to_world(cells[-1][0], cells[-1][1])
            out.append((float(wx), float(wy)))
        return out

    def _local_a_star_cost(
        self,
        goal_xy: Tuple[float, float],
        mission_phase: str,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
        plan_radius_m: float,
        inflate_cells: int,
        max_nodes: int,
        unknown_penalty: float,
        recent_penalty: float,
    ) -> Optional[float]:
        """
        Return total A* cost to goal (within local window) or None if unreachable.
        This variant is altitude-aware for static danger:
        - Entering a static-danger cell implicitly requires climbing to its `alt_m` (minimum altitude).
        - The A* cost includes the vertical climb cost (and an estimated descend-to-cruise cost).

        High-level Overview: Estimates the 'effort' required to reach a goal. 
        Crucially, this version understands that some hazards require flying 
        higher, and it includes the energy/time cost of climbing and 
        descending in its estimate.

        Implementation: A 2.5D path planner. The state space includes both 
        grid coordinates and quantized altitude steps. It enforces a 
        monotonically non-decreasing altitude for static danger zones (i.e., 
        you must climb to enter, and you stay high until the zone is passed). 
        The cost function includes both horizontal distance and vertical 
        motion costs.

        Args:
            goal_xy (Tuple[float, float]): Goal world position.
            mission_phase (str): Current mission phase.
            building_index (BuildingIndex): Building index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Map boundaries.
            plan_radius_m (float): Radius of the local planning window.
            inflate_cells (int): Obstacle inflation radius in cells.
            max_nodes (int): Maximum nodes to expand in A*.
            unknown_penalty (float): Penalty for unvisited cells.
            recent_penalty (float): Penalty for recently visited cells.

        Returns:
            Optional[float]: Total A* cost, or None if unreachable.
        """
        start = self.grid.world_to_cell(self.s.x, self.s.y)
        goal = self.grid.world_to_cell(goal_xy[0], goal_xy[1])
        if not self.grid.in_bounds_cell(start[0], start[1]):
            return None

        r_cells = max(3, int(max(10.0, float(plan_radius_m)) / self.grid.cell_size_m))
        sx, sy = start
        min_cx = max(0, sx - r_cells)
        max_cx = min(self.grid.cells - 1, sx + r_cells)
        min_cy = max(0, sy - r_cells)
        max_cy = min(self.grid.cells - 1, sy + r_cells)

        def in_window(c: Tuple[int, int]) -> bool:
            return (min_cx <= c[0] <= max_cx) and (min_cy <= c[1] <= max_cy)

        def neighbors(c: Tuple[int, int]) -> List[Tuple[int, int]]:
            cx, cy = c
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nc = (cx + dx, cy + dy)
                    if in_window(nc):
                        out.append(nc)
            return out

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if not in_window(goal):
            goal = (int(clamp(goal[0], min_cx, max_cx)), int(clamp(goal[1], min_cy, max_cy)))

        if self._cell_blocked_for_planning(goal, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z):
            return None

        # Cache per-cell checks: cost-eval A* can be called during avoidance decisions.
        blocked_cache: Dict[Tuple[int, int], bool] = {}
        geom_blocked_cache: Dict[Tuple[int, int], bool] = {}
        cell_cost_cache: Dict[Tuple[int, int], float] = {}
        req_alt_q_cache: Dict[Tuple[int, int], int] = {}

        # Altitude quantization for tractable 2.5D local planning.
        z_step = float(max(0.25, float(getattr(self, "local_a_star_altitude_step_m", 0.5))))
        v_mult = float(max(0.0, float(getattr(self, "planning_vertical_cost_mult", 3.0))))
        v_desc = float(max(0.0, float(getattr(self, "planning_descend_cost_factor", 0.7))))
        z_cruise = float(getattr(self.s, "z_cruise", self.s.z))

        def _z_to_q(zm: float) -> int:
            return int(round(float(zm) / z_step))

        def _q_to_z(q: int) -> float:
            return float(q) * float(z_step)

        def _descend_est(q: int) -> float:
            if v_mult <= 1e-9 or v_desc <= 1e-9:
                return 0.0
            return float(v_mult) * float(v_desc) * max(0.0, float(_q_to_z(int(q))) - float(z_cruise))

        def _req_alt_q(c: Tuple[int, int]) -> int:
            """Minimum altitude (quantized) required for this cell due to static danger, else 0."""
            if c in req_alt_q_cache:
                return int(req_alt_q_cache[c])
            qreq = 0
            try:
                mk = self.pher.danger.meta.get(c)
                if mk is not None:
                    k = str(getattr(mk, "kind", "") or "")
                    alt = getattr(mk, "alt_m", None)
                    if alt is not None:
                        # Static threats always enforce a minimum altitude.
                        if k.startswith("danger_static"):
                            qreq = int(math.ceil(float(alt) / z_step - 1e-9))
                        # Optional ablation: treat dynamic trail as static in planning.
                        elif k.startswith("danger_dyn_") and bool(getattr(self, "_dyn_trail_static_for_planning", False)):
                            qreq = int(math.ceil(float(alt) / z_step - 1e-9))
            except Exception:
                qreq = 0
            qreq = int(max(0, qreq))
            req_alt_q_cache[c] = int(qreq)
            return int(qreq)

        def _geom_blocked(c: Tuple[int, int]) -> bool:
            if c in geom_blocked_cache:
                return bool(geom_blocked_cache[c])
            wx, wy = self.grid.cell_to_world(c[0], c[1])
            v = bool(out_of_bounds(float(wx), float(wy), map_bounds_m)) or bool(
                building_index.is_obstacle_xy(float(wx), float(wy), float(self.s.z), float(safety_margin_z))
            )
            geom_blocked_cache[c] = v
            return v

        def _blocked(c: Tuple[int, int]) -> bool:
            if c in blocked_cache:
                return bool(blocked_cache[c])
            v = bool(
                self._cell_blocked_for_planning(
                    c, inflate_cells, building_index=building_index, safety_margin_z=safety_margin_z
                )
            ) or _geom_blocked(c)
            blocked_cache[c] = v
            return v

        def _cell_cost(c: Tuple[int, int]) -> float:
            if c in cell_cost_cache:
                return float(cell_cost_cache[c])
            v = float(
                self._cell_cost_for_planning(
                    c,
                    mission_phase=mission_phase,
                    unknown_penalty=unknown_penalty,
                    recent_penalty=recent_penalty,
                    # Use current altitude for the cached baseline. (Altitude-aware A* will override per-state.)
                    z_eff_override=None,
                )
            )
            cell_cost_cache[c] = v
            return v

        # State includes (cell, altitude_q). Altitude is monotonically non-decreasing in this planner
        # (we model "climb to clear static danger"; descent is handled by an admissible goal estimate).
        start_q = _z_to_q(float(self.s.z))
        open_heap: List[Tuple[float, Tuple[Tuple[int, int], int]]] = []
        heapq.heappush(open_heap, (0.0, (start, int(start_q))))
        g: Dict[Tuple[Tuple[int, int], int], float] = {(start, int(start_q)): 0.0}
        closed: Set[Tuple[Tuple[int, int], int]] = set()
        nodes = 0

        while open_heap and nodes < max(200, int(max_nodes)):
            _, state = heapq.heappop(open_heap)
            if state in closed:
                continue
            closed.add(state)
            nodes += 1
            cur, cur_q = state
            if cur == goal:
                return float(g.get((cur, int(cur_q)), 0.0)) + float(_descend_est(int(cur_q)))
            for nb in neighbors(cur):
                # Closed test is per-state; but we can early-skip if we've already expanded the same cell+alt.
                if (nb, int(cur_q)) in closed:
                    continue
                if _blocked(nb):
                    continue

                # Enforce minimum altitude for static-danger cells by climbing as needed.
                req_q = int(_req_alt_q(nb))
                new_q = int(cur_q) if int(cur_q) >= req_q else req_q
                dz_up = float(_q_to_z(new_q) - _q_to_z(int(cur_q)))
                vert_cost = float(v_mult) * max(0.0, float(dz_up))

                step_cost = math.hypot(nb[0] - cur[0], nb[1] - cur[1])
                z_eff = float(_q_to_z(new_q))
                # Use altitude-aware cell cost so static danger becomes non-penalizing once we climbed above it.
                cell_cost = float(
                    self._cell_cost_for_planning(
                        nb,
                        mission_phase=mission_phase,
                        unknown_penalty=unknown_penalty,
                        recent_penalty=recent_penalty,
                        z_eff_override=float(z_eff),
                    )
                )
                tentative = float(g.get((cur, int(cur_q)), 1e18)) + float(step_cost) + float(cell_cost) + float(vert_cost)
                nb_state = (nb, int(new_q))
                if tentative < g.get(nb_state, 1e18):
                    g[nb_state] = float(tentative)
                    f = float(tentative) + float(heuristic(nb, goal)) + float(_descend_est(int(new_q)))
                    heapq.heappush(open_heap, (f, nb_state))

        return None

    def _pierce_empty_goal_world(
        self,
        yaw: float,
        max_dist_m: float,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
    ) -> Optional[Tuple[float, float]]:
        """
        Ray-march and return the first promising EMPTY-space goal cell (world xy).

        High-level Overview: This 'scouts' ahead in a specific direction to 
        find a safe, open area to fly toward. It's like looking for clear 
        sky or an open corridor before committing to a move.

        Implementation: It performs a ray-march (step-by-step check) along a 
        given heading. It looks for cells that have been marked as 'empty' by 
        previous sensor scans. It rejects cells that contain known walls, 
        dangers, or building collisions at the target flight altitude.

        Args:
            yaw (float): Heading to scout.
            max_dist_m (float): Maximum scouting distance.
            building_index (BuildingIndex): For collision checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Geofence limits.

        Returns:
            Optional[Tuple[float, float]]: Coordinates of a promising empty cell, or None.
        """
        max_dist_m = float(max(0.0, max_dist_m))
        if max_dist_m <= 1e-6:
            return None
        step = float(max(0.5, self.grid.cell_size_m))
        steps = int(max_dist_m / step)
        if steps <= 0:
            return None

        # When overflying, prefer planning goals that are empty at *cruise altitude* (safe to descend).
        z_check = float(getattr(self.s, "z_cruise", self.s.z)) if bool(getattr(self.s, "overfly_active", False)) else float(self.s.z)
        dx, dy = math.cos(float(yaw)), math.sin(float(yaw))
        cur_c = self.grid.world_to_cell(self.s.x, self.s.y)
        dilate = 0
        try:
            dilate = int(getattr(self, "empty_goal_dilate_cells", 0))
        except Exception:
            dilate = 0

        def has_empty_near(c0: Tuple[int, int]) -> bool:
            if self.pher.empty.get(c0) > 1e-6:
                return True
            if dilate <= 0:
                return False
            cx0, cy0 = int(c0[0]), int(c0[1])
            for dx in range(-dilate, dilate + 1):
                for dy in range(-dilate, dilate + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cc = (cx0 + dx, cy0 + dy)
                    if self.pher.empty.get(cc) > 1e-6:
                        return True
            return False

        for i in range(2, steps + 1):
            px = float(self.s.x) + dx * float(i) * step
            py = float(self.s.y) + dy * float(i) * step
            if out_of_bounds(px, py, map_bounds_m):
                break
            cx, cy = self.grid.world_to_cell(px, py)
            c = (int(cx), int(cy))
            if c == cur_c:
                continue
            # Planning helper: allow a small dilation so we can pick usable goals even if beams are sparse.
            if not has_empty_near(c):
                continue
            # Never aim into a wall-marked cell.
            dm = self.pher.danger.meta.get(c)
            if dm is not None and str(dm.kind) == "nav_danger":
                continue
            # Must also be physically free at the altitude we plan to be at.
            wx, wy = self.grid.cell_to_world(c[0], c[1])
            if building_index.is_obstacle_xy(float(wx), float(wy), float(z_check), float(safety_margin_z)):
                continue
            # Also avoid aiming into known danger cells (even if empty is present due to bugs/races).
            if self.pher.danger.get(c) > 1e-6:
                continue
            return (float(wx), float(wy))

        return None

    def _line_blocked_known(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
        inflate_cells: int,
        building_index: Optional[BuildingIndex] = None,
        safety_margin_z: float = 0.0,
        max_steps: int = 400,
    ) -> bool:
        """
        Bresenham-like grid traversal between two cells using *known* occupancy.

        High-level Overview: Performs a 'line-of-sight' check between two grid 
        cells. It returns True if any part of the straight line is blocked 
        by buildings or obstacles.

        Implementation: Iterates through grid cells along the line using a 
        simplified Bresenham logic. For each cell, it calls 
        '_cell_blocked_for_planning'.

        Args:
            a (Tuple[int, int]): Start cell.
            b (Tuple[int, int]): End cell.
            inflate_cells (int): Obstacle inflation radius.
            building_index (Optional[BuildingIndex]): For altitude-aware checks.
            safety_margin_z (float): Vertical safety margin.
            max_steps (int): Safety limit on search distance.

        Returns:
            bool: True if the path is obstructed.
        """
        ax, ay = int(a[0]), int(a[1])
        bx, by = int(b[0]), int(b[1])
        dx = abs(bx - ax)
        dy = abs(by - ay)
        sx = 1 if ax < bx else -1
        sy = 1 if ay < by else -1
        err = dx - dy
        x, y = ax, ay
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                return False
            if self._cell_blocked_for_planning(
                (x, y),
                inflate_cells,
                building_index=building_index,
                safety_margin_z=safety_margin_z,
            ):
                return True
            if x == bx and y == by:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _unstick_move(
        self,
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
    ) -> Optional[Tuple[float, float, float]]:
        """
        When boxed in (e.g. tight corners), try a short escape move to the nearest free spot.

        High-level Overview: An emergency maneuver for when the drone is stuck 
        in a corner or very close to an obstacle. It searches in all directions 
        for the quickest path back to open air.

        Implementation: It probes points in a circular pattern around the 
        drone at increasing radii. It calculates a score for each point based 
        on its distance and the forward clearance (using '_ray_clearance_m'). 
        It returns the move that offers the best escape.

        Args:
            building_index (BuildingIndex): For obstacle checks.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Geofence limits.

        Returns:
            Optional[Tuple[float, float, float]]: (nx, ny, yaw) for the escape move, or None.
        """
        # Probe points around the drone at increasing radii; pick the one with the best clearance.
        radii = [
            self.grid.cell_size_m * 0.5,
            self.grid.cell_size_m * 1.0,
            self.grid.cell_size_m * 1.5,
            self.grid.cell_size_m * 2.0,
            self.grid.cell_size_m * 3.0,
        ]
        angles = 24
        best = None
        best_score = -1e9
        for r in radii:
            for k in range(angles):
                yaw = (2.0 * math.pi) * (k / angles)
                nx = self.s.x + math.cos(yaw) * r
                ny = self.s.y + math.sin(yaw) * r
                if out_of_bounds(float(nx), float(ny), map_bounds_m):
                    continue
                if building_index.is_obstacle_xy(nx, ny, self.s.z, safety_margin_z):
                    continue
                # Prefer headings that open into free space.
                lookahead = max(self.grid.cell_size_m * 4.0, 20.0)
                clearance = self._ray_clearance_m(
                    building_index=building_index,
                    safety_margin_z=safety_margin_z,
                    map_bounds_m=map_bounds_m,
                    yaw=yaw,
                    max_dist_m=lookahead,
                    step_m=self.grid.cell_size_m * 0.5,
                )
                # Prefer larger radii slightly (escape more decisively), but prioritize clearance.
                score = (2.0 * clearance) + (0.3 * r)
                if score > best_score:
                    best_score = score
                    best = (float(nx), float(ny), float(yaw))
            if best is not None:
                # If we found something at a smaller radius, still allow trying larger radii,
                # but stop early if clearance is already very good.
                if best_score >= 2.0 * max(self.grid.cell_size_m * 6.0, 30.0):
                    break
        return best

    def _update_energy(self, dist_m: float, maneuver_factor: float):
        """
        Calculate and subtract energy used during a move.

        High-level Overview: Simulated battery drain.

        Implementation: Multiplies distance by a constant cost-per-meter. 
        Includes a 'maneuver factor' to penalize sharp turns or high-effort 
        flight.

        Args:
            dist_m (float): Distance traveled.
            maneuver_factor (float): Multiplier for flight effort (e.g., sharp turns).
        """
        base_cost = dist_m * self.energy.cost_per_meter_units
        self.s.energy_units = max(0.0, self.s.energy_units - base_cost * maneuver_factor)

    def _set_speed_for_mode(self, return_speed_mps: float):
        """
        Adjust flight speed based on battery and mission status.

        High-level Overview: Governs how fast the drone flies.

        Implementation: 
        1. If battery is low, it forces a slow, energy-efficient speed.
        2. If in 'RETURN' mode, it uses the requested return speed.
        3. Otherwise, it uses the standard mission speed.

        Args:
            return_speed_mps (float): Requested speed for homing.
        """
        # Low-energy always caps speed
        if self.s.energy_units <= self.energy.low_energy_threshold_units:
            self.s.speed_mps = self.energy.low_energy_speed_mps
            return
        if self.s.mode == "RETURN":
            self.s.speed_mps = float(max(0.1, return_speed_mps))
            return
        self.s.speed_mps = self.energy.normal_speed_mps

    def _within_low_energy_range(self, base_xy: Tuple[float, float]) -> bool:
        """
        Check if the drone has enough energy to return home.

        High-level Overview: A safety check. If the battery is low, it 
        determines if the drone is still within its emergency glide/flight 
        range to the base.

        Implementation: Compares distance to base against a pre-calculated 
        'max range' value in the energy model.

        Args:
            base_xy (Tuple[float, float]): Home base coordinates.

        Returns:
            bool: True if safe or still has plenty of energy.
        """
        if self.s.energy_units > self.energy.low_energy_threshold_units:
            return True
        return math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) <= self.energy.low_energy_max_range_m + 1.0

    def reveal_with_mock_lidar(
        self,
        building_index: BuildingIndex,
        safety_margin_z: float,
        sense_radius_m: float,
        beam_count: int,
        t_sim: float,
        danger_cells: Optional[Set[Tuple[int, int]]] = None,
        danger_sources: Optional[Dict[Tuple[int, int], dict]] = None,
        # Dynamic danger paths (danger_id -> arrayOfCells list) from DangerMapManager.
        # Used for fast checks without scanning the full grid.
        dynamic_paths_by_id: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        # Dynamic danger heights (danger_id -> height_m above ground).
        dynamic_heights_by_id: Optional[Dict[str, float]] = None,
        sense_z_override: Optional[float] = None,
        inflate_cells: int = 1,
    ):
        """
        Simulate a 2D LiDAR scan to reveal obstacles and danger sources.

        High-level Overview: This is the drone's primary way of 'seeing' its 
        immediate surroundings. It simulates a laser scanner that rotates in a 
        circle, hitting buildings and detecting hazards.

        Implementation: It casts a fixed number of 'beams' (rays) in a circle. 
        For each beam, it steps forward cell-by-cell until it hits a building 
        (at the drone's altitude) or reaches max range. 
        - If a beam hits a building, that cell is marked 'occupied'.
        - Cells the beam passes through are marked 'free' and 'explored'.
        - Hazards (static/dynamic threats) are detected if they are within 
          range, even if they don't block the laser.
        - Special 'danger pheromones' are deposited at hazard locations.

        Args:
            building_index (BuildingIndex): Index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            sense_radius_m (float): LiDAR sensing range.
            beam_count (int): Number of beams in the scan.
            t_sim (float): Current simulation time.
            danger_cells (Optional[Set[Tuple[int, int]]]): Cells containing danger sources.
            danger_sources (Optional[Dict[Tuple[int, int], dict]]): Metadata for danger sources.
            dynamic_paths_by_id (Optional[Dict[str, List[Tuple[int, int]]]]): Reconstructed dynamic danger paths.
            dynamic_heights_by_id (Optional[Dict[str, float]]): Heights of dynamic dangers.
            sense_z_override (Optional[float]): Override for sensing altitude.
            inflate_cells (int, optional): Obstacle inflation radius.
        """
        # Mock 2D lidar with beams; reveals cells along beams until obstacle (height-aware).
        max_steps = int(sense_radius_m / self.grid.cell_size_m)
        inflate_cells = max(0, int(inflate_cells))

        def _deposit_danger_source_cell(c: Tuple[int, int]):
            """Deposit danger pheromone for a seen danger source cell (static or dynamic)."""
            if c in seen_danger_cells:
                return
            seen_danger_cells.add(c)
            # Discovered a danger source cell: store danger pheromone on it and optionally in a radius around it.
            # This lets a single-cell discovery “paint” an avoidance field larger than lidar range.
            r_cells = 0
            src_kind = ""
            src_id = ""
            src_speed = None
            src_height_m = None
            try:
                if danger_sources is not None:
                    info = danger_sources.get(c, None)
                    # Backward-compat: older versions used int radius as the value.
                    if isinstance(info, dict):
                        r_cells = int(info.get("radius", 0) or 0)
                        src_kind = str(info.get("kind", "") or "")
                        src_id = str(info.get("id", "") or "")
                        src_speed = info.get("speed", None)
                        src_height_m = info.get("height_m", None)
                    else:
                        r_cells = int(info or 0)
            except Exception:
                r_cells = 0
            r_cells = max(0, min(50, int(r_cells)))

            # Height above ground (meters). Default: 50m (car/gunmen intuition).
            height_m = 50.0
            try:
                if src_height_m is not None:
                    height_m = float(src_height_m)
            except Exception:
                height_m = 50.0
            height_m = float(max(0.0, min(999.0, float(height_m))))

            # Dynamic-threat speed classes (relative to drone speed):
            # - <= 0.89x: too slow -> never inspect (just mark what you saw)
            # - <= 1.10x: inspectable -> radius forced to 1 cell
            # - > 1.10x: too fast -> treat as static hazard (avoid where it was)
            slow_frac = 0.89
            trackable_frac = 1.10
            treat_dynamic_as_static = False
            allow_inspect = False

            # seconds per cell (from danger manager)
            sp_s = None
            try:
                if src_speed is not None:
                    sp_s = float(src_speed)
            except Exception:
                sp_s = None
            if sp_s is not None and float(sp_s) <= 1e-6:
                sp_s = None

            speed_ratio = None
            try:
                # Compare against max horizontal speed (not current speed), per spec.
                ref_speed = float(getattr(self.energy, "normal_speed_mps", self.s.speed_mps))
                if sp_s is not None and float(ref_speed) > 1e-6:
                    threat_mps = float(self.grid.cell_size_m) / float(sp_s)
                    speed_ratio = float(threat_mps) / float(ref_speed)
            except Exception:
                speed_ratio = None

            if src_kind == "dynamic" and speed_ratio is not None:
                if float(speed_ratio) <= float(slow_frac):
                    allow_inspect = False
                    # For dynamic threats up to 110% speed: danger radius is only 1 cell around.
                    r_cells = 1
                elif float(speed_ratio) <= float(trackable_frac):
                    allow_inspect = True
                    r_cells = 1
                else:
                    treat_dynamic_as_static = True
                    allow_inspect = False

            # Encode danger kind for downstream logic (e.g., special handling of dynamic threats).
            kind_kernel = "danger_map"
            kind_damage = "danger_map"
            if src_kind == "dynamic":
                if treat_dynamic_as_static:
                    kind_kernel = "danger_static"
                    kind_damage = "danger_static"
                    # Ensure we never enter inspector mode for this id.
                    try:
                        did = str(src_id or "")
                        if did:
                            self.s.dynamic_inspect_skip_ids.add(did)
                            if str(getattr(self.s, "dynamic_inspect_active_id", "") or "") == did:
                                self.s.dynamic_inspect_active_id = ""
                    except Exception:
                        pass
                else:
                    # Split dynamic danger into:
                    # - kernel trace: where the threat center was observed (stronger / used for intercept reasoning)
                    # - damage trace: lower repulsion, represents the radius-of-effect corridor
                    did = str(src_id or "")
                    kind_kernel = (f"danger_dyn_kernel:{did}" if did else "danger_dyn_kernel")
                    kind_damage = (f"danger_dyn_damage:{did}" if did else "danger_dyn_damage")
                    # Remember we know this dynamic danger exists (local knowledge).
                    try:
                        if did:
                            self.s.known_dynamic_danger_ids.add(did)
                    except Exception:
                        pass
            elif src_kind == "static":
                kind_kernel = "danger_static"
                kind_damage = "danger_static"

            # If dynamic danger is present, do not keep static danger pheromone in the same cell.
            # (Also symmetric: static should overwrite dynamic if a static threat is defined there.)
            try:
                old = self.pher.danger.meta.get(c)
                old_kind = str(getattr(old, "kind", "") or "")
                if kind_kernel.startswith("danger_dyn_") and old_kind.startswith("danger_static"):
                    self.pher.danger.set(c, 0.0, CellMeta(t=t_sim, conf=0.0, src=self.s.drone_uid, kind="cleared"))
                if kind_kernel.startswith("danger_static") and old_kind.startswith("danger_dyn_"):
                    self.pher.danger.set(c, 0.0, CellMeta(t=t_sim, conf=0.0, src=self.s.drone_uid, kind="cleared"))
            except Exception:
                pass

            # Center
            self.pher.deposit_danger(
                c,
                amount=0.8,
                t=t_sim,
                conf=0.85,
                src=self.s.drone_uid,
                kind=kind_kernel,
                alt_m=float(height_m),
                speed_s_per_cell=sp_s,
            )
            # Update local kernel cache + path reconstruction for reasoning/curiosity.
            try:
                if src_kind == "dynamic" and (not treat_dynamic_as_static):
                    did = str(src_id or "")
                    if did:
                        self._record_dyn_kernel_obs(did, (int(c[0]), int(c[1])), float(t_sim), sp_s, str(self.s.drone_uid))
                        # Real-time sighting (LiDAR): this is the sweet-spot signal for the inspector.
                        self._dyn_kernel_realtime[str(did)] = {
                            "cell": (int(c[0]), int(c[1])),
                            "t": float(t_sim),
                            "speed_s": sp_s,
                            "radius_cells": int(r_cells),
                        }
                        # Inspector logic disabled: no special inspector role/weights.
                        _ = allow_inspect
            except Exception:
                pass
            # Radius “field”
            #
            # Static threats emulate a "gun" with 3 concentric altitude bands:
            # - base radius (r): altitude = height_m
            # - second radius (2r): altitude = height_m/2
            # - third radius (3r): altitude = height_m/4
            #
            # Dynamic threats keep constant altitude (treated like other drones).
            if r_cells > 0:
                cx0, cy0 = int(c[0]), int(c[1])
                is_static = bool(src_kind == "static")
                base_r = int(max(0, int(r_cells)))
                # Keep runtime bounded: cap effective field radius.
                total_r = int(min(50, (base_r * 3) if is_static else base_r))
                if total_r > 0:
                    r1 = float(max(1, base_r))
                    r2 = float(2 * base_r)
                    r3 = float(3 * base_r)
                    for dx in range(-total_r, total_r + 1):
                        for dy in range(-total_r, total_r + 1):
                            if dx == 0 and dy == 0:
                                continue
                            if (dx * dx + dy * dy) > (total_r * total_r):
                                continue
                            cc = (cx0 + dx, cy0 + dy)
                            if not self.grid.in_bounds_cell(cc[0], cc[1]):
                                continue
                            # Kernel must always override damage trace:
                            # If this cell already contains a dynamic kernel pheromone, do not paint damage over it.
                            try:
                                oldm = self.pher.danger.meta.get(cc)
                                ok = str(getattr(oldm, "kind", "") or "")
                                if ok.startswith("danger_dyn_kernel"):
                                    continue
                            except Exception:
                                pass
                            dist = math.sqrt(float(dx * dx + dy * dy))
                            # Decay with distance (soft field).
                            amt = 0.25 * max(0.10, 1.0 - (dist / max(1.0, float(total_r))))
                            # Altitude falloff only for static threats.
                            alt_here = float(height_m)
                            if is_static and base_r > 0:
                                if dist <= r1 + 1e-6:
                                    alt_here = float(height_m)
                                elif dist <= r2 + 1e-6:
                                    alt_here = float(height_m) * 0.5
                                elif dist <= r3 + 1e-6:
                                    alt_here = float(height_m) * 0.25
                            self.pher.deposit_danger(
                                cc,
                                amount=float(amt),
                                t=t_sim,
                                conf=0.65,
                                src=self.s.drone_uid,
                                kind=kind_damage,
                                alt_m=float(alt_here),
                                speed_s_per_cell=sp_s,
                            )

        def _mark_occ(c: Tuple[int, int]):
            ix, iy = c
            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    cc = (ix + dx, iy + dy)
                    if not self.grid.in_bounds_cell(cc[0], cc[1]):
                        continue
                    self.known_occ[cc] = True
                    self.known_free.pop(cc, None)

        # Debug: record what this drone "saw" in the last scan
        self.last_lidar_beams: List[Tuple[float, float, bool]] = []  # (end_x, end_y, hit)
        self.last_lidar_t: float = float(t_sim)

        # Track whether we learned anything new this scan.
        # - perception_rev: any change (free/occ/danger)
        # - hazard_rev: only changes that can invalidate a committed ACO move (occ/danger)
        occ0 = len(self.known_occ)
        free0 = len(self.known_free)
        danger0 = len(self.pher.danger.v)
        # Avoid spamming repeated deposits for the same danger cell in one scan
        seen_danger_cells: Set[Tuple[int, int]] = set()

        sense_z = float(self.s.z if sense_z_override is None else sense_z_override)
        # Empty helper should be safe to aim for at cruise altitude (safe-to-descend).
        z_empty_check = float(getattr(self.s, "z_cruise", self.s.z))
        # Only store empty near wall danger to keep it sparse.
        empty_wall_r = int(getattr(self.s, "empty_near_wall_radius_cells", 2))

        # LiDAR beams: uniform scan (inspector-focused sensing disabled).
        angles: List[float] = [(2.0 * math.pi) * (i / max(1, int(beam_count))) for i in range(int(beam_count))]

        # -------- fast dynamic-threat check (before beams) --------
        # If a dynamic threat is within sensor radius, mark it as "seen" even if no beam exactly crosses it.
        # This is the intended "lidar checks threats first" shortcut requested by user.
        try:
            if danger_sources is not None and danger_cells is not None:
                sr2 = float(sense_radius_m) * float(sense_radius_m)
                for cc, info in (danger_sources or {}).items():
                    try:
                        if cc not in danger_cells:
                            continue
                        if not isinstance(info, dict):
                            continue
                        if str(info.get("kind", "") or "") != "dynamic":
                            continue
                        wx, wy = self.grid.cell_to_world(int(cc[0]), int(cc[1]))
                        if (float(wx) - float(self.s.x)) ** 2 + (float(wy) - float(self.s.y)) ** 2 > sr2 + 1e-6:
                            continue
                        # Optional visibility: if a building blocks the straight line at sensor altitude, skip.
                        # (Still O(1) for distance; LOS check is O(k) in path length.)
                        los_ok = True
                        try:
                            dx = float(wx) - float(self.s.x)
                            dy = float(wy) - float(self.s.y)
                            dist = math.hypot(dx, dy)
                            if dist > 1e-6:
                                steps = int(max(1, min(80, dist / max(1e-6, float(self.grid.cell_size_m) * 0.5))))
                                for i in range(1, steps + 1):
                                    px = float(self.s.x) + dx * (float(i) / float(steps))
                                    py = float(self.s.y) + dy * (float(i) / float(steps))
                                    if building_index.is_obstacle_xy(px, py, float(sense_z), float(safety_margin_z)):
                                        los_ok = False
                                        break
                        except Exception:
                            los_ok = True
                        if not los_ok:
                            continue
                        _deposit_danger_source_cell((int(cc[0]), int(cc[1])))
                    except Exception:
                        continue
        except Exception:
            pass

        for ang in angles:
            hit = False
            end_x = self.s.x + math.cos(ang) * sense_radius_m
            end_y = self.s.y + math.sin(ang) * sense_radius_m
            for step in range(1, max_steps + 1):
                px = self.s.x + math.cos(ang) * step * self.grid.cell_size_m
                py = self.s.y + math.sin(ang) * step * self.grid.cell_size_m
                cx, cy = self.grid.world_to_cell(px, py)
                if not self.grid.in_bounds_cell(cx, cy):
                    end_x, end_y = px, py
                    break
                c = (cx, cy)
                # Coverage: this cell was "seen" by lidar (either free or obstacle).
                try:
                    # Evidence quality: distance from drone to this cell along the beam.
                    obs_dist_m = float(step) * float(self.grid.cell_size_m)
                    self.pher.deposit_explored(c, t=t_sim, conf=0.5, src=self.s.drone_uid, obs_dist_m=obs_dist_m)
                except Exception:
                    pass
                # Danger-map hazards (static/dynamic): do NOT block the beam, but become "known" via pheromones once seen.
                if danger_cells is not None and c in danger_cells:
                    _deposit_danger_source_cell(c)
                is_occ = building_index.is_obstacle_xy(px, py, sense_z, safety_margin_z)
                if is_occ:
                    _mark_occ(c)
                    hit = True
                    end_x, end_y = px, py
                    # Mark navigation danger (walls/obstacles) with altitude metadata.
                    self.s.encounters += 1
                    # Wall hit -> danger(kind=nav_danger) with blocking altitude.
                    self.pher.deposit_nav_danger(c, amount=0.5, t=t_sim, conf=0.8, src=self.s.drone_uid, alt_m=float(sense_z))
                    # Empty helper: mark free cells around the wall footprint (sparse ring).
                    # This uses true geometry but only locally around a discovered wall.
                    try:
                        if empty_wall_r > 0:
                            cx0, cy0 = int(c[0]), int(c[1])
                            for dx in range(-empty_wall_r, empty_wall_r + 1):
                                for dy in range(-empty_wall_r, empty_wall_r + 1):
                                    if dx == 0 and dy == 0:
                                        continue
                                    cc = (cx0 + dx, cy0 + dy)
                                    if not self.grid.in_bounds_cell(cc[0], cc[1]):
                                        continue
                                    if self.pher.danger.get(cc) > 1e-9:
                                        continue
                                    wx, wy = self.grid.cell_to_world(cc[0], cc[1])
                                    if building_index.is_obstacle_xy(float(wx), float(wy), float(z_empty_check), float(safety_margin_z)):
                                        continue
                                    self.pher.deposit_empty(cc, t=t_sim, conf=0.55, src=self.s.drone_uid)
                    except Exception:
                        pass
                    break  # beam stops at obstacle
                else:
                    self.known_free[c] = True
                    self.known_occ.pop(c, None)
            self.last_lidar_beams.append((float(end_x), float(end_y), bool(hit)))

        # -------- simplified dynamic danger discovery --------
        # Once we know a dynamic danger id (from seeing at least one kernel cell or via comms),
        # we can "see" its trajectory kernels from danger_map within LiDAR radius.
        # This is a simplification: we don't require a beam hit for each kernel cell.
        try:
            if dynamic_paths_by_id and getattr(self.s, "known_dynamic_danger_ids", None):
                sr2 = float(sense_radius_m) * float(sense_radius_m)
                # Cap work per scan (path lists can be long).
                max_kernel_marks = int(getattr(self, "dyn_kernel_reveal_max_per_scan", 600))
                marked = 0
                for did in list(getattr(self.s, "known_dynamic_danger_ids", set()) or set()):
                    path = dynamic_paths_by_id.get(str(did))
                    if not path:
                        continue
                    # Height above ground for this dynamic danger id (if available).
                    hm = None
                    try:
                        if dynamic_heights_by_id is not None:
                            hm = dynamic_heights_by_id.get(str(did))
                    except Exception:
                        hm = None
                    for cc in path:
                        if marked >= max_kernel_marks:
                            break
                        cx, cy = int(cc[0]), int(cc[1])
                        if not self.grid.in_bounds_cell(cx, cy):
                            continue
                        wx, wy = self.grid.cell_to_world(cx, cy)
                        if (float(wx) - float(self.s.x)) ** 2 + (float(wy) - float(self.s.y)) ** 2 > sr2 + 1e-6:
                            continue
                        c0 = (int(cx), int(cy))
                        # Deposit kernel pheromone with the danger id encoded in kind.
                        self.pher.deposit_danger(
                            c0,
                            amount=0.35,
                            t=t_sim,
                            conf=0.65,
                            src=self.s.drone_uid,
                            kind=f"danger_dyn_kernel:{str(did)}",
                            alt_m=(float(hm) if hm is not None else None),
                            speed_s_per_cell=None,
                        )
                        # Update local kernel cache (used for sharing / visualization).
                        try:
                            self._record_dyn_kernel_obs(str(did), (int(cx), int(cy)), float(t_sim), None, str(self.s.drone_uid))
                        except Exception:
                            pass
                        marked += 1
                    if marked >= max_kernel_marks:
                        break
        except Exception:
            pass

        # If we discovered new occupancy/free info or new danger cells, bump perception revision.
        try:
            if len(self.known_occ) != occ0 or len(self.known_free) != free0 or len(self.pher.danger.v) != danger0:
                self.s.perception_rev = int(getattr(self.s, "perception_rev", 0)) + 1
        except Exception:
            pass
        # If we discovered new obstacles or new danger cells, bump hazard revision.
        try:
            if len(self.known_occ) != occ0 or len(self.pher.danger.v) != danger0:
                self.s.hazard_rev = int(getattr(self.s, "hazard_rev", 0)) + 1
        except Exception:
            pass

    def try_detect_targets(
        self,
        targets: List[Target],
        sense_radius_m: float,
        t_sim: float,
        base_xy: Optional[Tuple[float, float]] = None,
        exploration_area_radius_m: float = 0.0,
    ) -> List[Target]:
        """
        Check for targets within sensing range.

        High-level Overview: This is the drone's mission objective sensor. It 
        checks if any of the target objects (which the swarm is searching for) 
        are within range.

        Implementation: It calculates the distance to every target not yet marked 
        as found. If a target is within 'sense_radius_m' and also within the 
        valid mission area, it's marked as 'found' by this drone. This 
        information is then added to the swarm's collective knowledge.

        Args:
            targets (List[Target]): List of all potential targets in the world.
            sense_radius_m (float): Sensing range in meters.
            t_sim (float): Current simulation time.
            base_xy (Optional[Tuple[float, float]]): World coordinates of the home base.
            exploration_area_radius_m (float, optional): Radius of the exploration area around base.

        Returns:
            List[Target]: List of newly detected targets.
        """
        found = []
        for tgt in targets:
            # Targets are visible/detectable only inside exploration area radius (if enabled).
            if base_xy is not None and float(exploration_area_radius_m) > 1e-6:
                if math.hypot(float(tgt.x) - float(base_xy[0]), float(tgt.y) - float(base_xy[1])) > float(exploration_area_radius_m) + 1e-6:
                    continue
            # Allow "LiDAR discovery": if a target is in sensor range, a drone can learn it exists even if it was
            # not pre-shared by base/peers. This prevents beam-discretization edge cases where the drone is
            # clearly in range but a particular scan pattern didn't coincide with the exact target cell.
            kt = self.s.known_targets.get(tgt.target_id)
            if kt is not None and kt.found:
                continue
            # Detection: treat sense_radius as *horizontal* sensing range (2D).
            # This avoids coupling target marker altitude to the current drone cruise altitude.
            if math.hypot(float(self.s.x) - float(tgt.x), float(self.s.y) - float(tgt.y)) <= float(sense_radius_m) + 1e-9:
                # Locally found: update local knowledge, and update base target record (global truth) but do not
                # magically update other drones' knowledge (they learn via comm/base).
                if kt is None:
                    # First-sighting discovery.
                    try:
                        kt = TargetKnowledge(
                            target_id=tgt.target_id,
                            x=float(tgt.x),
                            y=float(tgt.y),
                            z=float(tgt.z),
                            found=True,
                            found_by=self.s.drone_uid,
                            found_t=float(t_sim),
                            updated_t=float(t_sim),
                        )
                        self.s.known_targets[tgt.target_id] = kt
                    except Exception:
                        # If we can't construct TargetKnowledge for any reason, still mark base target as found.
                        kt = None
                if kt is not None:
                    kt.found = True
                    kt.found_by = self.s.drone_uid
                    kt.found_t = t_sim
                    kt.updated_t = float(t_sim)
                    self.s.recent_target_updates.append(kt)
                tgt.found_by = self.s.drone_uid
                tgt.found_t = t_sim
                found.append(tgt)
        return found

    def step(
        self,
        dt: float,
        t_sim: float,
        base_xy: Tuple[float, float],
        mission_phase: str,  # "EXPLORE" | "EXPLOIT"
        building_index: BuildingIndex,
        safety_margin_z: float,
        map_bounds_m: MapBounds,
        aco_temperature: float,
        target_goal: Optional[Tuple[float, float]] = None,
        return_speed_mps: float = 10.0,
        return_use_aco_enabled: bool = True,
        base_no_nav_radius_m: float = 30.0,
        base_push_radius_m: float = 60.0,
        base_push_strength: float = 4.0,
        base_no_deposit_radius_m: float = 20.0,
        explore_min_radius_m: float = 200.0,
        explore_min_radius_strength: float = 10.0,
        recent_cell_penalty: float = 2.0,
        # Extra loop-breaking: increase penalty for repeated revisits within the recent_cells window.
        # 0 disables (backward-compatible).
        explore_revisit_penalty_repeat_mult: float = 0.0,
        # Optional: suppress nav pheromone deposit on recent revisits (reduces "knots" in nav layer).
        # 1.0 disables (backward-compatible). Typical: 0.2..0.6.
        explore_revisit_nav_deposit_scale: float = 1.0,
        wall_clearance_m: float = 15.0,
        wall_clearance_weight: float = 4.0,
        wall_avoid_start_factor: float = 2.0,
        wall_avoid_yaw_weight: float = 2.0,
        wall_corridor_relax: float = 0.25,
        corner_backoff_enabled: bool = True,
        unstick_move_enabled: bool = True,
        return_progress_weight: float = 8.0,
        return_danger_weight: float = 3.5,
        return_corridor_danger_relax: float = 0.25,
        local_plan_radius_m: float = 35.0,
        local_plan_inflate_cells: int = 1,
        local_plan_max_nodes: int = 1400,
        local_plan_unknown_penalty: float = 1.5,
        local_plan_trigger_clearance_frac: float = 0.75,
        explore_personal_bias_weight: float = 0.35,
        explore_score_noise: float = 0.08,
        explore_avoid_empty_weight: float = 0.0,
        explore_avoid_explored_weight: float = 0.0,
        empty_goal_dilate_cells: int = 0,
        # Exploration area (circle around base). 0 disables.
        exploration_area_radius_m: float = 0.0,
        exploration_radius_margin_m: float = 30.0,
        # Explore scoring: how attractive nav pheromone is during EXPLORE.
        explore_nav_weight: float = 1.2,
        # Far-ring "low explored density" shaping (EXPLORE only):
        # Sample probe cells at a ring distance X around the drone (in grid cells), compute explored-density
        # in a square neighborhood around each probe (radius Y, in cells), and bias headings toward the lowest density.
        # Weight 0 disables.
        explore_far_density_weight: float = 0.0,
        explore_far_density_ring_radius_cells: int = 0,
        explore_far_density_kernel_radius_cells: int = 3,
        explore_far_density_angle_step_deg: float = 30.0,
        # If True, when computing probe kernel density, ignore kernel cells that are closer than the ring radius X
        # to the drone. This focuses the score on the frontier/outward side of the ring.
        explore_far_density_exclude_inside_ring: bool = True,
        # Pheromone-based exploration reward: prefer low navigation pheromone (less visited).
        explore_low_nav_weight: float = 0.0,
        # Exploration reward based on explored freshness + observation quality.
        # These are intentionally "weights" so the GUI can expose exactly 3 knobs:
        # - explore_unexplored_reward_weight: overall reward for "less explored" space
        # - explore_explored_age_weight: how quickly old explored evidence becomes less trusted
        # - explore_explored_dist_weight: how strongly far observations count as weaker exploration
        explore_unexplored_reward_weight: float = 0.0,
        # ... and others ...
        explore_explored_age_weight: float = 0.0,
        explore_explored_dist_weight: float = 0.0,
        # Anti-crowding exploration: penalize aligning with peer exploration vectors.
        explore_vector_avoid_weight: float = 0.0,
        explore_vector_spatial_gate_m: float = 200.0,
        explore_vector_ttl_s: float = 120.0,
        # Danger inspection curiosity (EXPLORE only): reward positions that would reveal
        # unexplored cells adjacent to already-known (non-wall) danger cells.
        # This naturally turns off once the boundary band becomes explored.
        danger_inspect_weight: float = 0.0,
        danger_inspect_kernel_cells: int = 3,
        danger_inspect_danger_thr: float = 0.35,
        danger_inspect_max_cell_danger: float = 0.6,
        # Dynamic threat reasoning (optional):
        # - dynamic_threats is a list of dicts from DangerMapManager metadata (path/speed/radius/current cell).
        # - decision is visualized as a beam in RViz (yellow=cross, red=avoid).
        dynamic_threats: Optional[List[dict]] = None,
        dynamic_threat_decision_enabled: bool = True,
        dynamic_threat_cross_margin_s: float = 0.5,
        dynamic_threat_avoid_weight: float = 6.0,
        # If True, dynamic danger pheromone trails are treated as static danger for scoring.
        dynamic_danger_trail_as_static: bool = False,
        sense_radius_m: float = 0.0,
    ):
        """
        Advance the drone's simulation by one time step.
        Includes motion control, pheromone deposition, and path planning.

        High-level Overview: This is the 'main loop' for the drone's brain. 
        In every time step (tick), it assesses its state, updates its local 
        map, makes navigation decisions, and moves. It handles everything 
        from battery management to swarm coordination via pheromones.

        Implementation: 
        1. Energy Check: If low on battery, it switches to 'RETURN' mode.
        2. Static Danger Altitude Rule: If inside a high-altitude danger zone, 
           it climbs first before any horizontal movement.
        3. Local Planning: If blocked or in homing mode, it uses A* to find 
           a path.
        4. ACO Scoring (Exploration): In explore mode, it calculates scores 
           for various headings based on pheromones, wall avoidance, 
           crowding, and 'curiosity' (e.g., inspecting danger boundaries).
        5. Movement: Moves the drone to the new (x,y,yaw).
        6. Stigmergy: Deposits navigation or avoidance pheromones along its path.

        Args:
            dt (float): Time step in seconds.
            t_sim (float): Current simulation time.
            base_xy (Tuple[float, float]): Home base coordinates.
            mission_phase (str): 'EXPLORE' or 'EXPLOIT'.
            building_index (BuildingIndex): Index for obstacle checking.
            safety_margin_z (float): Vertical safety margin.
            map_bounds_m (MapBounds): Map boundaries.
            aco_temperature (float): Temperature for ACO sampling.
            target_goal (Optional[Tuple[float, float]]): Explicit goal point.
            return_speed_mps (float): Speed when returning to base.
            return_use_aco_enabled (bool): Whether to use ACO for return path.
            base_no_nav_radius_m (float): Radius around base where nav pheromones are ignored.
            base_push_radius_m (float): Radius for base repulsion.
            base_push_strength (float): Strength of base repulsion.
            base_no_deposit_radius_m (float): Radius around base where no pheromones are deposited.
            explore_min_radius_m (float): Minimum exploration distance from base.
            explore_min_radius_strength (float): Strength of minimum exploration radius bias.
            recent_cell_penalty (float): Penalty for recently visited cells.
            explore_revisit_penalty_repeat_mult (float): Multiplier for repeat visit penalty.
            explore_revisit_nav_deposit_scale (float): Scale for nav pheromone deposit on revisits.
            wall_clearance_m (float): Desired clearance from walls.
            wall_clearance_weight (float): Weight for wall clearance bias.
            wall_avoid_start_factor (float): Multiplier for clearance threshold to start avoidance.
            wall_avoid_yaw_weight (float): Weight for yaw-based wall avoidance.
            wall_corridor_relax (float): Relaxation factor for wall avoidance in tight corridors.
            corner_backoff_enabled (bool): Enable backoff when stuck in corners.
            unstick_move_enabled (bool): Enable short escape moves when boxed in.
            return_progress_weight (float): Weight for progress towards base in return mode.
            return_danger_weight (float): Weight for danger avoidance in return mode.
            return_corridor_danger_relax (float): Relaxation for danger in tight corridors.
            local_plan_radius_m (float): Radius for local A* planning.
            local_plan_inflate_cells (int): Inflation radius for A* planning.
            local_plan_max_nodes (int): Max nodes for A* search.
            local_plan_unknown_penalty (float): A* penalty for unvisited cells.
            local_plan_trigger_clearance_frac (float): Clearance fraction that triggers A* replanning.
            explore_personal_bias_weight (float): Weight for drone's personal yaw bias.
            explore_score_noise (float): Stochastic noise added to exploration scores.
            explore_avoid_empty_weight (float): Weight for avoiding empty-marked cells.
            explore_avoid_explored_weight (float): Weight for avoiding already explored cells.
            empty_goal_dilate_cells (int): Dilate empty cells for goal selection.
            exploration_area_radius_m (float): Maximum allowed exploration distance from base.
            exploration_radius_margin_m (float): Safety margin for exploration area.
            explore_nav_weight (float): Attraction to existing navigation pheromones.
            explore_far_density_weight (float): Weight for biasing towards low-density areas.
            explore_far_density_ring_radius_cells (int): Radius for density probe ring.
            explore_far_density_kernel_radius_cells (int): Radius for density probe kernel.
            explore_far_density_angle_step_deg (float): Angular step for density probes.
            explore_far_density_exclude_inside_ring (bool): Exclude cells inside probe ring.
            explore_low_nav_weight (float): Preference for low nav pheromone cells.
            explore_unexplored_reward_weight (float): Reward for moving towards unexplored areas.
            explore_explored_age_weight (float): How quickly old evidence becomes less valuable.
            explore_explored_dist_weight (float): Penalty for distant observations.
            explore_vector_avoid_weight (float): Penalty for following peer exploration vectors.
            explore_vector_spatial_gate_m (float): Range for peer vector avoidance.
            explore_vector_ttl_s (float): Time-to-live for shared exploration vectors.
            danger_inspect_weight (float): Reward for inspecting near-danger unexplored cells.
            danger_inspect_kernel_cells (int): Kernel size for danger inspection.
            danger_inspect_danger_thr (float): Threshold for danger intensity to trigger inspection.
            danger_inspect_max_cell_danger (float): Max danger intensity for inspection reward.
            dynamic_threats (Optional[List[dict]]): List of known dynamic threats.
            dynamic_threat_decision_enabled (bool): Enable logic for crossing/avoiding dynamic threats.
            dynamic_threat_cross_margin_s (float): Safety margin for crossing dynamic threat paths.
            dynamic_threat_avoid_weight (float): Weight for avoiding dynamic threat paths.
            dynamic_danger_trail_as_static (bool): Treat dynamic trails as static altitude hazards.
            sense_radius_m (float): Sensing radius.
        """
        # -------- dynamic danger inspection (simple rule) --------
        # Inspector is set when we FIRST deposit a kernel cell for some danger id.
        # Conflict resolution happens during comms: if two drones inspect the same id, the newer (higher t) stops.
        # Make planning helpers available to _pierce_empty_goal_world via per-drone attributes.
        # (We keep these as attributes so we don't have to thread params through many helpers.)
        try:
            self.empty_goal_dilate_cells = int(empty_goal_dilate_cells)
        except Exception:
            self.empty_goal_dilate_cells = 0
        if self.s.mode in ("IDLE", "RECHARGE"):
            return

        def _log_preclimb_cell(c: Tuple[int, int], req_alt: float, why: str):
            """Throttle log spam: print once per cell (or at most 1 Hz)."""
            try:
                last_c = getattr(self.s, "preclimb_static_last_log_cell", None)
                last_t = float(getattr(self.s, "preclimb_static_last_log_t", -1e9))
                if last_c == tuple(c) and (float(t_sim) - float(last_t)) < 1.0:
                    return
                self.s.preclimb_static_last_log_cell = (int(c[0]), int(c[1]))
                self.s.preclimb_static_last_log_t = float(t_sim)
            except Exception:
                pass
            try:
                print(
                    f"[static_alt] {str(getattr(self.s,'drone_uid','?'))} "
                    f"t={float(t_sim):.2f} cell=({int(c[0])},{int(c[1])}) "
                    f"z={float(self.s.z):.2f} req={float(req_alt):.2f} why={str(why)}"
                )
            except Exception:
                # Never crash step due to logging
                pass

        # Hard rule (per user): treat static danger altitude like building altitude.
        # If the drone is currently inside ANY static-danger pheromone cell whose required altitude > z,
        # it must climb first and may NOT move in XY.
        try:
            treat_dyn_trail_as_static_alt = bool(dynamic_danger_trail_as_static)
            cur_c = self.grid.world_to_cell(float(self.s.x), float(self.s.y))
            cur_cc = (int(cur_c[0]), int(cur_c[1]))
            mk0 = self.pher.danger.meta.get(cur_cc)
            if mk0 is not None:
                k0 = str(getattr(mk0, "kind", "") or "")
                is_static_like = bool(k0.startswith("danger_static")) or (
                    bool(treat_dyn_trail_as_static_alt) and bool(k0.startswith("danger_dyn_"))
                )
            else:
                is_static_like = False
            if mk0 is not None and bool(is_static_like):
                # Only enforce if the pheromone value is actually present (avoid stale meta edge cases).
                if float(self.pher.danger.get(cur_cc)) > 1e-6:
                    alt0 = getattr(mk0, "alt_m", None)
                    if alt0 is not None and float(self.s.z) < float(alt0) - 1e-6:
                        req = float(alt0)
                        self.s.overfly_active = True
                        self.s.z_target = float(max(req, float(getattr(self.s, "z_target", self.s.z))))
                        self.s.preclimb_static_hold = True
                        self.s.preclimb_static_req_alt = float(req)
                        self.last_move_source = "STATIC_ALT_HOLD"
                        _log_preclimb_cell(cur_cc, req_alt=req, why="inside_static_cell")
                        return
        except Exception:
            pass

        # If we triggered a "pre-climb before entering static danger" latch, do not move horizontally
        # until we reach (approximately) the target altitude.
        try:
            if bool(getattr(self.s, "preclimb_static_hold", False)):
                zt = float(getattr(self.s, "z_target", self.s.z))
                try:
                    zt = max(float(zt), float(getattr(self.s, "preclimb_static_req_alt", zt)))
                except Exception:
                    pass
                reached = float(self.s.z) >= float(zt) - 0.35
                if reached:
                    self.s.preclimb_static_hold = False
                else:
                    self.last_move_source = "PRECLIMB_STATIC_HOLD"
                    try:
                        cur_c = self.grid.world_to_cell(float(self.s.x), float(self.s.y))
                        cur_cc = (int(cur_c[0]), int(cur_c[1]))
                        _log_preclimb_cell(cur_cc, req_alt=float(zt), why="hold_wait")
                    except Exception:
                        pass
                    return
        except Exception:
            pass

        def _block_xy_until_static_alt_ok(nx: float, ny: float, yaw: float) -> bool:
            """Return True if we should freeze XY and only climb.

            This is intentionally analogous to building collision gating:
            - buildings: horizontal move is rejected if obstacle exists at (nx, ny) for current z
            - static danger: horizontal move is rejected if the motion segment would enter any danger_static cell
              whose required altitude is above current z (we enforce the max requirement along the segment).
            """
            try:
                treat_dyn_trail_as_static_alt = bool(dynamic_danger_trail_as_static)
                # Smoothness improvement (per user):
                # Start climbing a couple of cells BEFORE the highest-altitude static-danger pheromone
                # we are about to traverse, so we don't "hit a wall" and then stop to climb.
                try:
                    preclimb_cells = 2
                    look_cells = max(6, int(preclimb_cells) + 6)
                    stepm = float(max(0.5, float(self.grid.cell_size_m)))
                    best_alt = None
                    best_i = None
                    for i in range(1, int(look_cells) + 1):
                        px = float(self.s.x) + math.cos(float(yaw)) * float(i) * stepm
                        py = float(self.s.y) + math.sin(float(yaw)) * float(i) * stepm
                        cc = self.grid.world_to_cell(px, py)
                        cci = (int(cc[0]), int(cc[1]))
                        if float(self.pher.danger.get(cci)) <= 1e-6:
                            continue
                        mk = self.pher.danger.meta.get(cci)
                        if mk is None:
                            continue
                        kk = str(getattr(mk, "kind", "") or "")
                        is_static_like = bool(kk.startswith("danger_static")) or (
                            bool(treat_dyn_trail_as_static_alt) and bool(kk.startswith("danger_dyn_"))
                        )
                        if not bool(is_static_like):
                            continue
                        alt = getattr(mk, "alt_m", None)
                        if alt is None:
                            continue
                        altf = float(alt)
                        if best_alt is None or altf > float(best_alt) + 1e-9:
                            best_alt = float(altf)
                            best_i = int(i)
                    # If the highest-alt cell is close enough (<= 2 cells ahead), start climbing now (but do not hold).
                    if best_alt is not None and best_i is not None and int(best_i) <= int(preclimb_cells):
                        if float(self.s.z) < float(best_alt) - 1e-6:
                            self.s.overfly_active = True
                            self.s.z_target = float(max(float(best_alt), float(getattr(self.s, "z_target", self.s.z))))
                            try:
                                self.s.preclimb_static_req_alt = float(best_alt)
                            except Exception:
                                pass
                except Exception:
                    pass

                x0, y0 = float(self.s.x), float(self.s.y)
                x1, y1 = float(nx), float(ny)
                dist = math.hypot(x1 - x0, y1 - y0)
                if dist <= 1e-9:
                    return False

                step_m = max(0.25, float(self.grid.cell_size_m) * 0.5)
                steps = int(max(1, math.ceil(dist / max(1e-6, step_m))))
                alt_req = None
                last_cell = None
                # Include start cell (i=0) so moving within a cell still respects its altitude.
                for i in range(0, steps + 1):
                    a = float(i) / float(steps)
                    px = x0 + (x1 - x0) * a
                    py = y0 + (y1 - y0) * a
                    cc = self.grid.world_to_cell(px, py)
                    cci = (int(cc[0]), int(cc[1]))
                    if last_cell == cci:
                        continue
                    last_cell = cci
                    # Only enforce where danger pheromone is actually present.
                    if float(self.pher.danger.get(cci)) <= 1e-6:
                        continue
                    mk = self.pher.danger.meta.get(cci)
                    if mk is None:
                        continue
                    kk = str(getattr(mk, "kind", "") or "")
                    is_static_like = bool(kk.startswith("danger_static")) or (
                        bool(treat_dyn_trail_as_static_alt) and bool(kk.startswith("danger_dyn_"))
                    )
                    if not bool(is_static_like):
                        continue
                    alt = getattr(mk, "alt_m", None)
                    if alt is None:
                        continue
                    altf = float(alt)
                    alt_req = altf if alt_req is None else max(float(alt_req), altf)

                if alt_req is None:
                    return False
                if float(self.s.z) >= float(alt_req) - 1e-6:
                    return False

                # Need to climb before entering the max-required altitude along this move.
                self.s.overfly_active = True
                self.s.z_target = float(max(float(alt_req), float(getattr(self.s, "z_target", self.s.z))))
                self.s.overfly_start_x = float(getattr(self.s, "overfly_start_x", self.s.x))
                self.s.overfly_start_y = float(getattr(self.s, "overfly_start_y", self.s.y))
                self.s.overfly_start_t = float(getattr(self.s, "overfly_start_t", t_sim))
                self.s.preclimb_static_hold = True
                try:
                    self.s.preclimb_static_req_alt = float(alt_req)
                except Exception:
                    pass
                # If we need to climb, drop any committed plan/commitment so we replan cleanly after reaching altitude.
                try:
                    self._active_plan_world = []
                    self._active_plan_idx = 0
                    self._active_plan_until_t = -1e9
                except Exception:
                    pass
                try:
                    self._aco_commit_cell = None
                    self._aco_commit_until_t = -1e9
                except Exception:
                    pass
                self.last_move_source = "PRECLIMB_STATIC"
                # Face intended direction while climbing (nice visually; also keeps behavior consistent).
                try:
                    self.s.yaw = float(yaw)
                    self._last_yaw = float(yaw)
                except Exception:
                    pass
                try:
                    _log_preclimb_cell(self.grid.world_to_cell(float(self.s.x), float(self.s.y)), req_alt=float(alt_req), why="segment_gate")
                except Exception:
                    pass
                return True
            except Exception:
                return False

        # EXPLOIT: we want deterministic goal-seeking using the already-built pheromone map.
        # - Do not penalize "unknown" (we are not exploring).
        # - Do not inflate lidar-hit occupancy (we rely on pheromone nav_danger + true-geometry collision checks).
        if str(mission_phase).upper() == "EXPLOIT":
            try:
                local_plan_unknown_penalty = 0.0
            except Exception:
                pass
            try:
                local_plan_inflate_cells = 0
            except Exception:
                pass
            # Make dynamic-trail planning behavior available to A* cost evaluation.
            # - False: ignore dynamic danger pheromone trails (dynamic-aware optimization)
            # - True: treat them as static and bias A* to route around them
            try:
                self._dyn_trail_static_for_planning = bool(dynamic_danger_trail_as_static)
            except Exception:
                self._dyn_trail_static_for_planning = False
            # Parameters for the temporary "static overlay" penalty derived from dynamic trail intensity.
            try:
                pp = getattr(self, "_exploit_peer_params", None) or {}
                self._dyn_trail_overlay_strength = float(pp.get("dyn_overlay_strength", getattr(self, "_dyn_trail_overlay_strength", 3.0)))
            except Exception:
                self._dyn_trail_overlay_strength = 3.0
            try:
                pp = getattr(self, "_exploit_peer_params", None) or {}
                self._dyn_trail_overlay_gamma = float(pp.get("dyn_overlay_gamma", getattr(self, "_dyn_trail_overlay_gamma", 1.8)))
            except Exception:
                self._dyn_trail_overlay_gamma = 1.8

        # Dynamic danger hard no-fly (current location only, not pheromone trail):
        # - Only enabled for danger ids this drone knows (learned via LiDAR or comms).
        # - Used by A* (blocked cells) and by local candidate generation (skip aims).
        try:
            dyn_footprints: List[Tuple[int, int, int]] = []
            known_ids = set(getattr(self.s, "known_dynamic_danger_ids", set()) or set())
            if known_ids and dynamic_threats:
                cell_m = float(max(1e-6, float(self.grid.cell_size_m)))
                sr = float(max(0.0, float(sense_radius_m)))
                for dd in dynamic_threats or []:
                    did = str(dd.get("id", "") or "")
                    if not did or did not in known_ids:
                        continue
                    cell0 = dd.get("cell", None)
                    if cell0 is None or len(cell0) != 2:
                        continue
                    cx0, cy0 = int(cell0[0]), int(cell0[1])
                    rad = int(dd.get("radius", 0) or 0)
                    rad = int(max(0, min(50, rad)))
                    # Only avoid what we can currently "see" with LiDAR: within sensor radius of the footprint.
                    try:
                        wx0, wy0 = self.grid.cell_to_world(int(cx0), int(cy0))
                        eff = float(sr) + float(rad) * float(cell_m)
                        if eff > 1e-6:
                            if (float(wx0) - float(self.s.x)) ** 2 + (float(wy0) - float(self.s.y)) ** 2 > (eff * eff) + 1e-6:
                                continue
                    except Exception:
                        pass
                    # Store footprint as (center, radius^2) so checks remain O(#danger_ids).
                    dyn_footprints.append((int(cx0), int(cy0), int(rad * rad)))
            self._dyn_nofly_footprints = dyn_footprints
        except Exception:
            self._dyn_nofly_footprints = []

        # Avoidance (crab) mode: temporarily force a sideways heading to bypass obstacles.
        # This is purely a steering override; it should NOT create attractive navigation trails.
        if bool(getattr(self.s, "avoid_active", False)):
            try:
                dt_in = float(t_sim) - float(getattr(self.s, "avoid_start_t", t_sim))
                max_time = float(getattr(self.s, "avoid_max_time_s", 8.0))
                if dt_in >= max_time:
                    self.s.avoid_active = False
                else:
                    entry_yaw = float(getattr(self.s, "avoid_entry_yaw", float(self.s.yaw)))
                    side = int(getattr(self.s, "avoid_side", 1))
                    crab_yaw = (entry_yaw + (math.pi / 2.0) * float(side)) % (2.0 * math.pi)
                    self.s.avoid_target_yaw = float(crab_yaw)
                    # Exit condition: if we've shifted sideways enough and forward is reasonably clear again.
                    dx = float(self.s.x) - float(getattr(self.s, "avoid_start_x", float(self.s.x)))
                    dy = float(self.s.y) - float(getattr(self.s, "avoid_start_y", float(self.s.y)))
                    # Lateral offset wrt entry yaw
                    lat = (-math.sin(entry_yaw) * dx) + (math.cos(entry_yaw) * dy)
                    need = float(getattr(self.s, "avoid_need_lateral_m", float(self.grid.cell_size_m) * 3.0))
                    if abs(lat) >= need:
                        fwd_clear = self._ray_clearance_m(
                            building_index=building_index,
                            safety_margin_z=safety_margin_z,
                            map_bounds_m=map_bounds_m,
                            yaw=entry_yaw,
                            max_dist_m=max(10.0, float(wall_clearance_m) * 2.0),
                            step_m=self.grid.cell_size_m * 0.5,
                        )
                        if fwd_clear >= max(6.0, float(wall_clearance_m) * 0.9):
                            self.s.avoid_active = False
            except Exception:
                pass

        # mode transitions based on energy
        if self.s.mode == "EXPLORE" and self.s.energy_units <= self.energy.return_threshold_units:
            self.s.mode = "RETURN"
        if self.s.mode == "RETURN":
            # Snap-to-base if we can reach it within this integration step.
            # This avoids "orbiting" around base when dt is large (e.g., time acceleration).
            dist_to_base = math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1])
            if dist_to_base <= (self.s.speed_mps * dt + 1e-6):
                self.s.x, self.s.y = float(base_xy[0]), float(base_xy[1])
                self.s.mode = "RECHARGE"
                self.s.energy_units = min(self.energy.full_units, self.energy.recharge_to_units)
                return
            if math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) <= self.grid.cell_size_m * 2.0:
                # at base
                self.s.mode = "RECHARGE"
                self.s.energy_units = min(self.energy.full_units, self.energy.recharge_to_units)
                return

        self._set_speed_for_mode(return_speed_mps=return_speed_mps)

        # -----------------------------
        # RETURN behavior (legacy): pure local A*
        # -----------------------------
        # Historically RETURN used greedy A* (deterministic) and did not use ACO scoring at all.
        # We keep it as an option for ablations; default is now ACO-style RETURN.
        if self.s.mode == "RETURN" and (not bool(return_use_aco_enabled)):
            # Drop any ACO commitment and any previous local plan (RETURN should be deterministic).
            self._aco_commit_cell = None
            try:
                self.last_aco_candidates = []
                self.last_aco_choice_world = None
                self.last_aco_choice_t = float(t_sim)
            except Exception:
                pass
            try:
                self._active_plan_world = []
                self._active_plan_idx = 0
                self._active_plan_until_t = -1e9
            except Exception:
                pass

            def _line_clear_world_return(gx: float, gy: float) -> bool:
                dx = gx - float(self.s.x)
                dy = gy - float(self.s.y)
                dist = math.hypot(dx, dy)
                if dist <= 1e-6:
                    return True
                yaw = math.atan2(dy, dx)
                clear = self._ray_clearance_m(
                    building_index=building_index,
                    safety_margin_z=safety_margin_z,
                    map_bounds_m=map_bounds_m,
                    yaw=yaw,
                    max_dist_m=dist + 1e-3,
                    step_m=self.grid.cell_size_m * 0.5,
                )
                return clear >= (dist - self.grid.cell_size_m * 0.25)

            # Fastest case: straight-line sprint to base when unobstructed.
            bx, by = float(base_xy[0]), float(base_xy[1])
            if _line_clear_world_return(bx, by):
                step_m = float(self.s.speed_mps) * float(dt)
                dx = bx - float(self.s.x)
                dy = by - float(self.s.y)
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    yaw = math.atan2(dy, dx)
                    scale = min(1.0, float(step_m) / float(dist))
                    nx = float(self.s.x) + float(dx) * float(scale)
                    ny = float(self.s.y) + float(dy) * float(scale)
                    if (not out_of_bounds(float(nx), float(ny), map_bounds_m)) and (
                        not building_index.is_obstacle_xy(float(nx), float(ny), float(self.s.z), float(safety_margin_z))
                    ):
                        dxy = math.hypot(float(nx) - float(self.s.x), float(ny) - float(self.s.y))
                        self.s.x, self.s.y, self.s.yaw = float(nx), float(ny), float(yaw)
                        self.last_move_source = "A*"
                        dyaw = abs((float(yaw) - float(self._last_yaw) + math.pi) % (2.0 * math.pi) - math.pi)
                        maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)
                        self._update_energy(dxy, maneuver_factor=maneuver_factor)
                        self.s.total_dist_m += float(dxy)
                        self._last_yaw = float(yaw)
                        c = self.grid.world_to_cell(self.s.x, self.s.y)
                        self.s.recent_cells.append(c)
                        return

            # Otherwise: greedy local A* step toward base (A* includes danger/unknown/revisit costs;
            # the "no ACO" part is enforced by handling RETURN here and returning early).
            nc = self._local_a_star_next_cell(
                goal_xy=(bx, by),
                mission_phase=mission_phase,
                building_index=building_index,
                safety_margin_z=safety_margin_z,
                map_bounds_m=map_bounds_m,
                plan_radius_m=float(local_plan_radius_m),
                inflate_cells=max(0, int(local_plan_inflate_cells)),
                max_nodes=int(local_plan_max_nodes),
                unknown_penalty=float(local_plan_unknown_penalty),
                # When returning to base, prioritize making progress over "novelty".
                # The recent-cell revisit penalty can otherwise repel the drone from the only viable
                # corridor home (often the corridor it just used), causing oscillation/orbiting.
                recent_penalty=0.0,
            )
            if nc is not None:
                tx, ty = self.grid.cell_to_world(int(nc[0]), int(nc[1]))
                yaw = math.atan2(float(ty) - float(self.s.y), float(tx) - float(self.s.x))
                step_m = float(self.s.speed_mps) * float(dt)
                dist = math.hypot(float(tx) - float(self.s.x), float(ty) - float(self.s.y))
                if dist > 1e-6:
                    scale = min(1.0, float(step_m) / float(dist))
                    nx = float(self.s.x) + (float(tx) - float(self.s.x)) * float(scale)
                    ny = float(self.s.y) + (float(ty) - float(self.s.y)) * float(scale)
                    if (not out_of_bounds(float(nx), float(ny), map_bounds_m)) and (
                        not building_index.is_obstacle_xy(float(nx), float(ny), float(self.s.z), float(safety_margin_z))
                    ):
                        dxy = math.hypot(float(nx) - float(self.s.x), float(ny) - float(self.s.y))
                        self.s.x, self.s.y, self.s.yaw = float(nx), float(ny), float(yaw)
                        self.last_move_source = "A*"
                        dyaw = abs((float(yaw) - float(self._last_yaw) + math.pi) % (2.0 * math.pi) - math.pi)
                        maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)
                        self._update_energy(dxy, maneuver_factor=maneuver_factor)
                        self.s.total_dist_m += float(dxy)
                        self._last_yaw = float(yaw)
                        c = self.grid.world_to_cell(self.s.x, self.s.y)
                        self.s.recent_cells.append(c)
                        return

            # Fallback: small escape move if local A* couldn't find any step (rare).
            if bool(unstick_move_enabled):
                esc = self._unstick_move(building_index, safety_margin_z, map_bounds_m)
                if esc is not None:
                    nx, ny, yaw = esc
                    dxy = math.hypot(float(nx) - float(self.s.x), float(ny) - float(self.s.y))
                    self.s.x, self.s.y, self.s.yaw = float(nx), float(ny), float(yaw)
                    self.last_move_source = "A*"
                    self._update_energy(dxy, maneuver_factor=1.15)
                    self.s.total_dist_m += float(dxy)
                    self._last_yaw = float(yaw)
                    c = self.grid.world_to_cell(self.s.x, self.s.y)
                    self.s.recent_cells.append(c)
                    return

            # Still stuck: rotate slightly and retry next step.
            self.s.yaw = (float(self.s.yaw) + 0.3) % (2.0 * math.pi)
            self.last_move_source = "A*"
            return

        # ACO commitment: if we already chose a next cell, keep moving toward it until reached,
        # unless we learned something new about hazards (obstacles/danger) or it becomes invalid.
        try:
            commit_enabled = bool(getattr(self, "aco_commit_enabled", True))
        except Exception:
            commit_enabled = True
        if commit_enabled and self._aco_commit_cell is not None and self.s.mode == "EXPLORE" and mission_phase == "EXPLORE":
            # If avoidance mode is active, drop commitment and let avoidance steer.
            if bool(getattr(self.s, "avoid_active", False)):
                self._aco_commit_cell = None
            try:
                # Break commitment if new perception arrived.
                cur_rev = int(getattr(self.s, "hazard_rev", getattr(self.s, "perception_rev", 0)))
                if cur_rev != int(getattr(self, "_aco_commit_rev", 0)):
                    self._aco_commit_cell = None
                # Break on timeout (sim time).
                if self._aco_commit_cell is not None and float(t_sim) > float(getattr(self, "_aco_commit_until_t", -1e9)):
                    self._aco_commit_cell = None
            except Exception:
                self._aco_commit_cell = None

            if self._aco_commit_cell is not None:
                try:
                    tx, ty = self.grid.cell_to_world(int(self._aco_commit_cell[0]), int(self._aco_commit_cell[1]))
                    yaw = math.atan2(float(ty) - float(self.s.y), float(tx) - float(self.s.x))
                    step_m = float(self.s.speed_mps) * float(dt)
                    dist = math.hypot(float(tx) - float(self.s.x), float(ty) - float(self.s.y))
                    if dist > 1e-6:
                        scale = min(1.0, float(step_m) / float(dist))
                        nx = float(self.s.x) + (float(tx) - float(self.s.x)) * float(scale)
                        ny = float(self.s.y) + (float(ty) - float(self.s.y)) * float(scale)
                        # Still must be feasible.
                        if (not out_of_bounds(float(nx), float(ny), map_bounds_m)) and (not building_index.is_obstacle_xy(nx, ny, self.s.z, safety_margin_z)):
                            dxy = math.hypot(nx - float(self.s.x), ny - float(self.s.y))
                            self.s.x, self.s.y, self.s.yaw = float(nx), float(ny), float(yaw)
                            self.last_move_source = "ACO_COMMIT"
                            self.last_aco_choice_world = (float(tx), float(ty), float(yaw))
                            self.last_aco_choice_t = float(t_sim)
                            self._update_energy(dxy, maneuver_factor=1.01)
                            self.s.total_dist_m += dxy
                            self._last_yaw = float(yaw)
                            c = self.grid.world_to_cell(self.s.x, self.s.y)
                            self.s.recent_cells.append(c)
                            # Clear commitment once we enter the target cell.
                            if self.grid.world_to_cell(self.s.x, self.s.y) == (int(self._aco_commit_cell[0]), int(self._aco_commit_cell[1])):
                                self._aco_commit_cell = None
                            return
                        else:
                            # invalidated -> drop and replan
                            self._aco_commit_cell = None
                except Exception:
                    self._aco_commit_cell = None

        # Candidate headings (ACO-style direction decision)
        # NOTE: no full A*; cheap local decision.
        headings = 24
        candidates: List[Tuple[float, Tuple[float, float, float]]] = []  # score, (nx, ny, yaw)
        # Per-candidate dynamic threat decision (keyed by aim cell).
        dyn_decision_by_cell: Dict[Tuple[int, int], Tuple[str, Tuple[int, int]]] = {}

        # If we are actively inspecting a dynamic danger, disable "unexplored space" exploration shaping.
        inspector_active = False
        # Inspector target kernel (prefer realtime ground-truth while within LiDAR range; pheromones are fallback).
        inspector_did0 = ""
        inspector_kernel_rt_cell: Optional[Tuple[int, int]] = None
        inspector_kernel_rt_radius_cells: int = 0
        inspector_kernel_rt_speed_s: Optional[float] = None
        inspector_kernel_rt_visible: bool = False
        try:
            did0 = str(getattr(self.s, "dynamic_inspect_active_id", "") or "").strip()
            inspector_did0 = str(did0)
            if did0 and (did0 not in (getattr(self.s, "dynamic_inspect_skip_ids", set()) or set())):
                st0 = self._dyn_kernel_path.get(did0)
                inspector_active = (st0 is None) or (not bool(st0.get("complete", False)))
        except Exception:
            inspector_active = False
            inspector_did0 = ""

        # If we know the real (ground-truth) position of the hunted dynamic danger by id and it's within LiDAR range,
        # treat it as "realtime kernel" and keep following it. Pheromones are used only as fallback.
        try:
            if inspector_active and inspector_did0 and dynamic_threats:
                # Build O(1) lookup.
                dt_map = {str(d.get("id", "") or ""): d for d in (dynamic_threats or []) if str(d.get("id", "") or "")}
                dt0 = dt_map.get(inspector_did0)
                if dt0 is not None:
                    kc = dt0.get("cell", None)
                    if kc is not None and len(kc) == 2:
                        wx, wy = self.grid.cell_to_world(int(kc[0]), int(kc[1]))
                        dist = math.hypot(float(wx) - float(self.s.x), float(wy) - float(self.s.y))
                        if dist <= float(max(0.0, sense_radius_m)) + 1e-6:
                            inspector_kernel_rt_visible = True
                            inspector_kernel_rt_cell = (int(kc[0]), int(kc[1]))
                            try:
                                inspector_kernel_rt_radius_cells = int(dt0.get("radius", 0) or 0)
                            except Exception:
                                inspector_kernel_rt_radius_cells = 0
                            try:
                                inspector_kernel_rt_speed_s = float(dt0.get("speed", 0.0))
                            except Exception:
                                inspector_kernel_rt_speed_s = None
                            # Update realtime cache + path recon (acts like "perfect LiDAR" when within range).
                            try:
                                self._dyn_kernel_realtime[str(inspector_did0)] = {
                                    "cell": (int(inspector_kernel_rt_cell[0]), int(inspector_kernel_rt_cell[1])),
                                    "t": float(t_sim),
                                    "speed_s": inspector_kernel_rt_speed_s,
                                    "radius_cells": int(max(0, inspector_kernel_rt_radius_cells)),
                                }
                                self._record_dyn_kernel_obs(
                                    str(inspector_did0),
                                    (int(inspector_kernel_rt_cell[0]), int(inspector_kernel_rt_cell[1])),
                                    float(t_sim),
                                    inspector_kernel_rt_speed_s,
                                    str(self.s.drone_uid),
                                )
                            except Exception:
                                pass
        except Exception:
            pass

        # Inspector stop conditions:
        # - Only inspect while the threat is within LiDAR range (realtime visible in this step)
        # - Only inspect while we have enough battery budget
        if inspector_active and inspector_did0:
            try:
                min_e = float(getattr(self, "dyn_inspector_min_energy_units", getattr(self.energy, "return_threshold_units", 50.0)))
            except Exception:
                min_e = 50.0
            if (not bool(inspector_kernel_rt_visible)) or (float(self.s.energy_units) <= float(min_e)):
                try:
                    # Move on: stop being an inspector for this threat.
                    self.s.dynamic_inspect_active_id = ""
                    self.s.dynamic_inspect_skip_ids.add(str(inspector_did0))
                except Exception:
                    pass
                inspector_active = False
                inspector_did0 = ""

        def _danger_frontier_bonus(center: Tuple[int, int]) -> float:
            """Count unexplored cells near `center` that touch known non-wall danger.

            Uses only THIS drone's pheromone knowledge:
            - explored layer: where lidar has looked
            - danger layer: where hazards were observed
            """
            k = int(max(0, danger_inspect_kernel_cells))
            if k <= 0:
                return 0.0
            thr = float(danger_inspect_danger_thr)
            bonus = 0.0
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    ux = int(center[0]) + int(dx)
                    uy = int(center[1]) + int(dy)
                    if not self.grid.in_bounds_cell(ux, uy):
                        continue
                    u = (ux, uy)
                    # Already explored -> no curiosity value (shape already known here)
                    if float(self.pher.explored.get(u)) > 1e-6:
                        continue
                    # Frontier test: does u touch a known (non-wall) danger cell?
                    touches = False
                    for nx in (-1, 0, 1):
                        for ny in (-1, 0, 1):
                            if nx == 0 and ny == 0:
                                continue
                            vx = ux + int(nx)
                            vy = uy + int(ny)
                            if not self.grid.in_bounds_cell(vx, vy):
                                continue
                            v = (vx, vy)
                            dv = float(self.pher.danger.get(v))
                            if dv <= thr:
                                continue
                            mk = self.pher.danger.meta.get(v)
                            # Curiosity is ONLY for static/dynamic threats (not walls, not altitude nets,
                            # not generic danger deposited by other heuristics).
                            knd = str(getattr(mk, "kind", "")) if mk is not None else ""
                            if not (knd.startswith("danger_static") or knd.startswith("danger_dyn_") or knd.startswith("danger_dyn")):
                                continue
                            touches = True
                            break
                        if touches:
                            break
                    if touches:
                        bonus += 1.0
            return float(bonus)

        def _dyn_threat_penalty_for_cell(aim_cell: Tuple[int, int]) -> Tuple[float, str, Optional[Tuple[int, int]]]:
            """
            Compute a conservative dynamic-threat intercept penalty for aiming at `aim_cell`.

            Returns: (penalty, mode, threat_cell)
              - mode: "cross" if we think we can safely cross before interception, else "avoid"
              - threat_cell: a representative cell on the threat trajectory we are reasoning about (for viz beam)
            """
            # Simplification: dynamic danger pheromone trails are ignored.
            # Avoidance is handled by hard no-fly checks against the *current* dynamic danger location.
            return 0.0, "", None
            if not dynamic_threat_decision_enabled:
                return 0.0, "", None
            # Only meaningful if the aim point is inside a dynamic danger damage field (local pheromone knowledge).
            best_pen = 0.0
            best_mode = ""
            best_cell = None

            sr = float(max(0.0, sense_radius_m))
            cell_m = float(max(1e-6, self.grid.cell_size_m))
            margin_s = float(max(0.0, dynamic_threat_cross_margin_s))
            w_avoid = float(max(0.0, dynamic_threat_avoid_weight))

            # Parse dynamic danger id from this cell (we decide BEFORE entering a damage cell).
            mk = self.pher.danger.meta.get(aim_cell)
            k = str(getattr(mk, "kind", "") or "")
            if not (k.startswith("danger_dyn_damage") or k.startswith("danger_dyn_kernel")):
                return 0.0, "", None
            did = k.split(":", 1)[1] if ":" in k else ""
            if not did:
                # Without id we can't tie damage<->kernel, so only mild penalty.
                return float(0.4 * w_avoid), "cross", None

            # Use the latest known kernel cell from pheromone sharing (local + comm).
            info = self._dyn_kernel_latest.get(str(did))
            if not info:
                # If we only have damage trace but no kernel info, be cautious but don't hard-block.
                return float(1.2 * w_avoid), "avoid", None

            kcell = tuple(info.get("cell", aim_cell))
            t_obs = float(info.get("t", 0.0))
            src_obs = str(info.get("src", "") or "")
            speed_s = info.get("speed_s", None)
            if speed_s is None:
                # fallback: try meta at aim_cell
                speed_s = getattr(mk, "speed_s_per_cell", None)
            if speed_s is None:
                return float(1.2 * w_avoid), "avoid", kcell
            speed_s = float(max(0.01, float(speed_s)))
            threat_speed_mps = float(cell_m) / float(speed_s)
            threat_speed_mps = float(max(0.05, threat_speed_mps))

            # "Believe local sensing only": if we observed the kernel ourselves in THIS tick, trust it.
            seen_now = (src_obs == str(self.s.drone_uid)) and (abs(float(t_sim) - float(t_obs)) <= 1e-3)

            # Time for threat (kernel) to reach this aim cell, assuming it has been moving toward us since t_obs.
            kx, ky = self.grid.cell_to_world(int(kcell[0]), int(kcell[1]))
            ax, ay = self.grid.cell_to_world(int(aim_cell[0]), int(aim_cell[1]))
            dist_ka = math.hypot(float(ax) - float(kx), float(ay) - float(ky))
            # earliest arrival if it headed toward aim since observation time
            threat_time_from_obs = max(0.0, (dist_ka / threat_speed_mps) - max(0.0, float(t_sim) - float(t_obs)))

            threat_time_s = float(threat_time_from_obs)
            beam_cell = (int(kcell[0]), int(kcell[1]))

            # If we don't see the threat now, assume worst-case: it could be at the lidar boundary already
            # and appear next moment heading toward the aim cell.
            if (not seen_now) and sr > 1e-6:
                dist_da = math.hypot(float(ax) - float(self.s.x), float(ay) - float(self.s.y))
                boundary_dist_to_aim = max(0.0, float(dist_da) - float(sr))
                boundary_time = boundary_dist_to_aim / threat_speed_mps
                threat_time_s = float(min(threat_time_s, boundary_time))
                # beam points to "might be here": boundary cell along ray drone->aim
                try:
                    if dist_da > 1e-6:
                        ux = (float(ax) - float(self.s.x)) / float(dist_da)
                        uy = (float(ay) - float(self.s.y)) / float(dist_da)
                        bx = float(self.s.x) + float(ux) * float(sr)
                        by = float(self.s.y) + float(uy) * float(sr)
                        bc = self.grid.world_to_cell(float(bx), float(by))
                        beam_cell = (int(bc[0]), int(bc[1]))
                except Exception:
                    pass

            drone_speed = float(max(0.1, float(getattr(self.s, "speed_mps", 10.0))))
            drone_time_s = math.hypot(float(ax) - float(self.s.x), float(ay) - float(self.s.y)) / drone_speed

            if drone_time_s + margin_s < threat_time_s:
                mode = "cross"
                pen = 0.35 * w_avoid
            else:
                mode = "avoid"
                late = max(0.0, (drone_time_s + margin_s) - threat_time_s)
                pen = w_avoid * (4.0 + 1.8 * (late / max(0.2, margin_s + 0.2)))

            best_pen = float(min(float(pen), 80.0))
            best_mode = str(mode)
            best_cell = beam_cell

            # Cap to keep score stable.
            best_pen = float(min(best_pen, 80.0))
            return best_pen, best_mode, best_cell

        def _explored_effective(c: Tuple[int, int]) -> float:
            """Explored evidence in [0..1], discounted by age and observation distance."""
            v0 = float(self.pher.explored.get(c))
            if v0 <= 1e-6:
                return 0.0
            mk = self.pher.explored.meta.get(c)
            if mk is None:
                return float(max(0.0, min(1.0, v0)))
            # Age discount (older => less explored)
            try:
                age_s = max(0.0, float(t_sim) - float(getattr(mk, "t", 0.0)))
            except Exception:
                age_s = 0.0
            age_ref_s = 120.0
            wa = float(max(0.0, float(explore_explored_age_weight)))
            age_mult = math.exp(-wa * (age_s / max(1e-6, age_ref_s))) if wa > 1e-9 else 1.0

            # Distance discount (farther => less explored)
            try:
                d_obs = getattr(mk, "obs_dist_m", None)
            except Exception:
                d_obs = None
            try:
                d_obs = float(d_obs) if d_obs is not None else None
            except Exception:
                d_obs = None
            # If unknown, treat as "far"
            sr = float(max(1e-6, float(sense_radius_m) if float(sense_radius_m) > 1e-6 else 50.0))
            dn = float(d_obs / sr) if d_obs is not None else 1.0
            wd = float(max(0.0, float(explore_explored_dist_weight)))
            dist_mult = math.exp(-wd * (dn * dn)) if wd > 1e-9 else 1.0

            return float(max(0.0, min(1.0, v0 * age_mult * dist_mult)))

        # Precompute "far-ring low explored density" probes once per step (used as a directional bonus).
        far_density_probes: List[Tuple[float, float]] = []  # (probe_yaw, bonus=(1-mean_explored))
        if (not inspector_active) and self.s.mode == "EXPLORE" and mission_phase == "EXPLORE":
            try:
                w_far = float(explore_far_density_weight)
                x_ring = int(explore_far_density_ring_radius_cells)
                y_kern = int(explore_far_density_kernel_radius_cells)
                step_deg = float(explore_far_density_angle_step_deg)
                excl_inside = bool(explore_far_density_exclude_inside_ring)
                if w_far > 1e-9 and x_ring > 0 and y_kern >= 0 and step_deg > 1e-6:
                    cur_cell = self.grid.world_to_cell(self.s.x, self.s.y)
                    x2_ring = int(x_ring) * int(x_ring)
                    use_explore_area = float(exploration_area_radius_m) > 1e-6
                    base_x = float(base_xy[0])
                    base_y = float(base_xy[1])
                    step_rad = math.radians(step_deg)
                    n = int(max(1, round((2.0 * math.pi) / max(1e-9, step_rad))))
                    # Clamp to something sane; 360/30=12 by default.
                    n = int(clamp(float(n), 4.0, 72.0))
                    for i in range(int(n)):
                        yawp = (2.0 * math.pi) * (float(i) / float(n))
                        tx = int(cur_cell[0]) + int(round(math.cos(float(yawp)) * float(x_ring)))
                        ty = int(cur_cell[1]) + int(round(math.sin(float(yawp)) * float(x_ring)))
                        if not self.grid.in_bounds_cell(tx, ty):
                            continue
                        if use_explore_area:
                            wxp, wyp = self.grid.cell_to_world(int(tx), int(ty))
                            if math.hypot(float(wxp) - base_x, float(wyp) - base_y) > float(exploration_area_radius_m) + 1e-6:
                                continue
                        tot = 0.0
                        cnt = 0
                        for dx in range(-int(y_kern), int(y_kern) + 1):
                            for dy in range(-int(y_kern), int(y_kern) + 1):
                                x2 = int(tx) + int(dx)
                                y2 = int(ty) + int(dy)
                                if not self.grid.in_bounds_cell(x2, y2):
                                    continue
                                if excl_inside:
                                    ddx = int(x2) - int(cur_cell[0])
                                    ddy = int(y2) - int(cur_cell[1])
                                    if (ddx * ddx + ddy * ddy) < int(x2_ring):
                                        continue
                                if use_explore_area:
                                    # Important: cells outside the exploration area count as "explored"
                                    # so the boundary doesn't look artificially low-density.
                                    wx2, wy2 = self.grid.cell_to_world(int(x2), int(y2))
                                    if math.hypot(float(wx2) - base_x, float(wy2) - base_y) > float(exploration_area_radius_m) + 1e-6:
                                        tot += 1.0
                                        cnt += 1
                                        continue
                                vv = float(_explored_effective((x2, y2)))
                                tot += max(0.0, min(1.0, float(vv)))
                                cnt += 1
                        if cnt > 0:
                            mean_explored = tot / float(cnt)
                            far_density_probes.append((float(yawp), float(1.0 - max(0.0, min(1.0, mean_explored)))))
            except Exception:
                far_density_probes = []

        # Encourage exploration away from base until range constraints matter
        dist_to_base = math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1])
        base_dir = math.atan2(self.s.y - base_xy[1], self.s.x - base_xy[0])
        goal_yaw = None
        if target_goal is not None:
            goal_yaw = math.atan2(target_goal[1] - self.s.y, target_goal[0] - self.s.x)

            # If we have a concrete goal (base in RETURN, or a target), and the straight line is clear,
            # go straight immediately (no 90-degree "grid-like" path).
        direct_goal = None
        if self.s.mode == "RETURN":
            direct_goal = base_xy
        elif target_goal is not None:
            direct_goal = target_goal

        def _line_clear_world(gx: float, gy: float) -> bool:
            dx = gx - self.s.x
            dy = gy - self.s.y
            dist = math.hypot(dx, dy)
            if dist <= 1e-6:
                return True
            yaw = math.atan2(dy, dx)
            clear = self._ray_clearance_m(
                building_index=building_index,
                safety_margin_z=safety_margin_z,
                map_bounds_m=map_bounds_m,
                yaw=yaw,
                max_dist_m=dist + 1e-3,
                step_m=self.grid.cell_size_m * 0.5,
            )
            return clear >= (dist - self.grid.cell_size_m * 0.25)

        if direct_goal is not None and _line_clear_world(float(direct_goal[0]), float(direct_goal[1])):
            step_m = self.s.speed_mps * dt
            gx, gy = float(direct_goal[0]), float(direct_goal[1])
            dx = gx - self.s.x
            dy = gy - self.s.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                yaw = math.atan2(dy, dx)
                scale = min(1.0, step_m / dist)
                nx = self.s.x + dx * scale
                ny = self.s.y + dy * scale
                if not building_index.is_obstacle_xy(nx, ny, self.s.z, safety_margin_z):
                    distm = math.hypot(nx - self.s.x, ny - self.s.y)
                    self.s.x, self.s.y, self.s.yaw = float(nx), float(ny), float(yaw)
                    self.last_move_source = "ACO_DIRECT"
                    self.last_aco_choice_world = (float(nx), float(ny), float(yaw))
                    self.last_aco_choice_t = float(t_sim)
                    self._update_energy(distm, maneuver_factor=1.02)
                    self.s.total_dist_m += distm
                    self._last_yaw = float(yaw)
                    c = self.grid.world_to_cell(self.s.x, self.s.y)
                    self.s.recent_cells.append(c)
                    # Positive navigation pheromone with altitude metadata (only if we truly progressed).
                    # Do not leave attractive nav trail while in avoidance mode.
                    if (self.s.mode != "RETURN") and (not bool(getattr(self.s, "avoid_active", False))) and (
                        math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) > base_no_deposit_radius_m
                    ):
                        self.pher.deposit_nav(c, amount=0.2, t=t_sim, conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))
                    return

        # Wall edge-seeking: if we see a wall in front within ~2x clearance (or local plan radius),
        # immediately bias the motion towards a tangential direction (seek the edge/corner).
        preferred_avoid_yaw = None
        wc = max(0.0, float(wall_clearance_m))
        if wc > 1e-3:
            ahead = max(wc * float(wall_avoid_start_factor), 0.8 * float(local_plan_radius_m), 10.0)
            fwd_yaw = goal_yaw if goal_yaw is not None else float(self.s.yaw)
            fwd_clear = self._ray_clearance_m(
                building_index=building_index,
                safety_margin_z=safety_margin_z,
                map_bounds_m=map_bounds_m,
                yaw=fwd_yaw,
                max_dist_m=float(ahead),
                step_m=self.grid.cell_size_m * 0.5,
            )
            if fwd_clear < float(ahead) * 0.95:
                # sample turn directions; pick one with best forward clearance
                deltas = [-(math.pi / 2), -(math.pi / 3), -(math.pi / 4), (math.pi / 4), (math.pi / 3), (math.pi / 2)]
                best = None
                for dd in deltas:
                    yy = (fwd_yaw + dd) % (2.0 * math.pi)
                    cc = self._ray_clearance_m(
                        building_index=building_index,
                        safety_margin_z=safety_margin_z,
                        map_bounds_m=map_bounds_m,
                        yaw=yy,
                        max_dist_m=float(ahead),
                        step_m=self.grid.cell_size_m * 0.5,
                    )
                    align = math.cos(yy - (goal_yaw if goal_yaw is not None else fwd_yaw))
                    score = cc + 0.4 * float(ahead) * align
                    if best is None or score > best[0]:
                        best = (score, yy)
                if best is not None:
                    preferred_avoid_yaw = float(best[1])

        # If in avoidance mode, override the preferred yaw with crab heading.
        if bool(getattr(self.s, "avoid_active", False)):
            try:
                preferred_avoid_yaw = float(getattr(self.s, "avoid_target_yaw", float(self.s.yaw)))
            except Exception:
                preferred_avoid_yaw = float(self.s.yaw)

        # Step-size backoff helps escape tight building corners where a full step collides
        # but a shorter move would be feasible. Can be disabled for ablation / behavior comparison.
        step_scales = (1.0, 0.5, 0.25) if bool(corner_backoff_enabled) else (1.0,)
        for step_scale in step_scales:
            candidates.clear()
            step_m_base = self.s.speed_mps * dt * step_scale

            for k in range(headings):
                yaw = (2.0 * math.pi) * (k / headings)
                nx = self.s.x + math.cos(yaw) * step_m_base
                ny = self.s.y + math.sin(yaw) * step_m_base

                # keep within map bounds (square or rectangle)
                if out_of_bounds(float(nx), float(ny), map_bounds_m):
                    continue

                # hard obstacle check at next position (height-aware)
                if building_index.is_obstacle_xy(nx, ny, self.s.z, safety_margin_z):
                    continue

                # Lookahead clearance (prevents "hugging" walls head-on and getting stuck in corners).
                # This is cheap ray-marching in the direction of travel.
                lookahead = max(self.grid.cell_size_m * 4.0, 20.0)
                clearance = self._ray_clearance_m(
                    building_index=building_index,
                    safety_margin_z=safety_margin_z,
                    map_bounds_m=map_bounds_m,
                    yaw=yaw,
                    max_dist_m=lookahead,
                    step_m=self.grid.cell_size_m * 0.5,
                )

                # Keep-away-from-walls shaping (soft constraint): penalize being too close laterally.
                # Exception: narrow corridors where both sides are close (then relax penalty).
                wall_pen = 0.0
                corridor = False
                corridor_relax = 1.0
                wc = max(0.0, float(wall_clearance_m))
                if wc > 1e-6:
                    left_c = self._ray_clearance_m(
                        building_index=building_index,
                        safety_margin_z=safety_margin_z,
                        map_bounds_m=map_bounds_m,
                        yaw=(yaw + math.pi / 2.0),
                        max_dist_m=wc,
                        step_m=self.grid.cell_size_m * 0.5,
                    )
                    right_c = self._ray_clearance_m(
                        building_index=building_index,
                        safety_margin_z=safety_margin_z,
                        map_bounds_m=map_bounds_m,
                        yaw=(yaw - math.pi / 2.0),
                        max_dist_m=wc,
                        step_m=self.grid.cell_size_m * 0.5,
                    )
                    side_min = min(left_c, right_c)
                    corridor = (left_c < wc * 0.95) and (right_c < wc * 0.95)
                    corridor_relax = float(wall_corridor_relax) if corridor else 1.0
                    if side_min < wc:
                        frac = (wc - side_min) / max(1e-6, wc)
                        wall_pen += corridor_relax * float(wall_clearance_weight) * (frac * frac)
                    # Also penalize heading straight into a wall too early.
                    if clearance < wc:
                        frac2 = (wc - clearance) / max(1e-6, wc)
                        wall_pen += 2.0 * corridor_relax * float(wall_clearance_weight) * (frac2 * frac2)

                # local pheromone signals
                # IMPORTANT: score the *intended next cell*, not necessarily the cell we physically reach in this substep.
                # This avoids "waypoint points to current cell" behavior when dt is small.
                aim_dist = max(float(step_m_base), float(self.grid.cell_size_m))
                ax = self.s.x + math.cos(yaw) * aim_dist
                ay = self.s.y + math.sin(yaw) * aim_dist
                cc = self.grid.world_to_cell(ax, ay)
                cur_cc = self.grid.world_to_cell(self.s.x, self.s.y)
                if cc == cur_cc:
                    # Strongly discourage aiming at the current cell (nonsense for a waypoint).
                    continue
                # Hard no-fly: current dynamic danger footprint (do not even consider).
                if self._dyn_cell_blocked(tuple(cc)):
                    continue

                nav_tau = self.pher.nav.get(cc)
                danger_tau = self.pher.danger.get(cc)
                # Dynamic danger pheromone trails:
                # - default: informational only; ignore them for scoring (avoid only current footprint)
                # - exploit option: treat them as static danger (discourage paths through the trail)
                try:
                    mk0 = self.pher.danger.meta.get(cc)
                    k0 = str(getattr(mk0, "kind", "") or "")
                    if k0.startswith("danger_dyn_") and (not bool(dynamic_danger_trail_as_static)):
                        danger_tau = 0.0
                    if k0.startswith("danger_dyn_") and bool(dynamic_danger_trail_as_static):
                        # Increase trail cost in the "treat as static" comparison mode using the same
                        # overlay rule as A* (stronger for high/red values).
                        strength = float(getattr(self, "_dyn_trail_overlay_strength", 3.0))
                        gamma = float(getattr(self, "_dyn_trail_overlay_gamma", 1.8))
                        strength = max(0.0, strength)
                        gamma = max(0.5, gamma)
                        danger_tau = float(danger_tau) + float(strength) * (float(max(0.0, danger_tau)) ** float(gamma))
                except Exception:
                    pass
                # Static danger altitude semantics:
                # If aiming into a static-danger region while below its required altitude, add a penalty
                # proportional to how far below the altitude requirement we are.
                try:
                    mk0 = self.pher.danger.meta.get(cc)
                    if mk0 is not None:
                        k0 = str(getattr(mk0, "kind", "") or "")
                        alt0 = getattr(mk0, "alt_m", None)
                        if alt0 is not None:
                            is_static = bool(k0.startswith("danger_static"))
                            is_dyn_as_static = bool(k0.startswith("danger_dyn_")) and bool(dynamic_danger_trail_as_static)
                            if is_static or is_dyn_as_static:
                                z_eff = float(self.s.z)
                                if bool(getattr(self.s, "overfly_active", False)) or bool(getattr(self.s, "hop_active", False)):
                                    try:
                                        z_eff = max(z_eff, float(getattr(self.s, "z_target", z_eff)))
                                    except Exception:
                                        pass
                                if float(z_eff) < float(alt0) - 1e-6:
                                    dz = max(0.0, float(alt0) - float(z_eff))
                                    frac = float(dz) / max(1.0, float(alt0))
                                    w = float(getattr(self, "static_danger_altitude_violation_weight", 0.0))
                                    if w > 1e-9 and frac > 1e-9:
                                        danger_tau = float(danger_tau) + float(w) * float(frac)
                except Exception:
                    pass
                unknown_bonus = 0.0 if inspector_active else (1.0 if (cc not in self.known_free and cc not in self.known_occ) else 0.0)
                explored_tau = float(_explored_effective(cc))
                recent_pen = 0.0
                if self.s.mode == "EXPLORE" and cc in self.s.recent_cells:
                    # Base penalty for any revisit, plus optional extra penalty if we keep hitting the same cells.
                    try:
                        rp = float(recent_cell_penalty)
                        mult = float(explore_revisit_penalty_repeat_mult)
                        if mult > 1e-9:
                            cnt = 0
                            for rc in self.s.recent_cells:
                                if rc == cc:
                                    cnt += 1
                            recent_pen = rp * (1.0 + mult * float(max(0, int(cnt) - 1)))
                        else:
                            recent_pen = rp
                    except Exception:
                        recent_pen = float(recent_cell_penalty)

                # heuristic: exploration wants unknown, exploitation wants high nav / low danger
                # mild bias away from base for exploration, toward base for RETURN
                if self.s.mode == "RETURN":
                    # IMPORTANT: RETURN must not be attracted by nav pheromone "hot spots".
                    # It should primarily move toward base while avoiding danger/obstacles, even
                    # if that means temporarily increasing distance to base.
                    goal_yaw = math.atan2(base_xy[1] - self.s.y, base_xy[0] - self.s.x)
                    base_align = math.cos(yaw - goal_yaw)
                    base_bias = 4.0 * base_align
                else:
                    # explore: prefer moving outward early, but avoid violating low-energy range
                    yaw_align = math.cos(yaw - base_dir)
                    base_bias = 0.0 if inspector_active else (0.5 * yaw_align)

                # Anti-loitering near base:
                # - prevent a "nav pheromone sink" around base (common after recharge)
                # - strongly prefer moves that increase distance to base, but only inside a radius
                # Inspectors should not care about navigation pheromones (no attraction/repulsion from nav).
                nav_effect = 0.0 if inspector_active else nav_tau
                outward_bias = 0.0
                # Inspectors should not be pushed outward by base anti-loitering heuristics.
                if self.s.mode == "EXPLORE" and (not inspector_active):
                    if dist_to_base < base_no_nav_radius_m:
                        nav_effect = 0.0
                    if dist_to_base < base_push_radius_m:
                        new_dist = math.hypot(nx - base_xy[0], ny - base_xy[1])
                        outward_bias = base_push_strength * (new_dist - dist_to_base) / max(1.0, self.grid.cell_size_m)
                    # If drones tend to "orbit" around the base, force exploration to actually
                    # expand outward by rewarding distance increase until a minimum radius.
                    if dist_to_base < explore_min_radius_m:
                        new_dist = math.hypot(nx - base_xy[0], ny - base_xy[1])
                        frac = (explore_min_radius_m - dist_to_base) / max(1.0, explore_min_radius_m)
                        min_rad_bias = explore_min_radius_strength * (0.5 + frac) * (new_dist - dist_to_base) / max(
                            1.0, self.grid.cell_size_m
                        )
                        outward_bias += min_rad_bias
                    # If we have a hard exploration area boundary, stop pushing outward near the edge.
                    if float(exploration_area_radius_m) > 1e-6:
                        edge_m = max(0.0, float(exploration_radius_margin_m))
                        if dist_to_base >= float(exploration_area_radius_m) - edge_m:
                            outward_bias = 0.0

                # target seeking: if we have a known-unfound target, bias toward it.
                target_bias = 0.0
                if goal_yaw is not None and self.s.mode == "EXPLORE":
                    target_bias = 0.0 if inspector_active else (2.5 * math.cos(yaw - goal_yaw))

                # low-energy constraint: avoid increasing distance beyond max range
                if self.s.energy_units <= self.energy.low_energy_threshold_units:
                    # penalize moves that go further from base
                    new_dist = math.hypot(nx - base_xy[0], ny - base_xy[1])
                    if new_dist > self.energy.low_energy_max_range_m:
                        continue
                    base_bias += -1.0 * max(0.0, new_dist - dist_to_base)

                # Exploration area constraint (circle around base):
                # - When enabled, drones should not move further outward beyond the radius.
                # - If a drone is already outside (e.g. settings changed mid-run), allow moves that decrease distance.
                if float(exploration_area_radius_m) > 1e-6 and self.s.mode == "EXPLORE" and (not inspector_active):
                    r = float(exploration_area_radius_m)
                    new_dist = math.hypot(nx - base_xy[0], ny - base_xy[1])
                    if new_dist > r + 1e-6 and new_dist >= dist_to_base - 1e-6:
                        continue

                # combine score
                if self.s.mode == "RETURN":
                    # RETURN: ignore nav/unknown to avoid being "stuck" in a high-nav pocket.
                    new_dist = math.hypot(nx - base_xy[0], ny - base_xy[1])
                    progress = float(dist_to_base - new_dist)
                    # In narrow corridors, danger pheromone should be less discouraging.
                    dang_relax = float(return_corridor_danger_relax) if corridor else 1.0
                    score = (
                        (1.0 * base_bias)
                        + (float(return_progress_weight) * (progress / max(0.5, float(self.grid.cell_size_m))))
                        - (float(return_danger_weight) * dang_relax * danger_tau)
                        # RETURN: avoid "novelty"/revisit shaping; just get home safely.
                        - 0.0
                    )
                elif mission_phase == "EXPLOIT":
                    score = (3.0 * nav_effect) - (6.0 * danger_tau) + (0.2 * base_bias) + (0.5 * target_bias)
                else:
                    score = (
                        (float(explore_nav_weight) * nav_effect)
                        - (3.0 * danger_tau)
                        + (2.0 * unknown_bonus)
                        + (0.5 * base_bias)
                        + target_bias
                        + outward_bias
                        - recent_pen
                    )
                    if not inspector_active:
                        # Pheromone-based exploration reward: prefer low nav pheromone (less visited).
                        if float(explore_low_nav_weight) > 1e-9:
                            score += float(explore_low_nav_weight) * (1.0 / (1.0 + float(max(0.0, nav_tau))))
                        # Far-ring low-density shaping: bias headings toward directions whose probe cell
                        # at distance X has low explored density in a (2Y+1)x(2Y+1) neighborhood.
                        if far_density_probes and float(explore_far_density_weight) > 1e-9:
                            try:
                                best = 0.0
                                yy = float(yaw)
                                for yawp, bonus in far_density_probes:
                                    align = math.cos(yy - float(yawp))
                                    if align > 0.0:
                                        best = max(best, float(bonus) * float(align))
                                score += float(explore_far_density_weight) * float(best)
                            except Exception:
                                pass
                        # Optional: bias away from already explored space.
                        # This gently pushes exploration toward less-seen areas without asserting they are free.
                        w = float(explore_avoid_explored_weight)
                        if w <= 1e-9:
                            # backward-compat: explore_avoid_empty_weight previously did this job
                            w = float(explore_avoid_empty_weight)
                        if w > 1e-9 and float(explored_tau) > 1e-6:
                            score -= float(w) * float(explored_tau)
                        # Exploration reward: prefer cells that are less explored when taking into account:
                        # - age (old explored evidence decays)
                        # - observation distance (far sightings are weaker)
                        w_u = float(max(0.0, float(explore_unexplored_reward_weight)))
                        if w_u > 1e-9:
                            score += float(w_u) * float(max(0.0, 1.0 - float(explored_tau)))
                    # Danger-boundary inspection curiosity:
                    # - attraction is only toward *unexplored* cells adjacent to already-known (non-wall) danger
                    # - we only apply it when the candidate cell itself isn't already too dangerous
                    try:
                        w_ins = float(danger_inspect_weight)
                        if (not inspector_active) and w_ins > 1e-9 and float(danger_tau) <= float(danger_inspect_max_cell_danger):
                            score += float(w_ins) * _danger_frontier_bonus(cc)
                    except Exception:
                        pass
                    # Anti-crowding: penalize headings aligned with peer exploration vectors.
                    if float(explore_vector_avoid_weight) > 1e-9:
                        try:
                            ttl = max(0.0, float(explore_vector_ttl_s))
                            gate = max(0.0, float(explore_vector_spatial_gate_m))
                            for uid, vec in (self.s.known_explore_vectors or {}).items():
                                if not vec or str(uid) == str(self.s.drone_uid):
                                    continue
                                age = float(t_sim) - float(vec.t)
                                if ttl > 1e-6 and age > ttl:
                                    continue
                                if gate > 1e-6:
                                    if math.hypot(float(self.s.x) - float(vec.start_x), float(self.s.y) - float(vec.start_y)) > gate:
                                        continue
                                align = math.cos(float(yaw) - float(vec.yaw))
                                if align > 0.0:
                                    score -= float(explore_vector_avoid_weight) * float(align)
                        except Exception:
                            pass

                # Clearance shaping:
                # - reward open space so drones slide along walls instead of face-planting into them
                # - strongly penalize directions with very short clearance (near corners).
                # clearance is in meters, normalize by cell size for scale stability.
                cell = max(0.5, float(self.grid.cell_size_m))
                score += 0.06 * (clearance / cell)
                if clearance < 2.0 * cell:
                    score -= 1.5 * ((2.0 * cell - clearance) / cell)

                # Apply wall keep-away penalty and optional tangent bias.
                score -= float(wall_pen)
                if preferred_avoid_yaw is not None and float(wall_avoid_yaw_weight) > 0.0:
                    score += float(wall_avoid_yaw_weight) * math.cos(yaw - float(preferred_avoid_yaw))

                # Dynamic threat intercept reasoning (penalize unsafe crossings).
                try:
                    pen, mode, tcell = _dyn_threat_penalty_for_cell(cc)
                    if pen > 1e-9:
                        score -= float(pen)
                        if tcell is not None and mode:
                            dyn_decision_by_cell[tuple(cc)] = (str(mode), (int(tcell[0]), int(tcell[1])))
                except Exception:
                    pass

                # Dynamic danger kernel-path curiosity (exclusive inspector):
                # If we are the active inspector for some dynamic danger, bias toward staying close to its kernel.
                # IMPORTANT: when the drone is going to recharge (RETURN) or is otherwise not exploring,
                # the inspector reward should be 0. After recharging (mode becomes EXPLORE again), the reward resumes.
                try:
                    did = str(getattr(self.s, "dynamic_inspect_active_id", "") or "")
                    if (
                        did
                        and (did not in (getattr(self.s, "dynamic_inspect_skip_ids", set()) or set()))
                        and self.s.mode == "EXPLORE"
                        and str(mission_phase).upper() == "EXPLORE"
                        and float(getattr(self, "dyn_danger_inspect_weight", 0.0)) > 1e-9
                    ):
                        st = self._dyn_kernel_path.get(did)
                        is_complete = bool(st.get("complete", False)) if st is not None else False
                        if not is_complete:
                            # Prefer realtime (ground-truth if within LiDAR range, or LiDAR-seen kernel) while available.
                            # Fall back to historical pheromone/shared kernel only when realtime isn't available.
                            kc = None
                            rad0 = 0
                            if bool(inspector_kernel_rt_visible) and str(inspector_did0) == str(did) and inspector_kernel_rt_cell is not None:
                                kc = tuple(inspector_kernel_rt_cell)
                                rad0 = int(max(0, int(inspector_kernel_rt_radius_cells)))
                            if kc is None:
                                info = self._dyn_kernel_latest.get(did)
                                if info is not None:
                                    kc = tuple(info.get("cell", cc))
                            if kc is not None:
                                kx, ky = self.grid.cell_to_world(int(kc[0]), int(kc[1]))
                                # Prefer candidate moves that keep us within lidar proximity to the kernel.
                                dist_m = math.hypot(float(kx) - float(nx), float(ky) - float(ny))
                                # normalize by sense radius (if set), else by 50m.
                                sr = float(max(1.0, float(sense_radius_m) if float(sense_radius_m) > 1e-6 else 50.0))
                                closeness = max(0.0, 1.0 - (dist_m / sr))
                                # 1) Historical weight (pheromone/shared kernel) — used as fallback only.
                                if not (bool(inspector_kernel_rt_visible) and str(inspector_did0) == str(did)):
                                    score += float(getattr(self, "dyn_danger_inspect_weight", 0.0)) * float(closeness)

                                # 2) Realtime weight (LiDAR): prefer "sweet spot" ring outside damage radius.
                                try:
                                    # If realtime ground-truth is visible, treat it as realtime (O(1) radius).
                                    rt = None
                                    if bool(inspector_kernel_rt_visible) and str(inspector_did0) == str(did) and inspector_kernel_rt_cell is not None:
                                        rt = {
                                            "cell": tuple(inspector_kernel_rt_cell),
                                            "t": float(t_sim),
                                            "radius_cells": int(max(0, rad0)),
                                        }
                                    else:
                                        rt = self._dyn_kernel_realtime.get(str(did))
                                    ttl = float(getattr(self, "dyn_inspector_rt_ttl_s", 2.0))
                                    w_rt = float(getattr(self, "dyn_inspector_rt_weight", 0.0))
                                    if rt is not None and w_rt > 1e-9 and ttl > 1e-9:
                                        age = float(t_sim) - float(rt.get("t", -1e9))
                                        if age <= ttl + 1e-6:
                                            rc = tuple(rt.get("cell", kc))
                                            # Prefer using realtime-provided radius (from danger map) instead of scanning cells.
                                            rad = 0
                                            try:
                                                rad = int(rt.get("radius_cells", 0) or 0)
                                            except Exception:
                                                rad = 0
                                            if rad <= 0:
                                                # Fallback: estimate from locally stored damage pheromones (bounded scan).
                                                max_r = 30
                                                rad2 = 0
                                                for dx in range(-max_r, max_r + 1):
                                                    for dy in range(-max_r, max_r + 1):
                                                        cc2 = (int(rc[0]) + int(dx), int(rc[1]) + int(dy))
                                                        if not self.grid.in_bounds_cell(int(cc2[0]), int(cc2[1])):
                                                            continue
                                                        mkx = self.pher.danger.meta.get(cc2)
                                                        kk = str(getattr(mkx, "kind", "") or "")
                                                        if kk.startswith(f"danger_dyn_damage:{did}"):
                                                            rad2 = max(rad2, int(round(math.sqrt(float(dx * dx + dy * dy)))))
                                                rad = int(rad2)
                                            # Follow the realtime threat while it is "visible":
                                            # - primary: attraction to the newest kernel position (closeness)
                                            # - secondary: keep a standoff distance outside the damage radius
                                            rux, ruy = self.grid.cell_to_world(int(rc[0]), int(rc[1]))
                                            d_rt = math.hypot(float(rux) - float(nx), float(ruy) - float(ny))
                                            sr2 = float(max(1.0, float(sense_radius_m) if float(sense_radius_m) > 1e-6 else 50.0))
                                            rt_close = max(0.0, 1.0 - (float(d_rt) / float(sr2)))
                                            score += float(w_rt) * float(rt_close)

                                            standoff = int(getattr(self, "dyn_inspector_rt_standoff_cells", 1))
                                            standoff = int(max(0, standoff))
                                            safe_m = float((max(0, int(rad)) + int(standoff)) * float(self.grid.cell_size_m))
                                            if safe_m > 1e-6 and float(d_rt) < float(safe_m) - 1e-6:
                                                # Penalize stepping too close even if the damage pheromone isn't present.
                                                score -= float(getattr(self, "dyn_inspector_avoid_damage_weight", 25.0))
                                except Exception:
                                    pass
                                # Inspector should be afraid to actually step into the danger radius (damage trace):
                                # Apply extra penalty when the candidate cell is inside dynamic damage.
                                try:
                                    mk2 = self.pher.danger.meta.get(cc)
                                    k2 = str(getattr(mk2, "kind", "") or "")
                                    if k2.startswith("danger_dyn_damage"):
                                        thr = float(getattr(self, "dyn_inspector_avoid_damage_thr", 0.05))
                                        w = float(getattr(self, "dyn_inspector_avoid_damage_weight", 25.0))
                                        if float(self.pher.danger.get(cc)) >= thr:
                                            score -= float(w)
                                except Exception:
                                    pass
                                # No explicit "claim refresh" needed: inspector is coordinated via
                                # {dynamic_inspect_active_id, dynamic_inspect_active_t} conflict resolution on comms.
                except Exception:
                    pass

                # Break symmetry between drones in EXPLORE: per-drone "personality" bias + small noise.
                # BUT: active inspector should be deterministic (no ACO base randomness).
                try:
                    is_inspector = bool(str(getattr(self.s, "dynamic_inspect_active_id", "") or "").strip())
                except Exception:
                    is_inspector = False
                if (not is_inspector) and self.s.mode != "RETURN" and mission_phase == "EXPLORE":
                    if float(explore_personal_bias_weight) > 0.0:
                        score += float(explore_personal_bias_weight) * math.cos(yaw - float(self._explore_bias_yaw))
                    if float(explore_score_noise) > 0.0:
                        score += float(explore_score_noise) * (2.0 * float(self.rng.random()) - 1.0)

                candidates.append((score, (nx, ny, yaw)))

            if candidates:
                break

        # Record the ACO scoring snapshot for visualization even if we later decide to override with A*.
        # At this point we have a set of feasible candidate moves (or empty if boxed in).
        try:
            self.last_aco_candidates = list(candidates)
            if candidates:
                # "What ACO would do" (best-scoring candidate) — useful when A* overrides.
                _best = max(candidates, key=lambda t: float(t[0]))
                bx, by, byaw = _best[1]
                self.last_aco_choice_world = (float(bx), float(by), float(byaw))
                self.last_aco_choice_t = float(t_sim)
                # Set threat decision beam based on the best ACO aim cell (visual feedback).
                try:
                    if dynamic_threat_decision_enabled:
                        bcc = self.grid.world_to_cell(float(bx), float(by))
                        dd = dyn_decision_by_cell.get(tuple(bcc))
                        if dd is not None:
                            mode, tcell = dd
                            self.s.threat_decision_cell = (int(tcell[0]), int(tcell[1]))
                            self.s.threat_decision_mode = str(mode)
                            self.s.threat_decision_t = float(t_sim)
                        else:
                            self.s.threat_decision_cell = None
                            self.s.threat_decision_mode = ""
                    else:
                        self.s.threat_decision_cell = None
                        self.s.threat_decision_mode = ""
                except Exception:
                    pass
        except Exception:
            pass

        def _try_local_plan(preferred_yaw: float) -> Optional[Tuple[float, float, float]]:
            planned = _execute_active_plan()
            if planned is not None:
                return planned

            if self.s.mode == "RETURN":
                gx, gy = base_xy
            elif target_goal is not None:
                gx, gy = target_goal
            else:
                look = max(15.0, float(local_plan_radius_m))
                gx = self.s.x + math.cos(preferred_yaw) * look
                gy = self.s.y + math.sin(preferred_yaw) * look
                gx, gy = clamp_xy(gx, gy, map_bounds_m)

            # Build and score multiple candidate plans; pick ONE and commit to it.
            # Objective: go as far as possible in the journey, using as little energy as possible.
            try:
                # Desired journey goal for progress measurement
                if self.s.mode == "RETURN":
                    goal_x, goal_y = float(base_xy[0]), float(base_xy[1])
                elif target_goal is not None:
                    goal_x, goal_y = float(target_goal[0]), float(target_goal[1])
                else:
                    goal_x, goal_y = float(gx), float(gy)

                # Candidate goal points: preferred yaw + nearby yaws, with empty-space pierce first.
                cand_goals: List[Tuple[float, float]] = []
                pierce_yaw = float(preferred_yaw)
                if bool(getattr(self.s, "avoid_active", False)):
                    pierce_yaw = float(getattr(self.s, "avoid_entry_yaw", pierce_yaw))
                pierce_dist = max(float(local_plan_radius_m) * 2.0, float(wall_clearance_m) * 3.0, 30.0)
                empty_goal = self._pierce_empty_goal_world(
                    yaw=pierce_yaw,
                    max_dist_m=pierce_dist,
                    building_index=building_index,
                    safety_margin_z=safety_margin_z,
                    map_bounds_m=map_bounds_m,
                )
                if empty_goal is not None:
                    cand_goals.append((float(empty_goal[0]), float(empty_goal[1])))
                # Fallback geometric goals around preferred direction
                for dd in (0.0, math.pi / 6.0, -math.pi / 6.0, math.pi / 3.0, -math.pi / 3.0):
                    yy = float(preferred_yaw) + float(dd)
                    look = max(15.0, float(local_plan_radius_m))
                    cx, cy = clamp_xy(self.s.x + math.cos(yy) * look, self.s.y + math.sin(yy) * look, map_bounds_m)
                    cand_goals.append((float(cx), float(cy)))
                # Always include the base/target/forward goal (ensures RETURN has a solid plan).
                cand_goals.append((float(gx), float(gy)))

                def plan_cost(path: List[Tuple[float, float]]) -> float:
                    # approximate energy proportional to total horizontal distance
                    tot = 0.0
                    for i in range(1, len(path)):
                        tot += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                    return tot

                start_dist = math.hypot(self.s.x - goal_x, self.s.y - goal_y)
                best = None  # (progress, energy, path)
                tried = 0
                for (cgx, cgy) in cand_goals:
                    tried += 1
                    if tried > 8:
                        break
                    pw = self._local_a_star_path_world(
                        goal_xy=(float(cgx), float(cgy)),
                        mission_phase=mission_phase,
                        building_index=building_index,
                        safety_margin_z=safety_margin_z,
                        map_bounds_m=map_bounds_m,
                        plan_radius_m=float(local_plan_radius_m),
                        inflate_cells=max(0, int(local_plan_inflate_cells)),
                        max_nodes=int(min(900, int(local_plan_max_nodes))),
                        unknown_penalty=float(local_plan_unknown_penalty),
                        recent_penalty=float(recent_cell_penalty),
                    )
                    if pw is None or len(pw) < 2:
                        continue
                    endx, endy = pw[-1]
                    end_dist = math.hypot(endx - goal_x, endy - goal_y)
                    progress = float(start_dist - end_dist)
                    energy = float(plan_cost(pw) * float(self.energy.cost_per_meter_units))
                    # Primary: progress, Secondary: lower energy
                    if best is None:
                        best = (progress, energy, pw)
                    else:
                        bp, be, _ = best
                        # If progress is significantly better, take it.
                        if progress > bp + 1e-6 and progress >= bp * 1.02:
                            best = (progress, energy, pw)
                        else:
                            # If progress similar (within 2%), pick lower energy.
                            if abs(progress - bp) <= max(0.1, abs(bp) * 0.02) and energy < be - 1e-9:
                                best = (progress, energy, pw)

                if best is None:
                    return None
                _, _, plan_world = best
            except Exception:
                plan_world = None

            if plan_world is None or len(plan_world) < 2:
                return None

            # Store for debug visualization (planned, not executed)
            self.last_plan_world = list(plan_world)
            self.last_plan_t = float(t_sim)
            # Path "string-pull": if we can see further points directly, skip grid-like 90° turns.
            # Use true-geometry line check (this is sim; corresponds to "if no obstacles between current and finish").
            idx = 1
            for k in range(len(plan_world) - 1, 0, -1):
                wx, wy = plan_world[k]
                dx = wx - self.s.x
                dy = wy - self.s.y
                dist = math.hypot(dx, dy)
                if dist <= 1e-6:
                    continue
                yawk = math.atan2(dy, dx)
                clear = self._ray_clearance_m(
                    building_index=building_index,
                    safety_margin_z=safety_margin_z,
                    map_bounds_m=map_bounds_m,
                    yaw=yawk,
                    max_dist_m=dist + 1e-3,
                    step_m=self.grid.cell_size_m * 0.5,
                )
                if clear >= (dist - self.grid.cell_size_m * 0.25):
                    idx = k
                    break

            # Commit to this plan and execute it to completion (or until invalid).
            self._active_plan_world = list(plan_world)
            self._active_plan_idx = int(max(1, idx))
            # long TTL; will also end when plan finishes
            self._active_plan_until_t = float(t_sim) + 30.0
            # Execute first step now
            return _execute_active_plan()

        def _execute_active_plan() -> Optional[Tuple[float, float, float]]:
            """Execute committed A* plan if present; returns next step or None."""
            try:
                if not self._active_plan_world:
                    return None
                if float(t_sim) > float(self._active_plan_until_t):
                    self._active_plan_world = []
                    self._active_plan_idx = 0
                    self._active_plan_until_t = -1e9
                    return None

                # Advance idx while target is essentially current cell / too close
                cur_c = self.grid.world_to_cell(self.s.x, self.s.y)
                while self._active_plan_idx < len(self._active_plan_world):
                    tx, ty = self._active_plan_world[self._active_plan_idx]
                    if self.grid.world_to_cell(tx, ty) != cur_c and math.hypot(tx - self.s.x, ty - self.s.y) > max(
                        0.5, self.grid.cell_size_m * 0.25
                    ):
                        break
                    self._active_plan_idx += 1

                if self._active_plan_idx >= len(self._active_plan_world):
                    self._active_plan_world = []
                    self._active_plan_idx = 0
                    self._active_plan_until_t = -1e9
                    return None

                tx, ty = self._active_plan_world[self._active_plan_idx]
                step_m = self.s.speed_mps * dt
                dist = math.hypot(tx - self.s.x, ty - self.s.y)
                if dist <= 1e-6:
                    return None
                # Base direction toward the next planned point
                vx = (tx - float(self.s.x)) / float(dist)
                vy = (ty - float(self.s.y)) / float(dist)

                # EXPLOIT: apply peer-avoidance steering (penalty near comrades), but clamp deviation
                # so we still follow the A* path. Fade out the penalty near the final target.
                try:
                    if str(mission_phase).upper() == "EXPLOIT" and target_goal is not None and self.s.mode != "RETURN":
                        pp = getattr(self, "_exploit_peer_params", None) or {}
                        avoid_r = float(pp.get("avoid_r", 40.0))
                        avoid_w = float(pp.get("avoid_w", 2.0))
                        follow_w = float(pp.get("follow_w", 1.0))
                        fade0 = float(pp.get("fade0", 25.0))
                        fade_rng = float(pp.get("fade_rng", 50.0))
                        max_dev_deg = float(pp.get("max_dev_deg", 70.0))
                        yaw_rate = float(pp.get("yaw_rate", 1.2))
                        slow_r = float(pp.get("slowdown_r", 70.0))
                        landing_speed = float(pp.get("landing_speed", 3.0))

                        dg = math.hypot(float(self.s.x) - float(target_goal[0]), float(self.s.y) - float(target_goal[1]))
                        fade = 1.0
                        if fade_rng > 1e-6:
                            fade = max(0.0, min(1.0, (float(dg) - float(fade0)) / float(fade_rng)))
                        else:
                            fade = 0.0 if float(dg) <= float(fade0) + 1e-6 else 1.0

                        if avoid_r > 1e-6 and avoid_w > 1e-9 and fade > 1e-6:
                            repx = 0.0
                            repy = 0.0
                            for px, py, uid in (getattr(self, "_swarm_xy", None) or []):
                                try:
                                    if str(uid) == str(self.s.drone_uid):
                                        continue
                                except Exception:
                                    pass
                                dx = float(self.s.x) - float(px)
                                dy = float(self.s.y) - float(py)
                                dd = math.hypot(dx, dy)
                                if dd <= 1e-6 or dd >= avoid_r:
                                    continue
                                # Inverse-distance repulsion (strong close, weak far)
                                inv = 1.0 / max(1e-6, dd)
                                repx += (dx * inv) * inv
                                repy += (dy * inv) * inv

                            # Blend and normalize
                            fw = max(0.05, float(follow_w))
                            ax = float(fw) * float(vx) + float(avoid_w) * float(fade) * float(repx)
                            ay = float(fw) * float(vy) + float(avoid_w) * float(fade) * float(repy)
                            an = math.hypot(ax, ay)
                            if an > 1e-9:
                                ax /= an
                                ay /= an
                                # Clamp deviation from A* direction (reward staying on best path).
                                try:
                                    dot = max(-1.0, min(1.0, float(ax) * float(vx) + float(ay) * float(vy)))
                                    ang = math.degrees(math.acos(dot))
                                    if ang > max(5.0, max_dev_deg):
                                        # Blend back toward A* direction.
                                        keep = max(0.0, min(1.0, float(max_dev_deg) / max(1e-6, float(ang))))
                                        ax = float(vx) * (1.0 - keep) + float(ax) * keep
                                        ay = float(vy) * (1.0 - keep) + float(ay) * keep
                                        an2 = math.hypot(ax, ay)
                                        if an2 > 1e-9:
                                            ax /= an2
                                            ay /= an2
                                except Exception:
                                    pass
                                vx, vy = float(ax), float(ay)
                except Exception:
                    pass

                # Smooth yaw (no 90° snaps) in EXPLOIT by limiting yaw rate.
                desired_yaw = math.atan2(float(vy), float(vx))
                yaw = float(desired_yaw)
                try:
                    if str(mission_phase).upper() == "EXPLOIT":
                        yr = max(0.1, float(yaw_rate))
                        dy = ((float(desired_yaw) - float(self.s.yaw) + math.pi) % (2.0 * math.pi)) - math.pi
                        dy = max(-yr * float(dt), min(yr * float(dt), dy))
                        yaw = (float(self.s.yaw) + float(dy)) % (2.0 * math.pi)
                        # Move along the smoothed yaw (arc-like), not directly to waypoint.
                        vx = math.cos(float(yaw))
                        vy = math.sin(float(yaw))
                except Exception:
                    yaw = float(desired_yaw)

                # Slow down near the final target and during landing.
                step_use = min(float(step_m), float(dist))
                try:
                    if str(mission_phase).upper() == "EXPLOIT" and target_goal is not None:
                        dg2 = math.hypot(float(self.s.x) - float(target_goal[0]), float(self.s.y) - float(target_goal[1]))
                        if float(slow_r) > 1e-6 and dg2 < float(slow_r):
                            frac = max(0.25, min(1.0, float(dg2) / max(1e-6, float(slow_r))))
                            step_use *= float(frac)
                        if bool(getattr(self, "_exploit_land_active", False)):
                            # Cap step length based on landing speed.
                            step_use = min(float(step_use), float(max(0.2, float(landing_speed))) * float(dt))
                except Exception:
                    pass

                nx = float(self.s.x) + float(vx) * float(step_use)
                ny = float(self.s.y) + float(vy) * float(step_use)
                if building_index.is_obstacle_xy(nx, ny, self.s.z, safety_margin_z):
                    # invalidated
                    self._active_plan_world = []
                    self._active_plan_idx = 0
                    self._active_plan_until_t = -1e9
                    return None

                # Keep visual plan as the committed plan
                self.last_plan_world = list(self._active_plan_world)
                self.last_plan_t = float(t_sim)
                return float(nx), float(ny), float(yaw)
            except Exception:
                self._active_plan_world = []
                self._active_plan_idx = 0
                self._active_plan_until_t = -1e9
                return None

        # Always execute a committed plan first (prevents indecision).
        planned = _execute_active_plan()
        if planned is not None:
            nx, ny, yaw = planned
            self.last_move_source = "A*"
            if _block_xy_until_static_alt_ok(float(nx), float(ny), float(yaw)):
                return
            dyaw = abs((yaw - self._last_yaw + math.pi) % (2.0 * math.pi) - math.pi)
            maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)
            dist = math.hypot(nx - self.s.x, ny - self.s.y)
            self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
            self._update_energy(dist, maneuver_factor)
            self.s.total_dist_m += dist
            self._last_yaw = yaw
            c = self.grid.world_to_cell(self.s.x, self.s.y)
            self.s.recent_cells.append(c)
            if (not bool(getattr(self.s, "avoid_active", False))) and (
                math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) > base_no_deposit_radius_m
            ):
                if self.s.mode != "RETURN":
                    self.pher.deposit_nav(c, amount=0.25, t=t_sim, conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))
            return

        # EXPLOIT: if we have a concrete target, plan toward it immediately (A* on pheromone costs).
        # This avoids the expensive per-heading local scanning loop and prevents "stuck at base" dithering.
        if str(mission_phase).upper() == "EXPLOIT" and target_goal is not None and self.s.mode != "RETURN":
            try:
                gyaw = math.atan2(float(target_goal[1]) - float(self.s.y), float(target_goal[0]) - float(self.s.x))
                planned2 = _try_local_plan(float(gyaw))
                if planned2 is not None:
                    nx, ny, yaw = planned2
                    self.last_move_source = "A*"
                    if _block_xy_until_static_alt_ok(float(nx), float(ny), float(yaw)):
                        return
                    dyaw = abs((yaw - self._last_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                    maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)
                    dist = math.hypot(nx - self.s.x, ny - self.s.y)
                    self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
                    self._update_energy(dist, maneuver_factor)
                    self.s.total_dist_m += dist
                    self._last_yaw = yaw
                    c = self.grid.world_to_cell(self.s.x, self.s.y)
                    self.s.recent_cells.append(c)
                    # No exploration shaping in exploit: still deposit a small nav trace for stats/visualization.
                    if (not bool(getattr(self.s, "avoid_active", False))) and (
                        math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) > base_no_deposit_radius_m
                    ):
                        self.pher.deposit_nav(c, amount=0.15, t=t_sim, conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))
                    return
            except Exception:
                pass

        if candidates:
            # If the best heading is "into a corner" or the direct line is blocked by known obstacles,
            # use local A* instead (plan within lidar-known map + pheromone costs).
            best_yaw = max(candidates, key=lambda t: t[0])[1][2]
            lookahead = max(self.grid.cell_size_m * 4.0, 20.0)
            best_clearance = self._ray_clearance_m(
                building_index=building_index,
                safety_margin_z=safety_margin_z,
                map_bounds_m=map_bounds_m,
                yaw=best_yaw,
                max_dist_m=lookahead,
                step_m=self.grid.cell_size_m * 0.5,
            )
            # Build a short-horizon goal and see if the straight line is blocked in known map.
            if self.s.mode == "RETURN":
                gx, gy = base_xy
            elif target_goal is not None:
                gx, gy = target_goal
            else:
                look = max(15.0, float(local_plan_radius_m))
                gx, gy = clamp_xy(self.s.x + math.cos(best_yaw) * look, self.s.y + math.sin(best_yaw) * look, map_bounds_m)

            start_c = self.grid.world_to_cell(self.s.x, self.s.y)
            goal_c = self.grid.world_to_cell(gx, gy)
            los_blocked = self._line_blocked_known(
                start_c,
                goal_c,
                inflate_cells=max(0, int(local_plan_inflate_cells)),
                building_index=building_index,
                safety_margin_z=safety_margin_z,
            )

            # Prefer ACO randomness in level flight; reserve A* for altitude/overfly situations,
            # or true corner/boxed-in emergencies.
            need_alt_adjust = bool(getattr(self.s, "overfly_active", False)) or bool(getattr(self.s, "hop_active", False))
            try:
                zt = float(getattr(self.s, "z_target", self.s.z))
                zc = float(getattr(self.s, "z_cruise", self.s.z))
                if abs(zt - zc) > 0.5 or abs(self.s.z - zt) > 0.75:
                    need_alt_adjust = True
            except Exception:
                pass

            cell = max(0.5, float(self.grid.cell_size_m))
            trigger_clearance = float(local_plan_trigger_clearance_frac) * float(local_plan_radius_m)
            hard_corner = max(2.25 * cell, 0.0)
            hard_los = max(3.5 * cell, 0.0)
            want_a_star = False
            if need_alt_adjust:
                want_a_star = (best_clearance < max(3.0 * cell, trigger_clearance)) or bool(los_blocked)
            else:
                # level flight: only if very near a wall/corner (or LOS blocked AND we're close)
                want_a_star = (best_clearance < hard_corner) or (bool(los_blocked) and best_clearance < hard_los)

            if want_a_star:
                planned = _try_local_plan(best_yaw)
                if planned is not None:
                    nx, ny, yaw = planned
                    self.last_move_source = "A*"
                    if _block_xy_until_static_alt_ok(float(nx), float(ny), float(yaw)):
                        return
                    dyaw = abs((yaw - self._last_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                    maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)
                    dist = math.hypot(nx - self.s.x, ny - self.s.y)
                    self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
                    self._update_energy(dist, maneuver_factor)
                    self.s.total_dist_m += dist
                    self._last_yaw = yaw
                    c = self.grid.world_to_cell(self.s.x, self.s.y)
                    self.s.recent_cells.append(c)
                    if (not bool(getattr(self.s, "avoid_active", False))) and (
                        math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) > base_no_deposit_radius_m
                    ):
                        if self.s.mode != "RETURN":
                            self.pher.deposit_nav(c, amount=0.25, t=t_sim, conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))
                    return

        if not candidates:
            # Boxed in (typically in a corner or grazing a building margin): attempt an escape move.
            planned = _try_local_plan(self.s.yaw)
            if planned is not None:
                nx, ny, yaw = planned
                self.last_move_source = "A*"
                # Pre-climb before stepping into static danger (altitude requirement).
                try:
                    aim_dist = max(float(self.grid.cell_size_m), float(math.hypot(nx - self.s.x, ny - self.s.y)))
                    ax = float(self.s.x) + math.cos(float(yaw)) * float(aim_dist)
                    ay = float(self.s.y) + math.sin(float(yaw)) * float(aim_dist)
                    cc = self.grid.world_to_cell(ax, ay)
                    mk = self.pher.danger.meta.get((int(cc[0]), int(cc[1])))
                    if mk is not None:
                        kk = str(getattr(mk, "kind", "") or "")
                        is_static_like = bool(kk.startswith("danger_static")) or (
                            bool(dynamic_danger_trail_as_static) and bool(kk.startswith("danger_dyn_"))
                        )
                    else:
                        is_static_like = False
                    if mk is not None and bool(is_static_like):
                        alt = getattr(mk, "alt_m", None)
                        if alt is not None and float(self.s.z) < float(alt) - 1e-6:
                            self.s.overfly_active = True
                            self.s.z_target = float(max(float(alt), float(getattr(self.s, "z_target", self.s.z))))
                            self.s.overfly_start_x = float(self.s.x)
                            self.s.overfly_start_y = float(self.s.y)
                            self.s.overfly_start_t = float(t_sim)
                            self.s.preclimb_static_hold = True
                            self.last_move_source = "PRECLIMB_STATIC"
                            return
                except Exception:
                    pass

                dist = math.hypot(nx - self.s.x, ny - self.s.y)
                self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
                self._update_energy(dist, maneuver_factor=1.10)
                self.s.total_dist_m += dist
                self._last_yaw = yaw
                c = self.grid.world_to_cell(self.s.x, self.s.y)
                self.s.recent_cells.append(c)
                self.pher.deposit_danger(c, amount=0.4, t=t_sim, conf=0.7, src=self.s.drone_uid)
                return

            if bool(unstick_move_enabled):
                esc = self._unstick_move(building_index, safety_margin_z, map_bounds_m)
                if esc is not None:
                    nx, ny, yaw = esc
                    self.last_move_source = "A*"
                    dist = math.hypot(nx - self.s.x, ny - self.s.y)
                    self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
                    self._update_energy(dist, maneuver_factor=1.15)
                    self.s.total_dist_m += dist
                    self._last_yaw = yaw
                    c = self.grid.world_to_cell(self.s.x, self.s.y)
                    self.s.recent_cells.append(c)
                    # Mark danger where we got stuck to discourage others from repeating it.
                    self.pher.deposit_danger(c, amount=0.6, t=t_sim, conf=0.8, src=self.s.drone_uid)
                    return

            # Still stuck: rotate in place slightly and try again next step.
            self.s.yaw = (self.s.yaw + 0.3) % (2.0 * math.pi)
            # Backoff: reduce positive nav pheromone for the current cell so we don't keep retrying the same corner.
            try:
                cur_c = self.grid.world_to_cell(self.s.x, self.s.y)
                cur_v = self.pher.nav.get(cur_c)
                if cur_v > 1e-6:
                    meta = CellMeta(t=float(t_sim), conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))
                    self.pher.nav.set(cur_c, max(0.0, cur_v * 0.6), meta)
                    self.pher.recent_nav.append((cur_c, self.pher.nav.get(cur_c), meta))
                # Also mark navigation danger at this altitude (helps future planning avoid the corner).
                self.pher.deposit_nav_danger(cur_c, amount=0.2, t=t_sim, conf=0.7, src=self.s.drone_uid, alt_m=float(self.s.z))
            except Exception:
                pass
            return

        # Selection:
        # - EXPLOIT is greedy (otherwise it can dither and even increase distance temporarily).
        # - Active INSPECTOR is greedy (user requested removing base ACO randomness; "just follow the threat").
        # - Others use classical ACO softmax sampling (temperature).
        is_inspector = False
        try:
            is_inspector = bool(str(getattr(self.s, "dynamic_inspect_active_id", "") or "").strip())
        except Exception:
            is_inspector = False
        if mission_phase == "EXPLOIT" or is_inspector:
            nx, ny, yaw = max(candidates, key=lambda t: t[0])[1]
        else:
            nx, ny, yaw = softmax_sample(candidates, temperature=aco_temperature, rng=self.rng)
        # Store last ACO scoring snapshot for visualization (even if selection is greedy in RETURN/EXPLOIT).
        try:
            self.last_aco_candidates = list(candidates)
            self.last_aco_choice_world = (float(nx), float(ny), float(yaw))
            self.last_aco_choice_t = float(t_sim)
            self.last_move_source = "ACO"
        except Exception:
            pass

        # maneuver factor: extra cost if sharp turn
        dyaw = abs((yaw - self._last_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        maneuver_factor = 1.0 + (0.2 if dyaw > (math.pi / 3.0) else 0.1 if dyaw > (math.pi / 6.0) else 0.0)

        prev_x, prev_y = float(self.s.x), float(self.s.y)
        prev_cell = self.grid.world_to_cell(prev_x, prev_y)
        dist = math.hypot(nx - self.s.x, ny - self.s.y)

        # Pre-climb before stepping into static danger: if the *intended* next cell has a required
        # danger altitude higher than current z, climb first and delay horizontal motion.
        try:
            if _block_xy_until_static_alt_ok(float(nx), float(ny), float(yaw)):
                return
        except Exception:
            pass

        self.s.x, self.s.y, self.s.yaw = nx, ny, yaw
        self._update_energy(dist, maneuver_factor)
        self.s.total_dist_m += dist
        self._last_yaw = yaw

        # Commit to the chosen next cell (until reached) so we don't dither.
        if commit_enabled and self.s.mode == "EXPLORE" and mission_phase == "EXPLORE":
            try:
                # Commit the *intended* next cell (>= 1 cell away), not necessarily the cell reached by this substep.
                aim_dist = max(float(dist), float(self.grid.cell_size_m))
                ax = float(prev_x) + math.cos(float(yaw)) * float(aim_dist)
                ay = float(prev_y) + math.sin(float(yaw)) * float(aim_dist)
                cc = self.grid.world_to_cell(ax, ay)
                if cc != prev_cell:
                    self._aco_commit_cell = (int(cc[0]), int(cc[1]))
                    self._aco_commit_rev = int(getattr(self.s, "hazard_rev", getattr(self.s, "perception_rev", 0)))
                    tout = float(getattr(self, "aco_commit_timeout_s", 5.0))
                    self._aco_commit_until_t = float(t_sim) + max(0.2, float(tout))
            except Exception:
                pass

        # Inspector debug: print inspector + hunted threat location every 3 new cells entered.
        try:
            did_dbg = str(getattr(self.s, "dynamic_inspect_active_id", "") or "").strip()
            if did_dbg:
                cur_c = self.grid.world_to_cell(float(self.s.x), float(self.s.y))
                cur_c2 = (int(cur_c[0]), int(cur_c[1]))
                if self._insp_dbg_last_cell is None:
                    self._insp_dbg_last_cell = cur_c2
                if cur_c2 != tuple(self._insp_dbg_last_cell):
                    self._insp_dbg_last_cell = cur_c2
                    self._insp_dbg_cell_steps = int(self._insp_dbg_cell_steps) + 1
                    if (int(self._insp_dbg_cell_steps) % 3) == 0:
                        # Always print the *real* threat location by id if we have it (even if far away).
                        tcell = None
                        trad = None
                        try:
                            if dynamic_threats:
                                dt_map = {str(d.get("id", "") or ""): d for d in (dynamic_threats or []) if str(d.get("id", "") or "")}
                                dt0 = dt_map.get(did_dbg)
                                if dt0 is not None:
                                    tcell0 = dt0.get("cell", None)
                                    if tcell0 is not None and len(tcell0) == 2:
                                        tcell = (int(tcell0[0]), int(tcell0[1]))
                                    try:
                                        trad = int(dt0.get("radius", 0) or 0)
                                    except Exception:
                                        trad = None
                        except Exception:
                            tcell = None
                        wx, wy = self.grid.cell_to_world(int(cur_c2[0]), int(cur_c2[1]))
                        msg = f"[insp] {self.s.drone_uid} c={int(cur_c2[0])},{int(cur_c2[1])} w=({wx:.1f},{wy:.1f})"
                        if tcell is not None:
                            tx, ty = self.grid.cell_to_world(int(tcell[0]), int(tcell[1]))
                            dist_m = math.hypot(float(tx) - float(wx), float(ty) - float(wy))
                            msg += f" -> threat[{did_dbg}] c={int(tcell[0])},{int(tcell[1])} w=({tx:.1f},{ty:.1f}) d={dist_m:.1f}m"
                            if trad is not None:
                                msg += f" r={int(trad)}c"
                        else:
                            msg += f" -> threat[{did_dbg}] (unknown)"
                        msg += f" t={float(t_sim):.2f}s"
                        print(msg)
        except Exception:
            pass

        # deposit nav pheromone along path; reduced near obstacles via low amount when close to obstacle
        c = self.grid.world_to_cell(self.s.x, self.s.y)
        # Update short-term memory after movement (helps avoid small cycles).
        self.s.recent_cells.append(c)
        near_obs = building_index.is_obstacle_xy(self.s.x, self.s.y, self.s.z, safety_margin_z)
        # Avoid depositing nav pheromone at the base (prevents attraction sink after recharging).
        if math.hypot(self.s.x - base_xy[0], self.s.y - base_xy[1]) > base_no_deposit_radius_m:
            if bool(getattr(self.s, "avoid_active", False)):
                # Avoidance trail: deposit danger (non-attracting) instead of nav.
                self.pher.deposit_danger(
                    c,
                    amount=float(getattr(self.s, "avoid_deposit_danger_amount", 0.35)),
                    t=t_sim,
                    conf=0.6,
                    src=self.s.drone_uid,
                    kind="avoid",
                )
            elif self.s.mode != "RETURN":
                deposit = 0.1 if near_obs else 0.3
                # If we are repeatedly revisiting the same cell while exploring, suppress nav deposit so we
                # don't create strong "knots" that attract ourselves/others back into small loops.
                if self.s.mode == "EXPLORE" and mission_phase == "EXPLORE":
                    try:
                        rs = float(explore_revisit_nav_deposit_scale)
                        if rs < 0.999:
                            cnt = 0
                            for rc in self.s.recent_cells:
                                if rc == c:
                                    cnt += 1
                            if int(cnt) > 1:
                                deposit *= float(max(0.0, rs)) ** float(max(0, int(cnt) - 1))
                    except Exception:
                        pass
                self.pher.deposit_nav(c, amount=deposit, t=t_sim, conf=0.6, src=self.s.drone_uid, alt_m=float(self.s.z))


# -----------------------------
# Manager node
# -----------------------------


