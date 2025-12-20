from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Set


DroneType = str  # "REAL" | "PY"


@dataclass
class GridSpec:
    grid_size_m: float
    cell_size_m: float

    @property
    def half(self) -> float:
        return self.grid_size_m / 2.0

    @property
    def cells(self) -> int:
        return int(self.grid_size_m / self.cell_size_m)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int((x + self.half) / self.cell_size_m)
        cy = int((y + self.half) / self.cell_size_m)
        return cx, cy

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        x = (cx * self.cell_size_m) - self.half + (self.cell_size_m / 2.0)
        y = (cy * self.cell_size_m) - self.half + (self.cell_size_m / 2.0)
        return x, y

    def in_bounds_cell(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.cells and 0 <= cy < self.cells


@dataclass
class Building:
    x: float
    y: float
    z_center: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def top_z(self) -> float:
        return self.z_center + (self.size_z / 2.0)

    def contains_xy(self, px: float, py: float, margin: float) -> bool:
        hx = (self.size_x / 2.0) + margin
        hy = (self.size_y / 2.0) + margin
        return (abs(px - self.x) <= hx) and (abs(py - self.y) <= hy)

    def bbox_xy(self, margin: float) -> Tuple[float, float, float, float]:
        """Expanded axis-aligned bbox: (minx, maxx, miny, maxy)."""
        hx = (self.size_x / 2.0) + margin
        hy = (self.size_y / 2.0) + margin
        return (self.x - hx, self.x + hx, self.y - hy, self.y + hy)


@dataclass
class Target:
    target_id: str
    x: float
    y: float
    z: float
    found_by: Optional[str] = None
    found_t: Optional[float] = None

    def is_found(self) -> bool:
        return self.found_by is not None


@dataclass
class TargetKnowledge:
    target_id: str
    x: float
    y: float
    z: float
    found: bool = False
    found_by: Optional[str] = None
    found_t: Optional[float] = None
    updated_t: float = 0.0  # simulation time when last updated


@dataclass
class ExploreVector:
    """Lightweight exploration intent message shared via comms."""

    origin_uid: str
    start_x: float
    start_y: float
    yaw: float
    t: float  # sim time when vector was generated/updated


@dataclass
class DynamicInspectStatus:
    """Minimal inspector status exchanged on comms: {danger_id, timestamp}."""

    danger_id: str  # "" means not inspecting
    t: float  # sim time when this inspection started (tie-breaker: older wins)


@dataclass
class EnergyModel:
    full_units: float = 100.0
    cost_per_meter_units: float = 0.01  # per spec
    low_energy_threshold_units: float = 20.0  # triggers LOW_ENERGY mode
    return_threshold_units: float = 50.0  # return to base below 50%
    recharge_to_units: float = 80.0
    low_energy_speed_mps: float = 5.0
    normal_speed_mps: float = 10.0
    low_energy_max_range_m: float = 2000.0


@dataclass
class DroneState:
    drone_uid: str
    drone_type: DroneType  # "PY" or "REAL"
    seq: int
    x: float
    y: float
    z: float
    yaw: float = 0.0
    speed_mps: float = 10.0
    energy_units: float = 100.0
    mode: str = "IDLE"  # IDLE|EXPLORE|RETURN|RECHARGE
    total_dist_m: float = 0.0
    encounters: int = 0
    base_uploads: int = 0
    last_comm_t: Dict[str, float] = field(default_factory=dict)  # peer_uid -> last time we exchanged

    # targets knowledge (what this drone knows exists / is found)
    known_targets: Dict[str, TargetKnowledge] = field(default_factory=dict)
    recent_target_updates: Deque[TargetKnowledge] = field(default_factory=lambda: deque(maxlen=2000))
    last_target_comm_t: Dict[str, float] = field(default_factory=dict)

    # Exploration intent vectors shared via comms to avoid lockstep exploration.
    # Keys are origin drone_uid (including ourselves), values are the latest known vector.
    known_explore_vectors: Dict[str, ExploreVector] = field(default_factory=dict)
    recent_explore_vector_updates: Deque[ExploreVector] = field(default_factory=lambda: deque(maxlen=1500))
    last_explore_vector_comm_t: Dict[str, float] = field(default_factory=dict)
    explore_vector_cells_since_update: int = 0

    # Short-term memory of recently visited grid cells (used to avoid local loops / orbiting).
    recent_cells: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=80))

    # recharge handling
    recharge_until_t: float = 0.0

    # base pheromone sync bookkeeping (sim-time)
    last_base_pher_upload_t: float = -1e9
    last_base_pher_download_t: float = -1e9

    # Per-drone altitude control (sim). z is current altitude, z_target is setpoint, z_cruise is nominal.
    z_cruise: float = 8.0
    z_target: float = 8.0
    overfly_active: bool = False
    overfly_start_x: float = 0.0
    overfly_start_y: float = 0.0
    overfly_start_t: float = -1e9
    last_progress_t: float = -1e9
    last_progress_dist: float = 1e18
    last_overfly_check_t: float = -1e9
    last_sense_t: float = -1e9
    last_sense_wall_t: float = -1e9

    # Avoidance (crab) mode: temporary sideways motion to bypass obstacles.
    avoid_active: bool = False
    avoid_entry_yaw: float = 0.0
    avoid_side: int = 1  # +1 left, -1 right
    avoid_start_x: float = 0.0
    avoid_start_y: float = 0.0
    avoid_start_t: float = -1e9
    avoid_target_yaw: float = 0.0
    avoid_need_lateral_m: Optional[float] = None

    # Hop-over-lowest mode for L-corners:
    # climb just above the lowest blocking roof, move over it, then go straight.
    hop_active: bool = False
    hop_dir_yaw: float = 0.0
    hop_goal_yaw: float = 0.0
    hop_start_t: float = -1e9

    # Empty helper: how many cells around a wall to mark as empty (A* goal helpers).
    empty_near_wall_radius_cells: int = 2

    # Revision counter for "new perception" (obstacles/danger discovered). Used to break ACO commitment.
    perception_rev: int = 0

    # Revision counter for new *hazards* only (obstacles or danger pheromones).
    # ACO commitment uses this so merely discovering more free space doesn't break the committed step.
    hazard_rev: int = 0

    # Dynamic danger knowledge (learned via lidar discovery, not global truth).
    known_dynamic_danger_ids: Set[str] = field(default_factory=set)

    # Dynamic danger inspection coordination (simple "older inspector wins" protocol).
    dynamic_inspect_active_id: str = ""  # danger id we are inspecting (curiosity reward applies)
    dynamic_inspect_active_t: float = -1e9  # when we started inspecting this id
    dynamic_inspect_skip_ids: Set[str] = field(default_factory=set)  # ids we must not inspect (lost conflict or done)

    # Shared inspection status cache (for forwarding / debugging).
    known_dynamic_inspect_status: Dict[str, DynamicInspectStatus] = field(default_factory=dict)  # peer_uid -> status
    recent_dynamic_inspect_status_updates: Deque[DynamicInspectStatus] = field(default_factory=lambda: deque(maxlen=1000))
    last_dynamic_inspect_status_comm_t: Dict[str, float] = field(default_factory=dict)

    # Dynamic-threat decision visualization (set by planner, rendered by sim).
    threat_decision_cell: Optional[Tuple[int, int]] = None  # cell to point at (threat is/might be here)
    threat_decision_mode: str = ""  # "cross" | "avoid" | ""
    threat_decision_t: float = -1e9  # sim time when decision was made

    # Pre-climb static altitude handling
    preclimb_static_hold: bool = False
    preclimb_static_req_alt: Optional[float] = None
    preclimb_static_last_log_t: Optional[float] = None


@dataclass
class DroneConfig:
    """Configuration parameters for drone behavior (formerly accessed via getattr)."""
    
    # Empty layer and goal dilation
    empty_goal_dilate_cells: int = 0
    baseline_no_empty_layer: bool = False
    baseline_no_altitude_awareness: bool = False
    
    # Altitude and danger parameters
    static_danger_altitude_violation_weight: float = 0.0
    
    # Inspector parameters
    dyn_inspector_min_energy_units: Optional[float] = None
    
    # ACO commitment
    aco_commit_enabled: bool = True
    
    # Dynamic trail parameters (for EXPLOIT mode)
    dyn_trail_static_for_planning: bool = False
    dyn_trail_overlay_strength: float = 3.0
    dyn_trail_overlay_gamma: float = 1.8

