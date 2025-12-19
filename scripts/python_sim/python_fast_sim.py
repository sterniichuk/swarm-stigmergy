#!/usr/bin/env python3
"""
Python Fast Simulation (ROS2)

Single-node, fast, time-accelerated swarm simulation that publishes:
- Aggregated drone visualization: /swarm/markers/drones (MarkerArray)
- Aggregated path visualization: /swarm/markers/paths (MarkerArray)
- Targets visualization: /swarm/markers/targets (MarkerArray)
- Communication "wires": /swarm/markers/comm (MarkerArray)
- Optional PoseArray: /swarm/drones/poses
- /clock while running (for nodes with use_sim_time:=true)

Control:
- /swarm/cmd/start_python (Trigger)
- /swarm/cmd/stop_python (Trigger)
- /swarm/cmd/speed (Float32) time multiplier
- /swarm/cmd/drone_altitude_m (Float32) python drone flight altitude (meters)
- /swarm/cmd/target_add_mode (Bool): when true, RViz /clicked_point adds targets
- /swarm/targets/add (PointStamped): add target directly
- /swarm/cmd/clear_targets (Trigger)

Persistence:
- data/base_pheromone_full.json
- data/pheromone_data.json (compat export: cell_x/cell_y/intensity)
- data/python_sim_stats.json
"""

from __future__ import annotations

# Allow running this file directly (not as a package module).
# When executed as `python3 scripts/python_sim/python_fast_sim.py`, Python sets sys.path[0]
# to `scripts/python_sim/` which breaks absolute imports like `scripts.python_sim.*`.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parents[2]))

import json
import math
import os
import random
import time
import threading
import heapq
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Set, Union
from collections import deque

from scripts.python_sim.dto.pheromones import CellMeta, SparseLayer, PheromoneMap
from scripts.python_sim.dto.sim_types import (
    Building,
    DroneState,
    DroneType,
    DynamicInspectStatus,
    EnergyModel,
    ExploreVector,
    GridSpec,
    Target,
    TargetKnowledge,
)

from scripts.python_sim.lib.helpers import (
    make_drone_uid,
    yaw_to_quat,
    clamp,
    MapBounds,
    bounds_minmax,
    out_of_bounds,
    clamp_xy,
    softmax_sample,
)
from scripts.python_sim.lib.environment import load_buildings_from_sdf, BuildingIndex
from scripts.python_sim.lib.drone_agent import PythonDrone

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PointStamped, PoseArray, Pose, Quaternion, Point, PoseStamped
from std_msgs.msg import Bool, ColorRGBA, Float32, String
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock


class PythonFastSim(Node):
    def __init__(self):
        super().__init__("python_fast_sim")
        # Callback groups:
        # - sim: heavy work + state mutation
        # - viz: publishers/timers that must keep flowing even when sim is heavy
        self._sim_cbg = MutuallyExclusiveCallbackGroup()
        self._viz_cbg = MutuallyExclusiveCallbackGroup()
        self._state_lock = threading.Lock()
        # Dedicated lock for comm visualization buffer (avoid contending with sim state lock).
        self._comm_lock = threading.Lock()
        # Ensure this node's timers are driven by wall time (not /clock). `use_sim_time` is a
        # standard ROS parameter that may already be declared by rclpy; never redeclare it.
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        except Exception:
            pass

        # Core parameters
        self.declare_parameter("num_py_drones", 15)
        self.declare_parameter("z_start", 8.0)
        # Constant flight altitude for Python drones (meters). This influences height-aware obstacles:
        # a building is an obstacle only if drone_altitude_m <= building_top + safety_margin_z.
        self.declare_parameter("drone_altitude_m", float(self.get_parameter("z_start").value))
        # Altitude control parameters (sim)
        self.declare_parameter("min_flight_altitude_m", 5.0)
        self.declare_parameter("roof_clearance_margin_m", 3.0)
        self.declare_parameter("max_overfly_altitude_m", 60.0)
        self.declare_parameter("climb_rate_mps", 4.0)
        self.declare_parameter("descend_rate_mps", 3.0)
        # Vertical speed realism: multiplier relative to horizontal speed (0.1..1.0).
        # If enabled, effective climb/descend rate becomes `vertical_speed_mult * current_horizontal_speed_mps`.
        self.declare_parameter("vertical_speed_mult_enabled", True)
        self.declare_parameter("vertical_speed_mult", 0.30)
        self.declare_parameter("stuck_progress_timeout_s", 3.0)
        self.declare_parameter("progress_eps_m", 2.0)
        self.declare_parameter("grid_size", 11000.0)
        self.declare_parameter("cell_size", 5.0)
        self.declare_parameter("sense_radius", 75.0)
        self.declare_parameter("comm_radius", 200.0)
        # Communication visualization (RViz wires) – purely visual, should not affect sim.
        self.declare_parameter("comm_viz_enabled", True)
        # Don't draw wires for very close drones (too cluttered / obvious).
        self.declare_parameter("comm_viz_min_dist_m", 30.0)
        self.declare_parameter("comm_viz_expire_s", 0.5)
        self.declare_parameter("comm_viz_max_lines", 1500)
        # Base sync comm visualization (when a drone syncs with base).
        self.declare_parameter("base_comm_viz_enabled", True)
        self.declare_parameter("base_comm_viz_expire_s", 0.8)
        self.declare_parameter("base_comm_viz_line_width", 1.2)
        # How to visualize comms:
        # - "events": draw short-lived wires when data exchange happens (existing behavior)
        # - "clusters": draw one star per connected component (centroid -> drones) every RViz publish
        self.declare_parameter("comm_viz_mode", "clusters")  # events|clusters
        self.declare_parameter("comm_viz_cluster_line_width", 0.25)
        self.declare_parameter("safety_margin_z", 2.0)
        self.declare_parameter("building_xy_margin", 2.0)
        self.declare_parameter("world_file", "worlds/city_map.sdf")
        # Prevent "pheromone sink" at base that can cause loitering after recharge
        self.declare_parameter("base_no_nav_radius_m", 30.0)  # ignore nav pheromone inside this radius (EXPLORE)
        self.declare_parameter("base_no_deposit_radius_m", 20.0)  # don't deposit nav pheromone inside this radius
        self.declare_parameter("base_push_radius_m", 60.0)  # within this radius, prefer moving outward (EXPLORE)
        self.declare_parameter("base_push_strength", 4.0)  # weight for outward push
        # If drones still orbit near base, enforce outward expansion until this radius.
        self.declare_parameter("explore_min_radius_m", 200.0)
        self.declare_parameter("explore_min_radius_strength", 10.0)
        # Exploration area constraint (circle around base). 0 disables.
        self.declare_parameter("exploration_area_radius_m", 0.0)
        self.declare_parameter("exploration_radius_margin_m", 30.0)
        # Pheromone-based exploration: reward low nav pheromone (less visited).
        self.declare_parameter("explore_low_nav_weight", 0.0)
        # Explore scoring: how attractive nav pheromone is during EXPLORE.
        self.declare_parameter("explore_nav_weight", 1.2)
        # Danger-boundary inspection curiosity (EXPLORE only).
        # Drives drones to "go around" already-detected danger to reveal its boundary band (via explored layer),
        # without making danger itself preferable (still penalized by danger pheromone).
        self.declare_parameter("danger_inspect_weight", 0.0)
        self.declare_parameter("danger_inspect_kernel_cells", 3)
        self.declare_parameter("danger_inspect_danger_thr", 0.35)
        self.declare_parameter("danger_inspect_max_cell_danger", 0.6)
        # Dynamic danger kernel-path curiosity (exclusive inspector per danger id).
        # - historical: uses pheromone/shared kernel info (works even without current LiDAR sighting)
        # - realtime: uses current LiDAR kernel sighting, targets a "standoff ring" just outside damage radius
        self.declare_parameter("dyn_danger_inspect_weight", 8.0)  # historical follow weight
        self.declare_parameter("dyn_inspector_rt_weight", 12.0)
        self.declare_parameter("dyn_inspector_rt_ttl_s", 2.0)
        self.declare_parameter("dyn_inspector_rt_standoff_cells", 1)
        # Inspector: penalty for stepping into dynamic danger *damage* trace while inspecting.
        self.declare_parameter("dyn_inspector_avoid_damage_weight", 25.0)
        self.declare_parameter("dyn_inspector_avoid_damage_thr", 0.05)
        # Far-ring low explored density shaping (EXPLORE): probe density only at distance X around drone.
        self.declare_parameter("explore_far_density_weight", 0.0)
        self.declare_parameter("explore_far_density_ring_radius_cells", 0)
        self.declare_parameter("explore_far_density_kernel_radius_cells", 3)
        self.declare_parameter("explore_far_density_angle_step_deg", 30.0)
        self.declare_parameter("explore_far_density_exclude_inside_ring", True)
        # Anti-crowding exploration: share and avoid peer exploration vectors.
        self.declare_parameter("share_explore_vectors_comm", True)
        self.declare_parameter("explore_vector_comm_max", 120)
        self.declare_parameter("explore_vector_share_every_cells", 3)
        self.declare_parameter("explore_vector_ttl_s", 120.0)
        self.declare_parameter("explore_vector_avoid_weight", 0.0)
        self.declare_parameter("explore_vector_spatial_gate_m", float(self.get_parameter("comm_radius").value))
        # Penalize revisiting recently visited cells (breaks small loops).
        self.declare_parameter("recent_cell_penalty", 2.0)
        # Extra loop-breaking: scale revisit penalty by repeat count inside the recent-cells window.
        self.declare_parameter("explore_revisit_penalty_repeat_mult", 0.0)
        # Suppress nav deposit on repeated revisits (EXPLORE only). 1.0 disables.
        self.declare_parameter("explore_revisit_nav_deposit_scale", 1.0)
        # Wall avoidance shaping (behavioral, not collision):
        # - prefer keeping this clearance from obstacles (unless in narrow corridor)
        # - start turning tangentially when a wall is detected ahead
        self.declare_parameter("wall_clearance_m", 15.0)
        self.declare_parameter("wall_clearance_weight", 4.0)
        self.declare_parameter("wall_avoid_start_factor", 2.0)  # start turning when wall is within factor*clearance
        self.declare_parameter("wall_avoid_yaw_weight", 2.0)  # bias towards chosen tangent heading
        self.declare_parameter("wall_corridor_relax", 0.25)  # relax clearance penalty in narrow corridors
        # Corner helper: allow shorter integration steps when the full step would collide near building corners.
        # This helps escape "stuck at corner" situations when dt or speed is large.
        self.declare_parameter("corner_backoff_enabled", True)
        # Escape helper: when boxed in, optionally do a short "jump away" move to nearest free space.
        # Disable for ablations or to force overfly/hop to resolve tight corners.
        self.declare_parameter("unstick_move_enabled", True)
        # Avoidance mode (crab): temporary sideways motion to bypass obstacles without "attracting" others.
        self.declare_parameter("avoid_enabled", True)
        self.declare_parameter("avoid_side_random_on_entry", True)
        self.declare_parameter("avoid_crab_offset_cells", 3.0)  # sideways offset before trying forward again
        self.declare_parameter("avoid_max_time_s", 8.0)
        self.declare_parameter("avoid_angular_tolerance_deg", 25.0)
        self.declare_parameter("avoid_deposit_danger_amount", 0.35)

        # Overfly cost model: prefer going around unless overfly is clearly cheaper.
        self.declare_parameter("overfly_vertical_cost_mult", 3.0)
        self.declare_parameter("overfly_descend_cost_factor", 0.7)
        self.declare_parameter("overfly_must_be_fraction_of_around", 0.7)  # overfly_cost <= frac * around_cost
        # Static danger altitude semantics:
        # - Penalize being below the configured danger altitude, proportionally to the deficit.
        # - Local A* uses an altitude-aware cost with quantized altitude states (step in meters).
        self.declare_parameter("static_danger_altitude_violation_weight", 6.0)
        self.declare_parameter("local_a_star_altitude_step_m", 0.5)
        # Energy impact of vertical motion (battery): multiplier relative to horizontal energy cost per meter.
        self.declare_parameter("vertical_energy_cost_mult", 3.0)
        # RETURN-mode tuning (helps drones re-use narrow gaps instead of getting stuck in danger fields)
        # If True, RETURN uses the same ACO-style local scoring as EXPLORE (but with RETURN-specific weights).
        # If False, use the legacy deterministic local A* return behavior.
        self.declare_parameter("return_use_aco_enabled", True)
        self.declare_parameter("return_progress_weight", 8.0)      # reward reducing distance-to-base
        self.declare_parameter("return_danger_weight", 3.5)        # penalize danger pheromone (softer than explore)
        self.declare_parameter("return_corridor_danger_relax", 0.25)  # reduce danger penalty in narrow corridors

        # Time / loop
        self.declare_parameter("tick_hz", 30.0)  # wall tick
        self.declare_parameter("speed", 10.0)  # sim-time multiplier (python only)
        # Anti-freeze: cap how much wall time the sim tick is allowed to consume.
        # 0.0 disables budgeting (old behavior).
        # When enabled, the sim may fall behind (sim_debt increases) but GUI/ROS callbacks stay responsive.
        self.declare_parameter("sim_tick_wall_budget_s", 0.0)
        # Overload helper: when sim_debt exceeds this threshold, skip expensive sensing for that tick.
        # 0.0 disables.
        self.declare_parameter("sim_overload_debt_s", 0.0)
        # Movement integration: limit how far a drone can move per simulation substep.
        # This prevents "teleporting" across many cells when speed multiplier is high.
        self.declare_parameter("max_move_per_step_m", 0.0)  # 0 => auto (=0.8*cell_size)
        # Hard cap on compute per wall tick; any remaining simulated time is carried over (debt).
        self.declare_parameter("max_sim_substeps_per_tick", 60)
        # Performance diagnostics
        self.declare_parameter("perf_enabled", True)
        self.declare_parameter("perf_log_period_s", 5.0)
        # Throttle expensive sensing to avoid lockups at high speed multipliers.
        # Throttle expensive sensing by WALL time so time-acceleration doesn't multiply CPU cost.
        self.declare_parameter("sense_period_s", 0.20)  # wall seconds between mock-lidar scans
        # Sensing scheduling mode:
        # - "wall": scan every sense_period_s wall-seconds (cheap but sim-speed changes scan rate in sim time)
        # - "sim": scan every sense_period_sim_s simulated seconds (behavior consistent at high speed)
        self.declare_parameter("sense_period_mode", "sim")  # sim|wall
        self.declare_parameter("sense_period_sim_s", 0.25)  # simulated seconds between scans (mode=sim)
        self.declare_parameter("max_sense_scans_per_tick", 2)  # per drone, per wall tick (mode=sim)
        # Inflate lidar hits in cells (for planning). 1 means 1-cell padding around detected obstacle cells.
        self.declare_parameter("lidar_inflate_cells", 1)
        # Local planner (A*) horizon in meters (defaults to sense_radius).
        self.declare_parameter("local_plan_radius_m", float(self.get_parameter("sense_radius").value))
        # RViz publishing throttling (MarkerArray). Default: 6 Hz (not more than 6 updates/sec).
        self.declare_parameter("rviz_publish_hz", 6.0)
        # Path visualization history length (per drone).
        self.declare_parameter("path_history_len", 4000)
        # Downsample path markers for RViz performance.
        self.declare_parameter("path_viz_max_points", 350)
        self.declare_parameter("aco_temperature", 0.7)
        # ACO commitment: keep moving to the chosen next cell unless new perception arrives.
        self.declare_parameter("aco_commit_enabled", True)
        self.declare_parameter("aco_commit_timeout_s", 5.0)
        # EXPLORE symmetry breaking (prevents drones flying in lockstep):
        self.declare_parameter("explore_personal_bias_weight", 0.35)
        self.declare_parameter("explore_score_noise", 0.08)
        # Optional exploration heuristic: prefer less-explored space.
        # explored is binary and persistent; this is a soft bias only (0 disables).
        self.declare_parameter("explore_avoid_explored_weight", 0.0)
        # Explored-aware exploration reward (age + observation distance).
        self.declare_parameter("explore_unexplored_reward_weight", 0.0)
        self.declare_parameter("explore_explored_age_weight", 0.0)
        self.declare_parameter("explore_explored_dist_weight", 0.0)
        # Backward-compat: previously named explore_avoid_empty_weight.
        self.declare_parameter("explore_avoid_empty_weight", 0.0)
        # Planning helper: treat cells within N cells of an empty cell as \"empty enough\" for selecting
        # a pierce goal (does NOT change what is stored in the empty layer).
        self.declare_parameter("empty_goal_dilate_cells", 2)
        # Empty helper: only mark empty around discovered walls (nav_danger).
        self.declare_parameter("empty_near_wall_radius_cells", 2)
        self.declare_parameter("beam_count", 72)
        self.declare_parameter("seed", 1337)
        self.declare_parameter("keep_base_between_runs", False)
        # Compatibility: publish /pheromone_grid_params for nodes like danger_map_manager.py.
        # If you already run pheromone_heatmap.py (which also publishes /pheromone_grid_params),
        # set this to false to avoid conflicting publishers and RViz "blinking".
        self.declare_parameter("publish_grid_params", True)
        self.declare_parameter("grid_params_publish_period_sec", 1.0)

        # Evaporation (per second, in simulated time)
        self.declare_parameter("evap_nav_rate", 0.002)
        self.declare_parameter("evap_danger_rate", 0.001)
        # Wall/nav_danger evaporation multiplier (smaller = longer-lasting "TTL-like" walls).
        self.declare_parameter("wall_danger_evap_mult", 0.02)
        # Static vs dynamic danger evaporation multipliers:
        # Dynamic threats can evaporate slightly faster than static threats.
        self.declare_parameter("danger_evap_mult_static", 1.0)
        self.declare_parameter("danger_evap_mult_dynamic", 1.25)

        # Persistence
        self.declare_parameter("base_full_path", "data/base_pheromone_full.json")
        self.declare_parameter("compat_export_path", "data/pheromone_data.json")
        self.declare_parameter("stats_path", "data/python_sim_stats.json")
        # Named pheromone snapshots (save/load from GUI).
        # Path is relative to scripts/ (workspace_root) unless absolute.
        self.declare_parameter("pheromone_snapshot_dir", "data/pheromone_snapshots")
        # Load an existing compat pheromone file at startup for visualization (no map edits required).
        self.declare_parameter("load_compat_on_startup", True)

        # EXPLOIT mode options
        # - False: dynamic danger pheromone trails are informational only (avoid only current footprint)
        # - True: treat dynamic-danger pheromone trails as static danger for scoring/avoidance
        self.declare_parameter("exploit_dynamic_danger_as_static", False)
        # When a target goal is selected during EXPLOIT, hide other targets in RViz.
        self.declare_parameter("exploit_hide_other_targets", True)

        # EXPLOIT (comparison): dynamic trail -> temporary "static overlay" cost.
        # In "treat dynamic as static", we don't delete dynamic pheromones; instead we add an extra
        # penalty derived from the dynamic trail intensity:
        # - high/red trail cells get much stronger penalty
        # - low/green trail cells get small penalty
        self.declare_parameter("exploit_dyn_trail_overlay_strength", 3.0)
        self.declare_parameter("exploit_dyn_trail_overlay_gamma", 1.8)

        # EXPLOIT: swarm spacing (repel from peers but still follow A* plan).
        self.declare_parameter("exploit_peer_avoid_radius_m", 40.0)
        self.declare_parameter("exploit_peer_avoid_weight", 2.0)
        # "Reward" for sticking to A* direction while applying peer avoidance. Higher -> less deviation.
        self.declare_parameter("exploit_peer_path_follow_weight", 1.0)
        # Fade peer-avoidance near the target (so arrivals don't fight each other).
        self.declare_parameter("exploit_peer_avoid_fade_start_m", 25.0)
        self.declare_parameter("exploit_peer_avoid_fade_range_m", 50.0)
        # Limit how much peer-avoidance can steer away from the A* direction.
        self.declare_parameter("exploit_peer_avoid_max_deviation_deg", 70.0)

        # EXPLOIT: approach + landing smoothing.
        self.declare_parameter("exploit_approach_radius_m", 75.0)  # circle radius around target for approach points
        self.declare_parameter("exploit_approach_jitter_m", 6.0)   # random xy jitter on approach points
        self.declare_parameter("exploit_approach_reach_m", 10.0)   # when closer than this -> switch to final target
        self.declare_parameter("exploit_land_trigger_m", 12.0)     # start descending when within this of final target
        self.declare_parameter("exploit_land_z_done_m", 0.6)       # consider landed below this altitude
        self.declare_parameter("exploit_landing_speed_mps", 3.0)   # slow horizontal speed while landing
        self.declare_parameter("exploit_yaw_rate_rad_s", 1.2)      # yaw-rate limit for smooth turns (rad/s)
        self.declare_parameter("exploit_slowdown_radius_m", 70.0)  # start slowing down within this range of goal

        # Topics/services
        # In the "one roof" workflow, the GUI routes /clicked_point exclusively to exactly one feature
        # (targets OR danger creation). So the fast sim should NOT subscribe to /clicked_point by default.
        self.declare_parameter("enable_clicked_point_target_add", False)

        # Pheromone map visualization (replaces manual pheromone_heatmap.py in the "one roof" workflow)
        self.declare_parameter("pheromone_viz_enabled", True)
        self.declare_parameter("pheromone_viz_source", "base_danger")  # base_danger|base_nav
        # New selection model (preferred): owner + layer
        self.declare_parameter("pheromone_viz_owner", "base")  # base|combined|drone
        # Layers:
        # - danger: negative pheromones (walls + hazards)
        # - nav: positive navigation pheromone
        # - empty: sparse safe-goal helper for A* (near walls)
        # - explored: lidar coverage (seen-by-lidar)
        self.declare_parameter("pheromone_viz_layer", "danger")  # danger|nav|empty|explored
        self.declare_parameter("pheromone_viz_danger_kind", "all")  # all|nav_danger
        self.declare_parameter("pheromone_viz_drone_seq", 1)
        self.declare_parameter("pheromone_viz_z", 0.10)
        self.declare_parameter("pheromone_viz_alpha", 0.10)
        # "display_size" is a VISUAL crop/extent only; it does NOT change stored grid mapping.
        self.declare_parameter("pheromone_viz_display_size", 2000.0)
        # Publish pheromone visualization rarely (RViz-only). Default: once per 2 seconds.
        self.declare_parameter("pheromone_viz_publish_hz", 0.5)
        # Optional batching: take a snapshot periodically and publish it in N batches
        # to avoid long stalls when the map is large.
        # Example: snapshot_period=1.0, batch_count=4, publish_hz=4.0 -> full refresh each second.
        self.declare_parameter("pheromone_viz_snapshot_period_s", 1.0)
        self.declare_parameter("pheromone_viz_batch_count", 1)
        # Hard cap on number of cells rendered (visual-only performance safety).
        # If there are more cells, we downsample for display.
        self.declare_parameter("pheromone_viz_max_points", 12000)
        # Auto-batch (visual-only): if too many points are being rendered and batch_count is 1,
        # switch internally to N batches (still controlled by snapshot_period_s).
        self.declare_parameter("pheromone_viz_auto_batch_threshold", 6000)
        self.declare_parameter("pheromone_viz_auto_batch_count", 8)
        self.declare_parameter("pheromone_viz_show_grid", True)
        self.declare_parameter("pheromone_viz_grid_alpha", 0.06)
        self.declare_parameter("pheromone_viz_grid_line_width", 0.06)
        # Rendering:
        # - cubes: CUBE_LIST (lit; can look darker at shallow camera angles)
        # - points: POINTS (unlit; colors look consistent from any camera angle)
        self.declare_parameter("pheromone_viz_cell_render_mode", "points")  # points|cubes
        self.declare_parameter("pheromone_viz_context_cells", 12)  # show empty cells only near filled ones
        self.declare_parameter("pheromone_viz_blank_cells_radius", 50)  # 100x100 around center when blank
        self.declare_parameter("pheromone_viz_border_line_width", 0.8)
        self.declare_parameter("pheromone_viz_border_alpha", 0.8)
        self.declare_parameter("pheromone_viz_background_alpha", 0.0)  # transparent background plane

        # Target visualization (visual-only; does NOT affect detection radius)
        self.declare_parameter("target_viz_diameter", 10.0)  # meters
        self.declare_parameter("target_viz_alpha", 0.30)  # 0..1
        # Exploration area visualization (RViz-only)
        self.declare_parameter("exploration_radius_viz_enabled", True)
        self.declare_parameter("exploration_radius_viz_period_s", 15.0)
        self.declare_parameter("exploration_radius_viz_z", 0.03)
        self.declare_parameter("exploration_radius_viz_alpha", 0.55)
        self.declare_parameter("exploration_radius_viz_line_width", 0.8)

        # Drone marker scaling (RViz-only)
        self.declare_parameter("drone_marker_scale", 1.0)  # multiplier
        # Return-to-base speed (m/s) for python drones (when not in low-energy)
        self.declare_parameter("return_speed_mps", 10.0)

        # Target persistence
        self.declare_parameter("persist_targets", True)
        self.declare_parameter("targets_path", "data/targets.json")
        self.declare_parameter("targets_autosave_sec", 1.0)

        # Target knowledge sharing
        self.declare_parameter("share_targets_comm", True)
        self.declare_parameter("share_targets_at_base", True)
        self.declare_parameter("targets_comm_max", 200)  # max updates per exchange
        self.declare_parameter("targets_comm_radius", 200.0)

        # Pheromone sync with base (so "Base" pheromone visualization is meaningful)
        self.declare_parameter("share_pheromones_at_base", True)
        self.declare_parameter("base_pheromone_sync_radius_m", 10.0)
        self.declare_parameter("base_pheromone_sync_cooldown_s", 1.0)
        self.declare_parameter("download_pheromones_from_base", True)
        self.declare_parameter("base_pheromone_download_max_cells", 6000)  # per layer, per sync (newest cells)

        # Drone pointers (RViz)
        self.declare_parameter("drone_pointer_enabled", True)
        self.declare_parameter("drone_pointer_z", 8.0)
        self.declare_parameter("drone_pointer_scale", 1.0)
        self.declare_parameter("drone_pointer_alpha", 0.9)
        # Lidar debug visualization (per-drone known obstacle cells)
        self.declare_parameter("lidar_viz_enabled", False)
        self.declare_parameter("lidar_viz_drone_seq", 1)
        self.declare_parameter("lidar_viz_alpha", 0.9)
        self.declare_parameter("lidar_viz_line_width", 0.20)
        self.declare_parameter("lidar_viz_corner_size", 0.6)
        self.declare_parameter("lidar_viz_ttl_s", 1.0)
        # Planned path debug (local A* plan)
        self.declare_parameter("plan_viz_enabled", False)
        self.declare_parameter("plan_viz_drone_seq", 1)
        self.declare_parameter("plan_viz_alpha", 0.9)
        self.declare_parameter("plan_viz_line_width", 0.30)
        # Only show plan for a short time after it was created (otherwise last_plan_world looks "always on").
        self.declare_parameter("plan_viz_ttl_s", 2.5)
        # ACO decision visualization (chosen heading + optional candidate rays)
        self.declare_parameter("aco_viz_enabled", False)
        self.declare_parameter("aco_viz_drone_seq", 1)  # 0 => ALL
        self.declare_parameter("aco_viz_alpha", 0.55)
        self.declare_parameter("aco_viz_line_width", 0.10)
        self.declare_parameter("aco_viz_show_candidates", True)
        self.declare_parameter("aco_viz_top_k", 6)
        self.declare_parameter("aco_viz_ttl_s", 0.75)
        # Lift ACO viz above drone arrows so it stays visible.
        self.declare_parameter("aco_viz_z_offset_m", 0.75)
        # Optional: show candidate-cell score heatmap (small cubes).
        self.declare_parameter("aco_viz_show_heatmap", False)
        self.declare_parameter("aco_viz_heatmap_top_k", 12)
        self.declare_parameter("aco_viz_heatmap_scale_m", 1.75)
        # Keep ACO heatmap cells for a short wall-time trail (seconds). 0 disables history.
        self.declare_parameter("aco_viz_heatmap_history_s", 3.0)
        # Optional: show a single "best-step" orange pillar (trail) for ACO.
        self.declare_parameter("aco_viz_show_best_pillar", False)
        self.declare_parameter("aco_viz_best_pillar_history_s", 3.0)
        self.declare_parameter("aco_viz_best_pillar_line_width", 0.18)
        # ACO arrow shape controls (for readability in RViz).
        # - length_mult scales the arrow length relative to the chosen next-step vector.
        # - width_m controls shaft thickness (meters). Head size scales from width.
        self.declare_parameter("aco_viz_arrow_length_mult", 1.6)
        self.declare_parameter("aco_viz_arrow_width_m", 0.35)
        # Lidar scan visualization (beams + hit points)
        self.declare_parameter("lidar_scan_viz_enabled", False)
        self.declare_parameter("lidar_scan_viz_drone_seq", 1)
        self.declare_parameter("lidar_scan_viz_alpha", 0.35)
        self.declare_parameter("lidar_scan_viz_line_width", 0.05)
        self.declare_parameter("lidar_scan_viz_beam_stride", 2)
        self.declare_parameter("lidar_scan_viz_ttl_s", 0.75)

        # State
        self.running = False
        self.paused = False
        self.returning = False  # return-to-base phase requested (still simulated time)
        self.mission_phase = "EXPLORE"
        # Optional: GUI-selected concrete target to pursue (goal-seeking without RETURN).
        self.selected_target_id = ""
        self.base_xy = (0.0, 0.0)
        self.t_sim = 0.0
        self._last_wall = time.time()

        # GUI-facing event log (included in /swarm/gui_status).
        self.gui_log: str = ""

        # Exploit-run bookkeeping (for stats + automatic stop).
        self._exploit_active: bool = False
        self._exploit_start_t: float = 0.0
        self._exploit_active_uids: Set[str] = set()
        self._exploit_arrived_uids: Set[str] = set()
        # Per-run exploit statistics (reset on each exploit start):
        # uid -> {"start_t": float, "start_energy": float, "vert_m": float, "land_t": Optional[float], "land_energy": Optional[float]}
        self._exploit_stats_by_uid: Dict[str, dict] = {}

        # Config
        self.num_py = int(self.get_parameter("num_py_drones").value)
        self.z_start = float(self.get_parameter("z_start").value)
        self.drone_altitude_m = float(self.get_parameter("drone_altitude_m").value)
        self.min_flight_altitude_m = float(self.get_parameter("min_flight_altitude_m").value)
        self.roof_clearance_margin_m = float(self.get_parameter("roof_clearance_margin_m").value)
        self.max_overfly_altitude_m = float(self.get_parameter("max_overfly_altitude_m").value)
        self.climb_rate_mps = float(self.get_parameter("climb_rate_mps").value)
        self.descend_rate_mps = float(self.get_parameter("descend_rate_mps").value)
        self.vertical_speed_mult_enabled = bool(self.get_parameter("vertical_speed_mult_enabled").value)
        self.vertical_speed_mult = float(self.get_parameter("vertical_speed_mult").value)
        self.stuck_progress_timeout_s = float(self.get_parameter("stuck_progress_timeout_s").value)
        self.progress_eps_m = float(self.get_parameter("progress_eps_m").value)
        self.grid = GridSpec(
            grid_size_m=float(self.get_parameter("grid_size").value),
            cell_size_m=float(self.get_parameter("cell_size").value),
        )
        self.sense_radius = float(self.get_parameter("sense_radius").value)
        self.comm_radius = float(self.get_parameter("comm_radius").value)
        self.comm_viz_enabled = bool(self.get_parameter("comm_viz_enabled").value)
        self.comm_viz_min_dist_m = float(self.get_parameter("comm_viz_min_dist_m").value)
        self.comm_viz_expire_s = float(self.get_parameter("comm_viz_expire_s").value)
        self.comm_viz_max_lines = int(self.get_parameter("comm_viz_max_lines").value)
        self.base_comm_viz_enabled = bool(self.get_parameter("base_comm_viz_enabled").value)
        self.base_comm_viz_expire_s = float(self.get_parameter("base_comm_viz_expire_s").value)
        self.base_comm_viz_line_width = float(self.get_parameter("base_comm_viz_line_width").value)
        self.comm_viz_mode = str(self.get_parameter("comm_viz_mode").value).strip().lower()
        self.comm_viz_cluster_line_width = float(self.get_parameter("comm_viz_cluster_line_width").value)
        self.safety_margin_z = float(self.get_parameter("safety_margin_z").value)
        self.beam_count = int(self.get_parameter("beam_count").value)
        self.aco_temperature = float(self.get_parameter("aco_temperature").value)
        self.explore_personal_bias_weight = float(self.get_parameter("explore_personal_bias_weight").value)
        self.explore_score_noise = float(self.get_parameter("explore_score_noise").value)
        self.explore_avoid_explored_weight = float(self.get_parameter("explore_avoid_explored_weight").value)
        self.explore_avoid_empty_weight = float(self.get_parameter("explore_avoid_empty_weight").value)
        self.explore_unexplored_reward_weight = float(self.get_parameter("explore_unexplored_reward_weight").value)
        self.explore_explored_age_weight = float(self.get_parameter("explore_explored_age_weight").value)
        self.explore_explored_dist_weight = float(self.get_parameter("explore_explored_dist_weight").value)
        self.empty_goal_dilate_cells = int(self.get_parameter("empty_goal_dilate_cells").value)
        self.empty_near_wall_radius_cells = int(self.get_parameter("empty_near_wall_radius_cells").value)
        self.map_bounds = self.grid.half - self.grid.cell_size_m
        self.lidar_inflate_cells = int(self.get_parameter("lidar_inflate_cells").value)
        self.local_plan_radius_m = float(self.get_parameter("local_plan_radius_m").value)
        self.path_history_len = int(self.get_parameter("path_history_len").value)
        self.path_viz_max_points = int(self.get_parameter("path_viz_max_points").value)
        self.sense_period_mode = str(self.get_parameter("sense_period_mode").value).strip().lower()
        self.sense_period_sim_s = float(self.get_parameter("sense_period_sim_s").value)
        self.max_sense_scans_per_tick = int(self.get_parameter("max_sense_scans_per_tick").value)
        self.perf_enabled = bool(self.get_parameter("perf_enabled").value)
        self.perf_log_period_s = float(self.get_parameter("perf_log_period_s").value)

        @dataclass
        class _PerfAgg:
            n: int = 0
            total_s: float = 0.0
            max_s: float = 0.0

            def add(self, dt: float):
                self.n += 1
                self.total_s += float(dt)
                if dt > self.max_s:
                    self.max_s = float(dt)

        self._PerfAgg = _PerfAgg
        self._perf: Dict[str, _PerfAgg] = {}
        self._perf_last_log_wall = time.time()
        self._perf_tick_n = 0
        self._perf_sim_advanced_s = 0.0
        self._perf_sim_debt_s = 0.0
        self._perf_substeps = 0
        # Optional: only allow speed changes while paused (prevents runaway sim_debt when tuning live).
        self._pending_speed: Optional[float] = None
        self.base_no_nav_radius_m = float(self.get_parameter("base_no_nav_radius_m").value)
        self.base_no_deposit_radius_m = float(self.get_parameter("base_no_deposit_radius_m").value)
        self.base_push_radius_m = float(self.get_parameter("base_push_radius_m").value)
        self.base_push_strength = float(self.get_parameter("base_push_strength").value)
        self.explore_min_radius_m = float(self.get_parameter("explore_min_radius_m").value)
        self.explore_min_radius_strength = float(self.get_parameter("explore_min_radius_strength").value)
        self.exploration_area_radius_m = float(self.get_parameter("exploration_area_radius_m").value)
        self.exploration_radius_margin_m = float(self.get_parameter("exploration_radius_margin_m").value)
        self.explore_low_nav_weight = float(self.get_parameter("explore_low_nav_weight").value)
        self.explore_nav_weight = float(self.get_parameter("explore_nav_weight").value)
        self.danger_inspect_weight = float(self.get_parameter("danger_inspect_weight").value)
        self.danger_inspect_kernel_cells = int(self.get_parameter("danger_inspect_kernel_cells").value)
        self.danger_inspect_danger_thr = float(self.get_parameter("danger_inspect_danger_thr").value)
        self.danger_inspect_max_cell_danger = float(self.get_parameter("danger_inspect_max_cell_danger").value)
        self.dyn_danger_inspect_weight = float(self.get_parameter("dyn_danger_inspect_weight").value)
        self.dyn_inspector_rt_weight = float(self.get_parameter("dyn_inspector_rt_weight").value)
        self.dyn_inspector_rt_ttl_s = float(self.get_parameter("dyn_inspector_rt_ttl_s").value)
        self.dyn_inspector_rt_standoff_cells = int(self.get_parameter("dyn_inspector_rt_standoff_cells").value)
        self.dyn_inspector_avoid_damage_weight = float(self.get_parameter("dyn_inspector_avoid_damage_weight").value)
        self.dyn_inspector_avoid_damage_thr = float(self.get_parameter("dyn_inspector_avoid_damage_thr").value)
        self.explore_far_density_weight = float(self.get_parameter("explore_far_density_weight").value)
        self.explore_far_density_ring_radius_cells = int(self.get_parameter("explore_far_density_ring_radius_cells").value)
        self.explore_far_density_kernel_radius_cells = int(self.get_parameter("explore_far_density_kernel_radius_cells").value)
        self.explore_far_density_angle_step_deg = float(self.get_parameter("explore_far_density_angle_step_deg").value)
        self.explore_far_density_exclude_inside_ring = bool(self.get_parameter("explore_far_density_exclude_inside_ring").value)
        self.share_explore_vectors_comm = bool(self.get_parameter("share_explore_vectors_comm").value)
        self.explore_vector_comm_max = int(self.get_parameter("explore_vector_comm_max").value)
        self.explore_vector_share_every_cells = int(self.get_parameter("explore_vector_share_every_cells").value)
        self.explore_vector_ttl_s = float(self.get_parameter("explore_vector_ttl_s").value)
        self.explore_vector_avoid_weight = float(self.get_parameter("explore_vector_avoid_weight").value)
        self.explore_vector_spatial_gate_m = float(self.get_parameter("explore_vector_spatial_gate_m").value)
        self.recent_cell_penalty = float(self.get_parameter("recent_cell_penalty").value)
        self.explore_revisit_penalty_repeat_mult = float(self.get_parameter("explore_revisit_penalty_repeat_mult").value)
        self.explore_revisit_nav_deposit_scale = float(self.get_parameter("explore_revisit_nav_deposit_scale").value)
        self.wall_clearance_m = float(self.get_parameter("wall_clearance_m").value)
        self.wall_clearance_weight = float(self.get_parameter("wall_clearance_weight").value)
        self.wall_avoid_start_factor = float(self.get_parameter("wall_avoid_start_factor").value)
        self.wall_avoid_yaw_weight = float(self.get_parameter("wall_avoid_yaw_weight").value)
        self.wall_corridor_relax = float(self.get_parameter("wall_corridor_relax").value)
        self.corner_backoff_enabled = bool(self.get_parameter("corner_backoff_enabled").value)
        self.unstick_move_enabled = bool(self.get_parameter("unstick_move_enabled").value)
        self.avoid_enabled = bool(self.get_parameter("avoid_enabled").value)
        self.avoid_side_random_on_entry = bool(self.get_parameter("avoid_side_random_on_entry").value)
        self.avoid_crab_offset_cells = float(self.get_parameter("avoid_crab_offset_cells").value)
        self.avoid_max_time_s = float(self.get_parameter("avoid_max_time_s").value)
        self.avoid_angular_tolerance_deg = float(self.get_parameter("avoid_angular_tolerance_deg").value)
        self.avoid_deposit_danger_amount = float(self.get_parameter("avoid_deposit_danger_amount").value)
        self.overfly_vertical_cost_mult = float(self.get_parameter("overfly_vertical_cost_mult").value)
        self.overfly_descend_cost_factor = float(self.get_parameter("overfly_descend_cost_factor").value)
        self.overfly_must_be_fraction_of_around = float(self.get_parameter("overfly_must_be_fraction_of_around").value)
        self.static_danger_altitude_violation_weight = float(self.get_parameter("static_danger_altitude_violation_weight").value)
        self.local_a_star_altitude_step_m = float(self.get_parameter("local_a_star_altitude_step_m").value)
        self.vertical_energy_cost_mult = float(self.get_parameter("vertical_energy_cost_mult").value)
        self.return_use_aco_enabled = bool(self.get_parameter("return_use_aco_enabled").value)
        self.return_progress_weight = float(self.get_parameter("return_progress_weight").value)
        self.return_danger_weight = float(self.get_parameter("return_danger_weight").value)
        self.return_corridor_danger_relax = float(self.get_parameter("return_corridor_danger_relax").value)

        self.evap_nav_rate = float(self.get_parameter("evap_nav_rate").value)
        self.evap_danger_rate = float(self.get_parameter("evap_danger_rate").value)
        self.wall_danger_evap_mult = float(self.get_parameter("wall_danger_evap_mult").value)
        self.danger_evap_mult_static = float(self.get_parameter("danger_evap_mult_static").value)
        self.danger_evap_mult_dynamic = float(self.get_parameter("danger_evap_mult_dynamic").value)

        self.energy_model = EnergyModel()

        # Load buildings (height-aware obstacles)
        # NOTE: this file lives in scripts/python_sim/, so repo root is parents[2].
        repo_root = Path(__file__).resolve().parents[2]
        world_path = repo_root / str(self.get_parameter("world_file").value)
        buildings = load_buildings_from_sdf(world_path)
        self.declare_parameter("building_bucket_size_m", 50.0)
        # World boundary: allow rectangular bounds derived from buildings.
        self.declare_parameter("map_bounds_from_buildings", True)
        self.declare_parameter("map_border_padding_m", 20.0)
        self.building_index = BuildingIndex(
            buildings,
            margin_m=float(self.get_parameter("building_xy_margin").value),
            bucket_size_m=float(self.get_parameter("building_bucket_size_m").value),
        )
        self.get_logger().info(f"Loaded {len(buildings)} buildings from {world_path}")

        # Rectangular map bounds based on building extents (looks correct for non-square worlds).
        try:
            if bool(self.get_parameter("map_bounds_from_buildings").value) and len(buildings) > 0:
                pad = float(self.get_parameter("map_border_padding_m").value)
                # Start from grid bounds as a hard safety envelope.
                grid_min = -float(self.grid.half) + float(self.grid.cell_size_m)
                grid_max = float(self.grid.half) - float(self.grid.cell_size_m)
                minx = 1e18
                maxx = -1e18
                miny = 1e18
                maxy = -1e18
                for b in buildings:
                    bx0, bx1, by0, by1 = b.bbox_xy(self.building_index.margin_m)
                    minx = min(minx, float(bx0))
                    maxx = max(maxx, float(bx1))
                    miny = min(miny, float(by0))
                    maxy = max(maxy, float(by1))
                # expand + clamp to grid
                minx = clamp(minx - pad, grid_min, grid_max)
                maxx = clamp(maxx + pad, grid_min, grid_max)
                miny = clamp(miny - pad, grid_min, grid_max)
                maxy = clamp(maxy + pad, grid_min, grid_max)
                # Ensure base is inside.
                minx = min(minx, float(self.base_xy[0]) - pad)
                maxx = max(maxx, float(self.base_xy[0]) + pad)
                miny = min(miny, float(self.base_xy[1]) - pad)
                maxy = max(maxy, float(self.base_xy[1]) + pad)
                self.map_bounds = (float(minx), float(maxx), float(miny), float(maxy))
                self.get_logger().info(f"Map bounds set to rectangle: minx={minx:.1f} maxx={maxx:.1f} miny={miny:.1f} maxy={maxy:.1f}")
        except Exception:
            pass

        # Targets
        self.targets: List[Target] = []
        self.target_add_mode = False
        self._targets_dirty = False
        self._targets_version = 0

        # Pheromones
        self.base_map = PheromoneMap(owner_uid="BASE-0")

        # Drones
        self.rng = random.Random(int(self.get_parameter("seed").value))
        self.drones: List[PythonDrone] = []
        for i in range(1, self.num_py + 1):
            uid = make_drone_uid("PY", i)
            st = DroneState(
                drone_uid=uid,
                drone_type="PY",
                seq=i,
                x=self.base_xy[0],
                y=self.base_xy[1],
                z=self.drone_altitude_m,
                speed_mps=self.energy_model.normal_speed_mps,
                energy_units=self.energy_model.full_units,
                mode="IDLE",
            )
            st.z_cruise = float(self.drone_altitude_m)
            st.z_target = float(self.drone_altitude_m)
            st.last_progress_t = 0.0
            st.last_progress_dist = float("inf")
            st.recent_cells.append(self.grid.world_to_cell(st.x, st.y))
            pher = PheromoneMap(owner_uid=uid)
            agent = PythonDrone(state=st, grid=self.grid, pher=pher, energy=self.energy_model, rng=random.Random(1000 + i))
            # Inject sim-level tuning for consistent planning/scoring across drones.
            agent.static_danger_altitude_violation_weight = float(self.static_danger_altitude_violation_weight)
            agent.local_a_star_altitude_step_m = float(self.local_a_star_altitude_step_m)
            agent.planning_vertical_cost_mult = float(self.overfly_vertical_cost_mult)
            agent.planning_descend_cost_factor = float(self.overfly_descend_cost_factor)
            self.drones.append(agent)

        # Publishers
        self.pub_drones = self.create_publisher(MarkerArray, "/swarm/markers/drones", 10)
        self.pub_paths = self.create_publisher(MarkerArray, "/swarm/markers/paths", 10)
        self.pub_targets = self.create_publisher(MarkerArray, "/swarm/markers/targets", 10)
        self.pub_exploration_radius = self.create_publisher(Marker, "/swarm/markers/exploration_radius", 10)
        self.pub_comm = self.create_publisher(MarkerArray, "/swarm/markers/comm", 10)
        self.pub_lidar = self.create_publisher(MarkerArray, "/swarm/markers/lidar", 10)
        self.pub_plan = self.create_publisher(MarkerArray, "/swarm/markers/plan", 10)
        self.pub_aco = self.create_publisher(MarkerArray, "/swarm/markers/aco", 10)
        self.pub_lidar_scan = self.create_publisher(MarkerArray, "/swarm/markers/lidar_scan", 10)
        self.pub_poses = self.create_publisher(PoseArray, "/swarm/drones/poses", 10)
        self.pub_phase = self.create_publisher(String, "/swarm/mission_phase", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        # Latched so late-joiners (e.g., DangerMapManager) get grid params immediately.
        grid_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.pub_grid_params = self.create_publisher(Float64MultiArray, "/pheromone_grid_params", grid_qos)
        self.pub_pheromone = self.create_publisher(Marker, "/pheromone_heatmap", 10)
        self.pub_gui_status = self.create_publisher(String, "/swarm/gui_status", 10)

        # Control subscribers
        self.sub_speed = self.create_subscription(Float32, "/swarm/cmd/speed", self._on_speed, 10, callback_group=self._sim_cbg)
        self.sub_target_mode = self.create_subscription(Bool, "/swarm/cmd/target_add_mode", self._on_target_mode, 10, callback_group=self._sim_cbg)
        self.sub_add_target = self.create_subscription(PointStamped, "/swarm/targets/add", self._on_add_target, 10, callback_group=self._sim_cbg)

        # Target maintenance helpers (GUI convenience):
        # - Remember last "pose" (e.g. from RViz publish_pose / 2D Nav Goal) so we can delete nearest target.
        self.declare_parameter("delete_target_pose_topic", "/move_base_simple/goal")
        self._last_delete_pose: Optional[PoseStamped] = None
        try:
            pose_topic = str(self.get_parameter("delete_target_pose_topic").value)
            self.sub_delete_pose = self.create_subscription(
                PoseStamped,
                pose_topic,
                self._on_delete_pose,
                10,
                callback_group=self._viz_cbg,
            )
        except Exception:
            self.sub_delete_pose = None

        if bool(self.get_parameter("enable_clicked_point_target_add").value):
            self.sub_clicked = self.create_subscription(PointStamped, "/clicked_point", self._on_clicked_point, 10, callback_group=self._sim_cbg)
        else:
            self.sub_clicked = None

        # Services
        self.srv_start = self.create_service(Trigger, "/swarm/cmd/start_python", self._srv_start, callback_group=self._sim_cbg)
        self.srv_stop = self.create_service(Trigger, "/swarm/cmd/stop_python", self._srv_stop, callback_group=self._sim_cbg)
        self.srv_clear_targets = self.create_service(Trigger, "/swarm/cmd/clear_targets", self._srv_clear_targets, callback_group=self._sim_cbg)
        self.srv_clear_targets_found = self.create_service(
            Trigger, "/swarm/cmd/clear_targets_found", self._srv_clear_targets_found, callback_group=self._sim_cbg
        )
        self.srv_clear_targets_unfound = self.create_service(
            Trigger, "/swarm/cmd/clear_targets_unfound", self._srv_clear_targets_unfound, callback_group=self._sim_cbg
        )
        self.srv_set_all_targets_unfound = self.create_service(
            Trigger, "/swarm/cmd/set_all_targets_unfound", self._srv_set_all_targets_unfound, callback_group=self._sim_cbg
        )
        self.srv_delete_nearest_target = self.create_service(
            Trigger, "/swarm/cmd/delete_nearest_target", self._srv_delete_nearest_target, callback_group=self._sim_cbg
        )

        # Pheromone viz control (from GUI)
        self.sub_pher_viz_enable = self.create_subscription(Bool, "/swarm/cmd/pheromone_viz_enable", self._on_pheromone_viz_enable, 10, callback_group=self._sim_cbg)
        self.sub_pher_viz_params = self.create_subscription(Float64MultiArray, "/swarm/cmd/pheromone_viz_params", self._on_pheromone_viz_params, 10, callback_group=self._sim_cbg)
        self.sub_target_viz_params = self.create_subscription(Float64MultiArray, "/swarm/cmd/target_viz_params", self._on_target_viz_params, 10, callback_group=self._sim_cbg)
        self.sub_pause = self.create_subscription(Bool, "/swarm/cmd/pause_python", self._on_pause, 10, callback_group=self._sim_cbg)
        self.sub_drone_scale = self.create_subscription(Float32, "/swarm/cmd/drone_marker_scale", self._on_drone_scale, 10, callback_group=self._sim_cbg)
        self.sub_pher_select = self.create_subscription(String, "/swarm/cmd/pheromone_viz_select", self._on_pheromone_viz_select, 10, callback_group=self._sim_cbg)
        self.sub_pointer_params = self.create_subscription(Float64MultiArray, "/swarm/cmd/drone_pointer_params", self._on_pointer_params, 10, callback_group=self._sim_cbg)
        self.sub_return_speed = self.create_subscription(Float32, "/swarm/cmd/return_speed_mps", self._on_return_speed, 10, callback_group=self._sim_cbg)
        self.sub_drone_alt = self.create_subscription(Float32, "/swarm/cmd/drone_altitude_m", self._on_drone_altitude, 10, callback_group=self._sim_cbg)
        # Optional GUI control (works both in-proc and external GUIs): set vertical speed multiplier.
        self.sub_vert_speed_mult = self.create_subscription(Float32, "/swarm/cmd/vertical_speed_mult", self._on_vertical_speed_mult, 10, callback_group=self._sim_cbg)
        # Optional GUI control: select a concrete target by id for goal-seeking (EXPLOIT w/o RETURN).
        self.sub_selected_target = self.create_subscription(String, "/swarm/cmd/selected_target", self._on_selected_target, 10, callback_group=self._sim_cbg)
        # Pheromone map snapshot save/load (GUI).
        # Format (JSON in String):
        # - {"cmd":"save","name":"my_snapshot"}
        # - {"cmd":"load","path":"/abs/or/rel/path.json"}  (relative paths resolve against scripts/)
        self.sub_pheromone_map_storage = self.create_subscription(
            String, "/swarm/cmd/pheromone_map_storage", self._on_pheromone_map_storage, 10, callback_group=self._sim_cbg
        )
        # Start an exploit run (GUI). JSON:
        # {"target_id":"T1","drone_count":3,"dynamic_mode":"handled"|"static"}
        self.sub_exploit_start = self.create_subscription(String, "/swarm/cmd/exploit_start", self._on_exploit_start, 10, callback_group=self._sim_cbg)
        # Lidar debug control (from GUI): JSON {"enabled":true,"drone_seq":1}
        self.sub_lidar_viz = self.create_subscription(String, "/swarm/cmd/lidar_viz_select", self._on_lidar_viz_select, 10, callback_group=self._sim_cbg)
        self.sub_plan_viz = self.create_subscription(String, "/swarm/cmd/plan_viz_select", self._on_plan_viz_select, 10, callback_group=self._sim_cbg)
        self.sub_aco_viz = self.create_subscription(String, "/swarm/cmd/aco_viz_select", self._on_aco_viz_select, 10, callback_group=self._sim_cbg)
        self.sub_lidar_scan_viz = self.create_subscription(String, "/swarm/cmd/lidar_scan_viz_select", self._on_lidar_scan_viz_select, 10, callback_group=self._sim_cbg)

        # Danger map integration:
        # - Consume /danger_map (published by danger_map_manager.py) as "ground truth hazards".
        # - Drones only learn hazards when their (mock) lidar observes cells; observation deposits danger pheromone.
        self.declare_parameter("danger_map_in_lidar_enabled", True)
        self.declare_parameter("danger_map_topic", "/danger_map")
        self.declare_parameter("danger_map_cells_topic", "/danger_map_cells")

        # Dynamic threat visualization + decision-making (default ON).
        self.declare_parameter("dynamic_threat_viz_enabled", True)
        self.declare_parameter("dynamic_threat_viz_show_path", True)
        self.declare_parameter("dynamic_threat_viz_show_spray", True)
        self.declare_parameter("dynamic_threat_viz_show_speed_text", True)
        self.declare_parameter("dynamic_threat_decision_enabled", True)
        # If drone arrives at the crossing cell at least this many seconds before threat, we allow crossing.
        self.declare_parameter("dynamic_threat_cross_margin_s", 0.5)
        # How strongly to penalize moves that would be intercepted by a dynamic threat.
        self.declare_parameter("dynamic_threat_avoid_weight", 6.0)

        self._danger_cells_lock = threading.Lock()
        self._danger_cells: Set[Tuple[int, int]] = set()
        # Source-cell metadata from DangerMapManager (/danger_map_cells):
        # key=(cell_x,cell_y), value={"radius": int, "kind": str, "id": str}
        self._danger_sources: Dict[Tuple[int, int], dict] = {}
        # Rich dynamic danger info (id -> data with path/speed/radius/current_index).
        self._dynamic_dangers: Dict[str, dict] = {}
        try:
            danger_topic = str(self.get_parameter("danger_map_topic").value)
            self.sub_danger_map = self.create_subscription(
                Marker,
                danger_topic,
                self._on_danger_map_marker,
                10,
                callback_group=self._viz_cbg,
            )
            self.sub_danger_map_cells = self.create_subscription(
                String,
                str(self.get_parameter("danger_map_cells_topic").value),
                self._on_danger_map_cells,
                10,
                callback_group=self._viz_cbg,
            )
        except Exception:
            self.sub_danger_map = None
            self.sub_danger_map_cells = None

        # Timer
        tick_hz = float(self.get_parameter("tick_hz").value)
        self._timer = self.create_timer(1.0 / max(1.0, tick_hz), self._tick, callback_group=self._sim_cbg)
        self._grid_timer = None
        if bool(self.get_parameter("publish_grid_params").value):
            period = float(self.get_parameter("grid_params_publish_period_sec").value)
            self._grid_timer = self.create_timer(max(0.2, period), self._publish_grid_params, callback_group=self._viz_cbg)
        # publish once at startup so subscribers can initialize immediately
        self._publish_grid_params()

        # Pheromone visualization timer
        self._pher_timer = self.create_timer(
            1.0 / max(0.2, float(self.get_parameter("pheromone_viz_publish_hz").value)),
            self._publish_pheromone_viz,
            callback_group=self._viz_cbg,
        )
        # Pheromone viz batching state (purely visual; ok if slightly stale)
        self._pher_snapshot_wall_t = 0.0
        self._pher_snapshot_key = None  # type: ignore
        self._pher_snapshot_batches_points: List[List["Point"]] = []
        self._pher_snapshot_batches_colors: List[List["ColorRGBA"]] = []
        self._pher_snapshot_batch_i = 0
        self._pher_last_batch_count = int(self.get_parameter("pheromone_viz_batch_count").value)
        # ACO viz: unique id counter for heatmap history markers.
        self._aco_heat_id = 0
        self._aco_pillar_id = 0
        self._aco_viz_prev_enabled = False

        # Path viz cache (keep history per drone) — must exist before RViz timer fires.
        maxlen = max(50, int(self.path_history_len))
        self.path_hist: Dict[str, Deque[Tuple[float, float, float]]] = {d.s.drone_uid: deque(maxlen=maxlen) for d in self.drones}

        # Comms visualization: list of (t_expire, (x1,y1,z1,x2,y2,z2))
        self.comm_lines: Deque[Tuple[float, Tuple[float, float, float, float, float, float]]] = deque(maxlen=2000)
        # Base sync comm visualization: short-lived lines base<->drone during sync.
        self.base_comm_lines: Deque[Tuple[float, Tuple[float, float, float, float, float, float]]] = deque(maxlen=2000)

        # RViz publishing timer (throttled)
        self._last_t_ref = 0.0
        self._sim_debt = 0.0
        rviz_hz = float(self.get_parameter("rviz_publish_hz").value)
        self._rviz_timer = self.create_timer(1.0 / max(0.01, rviz_hz), self._publish_rviz_timer, callback_group=self._viz_cbg)

        # Optionally load compat pheromone file so visualization works even before running sim
        if bool(self.get_parameter("load_compat_on_startup").value):
            self._load_compat_pheromone_into_base()

        # Load persisted targets (if enabled)
        if bool(self.get_parameter("persist_targets").value):
            self._load_targets()
        # Autosave targets if dirty
        self._targets_timer = self.create_timer(
            max(0.2, float(self.get_parameter("targets_autosave_sec").value)),
            self._autosave_targets,
            callback_group=self._viz_cbg,
        )

        # GUI status timer
        self._gui_timer = self.create_timer(0.5, self._publish_gui_status, callback_group=self._viz_cbg)

        self.get_logger().info(
            "Python Fast Sim ready:\n"
            f"  drones: {self.num_py}\n"
            f"  grid: {self.grid.grid_size_m}m, cell: {self.grid.cell_size_m}m\n"
            f"  RViz topics: /swarm/markers/*\n"
            f"  services: /swarm/cmd/start_python, /swarm/cmd/stop_python\n"
        )

        # Live parameter updates (GUI / ros2 param set) should update cached fields immediately.
        self.add_on_set_parameters_callback(self._on_set_params)

        # Exploration radius visualization timer (wall time, visual-only).
        self._explore_radius_timer = None
        try:
            if bool(self.get_parameter("exploration_radius_viz_enabled").value):
                per = float(self.get_parameter("exploration_radius_viz_period_s").value)
                per = clamp(per, 1.0, 120.0)
                self._explore_radius_timer = self.create_timer(per, self._publish_exploration_radius, callback_group=self._viz_cbg)
        except Exception:
            self._explore_radius_timer = None

    def _perf_add(self, name: str, dt_s: float):
        if not getattr(self, "perf_enabled", False):
            return
        agg = self._perf.get(name)
        if agg is None:
            agg = self._PerfAgg()
            self._perf[name] = agg
        agg.add(float(dt_s))

    def _perf_maybe_log(self, now_wall: float):
        if not getattr(self, "perf_enabled", False):
            return
        period = max(1.0, float(getattr(self, "perf_log_period_s", 5.0)))
        if (now_wall - self._perf_last_log_wall) < period:
            return
        self._perf_last_log_wall = now_wall

        def fmt(name: str) -> str:
            a = self._perf.get(name)
            if not a or a.n <= 0:
                return f"{name}=—"
            avg_ms = (a.total_s / a.n) * 1000.0
            mx_ms = a.max_s * 1000.0
            return f"{name} avg={avg_ms:.2f}ms max={mx_ms:.1f}ms n={a.n}"

        msg = (
            "[perf] "
            + ", ".join(
                [
                    f"ticks={self._perf_tick_n}",
                    f"sim_advanced={self._perf_sim_advanced_s:.2f}s",
                    f"sim_debt={self._perf_sim_debt_s:.2f}s",
                    f"substeps={self._perf_substeps}",
                    f"rviz_markers={int(getattr(self, '_last_rviz_marker_count', 0))}",
                    f"rviz_points={int(getattr(self, '_last_rviz_point_count', 0))}",
                    fmt("tick_total"),
                    fmt("sim_step_total"),
                    fmt("sense_total"),
                    fmt("drone_step_total"),
                    fmt("rviz_publish"),
                    fmt("pher_viz"),
                ]
            )
        )
        try:
            self.get_logger().info(msg)
        except Exception:
            pass

        self._perf_tick_n = 0
        self._perf_sim_advanced_s = 0.0
        self._perf_sim_debt_s = float(getattr(self, "_sim_debt", 0.0))
        self._perf_substeps = 0

    def _on_set_params(self, params: List[Parameter]) -> SetParametersResult:
        try:
            for p in params:
                name = str(p.name)
                if name == "aco_temperature":
                    self.aco_temperature = clamp(float(p.value), 1e-6, 50.0)
                elif name == "num_py_drones":
                    new_n = int(clamp(float(p.value), 1.0, 200.0))
                    self._resize_drones(new_n)
                elif name == "sense_radius":
                    # Affects mock lidar range (meters)
                    self.sense_radius = clamp(float(p.value), 2.0, 2000.0)
                elif name == "evap_nav_rate":
                    self.evap_nav_rate = clamp(float(p.value), 0.0, 10.0)
                elif name == "evap_danger_rate":
                    self.evap_danger_rate = clamp(float(p.value), 0.0, 10.0)
                elif name == "wall_danger_evap_mult":
                    self.wall_danger_evap_mult = clamp(float(p.value), 0.0, 1.0)
                elif name == "danger_evap_mult_static":
                    self.danger_evap_mult_static = clamp(float(p.value), 0.0, 10.0)
                elif name == "danger_evap_mult_dynamic":
                    self.danger_evap_mult_dynamic = clamp(float(p.value), 0.0, 10.0)
                elif name == "base_no_nav_radius_m":
                    self.base_no_nav_radius_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "base_no_deposit_radius_m":
                    self.base_no_deposit_radius_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "base_push_radius_m":
                    self.base_push_radius_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "base_push_strength":
                    self.base_push_strength = clamp(float(p.value), 0.0, 1e6)
                elif name == "explore_min_radius_m":
                    self.explore_min_radius_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "explore_min_radius_strength":
                    self.explore_min_radius_strength = clamp(float(p.value), 0.0, 1e6)
                elif name == "recent_cell_penalty":
                    self.recent_cell_penalty = clamp(float(p.value), 0.0, 1e6)
                elif name == "explore_revisit_penalty_repeat_mult":
                    self.explore_revisit_penalty_repeat_mult = clamp(float(p.value), 0.0, 20.0)
                elif name == "explore_revisit_nav_deposit_scale":
                    self.explore_revisit_nav_deposit_scale = clamp(float(p.value), 0.0, 1.0)
                elif name == "wall_clearance_m":
                    self.wall_clearance_m = clamp(float(p.value), 0.0, 200.0)
                elif name == "wall_clearance_weight":
                    self.wall_clearance_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "wall_avoid_start_factor":
                    self.wall_avoid_start_factor = clamp(float(p.value), 0.5, 10.0)
                elif name == "wall_avoid_yaw_weight":
                    self.wall_avoid_yaw_weight = clamp(float(p.value), 0.0, 20.0)
                elif name == "wall_corridor_relax":
                    self.wall_corridor_relax = clamp(float(p.value), 0.0, 1.0)
                elif name == "corner_backoff_enabled":
                    self.corner_backoff_enabled = bool(p.value)
                elif name == "unstick_move_enabled":
                    self.unstick_move_enabled = bool(p.value)
                elif name == "avoid_enabled":
                    self.avoid_enabled = bool(p.value)
                elif name == "avoid_side_random_on_entry":
                    self.avoid_side_random_on_entry = bool(p.value)
                elif name == "avoid_crab_offset_cells":
                    self.avoid_crab_offset_cells = clamp(float(p.value), 0.0, 50.0)
                elif name == "avoid_max_time_s":
                    self.avoid_max_time_s = clamp(float(p.value), 0.1, 60.0)
                elif name == "avoid_angular_tolerance_deg":
                    self.avoid_angular_tolerance_deg = clamp(float(p.value), 1.0, 90.0)
                elif name == "avoid_deposit_danger_amount":
                    self.avoid_deposit_danger_amount = clamp(float(p.value), 0.0, 10.0)
                elif name == "overfly_vertical_cost_mult":
                    self.overfly_vertical_cost_mult = clamp(float(p.value), 0.1, 50.0)
                    # Keep per-drone local-planner cost model in sync.
                    for d in self.drones:
                        try:
                            d.planning_vertical_cost_mult = float(self.overfly_vertical_cost_mult)
                        except Exception:
                            pass
                elif name == "overfly_descend_cost_factor":
                    self.overfly_descend_cost_factor = clamp(float(p.value), 0.0, 2.0)
                    for d in self.drones:
                        try:
                            d.planning_descend_cost_factor = float(self.overfly_descend_cost_factor)
                        except Exception:
                            pass
                elif name == "overfly_must_be_fraction_of_around":
                    self.overfly_must_be_fraction_of_around = clamp(float(p.value), 0.05, 2.0)
                elif name == "static_danger_altitude_violation_weight":
                    self.static_danger_altitude_violation_weight = clamp(float(p.value), 0.0, 500.0)
                    for d in self.drones:
                        try:
                            d.static_danger_altitude_violation_weight = float(self.static_danger_altitude_violation_weight)
                        except Exception:
                            pass
                elif name == "local_a_star_altitude_step_m":
                    self.local_a_star_altitude_step_m = clamp(float(p.value), 0.25, 10.0)
                    for d in self.drones:
                        try:
                            d.local_a_star_altitude_step_m = float(self.local_a_star_altitude_step_m)
                        except Exception:
                            pass
                elif name == "vertical_energy_cost_mult":
                    self.vertical_energy_cost_mult = clamp(float(p.value), 0.0, 50.0)
                elif name == "return_progress_weight":
                    self.return_progress_weight = clamp(float(p.value), 0.0, 100.0)
                elif name == "return_danger_weight":
                    self.return_danger_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "return_corridor_danger_relax":
                    self.return_corridor_danger_relax = clamp(float(p.value), 0.0, 1.0)
                elif name == "return_use_aco_enabled":
                    self.return_use_aco_enabled = bool(p.value)
                elif name == "drone_altitude_m":
                    self.drone_altitude_m = clamp(float(p.value), -50.0, 500.0)
                    for d in self.drones:
                        d.s.z_cruise = float(self.drone_altitude_m)
                        if not bool(getattr(d.s, "overfly_active", False)):
                            d.s.z_target = float(self.drone_altitude_m)
                elif name == "lidar_inflate_cells":
                    self.lidar_inflate_cells = int(clamp(float(p.value), 0.0, 10.0))
                elif name == "min_flight_altitude_m":
                    self.min_flight_altitude_m = clamp(float(p.value), 0.0, 200.0)
                elif name == "roof_clearance_margin_m":
                    self.roof_clearance_margin_m = clamp(float(p.value), 0.0, 50.0)
                elif name == "max_overfly_altitude_m":
                    self.max_overfly_altitude_m = clamp(float(p.value), 1.0, 500.0)
                elif name == "climb_rate_mps":
                    self.climb_rate_mps = clamp(float(p.value), 0.1, 50.0)
                elif name == "descend_rate_mps":
                    self.descend_rate_mps = clamp(float(p.value), 0.1, 50.0)
                elif name == "vertical_speed_mult_enabled":
                    self.vertical_speed_mult_enabled = bool(p.value)
                elif name == "vertical_speed_mult":
                    self.vertical_speed_mult = clamp(float(p.value), 0.1, 1.0)
                elif name == "stuck_progress_timeout_s":
                    self.stuck_progress_timeout_s = clamp(float(p.value), 0.5, 60.0)
                elif name in ("danger_map_in_lidar_enabled", "danger_map_topic"):
                    # topic re-subscribe not supported here (restart needed for topic change)
                    pass
                elif name == "progress_eps_m":
                    self.progress_eps_m = clamp(float(p.value), 0.1, 50.0)
                elif name == "local_plan_radius_m":
                    self.local_plan_radius_m = clamp(float(p.value), 5.0, 500.0)
                elif name == "path_history_len":
                    self.path_history_len = int(clamp(float(p.value), 50.0, 200000.0))
                    # Rebuild deques with new maxlen (keep recent history).
                    new_hist: Dict[str, Deque[Tuple[float, float, float]]] = {}
                    for uid, old in self.path_hist.items():
                        d = deque(old, maxlen=int(self.path_history_len))
                        new_hist[uid] = d
                    self.path_hist = new_hist
                elif name == "path_viz_max_points":
                    self.path_viz_max_points = int(clamp(float(p.value), 50.0, 5000.0))
                elif name == "explore_avoid_empty_weight":
                    self.explore_avoid_empty_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_avoid_explored_weight":
                    self.explore_avoid_explored_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_unexplored_reward_weight":
                    self.explore_unexplored_reward_weight = clamp(float(p.value), 0.0, 200.0)
                elif name == "explore_explored_age_weight":
                    self.explore_explored_age_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_explored_dist_weight":
                    self.explore_explored_dist_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "exploration_area_radius_m":
                    self.exploration_area_radius_m = clamp(float(p.value), 0.0, 500000.0)
                elif name == "exploration_radius_margin_m":
                    self.exploration_radius_margin_m = clamp(float(p.value), 0.0, 100000.0)
                elif name == "explore_low_nav_weight":
                    self.explore_low_nav_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_nav_weight":
                    self.explore_nav_weight = clamp(float(p.value), -50.0, 50.0)
                elif name == "danger_inspect_weight":
                    self.danger_inspect_weight = clamp(float(p.value), 0.0, 500.0)
                elif name == "danger_inspect_kernel_cells":
                    self.danger_inspect_kernel_cells = int(clamp(float(p.value), 0.0, 50.0))
                elif name == "danger_inspect_danger_thr":
                    self.danger_inspect_danger_thr = clamp(float(p.value), 0.0, 50.0)
                elif name == "danger_inspect_max_cell_danger":
                    self.danger_inspect_max_cell_danger = clamp(float(p.value), 0.0, 50.0)
                elif name == "dyn_danger_inspect_weight":
                    self.dyn_danger_inspect_weight = clamp(float(p.value), 0.0, 500.0)
                elif name == "dyn_inspector_rt_weight":
                    self.dyn_inspector_rt_weight = clamp(float(p.value), 0.0, 500.0)
                elif name == "dyn_inspector_rt_ttl_s":
                    self.dyn_inspector_rt_ttl_s = clamp(float(p.value), 0.0, 60.0)
                elif name == "dyn_inspector_rt_standoff_cells":
                    self.dyn_inspector_rt_standoff_cells = int(clamp(float(p.value), 0.0, 50.0))
                elif name == "dyn_inspector_avoid_damage_weight":
                    self.dyn_inspector_avoid_damage_weight = clamp(float(p.value), 0.0, 500.0)
                elif name == "dyn_inspector_avoid_damage_thr":
                    self.dyn_inspector_avoid_damage_thr = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_far_density_weight":
                    self.explore_far_density_weight = clamp(float(p.value), 0.0, 200.0)
                elif name == "explore_far_density_ring_radius_cells":
                    self.explore_far_density_ring_radius_cells = int(clamp(float(p.value), 0.0, 200000.0))
                elif name == "explore_far_density_kernel_radius_cells":
                    self.explore_far_density_kernel_radius_cells = int(clamp(float(p.value), 0.0, 200000.0))
                elif name == "explore_far_density_angle_step_deg":
                    self.explore_far_density_angle_step_deg = clamp(float(p.value), 1.0, 90.0)
                elif name == "explore_far_density_exclude_inside_ring":
                    self.explore_far_density_exclude_inside_ring = bool(p.value)
                elif name == "share_explore_vectors_comm":
                    self.share_explore_vectors_comm = bool(p.value)
                elif name == "explore_vector_comm_max":
                    self.explore_vector_comm_max = int(clamp(float(p.value), 0.0, 5000.0))
                elif name == "explore_vector_share_every_cells":
                    self.explore_vector_share_every_cells = int(clamp(float(p.value), 1.0, 100.0))
                elif name == "explore_vector_ttl_s":
                    self.explore_vector_ttl_s = clamp(float(p.value), 0.0, 1e6)
                elif name == "explore_vector_avoid_weight":
                    self.explore_vector_avoid_weight = clamp(float(p.value), 0.0, 50.0)
                elif name == "explore_vector_spatial_gate_m":
                    self.explore_vector_spatial_gate_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "exploration_radius_viz_enabled":
                    self.exploration_radius_viz_enabled = bool(p.value)
                    try:
                        if self._explore_radius_timer is not None:
                            self._explore_radius_timer.cancel()
                    except Exception:
                        pass
                    self._explore_radius_timer = None
                    if bool(self.exploration_radius_viz_enabled):
                        per = clamp(float(getattr(self, "exploration_radius_viz_period_s", 15.0)), 1.0, 120.0)
                        self._explore_radius_timer = self.create_timer(per, self._publish_exploration_radius, callback_group=self._viz_cbg)
                    else:
                        try:
                            self._publish_exploration_radius(delete=True)
                        except Exception:
                            pass
                elif name == "exploration_radius_viz_period_s":
                    self.exploration_radius_viz_period_s = clamp(float(p.value), 1.0, 120.0)
                    try:
                        if bool(getattr(self, "exploration_radius_viz_enabled", True)):
                            if self._explore_radius_timer is not None:
                                self._explore_radius_timer.cancel()
                            self._explore_radius_timer = self.create_timer(
                                float(self.exploration_radius_viz_period_s), self._publish_exploration_radius, callback_group=self._viz_cbg
                            )
                    except Exception:
                        pass
                elif name == "exploration_radius_viz_z":
                    self.exploration_radius_viz_z = clamp(float(p.value), -50.0, 200.0)
                elif name == "exploration_radius_viz_alpha":
                    self.exploration_radius_viz_alpha = clamp(float(p.value), 0.0, 1.0)
                elif name == "exploration_radius_viz_line_width":
                    self.exploration_radius_viz_line_width = clamp(float(p.value), 0.01, 10.0)
                elif name == "empty_goal_dilate_cells":
                    self.empty_goal_dilate_cells = int(clamp(float(p.value), 0.0, 25.0))
                elif name == "empty_near_wall_radius_cells":
                    self.empty_near_wall_radius_cells = int(clamp(float(p.value), 0.0, 10.0))
                elif name == "sense_period_mode":
                    self.sense_period_mode = str(p.value).strip().lower()
                elif name == "sense_period_sim_s":
                    self.sense_period_sim_s = clamp(float(p.value), 0.02, 60.0)
                elif name == "max_sense_scans_per_tick":
                    self.max_sense_scans_per_tick = int(clamp(float(p.value), 0.0, 50.0))
                elif name == "comm_viz_enabled":
                    self.comm_viz_enabled = bool(p.value)
                elif name == "comm_viz_min_dist_m":
                    self.comm_viz_min_dist_m = clamp(float(p.value), 0.0, 1e6)
                elif name == "comm_viz_expire_s":
                    self.comm_viz_expire_s = clamp(float(p.value), 0.05, 10.0)
                elif name == "comm_viz_max_lines":
                    self.comm_viz_max_lines = int(clamp(float(p.value), 0.0, 50000.0))
                elif name == "base_comm_viz_enabled":
                    self.base_comm_viz_enabled = bool(p.value)
                elif name == "base_comm_viz_expire_s":
                    self.base_comm_viz_expire_s = clamp(float(p.value), 0.05, 10.0)
                elif name == "base_comm_viz_line_width":
                    self.base_comm_viz_line_width = clamp(float(p.value), 0.01, 10.0)
                elif name == "comm_viz_mode":
                    self.comm_viz_mode = str(p.value).strip().lower()
                elif name == "comm_viz_cluster_line_width":
                    self.comm_viz_cluster_line_width = clamp(float(p.value), 0.01, 10.0)
                elif name == "pheromone_viz_publish_hz":
                    # Update timer rate live (visual-only).
                    hz = clamp(float(p.value), 0.05, 50.0)
                    try:
                        if self._pher_timer is not None:
                            self._pher_timer.cancel()
                    except Exception:
                        pass
                    self._pher_timer = self.create_timer(1.0 / max(0.2, hz), self._publish_pheromone_viz, callback_group=self._viz_cbg)
                elif name == "pheromone_viz_snapshot_period_s":
                    # Affects snapshot refresh logic (no timer recreation needed).
                    _ = clamp(float(p.value), 0.1, 60.0)
                    # reset so next publish rebuilds
                    self._pher_snapshot_wall_t = 0.0
                elif name == "pheromone_viz_batch_count":
                    bc = int(clamp(float(p.value), 1.0, 64.0))
                    self._pher_last_batch_count = bc
                    # reset so next publish rebuilds + clears extra markers if needed
                    self._pher_snapshot_wall_t = 0.0
        except Exception as e:
            res = SetParametersResult()
            res.successful = False
            res.reason = str(e)
            return res
        res = SetParametersResult()
        res.successful = True
        res.reason = ""
        return res

    def _resize_drones(self, new_n: int):
        """Dynamically adjust number of Python drones (visual + sim)."""
        new_n = int(clamp(float(new_n), 1.0, 200.0))
        with self._state_lock:
            cur = len(self.drones)
            if new_n == cur:
                self.num_py = new_n
                return

            # Shrink: drop highest seq drones.
            if new_n < cur:
                drop = self.drones[new_n:]
                self.drones = self.drones[:new_n]
                for d in drop:
                    try:
                        self.path_hist.pop(d.s.drone_uid, None)
                    except Exception:
                        pass
                self.num_py = new_n
                return

            # Grow: add new drones.
            for i in range(cur + 1, new_n + 1):
                uid = make_drone_uid("PY", i)
                st = DroneState(
                    drone_uid=uid,
                    drone_type="PY",
                    seq=i,
                    x=float(self.base_xy[0]),
                    y=float(self.base_xy[1]),
                    z=float(self.drone_altitude_m),
                    speed_mps=float(self.energy_model.normal_speed_mps),
                    energy_units=float(self.energy_model.full_units),
                    mode="IDLE",
                )
                st.recent_cells.append(self.grid.world_to_cell(st.x, st.y))
                pher = PheromoneMap(owner_uid=uid)
                agent = PythonDrone(state=st, grid=self.grid, pher=pher, energy=self.energy_model, rng=random.Random(1000 + i))
                # Keep new drones consistent with current sim settings.
                agent.static_danger_altitude_violation_weight = float(getattr(self, "static_danger_altitude_violation_weight", 6.0))
                agent.local_a_star_altitude_step_m = float(getattr(self, "local_a_star_altitude_step_m", 0.5))
                agent.planning_vertical_cost_mult = float(getattr(self, "overfly_vertical_cost_mult", 3.0))
                agent.planning_descend_cost_factor = float(getattr(self, "overfly_descend_cost_factor", 0.7))
                self.drones.append(agent)
                self.path_hist[uid] = deque(maxlen=max(50, int(self.path_history_len)))

            self.num_py = new_n
    def _publish_rviz_timer(self):
        # Publish at a low rate for RViz responsiveness.
        t_ref = float(self._last_t_ref)
        # Don't block RViz publishing behind the full sim tick. Minor visual inconsistency
        # is acceptable; freezing is not.
        t0 = time.perf_counter()
        self._publish_rviz(t_ref)
        self._perf_add("rviz_publish", time.perf_counter() - t0)

    def _publish_exploration_radius(self, delete: bool = False):
        """
        Publish a circle marker representing the exploration area radius (if enabled).
        Visual-only and intentionally low-rate (default ~15s).
        """
        try:
            enabled = bool(getattr(self, "exploration_radius_viz_enabled", True))
        except Exception:
            enabled = True
        if delete or (not enabled):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "exploration_radius"
            m.id = 7000
            m.action = Marker.DELETE
            self.pub_exploration_radius.publish(m)
            return

        radius = float(getattr(self, "exploration_area_radius_m", 0.0))
        if radius <= 1e-6:
            # Disabled -> clear marker
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "exploration_radius"
            m.id = 7000
            m.action = Marker.DELETE
            self.pub_exploration_radius.publish(m)
            return

        from geometry_msgs.msg import Point

        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "exploration_radius"
        m.id = 7000
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(max(0.01, float(getattr(self, "exploration_radius_viz_line_width", 0.8))))
        m.color.r = 1.0
        m.color.g = 0.9
        m.color.b = 0.1
        m.color.a = float(clamp(float(getattr(self, "exploration_radius_viz_alpha", 0.55)), 0.0, 1.0))
        z = float(getattr(self, "exploration_radius_viz_z", 0.03))
        cx, cy = float(self.base_xy[0]), float(self.base_xy[1])
        pts: List[Point] = []
        steps = 96
        for i in range(steps + 1):
            ang = (2.0 * math.pi) * (float(i) / float(steps))
            p = Point()
            p.x = cx + float(radius) * math.cos(ang)
            p.y = cy + float(radius) * math.sin(ang)
            p.z = float(z)
            pts.append(p)
        m.points = pts
        self.pub_exploration_radius.publish(m)

    def _on_pause(self, msg: Bool):
        self.paused = bool(msg.data)
        # Apply pending speed only when entering paused state.
        if self.paused:
            try:
                if self._pending_speed is not None:
                    v = float(self._pending_speed)
                    self._pending_speed = None
                    self.set_parameters([rclpy.parameter.Parameter("speed", rclpy.Parameter.Type.DOUBLE, v)])
                    self.get_logger().info(f"Applied pending speed={v:.2f} (paused)")
            except Exception:
                pass

    def _on_drone_scale(self, msg: Float32):
        s = clamp(float(msg.data), 0.1, 10.0)
        self.set_parameters([rclpy.parameter.Parameter("drone_marker_scale", rclpy.Parameter.Type.DOUBLE, s)])

    def _on_return_speed(self, msg: Float32):
        v = clamp(float(msg.data), 0.1, 50.0)
        self.set_parameters([rclpy.parameter.Parameter("return_speed_mps", rclpy.Parameter.Type.DOUBLE, v)])

    def _on_drone_altitude(self, msg: Float32):
        z = clamp(float(msg.data), -50.0, 500.0)
        self.set_parameters([rclpy.parameter.Parameter("drone_altitude_m", rclpy.Parameter.Type.DOUBLE, z)])
        # NOTE: We intentionally do NOT mutate target Z when changing cruise altitude.
        # Targets are visual/ground-truth entities; detection treats sense_radius as horizontal range (2D),
        # so keeping original target z prevents RViz markers from "jumping to z=0" when user changes altitude.

    def _on_vertical_speed_mult(self, msg: Float32):
        # Keep robust: allow external GUI to drive the multiplier even if it cannot set ROS params.
        try:
            v = clamp(float(msg.data), 0.1, 1.0)
            self.vertical_speed_mult = float(v)
            # Best-effort: also keep parameter value in sync for `ros2 param get` / GUI persistence.
            try:
                self.set_parameters([rclpy.parameter.Parameter("vertical_speed_mult", rclpy.Parameter.Type.DOUBLE, float(v))])
            except Exception:
                pass
        except Exception:
            pass

    def _on_selected_target(self, msg: String):
        """
        Select a concrete target to pursue (goal-seeking without RETURN).
        Payload options:
        - JSON: {"id":"T1"} or {"target_id":"T1"} or {"id":""} to clear
        - Raw: "T1" or "" to clear
        """
        raw = (msg.data or "").strip()
        if not raw:
            self.selected_target_id = ""
            return
        tid = ""
        if raw.startswith("{") and raw.endswith("}"):
            try:
                data = json.loads(raw)
                tid = str(data.get("id") or data.get("target_id") or "").strip()
            except Exception:
                tid = ""
        else:
            tid = raw
        self.selected_target_id = tid

    def _workspace_root(self) -> Path:
        # scripts/
        return Path(__file__).resolve().parent.parent

    def _pheromone_snapshot_dir_path(self) -> Path:
        base = str(self.get_parameter("pheromone_snapshot_dir").value)
        p = Path(base)
        if not p.is_absolute():
            p = self._workspace_root() / p
        return p

    @staticmethod
    def _sanitize_snapshot_name(name: str) -> str:
        s = str(name or "").strip()
        if not s:
            return ""
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
            else:
                out.append("_")
        # Avoid leading dots (hidden files) and empty results
        s2 = "".join(out).strip("._")
        return s2

    def _dump_base_map_payload(self) -> dict:
        # Merge current drone maps into base then serialize full base_map.
        t_ref = float(getattr(self, "t_sim", 0.0))
        try:
            self._upload_all_to_base(t_ref)
        except Exception:
            pass

        def layer_to_list(layer: SparseLayer):
            out = []
            for (x, y), v in list(layer.v.items()):
                meta = layer.meta.get((x, y))
                row = {
                    "x": int(x),
                    "y": int(y),
                    "v": float(v),
                    "t": float(meta.t if meta else 0.0),
                    "conf": float(meta.conf if meta else 0.5),
                    "src": str(meta.src if meta else "unknown"),
                }
                if meta is not None and meta.alt_m is not None:
                    row["alt"] = float(meta.alt_m)
                if meta is not None and getattr(meta, "speed_s_per_cell", None) is not None:
                    try:
                        row["speed"] = float(getattr(meta, "speed_s_per_cell"))
                    except Exception:
                        pass
                if meta is not None and meta.kind:
                    row["kind"] = str(meta.kind)
                out.append(row)
            return out

        return {
            "version": 6,
            "grid": {"grid_size": self.grid.grid_size_m, "cell_size": self.grid.cell_size_m},
            "base_uid": self.base_map.owner_uid,
            "time": {"t_sim": float(getattr(self, "t_sim", 0.0)), "wall": time.time()},
            "layers": {
                "nav": layer_to_list(self.base_map.nav),
                "danger": layer_to_list(self.base_map.danger),
                "empty": layer_to_list(self.base_map.empty),
                "explored": layer_to_list(self.base_map.explored),
            },
        }

    def _save_pheromone_snapshot(self, name: str) -> Optional[Path]:
        safe = self._sanitize_snapshot_name(name)
        if not safe:
            return None
        out_dir = self._pheromone_snapshot_dir_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe}.json"
        payload = self._dump_base_map_payload()
        tmp = str(out_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, out_path)
        return out_path

    def _load_pheromone_snapshot(self, path_str: str) -> Optional[Path]:
        p = Path(str(path_str or "").strip())
        if not p.is_absolute():
            p = self._workspace_root() / p
        if not p.exists():
            return None
        with open(p, "r") as f:
            data = json.load(f)
        layers = (data or {}).get("layers", {}) or {}
        t_info = (data or {}).get("time", {}) or {}
        t_file = float(t_info.get("t_sim", 0.0) or 0.0)

        # Load into base_map (do not touch targets).
        self.base_map.clear()
        for layer_name in ("nav", "danger", "empty", "explored"):
            arr = layers.get(layer_name, []) or []
            target = getattr(self.base_map, layer_name, None)
            if target is None:
                continue
            for row in arr:
                try:
                    x = int(row.get("x"))
                    y = int(row.get("y"))
                    v = float(row.get("v", 0.0))
                    mt = float(row.get("t", 0.0))
                    conf = float(row.get("conf", 0.5))
                    src = str(row.get("src", "FILE"))
                    alt = row.get("alt", None)
                    kind = str(row.get("kind", "generic") or "generic")
                    sp = row.get("speed", None)
                    meta = CellMeta(
                        t=mt,
                        conf=conf,
                        src=src,
                        alt_m=(float(alt) if alt is not None else None),
                        kind=kind,
                        speed_s_per_cell=(float(sp) if sp is not None else None),
                    )
                    target.set((x, y), v, meta)
                except Exception:
                    continue

        # Restore mission timestamp from file to avoid evaporation / patch-timestamp drift.
        self.t_sim = float(t_file)
        self._last_wall = time.time()
        self._last_t_ref = float(self.t_sim)

        # Freeze time after loading until user starts something explicitly.
        self.running = False
        self.paused = True
        self.returning = False

        # Let drones at base re-download from base storage on next sync.
        try:
            for d in getattr(self, "drones", []) or []:
                d.s.last_base_pher_download_t = -1e9
        except Exception:
            pass
        return p

    def _on_pheromone_map_storage(self, msg: String):
        try:
            data = json.loads(str(msg.data or "{}"))
        except Exception:
            data = {}
        cmd = str(data.get("cmd", "") or "").strip().lower()
        if cmd == "save":
            name = str(data.get("name", "") or "")
            try:
                outp = self._save_pheromone_snapshot(name)
                if outp is None:
                    self.gui_log = "Snapshot save failed: invalid name"
                else:
                    self.gui_log = f"Saved pheromone snapshot: {outp}"
            except Exception as e:
                self.gui_log = f"Snapshot save failed: {e}"
        elif cmd == "load":
            path_s = str(data.get("path", "") or "")
            try:
                outp = self._load_pheromone_snapshot(path_s)
                if outp is None:
                    self.gui_log = "Snapshot load failed: file not found"
                else:
                    self.gui_log = f"Loaded pheromone snapshot: {outp}"
            except Exception as e:
                self.gui_log = f"Snapshot load failed: {e}"
        elif cmd in ("load_compat", "compat", "load-compat"):
            # Convenience: load legacy compat pheromone file (data/pheromone_data.json) into base_map.
            # This is useful when starting an exploit run without any exploration/snapshot loaded yet.
            try:
                self._load_compat_pheromone_into_base()
                # Freeze after loading until user explicitly starts a run (matches snapshot-load behavior).
                self.running = False
                self.paused = True
                self.returning = False
                # Let drones re-download from base storage on next sync.
                try:
                    for d in getattr(self, "drones", []) or []:
                        d.s.last_base_pher_download_t = -1e9
                except Exception:
                    pass
                self.gui_log = "Loaded compat pheromone file into base map"
            except Exception as e:
                self.gui_log = f"Compat pheromone load failed: {e}"
        else:
            self.gui_log = f"Unknown pheromone_map_storage cmd: {cmd or '—'}"

    def _on_exploit_start(self, msg: String):
        """
        Start an exploit run toward a selected target (from base).

        JSON payload:
          {"target_id":"T1","drone_count":3,"dynamic_mode":"handled"|"static"}
        """
        try:
            data = json.loads(str(msg.data or "{}"))
        except Exception:
            data = {}

        tid = str(data.get("target_id", "") or data.get("id", "") or "").strip()
        if not tid:
            self.gui_log = "Exploit start: missing target_id"
            return

        # Validate target exists
        tgt = None
        try:
            for t in self.targets:
                if str(t.target_id) == tid:
                    tgt = t
                    break
        except Exception:
            tgt = None
        if tgt is None:
            self.gui_log = f"Exploit start: unknown target_id={tid}"
            return

        # Apply mode
        mode = str(data.get("dynamic_mode", "handled") or "handled").strip().lower()
        treat_static = True if mode in ("static", "treat_static", "as_static", "treated_as_static") else False
        try:
            self.set_parameters(
                [rclpy.parameter.Parameter("exploit_dynamic_danger_as_static", rclpy.Parameter.Type.BOOL, bool(treat_static))]
            )
        except Exception:
            pass

        # Drone count / resize
        try:
            n_req = int(data.get("drone_count", 3) or 3)
        except Exception:
            n_req = 3
        n_req = int(clamp(float(n_req), 1.0, 200.0))
        try:
            if int(getattr(self, "num_py", len(self.drones))) != int(n_req):
                self._resize_drones(int(n_req))
                self.num_py = int(n_req)
                try:
                    self.set_parameters([rclpy.parameter.Parameter("num_py_drones", rclpy.Parameter.Type.INTEGER, int(n_req))])
                except Exception:
                    pass
        except Exception:
            pass

        # Mark selection (also hides other targets in RViz if enabled).
        self.selected_target_id = str(tid)

        # Reset per-run bookkeeping
        self._exploit_active = True
        self._exploit_start_t = float(getattr(self, "t_sim", 0.0))
        self._exploit_active_uids = set()
        self._exploit_arrived_uids = set()
        self._exploit_stats_by_uid = {}

        # Start/continue simulation time from current timestamp (do not reset t_sim).
        self.mission_phase = "EXPLOIT"
        self.returning = False
        self.running = True
        self.paused = False
        self._last_wall = time.time()

        # Initialize drones at base, clear local maps, and download from base storage.
        z = float(self.get_parameter("drone_altitude_m").value)
        for d in self.drones:
            # All drones start at base for exploit (repeatable trials).
            d.s.x, d.s.y, d.s.z = self.base_xy[0], self.base_xy[1], z
            d.s.energy_units = self.energy_model.full_units
            d.s.speed_mps = self.energy_model.normal_speed_mps
            d.s.total_dist_m = 0.0
            d.s.encounters = 0
            d.s.base_uploads = 0
            d.s.recharge_until_t = 0.0
            try:
                d.s.last_base_pher_upload_t = -1e9
                d.s.last_base_pher_download_t = -1e9
            except Exception:
                pass
            try:
                d.pher.clear()
            except Exception:
                pass
            try:
                self._download_base_to_drone(d, since_t=-1e9, max_cells=200000)
            except Exception:
                pass
            try:
                self._sync_targets_base_to_drone(d, t_ref=float(getattr(self, "t_sim", 0.0)), force=True)
            except Exception:
                pass
            d.s.mode = "EXPLORE"
            # EXPLOIT per-drone approach/landing state (kept on agent for simplicity).
            try:
                d._exploit_approach_done = False  # type: ignore[attr-defined]
                d._exploit_land_active = False  # type: ignore[attr-defined]
            except Exception:
                pass
            # Exploit stats init (per run).
            try:
                uid0 = str(d.s.drone_uid)
            except Exception:
                uid0 = ""
            if uid0:
                self._exploit_stats_by_uid[uid0] = {
                    "start_t": float(self._exploit_start_t),
                    "start_energy": float(self.energy_model.full_units),
                    "vert_m": 0.0,
                    "land_t": None,
                    "land_energy": None,
                }
            try:
                self._exploit_active_uids.add(str(d.s.drone_uid))
            except Exception:
                pass

        # Assign per-drone approach points around the selected target (different arrival angles).
        try:
            ar = float(self.get_parameter("exploit_approach_radius_m").value)
            jitter = float(self.get_parameter("exploit_approach_jitter_m").value)
            ar = max(0.0, min(250.0, ar))
            jitter = max(0.0, min(50.0, jitter))
        except Exception:
            ar = 35.0
            jitter = 6.0
        try:
            # Spread angles evenly, but offset by a random global rotation for variety.
            n0 = max(1, len(self.drones))
            rot0 = random.Random(int(time.time()) & 0x7FFFFFFF).uniform(-math.pi, math.pi)
            for i, d in enumerate(self.drones):
                ang = float(rot0) + (2.0 * math.pi) * (float(i) / float(n0))
                # Small per-drone jitter for non-symmetric approaches.
                rng = getattr(d, "rng", None) or random.Random(1234 + i)
                jx = float(rng.uniform(-jitter, jitter)) if jitter > 1e-9 else 0.0
                jy = float(rng.uniform(-jitter, jitter)) if jitter > 1e-9 else 0.0
                ax = float(tgt.x) + math.cos(ang) * float(ar) + jx
                ay = float(tgt.y) + math.sin(ang) * float(ar) + jy
                # Keep inside map bounds.
                try:
                    minx, maxx, miny, maxy = self.map_bounds
                    ax = float(clamp(float(ax), float(minx) + 1.0, float(maxx) - 1.0))
                    ay = float(clamp(float(ay), float(miny) + 1.0, float(maxy) - 1.0))
                except Exception:
                    pass
                d._exploit_approach_world = (float(ax), float(ay))  # type: ignore[attr-defined]
                d._exploit_final_world = (float(tgt.x), float(tgt.y))  # type: ignore[attr-defined]
        except Exception:
            pass

        self.gui_log = f"Exploit start: target={tid}, drones={len(self._exploit_active_uids)}, dynamic_mode={'static' if treat_static else 'handled'}"

    def _on_pheromone_viz_select(self, msg: String):
        # JSON: {"owner":"base|combined|drone","drone_seq":1,"layer":"danger|nav|empty"}
        try:
            data = json.loads(msg.data)
            owner = str(data.get("owner", "base")).lower()
            layer = str(data.get("layer", "danger")).lower()
            seq = int(data.get("drone_seq", 1))
            if owner not in ("base", "combined", "drone"):
                owner = "base"
            if layer not in ("danger", "nav", "empty", "explored"):
                layer = "danger"
            seq = max(1, min(self.num_py, seq))
            self.set_parameters(
                [
                    rclpy.parameter.Parameter("pheromone_viz_owner", rclpy.Parameter.Type.STRING, owner),
                    rclpy.parameter.Parameter("pheromone_viz_layer", rclpy.Parameter.Type.STRING, layer),
                    rclpy.parameter.Parameter("pheromone_viz_drone_seq", rclpy.Parameter.Type.INTEGER, seq),
                ]
            )
            # Force next viz publish to rebuild snapshot immediately (batched mode).
            try:
                self._pher_snapshot_wall_t = 0.0
                self._pher_snapshot_key = None  # type: ignore
                self._pher_snapshot_batch_i = 0
            except Exception:
                pass
            self.get_logger().info(f"Pheromone viz select: owner={owner}, layer={layer}, drone_seq={seq}")
        except Exception:
            return

    def _on_lidar_viz_select(self, msg: String):
        try:
            data = json.loads(msg.data)
            enabled = bool(data.get("enabled", False))
            seq = int(data.get("drone_seq", 1))
            seq = max(1, min(self.num_py, seq))
            self.set_parameters(
                [
                    rclpy.parameter.Parameter("lidar_viz_enabled", rclpy.Parameter.Type.BOOL, enabled),
                    rclpy.parameter.Parameter("lidar_viz_drone_seq", rclpy.Parameter.Type.INTEGER, seq),
                ]
            )
        except Exception:
            return

    def _on_plan_viz_select(self, msg: String):
        # JSON: {"enabled":true,"drone_seq":1}
        # Special: drone_seq==0 means "ALL drones".
        try:
            data = json.loads(msg.data)
            enabled = bool(data.get("enabled", False))
            seq = int(data.get("drone_seq", 1))
            if seq != 0:
                seq = max(1, min(self.num_py, seq))
            self.set_parameters(
                [
                    rclpy.parameter.Parameter("plan_viz_enabled", rclpy.Parameter.Type.BOOL, enabled),
                    rclpy.parameter.Parameter("plan_viz_drone_seq", rclpy.Parameter.Type.INTEGER, seq),
                ]
            )
        except Exception:
            return

    def _on_aco_viz_select(self, msg: String):
        # JSON: {"enabled":true,"drone_seq":1} ; drone_seq==0 => ALL drones
        try:
            data = json.loads(msg.data)
            enabled = bool(data.get("enabled", False))
            seq = int(data.get("drone_seq", 1))
            if seq != 0:
                seq = max(1, min(self.num_py, seq))
            self.set_parameters(
                [
                    rclpy.parameter.Parameter("aco_viz_enabled", rclpy.Parameter.Type.BOOL, enabled),
                    rclpy.parameter.Parameter("aco_viz_drone_seq", rclpy.Parameter.Type.INTEGER, seq),
                ]
            )
        except Exception:
            return

    def _on_lidar_scan_viz_select(self, msg: String):
        # JSON: {"enabled":true,"drone_seq":1}
        try:
            data = json.loads(msg.data)
            enabled = bool(data.get("enabled", False))
            seq = int(data.get("drone_seq", 1))
            seq = max(1, min(self.num_py, seq))
            self.set_parameters(
                [
                    rclpy.parameter.Parameter("lidar_scan_viz_enabled", rclpy.Parameter.Type.BOOL, enabled),
                    rclpy.parameter.Parameter("lidar_scan_viz_drone_seq", rclpy.Parameter.Type.INTEGER, seq),
                ]
            )
        except Exception:
            return

    def _on_pointer_params(self, msg: Float64MultiArray):
        # Format: [enabled(0/1), z, scale, alpha]
        if len(msg.data) < 4:
            return
        enabled = bool(int(msg.data[0]) != 0)
        z = clamp(float(msg.data[1]), -50.0, 200.0)
        scale = clamp(float(msg.data[2]), 0.1, 10.0)
        alpha = clamp(float(msg.data[3]), 0.0, 1.0)
        self.set_parameters(
            [
                rclpy.parameter.Parameter("drone_pointer_enabled", rclpy.Parameter.Type.BOOL, enabled),
                rclpy.parameter.Parameter("drone_pointer_z", rclpy.Parameter.Type.DOUBLE, z),
                rclpy.parameter.Parameter("drone_pointer_scale", rclpy.Parameter.Type.DOUBLE, scale),
                rclpy.parameter.Parameter("drone_pointer_alpha", rclpy.Parameter.Type.DOUBLE, alpha),
            ]
        )

    def _on_danger_map_marker(self, msg: Marker):
        """Receive current danger cells from DangerMapManager (/danger_map).

        We only consume marker id=1, type=CUBE_LIST which contains the active danger cells.
        """
        try:
            if int(msg.id) != 1 or int(msg.type) != int(Marker.CUBE_LIST):
                return
            pts = list(msg.points) if msg.points is not None else []
            if not pts:
                with self._danger_cells_lock:
                    self._danger_cells = set()
                return
            s: Set[Tuple[int, int]] = set()
            for p in pts:
                cx, cy = self.grid.world_to_cell(float(p.x), float(p.y))
                if self.grid.in_bounds_cell(int(cx), int(cy)):
                    s.add((int(cx), int(cy)))
            with self._danger_cells_lock:
                self._danger_cells = s
        except Exception:
            # keep last known danger cells on errors
            pass

    def _on_danger_map_cells(self, msg: String):
        """Receive danger source cells + radius metadata (JSON) from DangerMapManager (/danger_map_cells)."""
        try:
            data = json.loads(str(msg.data or "{}"))
            cells = data.get("cells", [])
            out: Dict[Tuple[int, int], dict] = {}
            for it in cells:
                cx = int(it.get("cell_x"))
                cy = int(it.get("cell_y"))
                r = int(it.get("radius", 0) or 0)
                kind = str(it.get("kind", "") or "")
                danger_id = str(it.get("id", "") or "")
                info = {"radius": max(0, int(r)), "kind": kind, "id": danger_id}
                # Optional dynamic speed (seconds per cell) for threat reasoning / viz.
                sp = it.get("speed", None)
                if sp is not None:
                    try:
                        info["speed"] = float(sp)
                    except Exception:
                        pass
                # Optional height above ground (meters). Static threats default to 50m if absent.
                hm = it.get("height_m", None)
                if hm is not None:
                    try:
                        info["height_m"] = float(hm)
                    except Exception:
                        pass
                out[(cx, cy)] = info

            # Optional: rich dynamic metadata (trajectory, speed, current index).
            dyn = data.get("dynamic", []) or []
            dyn_out: Dict[str, dict] = {}
            dyn_paths_by_id: Dict[str, List[Tuple[int, int]]] = {}
            dyn_heights_by_id: Dict[str, float] = {}
            for d in dyn:
                try:
                    did = str(d.get("id", "") or "")
                    if not did:
                        continue
                    speed = float(d.get("speed", 4.0))
                    speed = float(max(0.01, speed))
                    radius = int(d.get("radius", 0) or 0)
                    radius = int(max(0, radius))
                    height_m = float(d.get("height_m", 50.0) or 50.0)
                    cur_i = int(d.get("current_index", 0) or 0)
                    arr = d.get("arrayOfCells", []) or []
                    path: List[Tuple[int, int]] = []
                    for cc in arr:
                        path.append((int(cc.get("cell_x")), int(cc.get("cell_y"))))
                    if not path:
                        continue
                    cur_i = int(cur_i) % len(path)
                    dyn_out[did] = {
                        "id": did,
                        "speed": speed,
                        "radius": radius,
                        "height_m": float(height_m),
                        "current_index": int(cur_i),
                        "path": path,
                        "cell": (int(path[cur_i][0]), int(path[cur_i][1])),
                    }
                    dyn_paths_by_id[did] = list(path)
                    dyn_heights_by_id[did] = float(height_m)
                except Exception:
                    continue
            with self._danger_cells_lock:
                self._danger_sources = out
                self._dynamic_dangers = dyn_out
                # Convenience: dynamic danger trajectory cells by id (from danger_map.json arrayOfCells).
                # Used for fast LiDAR checks without scanning the whole grid.
                self._dynamic_paths_by_id = dyn_paths_by_id
                self._dynamic_heights_by_id = dyn_heights_by_id
        except Exception:
            pass

    def _on_pheromone_viz_enable(self, msg: Bool):
        self.set_parameters([rclpy.parameter.Parameter("pheromone_viz_enabled", rclpy.Parameter.Type.BOOL, bool(msg.data))])

    def _on_pheromone_viz_params(self, msg: Float64MultiArray):
        # Format: [display_size_m, z, alpha]
        if len(msg.data) < 3:
            return
        disp = float(msg.data[0])
        z = float(msg.data[1])
        a = float(msg.data[2])
        disp = clamp(disp, 50.0, self.grid.grid_size_m)
        z = clamp(z, -50.0, 200.0)
        a = clamp(a, 0.0, 1.0)
        self.set_parameters(
            [
                rclpy.parameter.Parameter("pheromone_viz_display_size", rclpy.Parameter.Type.DOUBLE, disp),
                rclpy.parameter.Parameter("pheromone_viz_z", rclpy.Parameter.Type.DOUBLE, z),
                rclpy.parameter.Parameter("pheromone_viz_alpha", rclpy.Parameter.Type.DOUBLE, a),
            ]
        )

    def _on_target_viz_params(self, msg: Float64MultiArray):
        # Format: [diameter_m, alpha]
        if len(msg.data) < 2:
            return
        d = clamp(float(msg.data[0]), 0.5, 200.0)
        a = clamp(float(msg.data[1]), 0.0, 1.0)
        self.set_parameters(
            [
                rclpy.parameter.Parameter("target_viz_diameter", rclpy.Parameter.Type.DOUBLE, d),
                rclpy.parameter.Parameter("target_viz_alpha", rclpy.Parameter.Type.DOUBLE, a),
            ]
        )

    def _load_compat_pheromone_into_base(self):
        # Loads data/pheromone_data.json into base_map.danger with simple intensity->value mapping.
        repo_root = Path(__file__).resolve().parents[2]
        compat_path = repo_root / str(self.get_parameter("compat_export_path").value)
        if not compat_path.exists():
            return
        try:
            with open(compat_path, "r") as f:
                data = json.load(f)
            # Do NOT clear base map; just merge in with old timestamp (0) so new data wins later
            t0 = 0.0
            for cell in data:
                cx = int(cell["cell_x"])
                cy = int(cell["cell_y"])
                intensity = int(cell.get("intensity", 0))
                if intensity <= 0:
                    continue
                # store as float value roughly aligned with our binning thresholds
                v = 0.6 if intensity == 1 else 1.2 if intensity == 2 else 2.0
                meta = CellMeta(t=t0, conf=0.5, src="FILE")
                # Legacy compat file represents generic "danger". Keep it there (not nav_danger).
                self.base_map.danger.merge_cell((cx, cy), v, meta)
            self.get_logger().info(f"Loaded compat pheromone file for viz: {compat_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load compat pheromone file {compat_path}: {e}")

    def _color_for_intensity(self, intensity: int) -> Tuple[float, float, float]:
        # Match scripts/pheromone_heatmap.py mapping
        if intensity <= 0:
            return (0.5, 1.0, 0.5)  # light green
        if intensity == 1:
            return (0.7, 1.0, 0.3)  # yellow-green
        if intensity == 2:
            return (1.0, 1.0, 0.0)  # yellow
        return (1.0, 0.0, 0.0)  # red

    def _publish_pheromone_viz(self):
        t0 = time.perf_counter()
        with self._state_lock:
            if not bool(self.get_parameter("pheromone_viz_enabled").value):
                return

        z = float(self.get_parameter("pheromone_viz_z").value)
        alpha = float(self.get_parameter("pheromone_viz_alpha").value)
        display_size = float(self.get_parameter("pheromone_viz_display_size").value)
        # Backward-compat (older param):
        source = str(self.get_parameter("pheromone_viz_source").value)
        show_grid = bool(self.get_parameter("pheromone_viz_show_grid").value)
        grid_alpha = float(self.get_parameter("pheromone_viz_grid_alpha").value)
        grid_w = float(self.get_parameter("pheromone_viz_grid_line_width").value)
        render_mode = str(self.get_parameter("pheromone_viz_cell_render_mode").value).strip().lower()
        ctx_cells = int(self.get_parameter("pheromone_viz_context_cells").value)
        blank_r = int(self.get_parameter("pheromone_viz_blank_cells_radius").value)
        border_w = float(self.get_parameter("pheromone_viz_border_line_width").value)
        border_a = float(self.get_parameter("pheromone_viz_border_alpha").value)
        bg_a = float(self.get_parameter("pheromone_viz_background_alpha").value)

        owner = str(self.get_parameter("pheromone_viz_owner").value).lower()
        layer_name = str(self.get_parameter("pheromone_viz_layer").value).lower()
        drone_seq = int(self.get_parameter("pheromone_viz_drone_seq").value)
        snapshot_period_s = float(self.get_parameter("pheromone_viz_snapshot_period_s").value)
        batch_count = int(self.get_parameter("pheromone_viz_batch_count").value)
        max_points = int(self.get_parameter("pheromone_viz_max_points").value)
        auto_batch_threshold = int(self.get_parameter("pheromone_viz_auto_batch_threshold").value)
        auto_batch_count = int(self.get_parameter("pheromone_viz_auto_batch_count").value)
        snapshot_period_s = clamp(snapshot_period_s, 0.1, 60.0)
        batch_count = int(clamp(float(batch_count), 1.0, 64.0))
        if owner not in ("base", "combined", "drone"):
            # map old style source into new owner/layer where possible
            if "nav" in source:
                layer_name = "nav"
            else:
                layer_name = "danger"
            owner = "base"
        if layer_name not in ("nav", "danger", "empty", "explored"):
            layer_name = "danger"
        drone_seq = max(1, min(self.num_py, drone_seq))

        # Optional filter for danger kind(s).
        # Supports:
        # - "all"
        # - single kind: "nav_danger", "danger_static", "danger_dyn_kernel", "danger_dyn_damage"
        # - comma-separated list of kinds (prefix match, ':' suffix allowed in stored meta)
        kind_sel = ""
        if layer_name == "danger":
            kind_sel = str(self.get_parameter("pheromone_viz_danger_kind").value).strip().lower()

        # Choose layer dict to visualize
        def get_layer(agent: Optional[PythonDrone]) -> SparseLayer:
            if agent is None:
                if layer_name == "danger":
                    return self.base_map.danger
                if layer_name == "empty":
                    return self.base_map.empty
                if layer_name == "explored":
                    return self.base_map.explored
                return self.base_map.nav
            if layer_name == "danger":
                return agent.pher.danger
            if layer_name == "empty":
                return agent.pher.empty
            if layer_name == "explored":
                return agent.pher.explored
            return agent.pher.nav

        def safe_items(layer_obj: SparseLayer) -> List[Tuple[Tuple[int, int], float]]:
            # Avoid "dictionary changed size during iteration" when sim tick mutates concurrently.
            for _ in range(3):
                try:
                    return list(layer_obj.v.items())
                except RuntimeError:
                    time.sleep(0)
            return list(layer_obj.v.items())

        # Compute a display-window filter in cell coordinates (optional optimization).
        # This is a hard cap for viz workload: cells outside display_size are ignored.
        try:
            half = float(display_size) * 0.5
            cmin = self.grid.world_to_cell(-half, -half)
            cmax = self.grid.world_to_cell(half, half)
            win_min_cx = int(min(cmin[0], cmax[0])); win_max_cx = int(max(cmin[0], cmax[0]))
            win_min_cy = int(min(cmin[1], cmax[1])); win_max_cy = int(max(cmin[1], cmax[1]))
        except Exception:
            win_min_cx = 0; win_min_cy = 0; win_max_cx = int(self.grid.cells - 1); win_max_cy = int(self.grid.cells - 1)

        def in_viz_window(c: Tuple[int, int]) -> bool:
            return (win_min_cx <= int(c[0]) <= win_max_cx) and (win_min_cy <= int(c[1]) <= win_max_cy)

        if owner == "base":
            layer = get_layer(None)
        elif owner == "drone":
            agent = self.drones[drone_seq - 1] if 1 <= drone_seq <= len(self.drones) else self.drones[0]
            layer = get_layer(agent)
        else:
            # combined of all drones (max per-cell), optionally include base too (base acts like global cache)
            # NOTE: For danger layer we MUST keep meta.kind to support kind-specific coloring and filtering.
            # So danger always uses the meta-aware merge path.
            if layer_name in ("nav", "empty"):
                # Fast path: we don't need meta merging for nav/empty visualization.
                maxv: Dict[Tuple[int, int], float] = {}
                base_layer = get_layer(None)
                for c, v in safe_items(base_layer):
                    if not in_viz_window(c):
                        continue
                    prev = maxv.get(c)
                    if prev is None or v > prev:
                        maxv[c] = float(v)
                for agent in self.drones:
                    l = get_layer(agent)
                    for c, v in safe_items(l):
                        if not in_viz_window(c):
                            continue
                        prev = maxv.get(c)
                        if prev is None or v > prev:
                            maxv[c] = float(v)
                layer = SparseLayer()
                layer.v = maxv
                layer.meta = {}  # not needed
            elif layer_name == "explored":
                # We DO need meta for explored visualization: obs_dist_m controls the blue->white tint.
                maxv: Dict[Tuple[int, int], float] = {}
                best_meta: Dict[Tuple[int, int], CellMeta] = {}

                def _upd_explored_from(src_layer: SparseLayer):
                    for c, v in safe_items(src_layer):
                        if not in_viz_window(c):
                            continue
                        prev = maxv.get(c)
                        if prev is None or v > prev:
                            maxv[c] = float(v)

                        mk = src_layer.meta.get(c)
                        if mk is None:
                            continue
                        try:
                            tt = float(getattr(mk, "t", 0.0))
                        except Exception:
                            tt = 0.0
                        try:
                            cc = float(getattr(mk, "conf", 0.5))
                        except Exception:
                            cc = 0.5
                        try:
                            dd = getattr(mk, "obs_dist_m", None)
                        except Exception:
                            dd = None
                        try:
                            dd = float(dd) if dd is not None else None
                        except Exception:
                            dd = None

                        pm = best_meta.get(c)
                        if pm is None:
                            best_meta[c] = CellMeta(t=tt, conf=cc, src="COMBINED", kind="explored", obs_dist_m=dd)
                        else:
                            # Combine across agents:
                            # - newer timestamp for freshness
                            # - higher confidence
                            # - smaller obs_dist_m for "how closely explored"
                            t2 = max(float(pm.t), float(tt))
                            c2 = max(float(pm.conf), float(cc))
                            d2 = pm.obs_dist_m
                            if d2 is None:
                                d2 = dd
                            elif dd is not None:
                                d2 = min(float(d2), float(dd))
                            best_meta[c] = CellMeta(t=float(t2), conf=float(c2), src="COMBINED", kind="explored", obs_dist_m=d2)

                base_layer = get_layer(None)
                _upd_explored_from(base_layer)
                for agent in self.drones:
                    _upd_explored_from(get_layer(agent))

                layer = SparseLayer()
                layer.v = maxv
                layer.meta = best_meta
            else:
                combined = SparseLayer()
                # base first
                base_layer = get_layer(None)
                for c, v in safe_items(base_layer):
                    if not in_viz_window(c):
                        continue
                    meta = base_layer.meta.get(c)
                    if meta:
                        combined.merge_cell(c, v, meta)
                for agent in self.drones:
                    l = get_layer(agent)
                    for c, v in safe_items(l):
                        if not in_viz_window(c):
                            continue
                        meta = l.meta.get(c)
                        if meta:
                            prev = combined.get(c)
                            if v > prev:
                                combined.merge_cell(c, v, meta)
                layer = combined

        # Take a snapshot of items to avoid concurrent mutation from sim tick.
        layer_items: List[Tuple[Tuple[int, int], float]] = [(c, v) for (c, v) in safe_items(layer) if in_viz_window(c)]

        # Optional filter for danger kind(s).
        if layer_name == "danger" and kind_sel and kind_sel != "all":
            sels = [s.strip().lower() for s in str(kind_sel).split(",") if s.strip()]
            if sels:
                def _kind_ok(c: Tuple[int, int]) -> bool:
                    mk = layer.meta.get(c)
                    if mk is None:
                        return False
                    k = str(getattr(mk, "kind", "") or "").lower()
                    base = k.split(":", 1)[0] if ":" in k else k
                    return (base in sels) or any(base.startswith(ss) for ss in sels)
                layer_items = [(c, v) for (c, v) in layer_items if _kind_ok(c)]

        # Visual-only cap/downsample.
        if max_points > 0 and len(layer_items) > max_points:
            stride = max(2, int(math.ceil(len(layer_items) / float(max_points))))
            layer_items = layer_items[::stride]

        # Auto-batch when showing many points, even if batch_count is configured as 1.
        if batch_count <= 1 and auto_batch_threshold > 0 and len(layer_items) > auto_batch_threshold:
            batch_count = int(clamp(float(auto_batch_count), 2.0, 64.0))

        # Compute visual extent: show empty cells only near filled ones; when blank show 100x100 around origin.
        cell = float(self.grid.cell_size_m)
        if layer_items:
            xs = [c[0] for (c, _) in layer_items]
            ys = [c[1] for (c, _) in layer_items]
            min_cx = max(0, min(xs) - ctx_cells)
            max_cx = min(self.grid.cells - 1, max(xs) + ctx_cells)
            min_cy = max(0, min(ys) - ctx_cells)
            max_cy = min(self.grid.cells - 1, max(ys) + ctx_cells)
            # Convert to world edges (cell boundary)
            min_wx, min_wy = self.grid.cell_to_world(min_cx, min_cy)
            max_wx, max_wy = self.grid.cell_to_world(max_cx, max_cy)
            left = min_wx - (cell / 2.0)
            right = max_wx + (cell / 2.0)
            bottom = min_wy - (cell / 2.0)
            top = max_wy + (cell / 2.0)
        else:
            # 100x100 around center
            left = -blank_r * cell
            right = blank_r * cell
            bottom = -blank_r * cell
            top = blank_r * cell

        # Cap extent by display_size (user-defined max zoom area)
        half_cap = display_size / 2.0
        left = clamp(left, -half_cap, half_cap)
        right = clamp(right, -half_cap, half_cap)
        bottom = clamp(bottom, -half_cap, half_cap)
        top = clamp(top, -half_cap, half_cap)

        # Background plane (single cube) for "free space" tint
        bg = Marker()
        bg.header.frame_id = "world"
        bg.header.stamp = self.get_clock().now().to_msg()
        bg.ns = "pheromone_heatmap_bg"
        bg.id = 0
        bg.type = Marker.CUBE
        bg.action = Marker.ADD
        bg.pose.position.x = float((left + right) / 2.0)
        bg.pose.position.y = float((bottom + top) / 2.0)
        bg.pose.position.z = float(z)
        bg.pose.orientation.w = 1.0
        bg.scale.x = float(max(0.1, right - left))
        bg.scale.y = float(max(0.1, top - bottom))
        bg.scale.z = 0.05
        # Default unfilled background: transparent block (edges will show extent)
        bg.color.r = 0.0
        bg.color.g = 0.0
        bg.color.b = 0.0
        bg.color.a = float(clamp(bg_a, 0.0, 1.0))

        # Publish background first
        now_wall = time.time()

        # Border (thick white edges) to make the extent visible even when blank
        from geometry_msgs.msg import Point
        def _publish_bg_border_and_grid(stamp_msg):
            bg.header.stamp = stamp_msg
            self.pub_pheromone.publish(bg)

            border = Marker()
            border.header.frame_id = "world"
            border.header.stamp = stamp_msg
            border.ns = "pheromone_heatmap_border"
            border.id = 3
            border.type = Marker.LINE_STRIP
            border.action = Marker.ADD
            border.scale.x = float(max(0.01, border_w))
            border.color.r = 1.0
            border.color.g = 1.0
            border.color.b = 1.0
            border.color.a = float(clamp(border_a, 0.0, 1.0))
            border.points = []
            p1 = Point(); p1.x = float(left); p1.y = float(bottom); p1.z = float(z + 0.04)
            p2 = Point(); p2.x = float(right); p2.y = float(bottom); p2.z = float(z + 0.04)
            p3 = Point(); p3.x = float(right); p3.y = float(top); p3.z = float(z + 0.04)
            p4 = Point(); p4.x = float(left); p4.y = float(top); p4.z = float(z + 0.04)
            border.points.extend([p1, p2, p3, p4, p1])
            self.pub_pheromone.publish(border)

            # Optional grid overlay (cheap LINE_LIST) so cells are visible like in danger_map_manager.py
            if show_grid and grid_w > 0.0 and grid_alpha > 0.0:
                g = Marker()
                g.header.frame_id = "world"
                g.header.stamp = stamp_msg
                g.ns = "pheromone_heatmap_grid"
                g.id = 2
                g.type = Marker.LINE_LIST
                g.action = Marker.ADD
                g.scale.x = float(grid_w)
                g.color.r = 1.0
                g.color.g = 1.0
                g.color.b = 1.0
                g.color.a = float(clamp(grid_alpha, 0.0, 1.0))
                g.points = []

                # Draw grid only within current extent (empty cells only around filled cells / small blank map)
                x = left
                while x <= right + 1e-6:
                    q1 = Point(); q1.x = float(x); q1.y = float(bottom); q1.z = float(z + 0.03)
                    q2 = Point(); q2.x = float(x); q2.y = float(top); q2.z = float(z + 0.03)
                    g.points.append(q1); g.points.append(q2)
                    x += cell

                y = bottom
                while y <= top + 1e-6:
                    q1 = Point(); q1.x = float(left); q1.y = float(y); q1.z = float(z + 0.03)
                    q2 = Point(); q2.x = float(right); q2.y = float(y); q2.z = float(z + 0.03)
                    g.points.append(q1); g.points.append(q2)
                    y += cell
                self.pub_pheromone.publish(g)

        def _make_cells_marker(stamp_msg, marker_id: int):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = stamp_msg
            m.ns = "pheromone_heatmap_cells"
            m.id = int(marker_id)
            m.type = Marker.POINTS if render_mode == "points" else Marker.CUBE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            # RViz safety: some builds treat Marker.color.a==0 as invisible even when per-point colors are present.
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = float(alpha)
            m.scale.x = float(self.grid.cell_size_m)
            m.scale.y = float(self.grid.cell_size_m)
            m.scale.z = float(self.grid.cell_size_m if m.type == Marker.CUBE_LIST else 0.0)
            m.points = []
            m.colors = []
            return m

        def _explored_rgb(cxy: Tuple[int, int]) -> Tuple[float, float, float]:
            # Close observations => more blue; far observations => more white.
            blue = (0.2, 0.6, 1.0)
            white = (1.0, 1.0, 1.0)
            mk = None
            try:
                mk = layer.meta.get(cxy) if hasattr(layer, "meta") else None
            except Exception:
                mk = None
            try:
                d = getattr(mk, "obs_dist_m", None) if mk is not None else None
            except Exception:
                d = None
            try:
                d = float(d) if d is not None else None
            except Exception:
                d = None
            try:
                sr = float(getattr(self, "sense_radius", 0.0))
            except Exception:
                sr = 0.0
            if sr <= 1e-6:
                sr = 50.0
            t = float(clamp((float(d) / float(sr)) if d is not None else 1.0, 0.0, 1.0))
            rr = blue[0] * (1.0 - t) + white[0] * t
            gg = blue[1] * (1.0 - t) + white[1] * t
            bb = blue[2] * (1.0 - t) + white[2] * t
            return float(rr), float(gg), float(bb)

        stamp_msg = self.get_clock().now().to_msg()
        key = (
            owner,
            layer_name,
            int(drone_seq),
            float(display_size),
            float(z),
            float(alpha),
            str(render_mode),
            int(ctx_cells),
            int(blank_r),
            float(border_w),
            float(border_a),
            bool(show_grid),
            float(grid_w),
            float(grid_alpha),
            float(snapshot_period_s),
            int(batch_count),
            int(max_points),
            str(kind_sel),
        )
        # Debug: log when selection changes even if the select topic is starved.
        try:
            if getattr(self, "_pher_last_debug_key", None) != key:
                self._pher_last_debug_key = key
                self.get_logger().info(f"[pher_viz] owner={owner} layer={layer_name} drone_seq={int(drone_seq)} batches={batch_count} snap_s={snapshot_period_s:.2f}")
        except Exception:
            pass

        if batch_count <= 1:
            # If we previously published batched markers (ids 100..), clear them so old layers
            # don't remain visible when switching back to single-marker mode.
            try:
                prev_batches = int(getattr(self, "_pher_last_batch_count", 1))
                if prev_batches > 1:
                    for j in range(0, prev_batches):
                        mm = _make_cells_marker(stamp_msg, marker_id=100 + j)
                        mm.points = []
                        mm.colors = []
                        self.pub_pheromone.publish(mm)
                self._pher_last_batch_count = 1
            except Exception:
                pass
            _publish_bg_border_and_grid(stamp_msg)
            m = _make_cells_marker(stamp_msg, marker_id=1)
            for (cx, cy), v in layer_items:
                wx, wy = self.grid.cell_to_world(cx, cy)
                if wx < left or wx > right or wy < bottom or wy > top:
                    continue
                if layer_name == "empty":
                    rr, gg, bb = (1.0, 1.0, 0.0)  # yellow
                elif layer_name == "explored":
                    rr, gg, bb = _explored_rgb((cx, cy))
                else:
                    intensity = 1 if v <= 0.8 else 2 if v <= 1.6 else 3
                    rr, gg, bb = self._color_for_intensity(intensity)
                # Default: draw on the configured pheromone plane z.
                pz = float(z + 0.02)
                # Static threats: show the cell at its danger altitude (imitate a "gun" height profile).
                try:
                    if layer_name == "danger":
                        mk = layer.meta.get((cx, cy))
                        k = str(getattr(mk, "kind", "") or "") if mk is not None else ""
                        if k.startswith("danger_static") and (mk is not None) and (getattr(mk, "alt_m", None) is not None):
                            pz = float(getattr(mk, "alt_m")) + 0.02
                        # Dynamic threats: show the pheromone at its threat height above ground.
                        elif k.startswith("danger_dyn_") and (mk is not None) and (getattr(mk, "alt_m", None) is not None):
                            pz = float(getattr(mk, "alt_m")) + 0.02
                except Exception:
                    pass
                p = Point(); p.x = float(wx); p.y = float(wy); p.z = float(pz)
                m.points.append(p)
                col = ColorRGBA(); col.r = float(rr); col.g = float(gg); col.b = float(bb); col.a = float(alpha)
                m.colors.append(col)
            self.pub_pheromone.publish(m)
        else:
            # Batched mode:
            # - refresh snapshot periodically
            # - publish one batch per callback using stable marker IDs (100..)
            need_rebuild = (self._pher_snapshot_key != key) or (self._pher_snapshot_wall_t <= 0.0) or ((now_wall - self._pher_snapshot_wall_t) >= snapshot_period_s)
            prev_batches = int(getattr(self, "_pher_last_batch_count", batch_count))
            if need_rebuild:
                # clear any extra old batch markers if batch_count shrank
                if prev_batches > batch_count:
                    for j in range(batch_count, prev_batches):
                        mm = _make_cells_marker(stamp_msg, marker_id=100 + j)
                        mm.points = []
                        mm.colors = []
                        self.pub_pheromone.publish(mm)
                self._pher_snapshot_key = key
                self._pher_snapshot_wall_t = now_wall
                self._pher_snapshot_batch_i = 0
                self._pher_last_batch_count = batch_count

                # build full snapshot once, directly into batches (avoid large intermediate lists)
                b = int(max(2, batch_count))
                self._pher_snapshot_batches_points = [[] for _ in range(b)]
                self._pher_snapshot_batches_colors = [[] for _ in range(b)]
                i = 0
                for (cx, cy), v in layer_items:
                    wx, wy = self.grid.cell_to_world(cx, cy)
                    if wx < left or wx > right or wy < bottom or wy > top:
                        continue
                    if layer_name == "empty":
                        rr, gg, bb = (1.0, 1.0, 0.0)
                    elif layer_name == "explored":
                        rr, gg, bb = _explored_rgb((cx, cy))
                    else:
                        intensity = 1 if v <= 0.8 else 2 if v <= 1.6 else 3
                        rr, gg, bb = self._color_for_intensity(intensity)
                    pz = float(z + 0.02)
                    try:
                        # Height-aware rendering:
                        # - danger: use stored danger altitude metadata when present
                        # - nav: use stored minimum successful traversal altitude when present
                        mk = layer.meta.get((cx, cy))
                        if mk is not None and (getattr(mk, "alt_m", None) is not None):
                            if layer_name == "danger":
                                k = str(getattr(mk, "kind", "") or "")
                                if k.startswith("danger_static") or k.startswith("danger_dyn_") or k.startswith("nav_danger") or k.startswith("danger_map"):
                                    pz = float(getattr(mk, "alt_m")) + 0.02
                            elif layer_name == "nav":
                                pz = float(getattr(mk, "alt_m")) + 0.02
                    except Exception:
                        pass
                    p = Point(); p.x = float(wx); p.y = float(wy); p.z = float(pz)
                    col = ColorRGBA(); col.r = float(rr); col.g = float(gg); col.b = float(bb); col.a = float(alpha)
                    bi = int(i % b)
                    self._pher_snapshot_batches_points[bi].append(p)
                    self._pher_snapshot_batches_colors[bi].append(col)
                    i += 1

            # publish bg/border/grid once per full cycle (batch_i==0)
            if (self._pher_snapshot_batch_i % max(1, batch_count)) == 0:
                _publish_bg_border_and_grid(stamp_msg)

            bi = int(self._pher_snapshot_batch_i % max(1, batch_count))
            m = _make_cells_marker(stamp_msg, marker_id=100 + bi)
            if bi < len(self._pher_snapshot_batches_points):
                m.points = self._pher_snapshot_batches_points[bi]
                m.colors = self._pher_snapshot_batches_colors[bi]
            self.pub_pheromone.publish(m)
            self._pher_snapshot_batch_i += 1

        self._perf_add("pher_viz", time.perf_counter() - t0)

    def _publish_grid_params(self):
        if not bool(self.get_parameter("publish_grid_params").value):
            return
        msg = Float64MultiArray()
        msg.data = [float(self.grid.grid_size_m), float(self.grid.cell_size_m)]
        self.pub_grid_params.publish(msg)

    # -------- control --------

    def _on_speed(self, msg: Float32):
        v = float(msg.data)
        v = clamp(v, 0.1, 200.0)
        # If we're actively running and not paused, delay applying speed changes.
        # This prevents sudden dt_sim spikes that immediately accumulate large sim_debt.
        try:
            if bool(getattr(self, "running", False)) and (not bool(getattr(self, "paused", False))):
                self._pending_speed = float(v)
                self.get_logger().info(f"Queued speed change to {v:.2f}; will apply when paused")
                return
        except Exception:
            pass
        self.set_parameters([rclpy.parameter.Parameter("speed", rclpy.Parameter.Type.DOUBLE, v)])

    def _on_target_mode(self, msg: Bool):
        self.target_add_mode = bool(msg.data)

    def _on_clicked_point(self, msg: PointStamped):
        if not self.target_add_mode:
            return
        self._add_target(msg.point.x, msg.point.y, msg.point.z)

    def _on_add_target(self, msg: PointStamped):
        self._add_target(msg.point.x, msg.point.y, msg.point.z)

    def _add_target(self, x: float, y: float, z: float):
        # Targets are detected within the drone's sensing radius at the drone's flight altitude.
        # So we keep target.z == cruise altitude.
        tz = float(self.drone_altitude_m)
        tx, ty = float(x), float(y)
        tx, ty = clamp_xy(tx, ty, self.map_bounds)

        # If the target was placed on a building footprint, move it to the nearest free cell.
        try:
            if self.building_index.is_footprint_xy(tx, ty):
                start_c = self.grid.world_to_cell(tx, ty)
                best_xy = None
                # search expanding rings
                max_r = max(20, int(200.0 / max(1e-6, float(self.grid.cell_size_m))))
                for rr in range(0, max_r + 1):
                    for dx in range(-rr, rr + 1):
                        for dy in range(-rr, rr + 1):
                            if abs(dx) != rr and abs(dy) != rr:
                                continue
                            cc = (int(start_c[0] + dx), int(start_c[1] + dy))
                            if not self.grid.in_bounds_cell(cc[0], cc[1]):
                                continue
                            wx, wy = self.grid.cell_to_world(cc[0], cc[1])
                            if out_of_bounds(float(wx), float(wy), self.map_bounds):
                                continue
                            if not self.building_index.is_footprint_xy(float(wx), float(wy)):
                                best_xy = (float(wx), float(wy))
                                break
                        if best_xy is not None:
                            break
                    if best_xy is not None:
                        break
                if best_xy is not None:
                    tx, ty = best_xy
        except Exception:
            pass

        tid = f"T{len(self.targets)+1}"
        self.targets.append(Target(target_id=tid, x=float(tx), y=float(ty), z=float(tz)))
        self.get_logger().info(f"Added target {tid} at ({tx:.1f},{ty:.1f},{tz:.1f})")
        self._targets_dirty = True
        self._targets_version += 1

    def _srv_start(self, req, res):
        if self.running:
            res.success = True
            res.message = "Already running"
            return res
        # Starting a new mission => reset any previous exploit-run stats.
        try:
            self._exploit_active = False
            self._exploit_active_uids = set()
            self._exploit_arrived_uids = set()
            self._exploit_stats_by_uid = {}
            self.selected_target_id = ""
        except Exception:
            pass
        self.running = True
        self.returning = False
        self.mission_phase = "EXPLORE"
        self.t_sim = 0.0
        self._last_wall = time.time()
        if not bool(self.get_parameter("keep_base_between_runs").value):
            self.base_map.clear()
        z = float(self.get_parameter("drone_altitude_m").value)
        for d in self.drones:
            d.s.x, d.s.y, d.s.z = self.base_xy[0], self.base_xy[1], z
            d.s.energy_units = self.energy_model.full_units
            d.s.speed_mps = self.energy_model.normal_speed_mps
            d.s.mode = "EXPLORE"
            d.s.total_dist_m = 0.0
            d.s.encounters = 0
            d.s.base_uploads = 0
            d.s.last_comm_t = {}
            d.s.known_targets = {}
            d.s.recent_target_updates.clear()
            d.s.last_target_comm_t = {}
            d.s.recharge_until_t = 0.0
            d.pher.clear()
            self.path_hist[d.s.drone_uid].clear()
            # initial base download: drones start at base, so they know current targets
            self._sync_targets_base_to_drone(d, t_ref=0.0, force=True)
        res.success = True
        res.message = "Started"
        self.get_logger().info("Python sim started")
        return res

    def _srv_stop(self, req, res):
        if not self.running:
            res.success = True
            res.message = "Already stopped"
            return res
        # stop exploration: switch to exploit/return and finalize
        self.mission_phase = "EXPLOIT"
        for d in self.drones:
            d.s.mode = "RETURN"
        # IMPORTANT: Return-to-base is still part of Python simulation time.
        # Keep `running=True` so time acceleration (dt_sim) stays active while drones are moving.
        self.returning = True
        res.success = True
        res.message = "Return to base (accelerated time continues) + persist on arrival"
        self.get_logger().info("Python sim: return requested (accelerated time continues until all drones are at base)")
        return res

    def _srv_clear_targets(self, req, res):
        self.targets = []
        self._targets_dirty = True
        try:
            self._targets_version += 1
        except Exception:
            pass
        # If the currently selected target is gone, clear selection.
        try:
            self.selected_target_id = ""
        except Exception:
            pass
        res.success = True
        res.message = "Targets cleared"
        return res

    def _srv_clear_targets_found(self, req, res):
        """Remove only targets that are already found."""
        before = len(self.targets)
        kept: List[Target] = []
        for t in self.targets:
            try:
                if bool(t.is_found()):
                    continue
            except Exception:
                pass
            kept.append(t)
        self.targets = kept
        removed = before - len(self.targets)
        self._targets_dirty = True
        try:
            self._targets_version += 1
        except Exception:
            pass
        # If selection points to a removed target, clear it.
        try:
            sel = str(getattr(self, "selected_target_id", "") or "").strip()
            if sel and all(sel != tt.target_id for tt in self.targets):
                self.selected_target_id = ""
        except Exception:
            pass
        res.success = True
        res.message = f"Cleared found targets: removed {removed}"
        return res

    def _srv_clear_targets_unfound(self, req, res):
        """Remove only targets that are NOT found yet."""
        before = len(self.targets)
        kept: List[Target] = []
        for t in self.targets:
            try:
                if not bool(t.is_found()):
                    continue
            except Exception:
                # If unsure, treat as unfound -> clear it.
                continue
            kept.append(t)
        self.targets = kept
        removed = before - len(self.targets)
        self._targets_dirty = True
        try:
            self._targets_version += 1
        except Exception:
            pass
        # Selection is only meaningful for unfound goal-seeking; clear if not present.
        try:
            sel = str(getattr(self, "selected_target_id", "") or "").strip()
            if sel and all(sel != tt.target_id for tt in self.targets):
                self.selected_target_id = ""
        except Exception:
            pass
        res.success = True
        res.message = f"Cleared unfound targets: removed {removed}"
        return res

    def _srv_set_all_targets_unfound(self, req, res):
        """Mark all targets as UNFOUND (clear found_by/found_t), preserving positions and ids."""
        changed = 0
        for t in self.targets:
            try:
                if (t.found_by is not None) or (t.found_t is not None):
                    changed += 1
            except Exception:
                # If the shape is unexpected, still try to reset.
                changed += 1
            try:
                t.found_by = None
                t.found_t = None
            except Exception:
                pass

        if changed > 0:
            self._targets_dirty = True
            try:
                self._targets_version += 1
            except Exception:
                pass

            # Force base->drone re-sync so drones also "unfind" targets (the normal sync path only upgrades found->true).
            t_ref = float(getattr(self, "t_sim", 0.0))
            for d in getattr(self, "drones", []) or []:
                try:
                    self._sync_targets_base_to_drone(d, t_ref=t_ref, force=True)
                except Exception:
                    pass

            # Cancel "returning" phase if we are at base for other reasons (manual stop sets it too).
            try:
                if bool(getattr(self, "returning", False)):
                    self.returning = False
                    if str(getattr(self, "mission_phase", "") or "") == "EXPLOIT":
                        self.mission_phase = "EXPLORE"
                    for d in getattr(self, "drones", []) or []:
                        try:
                            if str(getattr(d.s, "mode", "") or "") == "RETURN":
                                d.s.mode = "EXPLORE"
                        except Exception:
                            pass
            except Exception:
                pass

        res.success = True
        res.message = f"Set all targets UNFOUND: reset {changed}"
        return res

    def _on_delete_pose(self, msg: PoseStamped):
        # Store last pose from external tool (RViz publish_pose / 2D Nav Goal / GUI).
        self._last_delete_pose = msg

    def _srv_delete_nearest_target(self, req, res):
        """Delete nearest target to the last received PoseStamped on delete_target_pose_topic."""
        if not self.targets:
            res.success = False
            res.message = "No targets to delete"
            return res
        pose = getattr(self, "_last_delete_pose", None)
        if pose is None:
            res.success = False
            res.message = "No pose received yet (publish a PoseStamped first)"
            return res
        px = float(pose.pose.position.x)
        py = float(pose.pose.position.y)
        # Use horizontal distance (XY) since targets are on a plane.
        best_i = -1
        best_d2 = float("inf")
        for i, t in enumerate(self.targets):
            dx = float(t.x) - px
            dy = float(t.y) - py
            d2 = (dx * dx) + (dy * dy)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_i < 0:
            res.success = False
            res.message = "Failed to find nearest target"
            return res
        removed_t = self.targets.pop(best_i)
        self._targets_dirty = True
        try:
            self._targets_version += 1
        except Exception:
            pass
        # Clear selection if it pointed to the removed target.
        try:
            if str(getattr(self, "selected_target_id", "") or "") == str(removed_t.target_id):
                self.selected_target_id = ""
        except Exception:
            pass
        res.success = True
        res.message = f"Deleted nearest target {removed_t.target_id} (dist={math.sqrt(best_d2):.2f}m)"
        return res

    # -------- core tick --------

    def _tick(self):
        # IMPORTANT: do not hold a global lock for the entire tick; it will starve RViz publishing
        # and make pointers/poses look frozen while the sim is running.
        return self._tick_locked()

    def _tick_locked(self):
        t_tick0 = time.perf_counter()
        now = time.time()
        dt_wall = now - self._last_wall
        self._last_wall = now
        if dt_wall <= 0:
            return

        # Do NOT overwrite per-drone altitude each tick; drones can climb/descend for overfly recovery.

        speed = float(self.get_parameter("speed").value)
        if self.paused:
            dt_sim_total = 0.0
        else:
            dt_sim_total = dt_wall * speed if self.running else dt_wall  # when stopped: 1x wall tick

        # publish mission phase periodically
        phase_msg = String()
        phase_msg.data = self.mission_phase
        self.pub_phase.publish(phase_msg)

        # Evaporation follows sim-time while running, and 1x wall-time when not running.
        evap_dt = dt_sim_total
        danger_kind_mult = {
            "nav_danger": float(getattr(self, "wall_danger_evap_mult", 0.02)),
            # base kinds (':<id>' suffix is handled by SparseLayer.evaporate fallback)
            "danger_static": float(getattr(self, "danger_evap_mult_static", 1.0)),
            "danger_dyn_kernel": float(getattr(self, "danger_evap_mult_dynamic", 1.25)),
            "danger_dyn_kernel_done": float(getattr(self, "danger_evap_mult_dynamic", 1.25)),
            "danger_dyn_damage": float(getattr(self, "danger_evap_mult_dynamic", 1.25)),
        }
        for d in self.drones:
            d.pher.evaporate(evap_dt, self.evap_nav_rate, self.evap_danger_rate, danger_kind_rate_mult=danger_kind_mult)
        self.base_map.evaporate(evap_dt, self.evap_nav_rate, self.evap_danger_rate, danger_kind_rate_mult=danger_kind_mult)

        # When not running, keep publishing /clock at 1x so other nodes using sim time don't freeze.
        if not self.running:
            self.t_sim += dt_sim_total
            clk = Clock()
            sec = int(self.t_sim)
            nsec = int((self.t_sim - sec) * 1e9)
            clk.clock.sec = sec
            clk.clock.nanosec = nsec
            self.pub_clock.publish(clk)
            self._last_t_ref = float(self.t_sim)
            return

        # Paused: freeze sim time, do not move/sense/communicate, but keep publishing /clock.
        if self.paused:
            clk = Clock()
            sec = int(self.t_sim)
            nsec = int((self.t_sim - sec) * 1e9)
            clk.clock.sec = sec
            clk.clock.nanosec = nsec
            self.pub_clock.publish(clk)
            self._last_t_ref = float(self.t_sim)
            return

        def _sim_step(dt_step: float, t_ref: float) -> bool:
            """One small simulation integration step."""
            any_found_local = False
            sense_period_wall = float(self.get_parameter("sense_period_s").value)
            sense_mode = str(self.sense_period_mode).strip().lower()
            sense_period_sim = float(self.sense_period_sim_s)
            max_scans_tick = int(self.max_sense_scans_per_tick)
            # Overload mode: if sim debt is high, skip sensing to reduce CPU load and keep UI responsive.
            try:
                overload_debt = float(self.get_parameter("sim_overload_debt_s").value)
                if overload_debt > 1e-9 and float(getattr(self, "_sim_debt", 0.0)) >= overload_debt:
                    max_scans_tick = 0
            except Exception:
                pass
            # Base pheromone sync settings (sim-time)
            share_pher_base = bool(self.get_parameter("share_pheromones_at_base").value)
            base_sync_r = float(self.get_parameter("base_pheromone_sync_radius_m").value)
            base_sync_cd = float(self.get_parameter("base_pheromone_sync_cooldown_s").value)
            download_pher_base = bool(self.get_parameter("download_pheromones_from_base").value)
            base_dl_max = int(self.get_parameter("base_pheromone_download_max_cells").value)

            # Snapshot all drone XY positions once per integration step (used for EXPLOIT peer-avoidance).
            try:
                swarm_xy = [(float(dd.s.x), float(dd.s.y), str(dd.s.drone_uid)) for dd in (self.drones or [])]
            except Exception:
                swarm_xy = []
            try:
                exploit_peer_params = {
                    "avoid_r": float(self.get_parameter("exploit_peer_avoid_radius_m").value),
                    "avoid_w": float(self.get_parameter("exploit_peer_avoid_weight").value),
                    "follow_w": float(self.get_parameter("exploit_peer_path_follow_weight").value),
                    "fade0": float(self.get_parameter("exploit_peer_avoid_fade_start_m").value),
                    "fade_rng": float(self.get_parameter("exploit_peer_avoid_fade_range_m").value),
                    "max_dev_deg": float(self.get_parameter("exploit_peer_avoid_max_deviation_deg").value),
                    # Motion smoothing
                    "yaw_rate": float(self.get_parameter("exploit_yaw_rate_rad_s").value),
                    "slowdown_r": float(self.get_parameter("exploit_slowdown_radius_m").value),
                    "landing_speed": float(self.get_parameter("exploit_landing_speed_mps").value),
                }
            except Exception:
                exploit_peer_params = {}
            # Snapshot exploit overlay params too (used by A* cost evaluation inside PythonDrone).
            try:
                exploit_peer_params["dyn_overlay_strength"] = float(self.get_parameter("exploit_dyn_trail_overlay_strength").value)
                exploit_peer_params["dyn_overlay_gamma"] = float(self.get_parameter("exploit_dyn_trail_overlay_gamma").value)
            except Exception:
                pass

            for d in self.drones:
                if d.s.mode == "IDLE":
                    continue
                # Make peer positions available to the drone step logic (cheap: one shared snapshot list).
                try:
                    d._swarm_xy = swarm_xy  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    d._exploit_peer_params = exploit_peer_params  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Keep per-drone wall-empty radius in sync with sim parameter.
                try:
                    d.s.empty_near_wall_radius_cells = int(self.empty_near_wall_radius_cells)
                except Exception:
                    pass

                # Altitude control (per-drone): move z toward z_target at configured rates.
                try:
                    z_min = max(0.0, float(self.min_flight_altitude_m))
                    # EXPLOIT landing must be able to reach z=0 even if min_flight_altitude_m > 0.
                    try:
                        if bool(getattr(d, "_exploit_land_active", False)) and str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT":
                            z_min = 0.0
                    except Exception:
                        pass
                    # Keep cruise/target initialized
                    if not hasattr(d.s, "z_cruise"):
                        d.s.z_cruise = float(self.drone_altitude_m)
                    if not hasattr(d.s, "z_target"):
                        d.s.z_target = float(d.s.z_cruise)
                    # If a drone is in "static danger altitude hold", force overfly_active and
                    # force z_target high enough BEFORE we apply the normal "not overfly_active -> reset z_target" logic.
                    try:
                        if bool(getattr(d.s, "preclimb_static_hold", False)):
                            req = getattr(d.s, "preclimb_static_req_alt", None)
                            if req is not None:
                                reqf = float(req)
                                if reqf > float(getattr(d.s, "z_target", d.s.z)):
                                    d.s.z_target = float(reqf)
                                d.s.overfly_active = True
                    except Exception:
                        pass
                    if not bool(getattr(d.s, "overfly_active", False)):
                        d.s.z_cruise = float(self.drone_altitude_m)
                        # EXPLOIT landing: allow setting z_target below cruise when landing is active.
                        if bool(getattr(d, "_exploit_land_active", False)) and str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT":
                            try:
                                d.s.z_target = 0.0
                            except Exception:
                                d.s.z_target = float(z_min)
                        else:
                            d.s.z_target = max(z_min, float(d.s.z_cruise))
                    # Integrate
                    tgt = max(z_min, float(d.s.z_target))
                    # Also enforce any pending static-danger requirement on the integrator target.
                    try:
                        if bool(getattr(d.s, "preclimb_static_hold", False)):
                            req = getattr(d.s, "preclimb_static_req_alt", None)
                            if req is not None:
                                tgt = max(float(tgt), float(req))
                    except Exception:
                        pass
                    curz = float(d.s.z)
                    if abs(tgt - curz) > 1e-3:
                        # Vertical speed realism: optionally tie climb/descend to horizontal speed.
                        if bool(getattr(self, "vertical_speed_mult_enabled", False)):
                            v_mult = float(getattr(self, "vertical_speed_mult", 0.30))
                            v_mult = clamp(v_mult, 0.1, 1.0)
                            # Use current commanded horizontal speed as reference (more realistic under slow-down).
                            v_ref = float(getattr(d.s, "speed_mps", float(self.energy_model.normal_speed_mps)))
                            rate = max(0.05, float(v_mult) * max(0.1, float(v_ref)))
                        else:
                            rate = float(self.climb_rate_mps) if tgt > curz else float(self.descend_rate_mps)
                        dz = math.copysign(min(abs(tgt - curz), max(0.05, rate) * float(dt_step)), tgt - curz)
                        z_next = float(curz + dz)
                        # Never descend into a building footprint: if we're above a roof, keep at least
                        # top_z + safety_margin_z + eps. This prevents "ending up inside" a building due
                        # to altitude changes after overflying.
                        try:
                            roof = self.building_index.top_z_at_xy(float(d.s.x), float(d.s.y))
                            if roof is not None:
                                min_safe = float(roof) + float(self.safety_margin_z) + 0.05
                                if z_next < min_safe:
                                    z_next = float(min_safe)
                        except Exception:
                            pass
                        d.s.z = float(z_next)
                        # Exploit stats: vertical meters flown (sum |dz|).
                        try:
                            if bool(getattr(self, "_exploit_active", False)) and str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT":
                                uid = str(getattr(d.s, "drone_uid", "") or "")
                                if uid and uid in (getattr(self, "_exploit_active_uids", set()) or set()):
                                    st = (getattr(self, "_exploit_stats_by_uid", {}) or {}).get(uid)
                                    if isinstance(st, dict):
                                        st["vert_m"] = float(st.get("vert_m", 0.0)) + abs(float(z_next) - float(curz))
                        except Exception:
                            pass
                        # Battery cost for vertical motion (approx): scale horizontal energy per meter by multiplier.
                        try:
                            vmult = float(getattr(self, "vertical_energy_cost_mult", 0.0))
                            if vmult > 1e-6:
                                d.s.energy_units = max(
                                    0.0,
                                    float(d.s.energy_units)
                                    - (abs(float(d.s.z - curz)) * float(self.energy_model.cost_per_meter_units) * vmult),
                                )
                        except Exception:
                            pass
                    else:
                        z_next = float(tgt)
                        try:
                            roof = self.building_index.top_z_at_xy(float(d.s.x), float(d.s.y))
                            if roof is not None:
                                min_safe = float(roof) + float(self.safety_margin_z) + 0.05
                                if z_next < min_safe:
                                    z_next = float(min_safe)
                        except Exception:
                            pass
                        d.s.z = float(z_next)
                        # Exploit stats: vertical meters flown (sum |dz|).
                        try:
                            if bool(getattr(self, "_exploit_active", False)) and str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT":
                                uid = str(getattr(d.s, "drone_uid", "") or "")
                                if uid and uid in (getattr(self, "_exploit_active_uids", set()) or set()):
                                    st = (getattr(self, "_exploit_stats_by_uid", {}) or {}).get(uid)
                                    if isinstance(st, dict):
                                        st["vert_m"] = float(st.get("vert_m", 0.0)) + abs(float(z_next) - float(curz))
                        except Exception:
                            pass
                except Exception:
                    pass

                # Recharge handling: after recharge delay, resume exploring
                if d.s.mode == "RECHARGE":
                    # While at base, keep syncing pheromones (upload to base, optional download back).
                    if share_pher_base and math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1]) <= base_sync_r:
                        did_sync = False
                        if (t_ref - float(d.s.last_base_pher_upload_t)) >= max(0.1, base_sync_cd):
                            if self._upload_drone_to_base(d) > 0:
                                did_sync = True
                            d.s.last_base_pher_upload_t = float(t_ref)
                        if download_pher_base and (t_ref - float(d.s.last_base_pher_download_t)) >= max(0.1, base_sync_cd):
                            # download changes since last download timestamp
                            if self._download_base_to_drone(d, since_t=float(d.s.last_base_pher_download_t), max_cells=base_dl_max) > 0:
                                did_sync = True
                            d.s.last_base_pher_download_t = float(t_ref)
                        if did_sync and self.base_comm_viz_enabled:
                            # Connect to the "base pillar" tip (same height as BASE pointer)
                            base_tip_z = float(self.get_parameter("drone_pointer_z").value)
                            with self._comm_lock:
                                self.base_comm_lines.append(
                                    (
                                        time.time() + float(self.base_comm_viz_expire_s),
                                        (d.s.x, d.s.y, d.s.z, self.base_xy[0], self.base_xy[1], base_tip_z - 1.0),
                                    )
                                )
                    if t_ref >= d.s.recharge_until_t:
                        # One last download before leaving base.
                        if share_pher_base and download_pher_base:
                            self._download_base_to_drone(d, since_t=float(d.s.last_base_pher_download_t), max_cells=base_dl_max)
                            d.s.last_base_pher_download_t = float(t_ref)
                        self._sync_targets_base_to_drone(d, t_ref=t_ref)
                        # After recharging:
                        # - Else if this drone believes all targets are found (via comm/base), stay at base (IDLE).
                        # - Else continue exploring.
                        try:
                            # "distributed mission complete": only if the drone knows ALL target ids and all are found.
                            known_map = getattr(d.s, "known_targets", {}) or {}
                            knows_all_ids = len(known_map) >= len(self.targets)
                            all_found_known = bool(known_map) and all(bool(kt.found) for kt in known_map.values())
                            if knows_all_ids and all_found_known:
                                d.s.mode = "IDLE"
                            else:
                                d.s.mode = "EXPLORE"
                        except Exception:
                            d.s.mode = "EXPLORE"
                    continue

                # Distributed return-to-base decision (no global trigger):
                # If this drone knows ALL target ids and all are found (from comm/base), return to base.
                try:
                    # In EXPLOIT (goal-seeking trial), do NOT auto-return just because targets are marked found.
                    if self.targets and d.s.mode == "EXPLORE" and str(getattr(self, "mission_phase", "") or "").upper() != "EXPLOIT":
                        known_map = getattr(d.s, "known_targets", {}) or {}
                        knows_all_ids = len(known_map) >= len(self.targets)
                        all_found_known = bool(known_map) and all(bool(kt.found) for kt in known_map.values())
                        if knows_all_ids and all_found_known:
                            d.s.mode = "RETURN"
                except Exception:
                    pass

                # Throttle sensing & target detection.
                do_sense = False
                if sense_mode == "wall":
                    if (now - d.s.last_sense_wall_t) >= max(0.02, sense_period_wall):
                        do_sense = True
                else:
                    # sim-time based sensing (keeps behavior consistent when speed is high)
                    if (t_ref - float(d.s.last_sense_t)) >= max(0.02, sense_period_sim):
                        # cap per wall tick to avoid CPU blowups when sim time jumps a lot
                        if max_scans_tick <= 0:
                            do_sense = False
                        else:
                            cnt = sense_counts.get(d.s.drone_uid, 0)
                            if cnt < max_scans_tick:
                                do_sense = True

                if do_sense:
                    t_sense0 = time.perf_counter()
                    d.s.last_sense_wall_t = float(now)
                    d.s.last_sense_t = float(t_ref)
                    sense_counts[d.s.drone_uid] = sense_counts.get(d.s.drone_uid, 0) + 1

                    danger_cells = None
                    danger_sources = {}
                    dyn_paths_by_id = {}
                    dyn_heights_by_id = {}
                    if bool(self.get_parameter("danger_map_in_lidar_enabled").value):
                        with self._danger_cells_lock:
                            danger_cells = set(self._danger_cells)
                            danger_sources = dict(getattr(self, "_danger_sources", {}))
                            dyn_paths_by_id = dict(getattr(self, "_dynamic_paths_by_id", {}) or {})
                            dyn_heights_by_id = dict(getattr(self, "_dynamic_heights_by_id", {}) or {})

                    d.reveal_with_mock_lidar(
                        building_index=self.building_index,
                        safety_margin_z=self.safety_margin_z,
                        sense_radius_m=self.sense_radius,
                        # In EXPLOIT, rely on the already-built pheromone map; keep only the O(1)-per-danger-id
                        # dynamic hazard shortcuts inside reveal_with_mock_lidar (beams disabled).
                        beam_count=(
                            0
                            if str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT"
                            and str(getattr(self, "selected_target_id", "") or "").strip()
                            else self.beam_count
                        ),
                        t_sim=t_ref,
                        danger_cells=danger_cells,
                        danger_sources=danger_sources,
                        dynamic_paths_by_id=dyn_paths_by_id,
                        dynamic_heights_by_id=dyn_heights_by_id,
                        # While overflying, keep scanning at cruise altitude to discover the building edge and
                        # mark empty space behind it. This also makes wall footprints persistently known.
                        sense_z_override=(float(getattr(d.s, "z_cruise", d.s.z)) if bool(getattr(d.s, "overfly_active", False)) else None),
                        inflate_cells=int(self.lidar_inflate_cells),
                    )
                    self._perf_add("sense_total", time.perf_counter() - t_sense0)

                # Target detection should not depend on beam discretization.
                # Run it even if we skipped this tick's LiDAR scan (prevents "in range but missed" cases).
                found = d.try_detect_targets(
                    self.targets,
                    self.sense_radius,
                    t_ref,
                    base_xy=self.base_xy,
                    exploration_area_radius_m=float(getattr(self, "exploration_area_radius_m", 0.0)),
                )
                if found:
                    any_found_local = True
                    self._targets_dirty = True

                # Snapshot dynamic threats for decision-making (even if we skipped sensing this tick).
                dyn_threats = None
                try:
                    if bool(self.get_parameter("dynamic_threat_decision_enabled").value):
                        with self._danger_cells_lock:
                            dyn_threats = list((getattr(self, "_dynamic_dangers", {}) or {}).values())
                        # Apply speed-class conventions:
                        # - For threats up to 110% of max drone horizontal speed: danger radius is 1 cell.
                        try:
                            ref_speed = float(getattr(self.energy, "normal_speed_mps", 10.0))
                            cell_m = float(max(1e-6, self.grid.cell_size_m))
                            for dd in dyn_threats or []:
                                try:
                                    sp_s = dd.get("speed", None)
                                    if sp_s is None:
                                        continue
                                    sp_s = float(sp_s)
                                    if sp_s <= 1e-6 or ref_speed <= 1e-6:
                                        continue
                                    threat_mps = cell_m / float(sp_s)
                                    ratio = float(threat_mps) / float(ref_speed)
                                    if ratio <= 1.10 + 1e-9:
                                        dd["radius"] = 1
                                except Exception:
                                    continue
                        except Exception:
                            pass
                except Exception:
                    dyn_threats = None

                # Target bias can make it look like drones "know where to look".
                # Keep this disabled by default; exploration should find targets opportunistically.
                target_goal = None
                # Optional: if GUI selected a specific target id, pursue it (goal-seeking) without switching to RETURN.
                try:
                    tid = str(getattr(self, "selected_target_id", "") or "").strip()
                    if tid:
                        for t in self.targets:
                            if str(t.target_id) == tid:
                                # EXPLOIT: use per-drone approach points so drones arrive from different angles,
                                # then switch to the final target for landing.
                                goal_xy = (float(t.x), float(t.y))
                                if str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT":
                                    try:
                                        ap = getattr(d, "_exploit_approach_world", None)
                                        fin = getattr(d, "_exploit_final_world", None)
                                        if fin is None:
                                            fin = goal_xy
                                        # Decide whether we are done with approach.
                                        reach = float(self.get_parameter("exploit_approach_reach_m").value)
                                        reach = max(1.0, reach)
                                        if ap is not None and (not bool(getattr(d, "_exploit_approach_done", False))):
                                            if math.hypot(float(d.s.x) - float(ap[0]), float(d.s.y) - float(ap[1])) <= reach:
                                                d._exploit_approach_done = True  # type: ignore[attr-defined]
                                        # Choose current goal.
                                        if ap is not None and (not bool(getattr(d, "_exploit_approach_done", False))):
                                            goal_xy = (float(ap[0]), float(ap[1]))
                                        else:
                                            goal_xy = (float(fin[0]), float(fin[1]))
                                        # Landing trigger: once approaching final goal, start descent.
                                        try:
                                            land_r = float(self.get_parameter("exploit_land_trigger_m").value)
                                            land_r = max(1.0, land_r)
                                        except Exception:
                                            land_r = 12.0
                                        if bool(getattr(d, "_exploit_approach_done", False)) and math.hypot(
                                            float(d.s.x) - float(goal_xy[0]), float(d.s.y) - float(goal_xy[1])
                                        ) <= land_r:
                                            d._exploit_land_active = True  # type: ignore[attr-defined]
                                        else:
                                            d._exploit_land_active = False  # type: ignore[attr-defined]
                                    except Exception:
                                        goal_xy = (float(t.x), float(t.y))
                                target_goal = goal_xy
                                break
                except Exception:
                    target_goal = None

                t_step0 = time.perf_counter()
                prev_db = math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1])
                prev_cell = self.grid.world_to_cell(d.s.x, d.s.y)
                # Mission phase for this step: if a concrete target is selected, act like EXPLOIT (greedy) even if
                # the global phase is still EXPLORE and we're not in RETURN.
                mp = ("EXPLOIT" if (target_goal is not None and not self.returning) else self.mission_phase)
                # Exploit option: treat dynamic-danger pheromone trails as static danger (comparison mode).
                dyn_trail_static = bool(self.get_parameter("exploit_dynamic_danger_as_static").value) if str(mp).upper() == "EXPLOIT" else False
                d.step(
                    dt=dt_step,
                    t_sim=t_ref,
                    base_xy=self.base_xy,
                    # If we have a concrete goal (selected target), behave like EXPLOIT (greedy) without RETURN.
                    mission_phase=mp,
                    building_index=self.building_index,
                    safety_margin_z=self.safety_margin_z,
                    map_bounds_m=self.map_bounds,
                    aco_temperature=self.aco_temperature,
                    target_goal=target_goal,
                    return_speed_mps=float(self.get_parameter("return_speed_mps").value),
                    return_use_aco_enabled=bool(getattr(self, "return_use_aco_enabled", True)),
                    base_no_nav_radius_m=float(self.base_no_nav_radius_m),
                    base_push_radius_m=float(self.base_push_radius_m),
                    base_push_strength=float(self.base_push_strength),
                    base_no_deposit_radius_m=float(self.base_no_deposit_radius_m),
                    explore_min_radius_m=float(self.explore_min_radius_m),
                    explore_min_radius_strength=float(self.explore_min_radius_strength),
                    explore_personal_bias_weight=float(self.explore_personal_bias_weight),
                    explore_score_noise=float(self.explore_score_noise),
                    explore_avoid_empty_weight=float(self.explore_avoid_empty_weight),
                    explore_avoid_explored_weight=float(self.explore_avoid_explored_weight),
                    empty_goal_dilate_cells=int(self.empty_goal_dilate_cells),
                    recent_cell_penalty=float(self.recent_cell_penalty),
                    explore_revisit_penalty_repeat_mult=float(getattr(self, "explore_revisit_penalty_repeat_mult", 0.0)),
                    explore_revisit_nav_deposit_scale=float(getattr(self, "explore_revisit_nav_deposit_scale", 1.0)),
                    wall_clearance_m=float(self.wall_clearance_m),
                    wall_clearance_weight=float(self.wall_clearance_weight),
                    wall_avoid_start_factor=float(self.wall_avoid_start_factor),
                    wall_avoid_yaw_weight=float(self.wall_avoid_yaw_weight),
                    wall_corridor_relax=float(self.wall_corridor_relax),
                    corner_backoff_enabled=bool(getattr(self, "corner_backoff_enabled", True)),
                    unstick_move_enabled=bool(getattr(self, "unstick_move_enabled", True)),
                    return_progress_weight=float(self.return_progress_weight),
                    return_danger_weight=float(self.return_danger_weight),
                    return_corridor_danger_relax=float(self.return_corridor_danger_relax),
                    local_plan_radius_m=float(self.local_plan_radius_m),
                    local_plan_inflate_cells=int(self.lidar_inflate_cells),
                    exploration_area_radius_m=float(getattr(self, "exploration_area_radius_m", 0.0)),
                    exploration_radius_margin_m=float(getattr(self, "exploration_radius_margin_m", 30.0)),
                    explore_nav_weight=float(getattr(self, "explore_nav_weight", 1.2)),
                    explore_far_density_weight=float(getattr(self, "explore_far_density_weight", 0.0)),
                    explore_far_density_ring_radius_cells=int(getattr(self, "explore_far_density_ring_radius_cells", 0)),
                    explore_far_density_kernel_radius_cells=int(getattr(self, "explore_far_density_kernel_radius_cells", 3)),
                    explore_far_density_angle_step_deg=float(getattr(self, "explore_far_density_angle_step_deg", 30.0)),
                    explore_far_density_exclude_inside_ring=bool(getattr(self, "explore_far_density_exclude_inside_ring", True)),
                    explore_low_nav_weight=float(getattr(self, "explore_low_nav_weight", 0.0)),
                    explore_unexplored_reward_weight=float(getattr(self, "explore_unexplored_reward_weight", 0.0)),
                    explore_explored_age_weight=float(getattr(self, "explore_explored_age_weight", 0.0)),
                    explore_explored_dist_weight=float(getattr(self, "explore_explored_dist_weight", 0.0)),
                    explore_vector_avoid_weight=float(getattr(self, "explore_vector_avoid_weight", 0.0)),
                    explore_vector_spatial_gate_m=float(getattr(self, "explore_vector_spatial_gate_m", float(self.comm_radius))),
                    explore_vector_ttl_s=float(getattr(self, "explore_vector_ttl_s", 120.0)),
                    danger_inspect_weight=float(getattr(self, "danger_inspect_weight", 0.0)),
                    danger_inspect_kernel_cells=int(getattr(self, "danger_inspect_kernel_cells", 3)),
                    danger_inspect_danger_thr=float(getattr(self, "danger_inspect_danger_thr", 0.35)),
                    danger_inspect_max_cell_danger=float(getattr(self, "danger_inspect_max_cell_danger", 0.6)),
                    dynamic_threats=dyn_threats,
                    dynamic_threat_decision_enabled=bool(self.get_parameter("dynamic_threat_decision_enabled").value),
                    dynamic_threat_cross_margin_s=float(self.get_parameter("dynamic_threat_cross_margin_s").value),
                    dynamic_threat_avoid_weight=float(self.get_parameter("dynamic_threat_avoid_weight").value),
                    dynamic_danger_trail_as_static=bool(dyn_trail_static),
                    sense_radius_m=float(self.sense_radius),
                )
                self._perf_add("drone_step_total", time.perf_counter() - t_step0)

                # Exploration vector update every N cells (for comm sharing / anti-crowding).
                try:
                    # If the avoidance weight is zero, don't generate/share vectors at all.
                    # This avoids CPU + comm overhead when the feature is disabled.
                    if float(getattr(self, "explore_vector_avoid_weight", 0.0)) <= 1e-9:
                        raise RuntimeError("explore_vector_avoid_weight=0 -> skip explore vectors")
                    share_every = int(getattr(self, "explore_vector_share_every_cells", 3))
                    share_every = int(clamp(float(share_every), 1.0, 100.0))
                    cur_cell = self.grid.world_to_cell(d.s.x, d.s.y)
                    if cur_cell != prev_cell and d.s.mode == "EXPLORE":
                        d.s.explore_vector_cells_since_update = int(getattr(d.s, "explore_vector_cells_since_update", 0)) + 1
                        if int(d.s.explore_vector_cells_since_update) >= int(share_every):
                            try:
                                # Prefer the last ACO choice yaw (more "intent") if available.
                                yaw_intent = float(getattr(d, "last_aco_choice_world", None)[2])  # type: ignore[index]
                            except Exception:
                                yaw_intent = float(d.s.yaw)
                            ev = ExploreVector(
                                origin_uid=str(d.s.drone_uid),
                                start_x=float(d.s.x),
                                start_y=float(d.s.y),
                                yaw=float(yaw_intent),
                                t=float(t_ref),
                            )
                            d.s.known_explore_vectors[str(d.s.drone_uid)] = ev
                            d.s.recent_explore_vector_updates.append(ev)
                            d.s.explore_vector_cells_since_update = 0
                except Exception:
                    pass

                # Stuck recovery / corridor overfly:
                # - If we are in a narrow corridor and a wall is close ahead, trigger overfly immediately.
                # - Otherwise, if we are not making progress for a while and a wall is close ahead, try to overfly.
                try:
                    new_db = math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1])
                    if prev_db - new_db > float(self.progress_eps_m):
                        d.s.last_progress_t = float(t_ref)
                        d.s.last_progress_dist = float(new_db)
                    else:
                        # initialize if unset
                        if float(getattr(d.s, "last_progress_t", -1e9)) < -1e8:
                            d.s.last_progress_t = float(t_ref)
                            d.s.last_progress_dist = float(new_db)

                    # Heading used for "blocked ahead" checks.
                    heading_yaw = float(d.s.yaw)
                    if d.s.mode == "RETURN":
                        heading_yaw = math.atan2(self.base_xy[1] - d.s.y, self.base_xy[0] - d.s.x)

                    # Throttle this check so it doesn't dominate CPU at high sim speed.
                    last_chk = float(getattr(d.s, "last_overfly_check_t", -1e9))
                    if (float(t_ref) - last_chk) >= 0.25:
                        d.s.last_overfly_check_t = float(t_ref)

                        # If we are in crab/avoid mode, evaluate "blocked ahead" in the *crab direction*,
                        # otherwise we'll miss the smaller side-building we are actually trying to pass.
                        probe_yaw = float(heading_yaw)
                        try:
                            if bool(getattr(d.s, "avoid_active", False)):
                                probe_yaw = float(getattr(d.s, "avoid_target_yaw", probe_yaw))
                        except Exception:
                            probe_yaw = float(heading_yaw)

                        # Measure clearances at current altitude (very cheap: a few short ray probes).
                        probe_max = max(10.0, float(self.wall_clearance_m) * 2.0)
                        probe_step = self.grid.cell_size_m * 0.5
                        fwd_clear = d._ray_clearance_m(
                            building_index=self.building_index,
                            safety_margin_z=self.safety_margin_z,
                            map_bounds_m=self.map_bounds,
                            yaw=probe_yaw,
                            max_dist_m=probe_max,
                            step_m=probe_step,
                        )
                        left_clear = d._ray_clearance_m(
                            building_index=self.building_index,
                            safety_margin_z=self.safety_margin_z,
                            map_bounds_m=self.map_bounds,
                            yaw=probe_yaw + math.pi * 0.5,
                            max_dist_m=probe_max,
                            step_m=probe_step,
                        )
                        right_clear = d._ray_clearance_m(
                            building_index=self.building_index,
                            safety_margin_z=self.safety_margin_z,
                            map_bounds_m=self.map_bounds,
                            yaw=probe_yaw - math.pi * 0.5,
                            max_dist_m=probe_max,
                            step_m=probe_step,
                        )

                        corridor_narrow = (left_clear < float(self.wall_clearance_m) * 0.9) and (right_clear < float(self.wall_clearance_m) * 0.9)
                        blocked_ahead = fwd_clear < max(6.0, float(self.wall_clearance_m) * 1.2)

                        # Static-threat "gun" overfly:
                        # If we are already inside a static-danger cell and below its required altitude,
                        # immediately climb to comply (otherwise the "ahead" probe can miss this case).
                        try:
                            cur_c = self.grid.world_to_cell(float(d.s.x), float(d.s.y))
                            cur_cc = (int(cur_c[0]), int(cur_c[1]))
                            mk_here = d.pher.danger.meta.get(cur_cc)
                            if mk_here is not None:
                                k_here = str(getattr(mk_here, "kind", "") or "")
                                alt_here = getattr(mk_here, "alt_m", None)
                                treat_dyn_trail_as_static = False
                                try:
                                    if str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT" and bool(
                                        self.get_parameter("exploit_dynamic_danger_as_static").value
                                    ):
                                        treat_dyn_trail_as_static = True
                                except Exception:
                                    treat_dyn_trail_as_static = False
                                is_static_like = bool(k_here.startswith("danger_static")) or (
                                    bool(treat_dyn_trail_as_static) and bool(k_here.startswith("danger_dyn_"))
                                )
                                if bool(is_static_like) and alt_here is not None:
                                    if (not bool(getattr(d.s, "hop_active", False))) and (not bool(getattr(d.s, "overfly_active", False))):
                                        if float(d.s.z) < float(alt_here) - 1e-6:
                                            desired = max(float(self.min_flight_altitude_m), float(alt_here))
                                            desired = min(float(desired), float(self.max_overfly_altitude_m))
                                            if float(desired) > float(d.s.z) + 0.2:
                                                d.s.overfly_active = True
                                                d.s.z_target = float(desired)
                                                d.s.overfly_start_x = float(d.s.x)
                                                d.s.overfly_start_y = float(d.s.y)
                                                d.s.overfly_start_t = float(t_ref)
                                                d.s.last_progress_t = float(t_ref)
                        except Exception:
                            pass

                        # If a static danger field is ahead and our current altitude is below its danger altitude,
                        # we may choose to climb (NO extra safety margin, unlike buildings).
                        danger_needed = None
                        try:
                            # EXPLOIT option: treat dynamic-danger pheromone trails as static "gun" fields too.
                            treat_dyn_trail_as_static = False
                            try:
                                if str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT" and bool(
                                    self.get_parameter("exploit_dynamic_danger_as_static").value
                                ):
                                    treat_dyn_trail_as_static = True
                            except Exception:
                                treat_dyn_trail_as_static = False

                            def _static_danger_required_alt(yaw0: float) -> Optional[float]:
                                stepm = max(0.5, float(self.grid.cell_size_m) * 0.5)
                                steps = int(max(1.0, float(probe_max)) / stepm)
                                req = None
                                for ii in range(1, steps + 1):
                                    px = float(d.s.x) + math.cos(float(yaw0)) * float(ii) * stepm
                                    py = float(d.s.y) + math.sin(float(yaw0)) * float(ii) * stepm
                                    if out_of_bounds(px, py, self.map_bounds):
                                        break
                                    cc = self.grid.world_to_cell(px, py)
                                    ccc = (int(cc[0]), int(cc[1]))
                                    mk = d.pher.danger.meta.get(ccc)
                                    if mk is None:
                                        continue
                                    k = str(getattr(mk, "kind", "") or "")
                                    is_static = bool(k.startswith("danger_static"))
                                    is_dyn_trail_as_static = bool(treat_dyn_trail_as_static) and bool(k.startswith("danger_dyn_"))
                                    if not (is_static or is_dyn_trail_as_static):
                                        continue
                                    alt = getattr(mk, "alt_m", None)
                                    if alt is None:
                                        continue
                                    altf = float(alt)
                                    # Only consider it blocking if we're below the danger altitude.
                                    if float(d.s.z) >= altf - 1e-6:
                                        continue
                                    req = altf if req is None else max(float(req), altf)
                                return req

                            danger_needed = _static_danger_required_alt(float(probe_yaw))
                        except Exception:
                            danger_needed = None

                        # L-corner "hop over lowest" mode:
                        # If blocked in 3 directions (front/left/right), climb just above the lowest blocking roof,
                        # move in that direction while high, and once the forward (goal) direction is clear,
                        # stop hopping and go straight.
                        try:
                            # Determine the goal-forward direction we want to be able to proceed in.
                            goal_yaw = float(heading_yaw)
                            if bool(getattr(d.s, "avoid_active", False)):
                                goal_yaw = float(getattr(d.s, "avoid_entry_yaw", goal_yaw))

                            def _first_hit_roof(yaw0: float):
                                stepm = max(0.5, float(self.grid.cell_size_m) * 0.5)
                                steps = int(max(1.0, float(probe_max)) / stepm)
                                for ii in range(1, steps + 1):
                                    px = float(d.s.x) + math.cos(float(yaw0)) * float(ii) * stepm
                                    py = float(d.s.y) + math.sin(float(yaw0)) * float(ii) * stepm
                                    if self.building_index.is_obstacle_xy(px, py, float(d.s.z), self.safety_margin_z):
                                        roof = self.building_index.max_top_z_near(px, py, radius_m=max(2.0, float(self.grid.cell_size_m) * 1.5))
                                        if roof is None:
                                            return None
                                        return float(roof) + float(self.roof_clearance_margin_m)
                                return None

                            # detect "blocked 3 ways"
                            blk_front = bool(blocked_ahead)
                            blk_left = bool(left_clear < max(6.0, float(self.wall_clearance_m) * 1.2))
                            blk_right = bool(right_clear < max(6.0, float(self.wall_clearance_m) * 1.2))
                            stuck3 = blk_front and blk_left and blk_right

                            if stuck3 and not bool(getattr(d.s, "hop_active", False)):
                                rf = _first_hit_roof(goal_yaw)
                                rl = _first_hit_roof(goal_yaw + math.pi * 0.5)
                                rr = _first_hit_roof(goal_yaw - math.pi * 0.5)
                                opts = []
                                if rf is not None:
                                    opts.append((rf, goal_yaw))
                                if rl is not None:
                                    opts.append((rl, goal_yaw + math.pi * 0.5))
                                if rr is not None:
                                    opts.append((rr, goal_yaw - math.pi * 0.5))
                                if len(opts) >= 1:
                                    # Pick the lowest roof to hop over
                                    opts.sort(key=lambda t: t[0])
                                    lowest_need, hop_dir = opts[0]
                                    desired = min(float(lowest_need), float(self.max_overfly_altitude_m))
                                    if desired > float(d.s.z) + 0.5:
                                        d.s.hop_active = True
                                        d.s.hop_dir_yaw = float(hop_dir) % (2.0 * math.pi)
                                        d.s.hop_goal_yaw = float(goal_yaw) % (2.0 * math.pi)
                                        d.s.hop_start_t = float(t_ref)
                                        # Use overfly machinery to climb, and reuse avoidance steering to move sideways/over.
                                        d.s.overfly_active = True
                                        d.s.z_target = float(desired)
                                        d.s.overfly_start_x = float(d.s.x)
                                        d.s.overfly_start_y = float(d.s.y)
                                        d.s.overfly_start_t = float(t_ref)
                                        d.s.avoid_active = True
                                        d.s.avoid_entry_yaw = float(goal_yaw)
                                        d.s.avoid_target_yaw = float(d.s.hop_dir_yaw)
                                        d.s.last_progress_t = float(t_ref)

                            # While hopping: once forward (goal) direction is clear at current altitude, stop hopping and go straight.
                            if bool(getattr(d.s, "hop_active", False)):
                                goal_yaw2 = float(getattr(d.s, "hop_goal_yaw", goal_yaw))
                                fwd2 = d._ray_clearance_m(
                                    building_index=self.building_index,
                                    safety_margin_z=self.safety_margin_z,
                                    map_bounds_m=self.map_bounds,
                                    yaw=goal_yaw2,
                                    max_dist_m=float(probe_max),
                                    step_m=probe_step,
                                )
                                if float(fwd2) >= float(probe_max) * 0.95:
                                    d.s.hop_active = False
                                    d.s.avoid_active = False
                                    # End overfly and go straight; descend to cruise.
                                    d.s.overfly_active = False
                                    d.s.z_target = max(float(self.min_flight_altitude_m), float(getattr(d.s, "z_cruise", self.drone_altitude_m)))
                        except Exception:
                            pass

                        # Identify the *specific* blocking building ahead (not the max roof in a wide radius).
                        # This prevents "climb to max altitude" behavior when a very tall building is nearby
                        # but the actual obstacle we want to clear (while going around) is shorter.
                        roof_ahead = None
                        roof_needed = None
                        hit_cell = None
                        try:
                            if blocked_ahead:
                                hit = None
                                # step along heading until we hit an obstacle at current altitude
                                stepm = max(0.5, float(self.grid.cell_size_m) * 0.5)
                                steps = int(max(1.0, float(probe_max)) / stepm)
                                for ii in range(1, steps + 1):
                                    px = float(d.s.x) + math.cos(probe_yaw) * float(ii) * stepm
                                    py = float(d.s.y) + math.sin(probe_yaw) * float(ii) * stepm
                                    if self.building_index.is_obstacle_xy(px, py, float(d.s.z), self.safety_margin_z):
                                        hit = (px, py)
                                        hit_cell = self.grid.world_to_cell(px, py)
                                        break
                                if hit is not None:
                                    roof_ahead = self.building_index.max_top_z_near(hit[0], hit[1], radius_m=max(2.0, float(self.grid.cell_size_m) * 1.5))
                                    if roof_ahead is not None:
                                        roof_needed = float(roof_ahead) + float(self.roof_clearance_margin_m)
                        except Exception:
                            roof_ahead = None
                            roof_needed = None
                            hit_cell = None

                        # Trigger overfly immediately in narrow corridors when blocked ahead.
                        should_overfly_now = corridor_narrow and blocked_ahead
                        # In avoidance/crab mode, prefer the first feasible path: if blocked in the chosen side direction,
                        # attempt overfly immediately (don't wait for "stuck").
                        if (not should_overfly_now) and bool(getattr(d.s, "avoid_active", False)) and blocked_ahead:
                            should_overfly_now = True

                        # Otherwise, only overfly when we're clearly stuck (mode-independent).
                        stuck_t = float(t_ref) - float(getattr(d.s, "last_progress_t", float(t_ref)))
                        if (not should_overfly_now) and stuck_t >= float(self.stuck_progress_timeout_s):
                            should_overfly_now = fwd_clear < max(8.0, float(self.wall_clearance_m) * 1.5)

                        if should_overfly_now:
                            roof = roof_ahead
                            if roof is None:
                                roof = self.building_index.max_top_z_near(d.s.x, d.s.y, radius_m=probe_max)
                            if roof is not None:
                                desired = max(
                                    float(self.min_flight_altitude_m),
                                    float(roof) + float(self.roof_clearance_margin_m),
                                )
                                desired = min(float(desired), float(self.max_overfly_altitude_m))
                                # If the required roof clearance is above our max overfly altitude, don't try to overfly.
                                too_high = False
                                try:
                                    if roof_needed is not None and float(roof_needed) > float(self.max_overfly_altitude_m) - 0.1:
                                        too_high = True
                                except Exception:
                                    too_high = False
                                # Decide: go around (preferred) vs overfly (vertical cost).
                                dz_up = max(0.0, float(desired) - float(d.s.z))
                                dz_down = max(0.0, float(desired) - float(d.s.z_cruise))
                                horiz = float(max(0.0, fwd_clear))
                                overfly_cost = float(self.overfly_vertical_cost_mult) * (dz_up + float(self.overfly_descend_cost_factor) * dz_down) + horiz

                                # Around cost: local A* to a point ahead (or base when returning).
                                if d.s.mode == "RETURN":
                                    gx, gy = float(self.base_xy[0]), float(self.base_xy[1])
                                else:
                                    gx = float(d.s.x) + math.cos(heading_yaw) * float(probe_max)
                                    gy = float(d.s.y) + math.sin(heading_yaw) * float(probe_max)
                                    gx, gy = clamp_xy(gx, gy, self.map_bounds)

                                around_cost = d._local_a_star_cost(
                                    goal_xy=(gx, gy),
                                    mission_phase=str(self.mission_phase),
                                    building_index=self.building_index,
                                    safety_margin_z=self.safety_margin_z,
                                    map_bounds_m=self.map_bounds,
                                    plan_radius_m=float(self.local_plan_radius_m),
                                    inflate_cells=int(self.lidar_inflate_cells),
                                    max_nodes=600,
                                    unknown_penalty=1.0,
                                    recent_penalty=float(self.recent_cell_penalty),
                                )

                                want_overfly = not bool(too_high)
                                if around_cost is not None and around_cost > 1e-6:
                                    frac = float(self.overfly_must_be_fraction_of_around)
                                    # In avoidance mode, relax the requirement: overfly is acceptable if it's not worse than going around.
                                    if bool(getattr(d.s, "avoid_active", False)):
                                        frac = max(1.0, frac)
                                    want_overfly = bool(overfly_cost <= frac * float(around_cost))

                                if want_overfly and (desired > float(d.s.z_cruise) + 1.0) and (desired > float(d.s.z) + 0.5):
                                    d.s.overfly_active = True
                                    d.s.z_target = float(desired)
                                    d.s.overfly_start_x = float(d.s.x)
                                    d.s.overfly_start_y = float(d.s.y)
                                    d.s.overfly_start_t = float(t_ref)
                                    # reset timer so we don't spam
                                    d.s.last_progress_t = float(t_ref)
                                else:
                                    # If the obstacle is too high to overfly, mark the blocking cell as "blocked"
                                    # so piercing/A* won't keep aiming into it.
                                    if bool(too_high) and hit_cell is not None:
                                        try:
                                            d.pher.deposit_danger(
                                                (int(hit_cell[0]), int(hit_cell[1])),
                                                amount=1.2,
                                                t=t_ref,
                                                conf=0.9,
                                                src="blocked",
                                                kind="blocked",
                                            )
                                        except Exception:
                                            pass
                                    # Prefer going around: enter crab avoidance instead of climbing.
                                    if bool(self.avoid_enabled) and not bool(getattr(d.s, "avoid_active", False)):
                                        side = int(getattr(d, "crab_side", 1))
                                        if bool(self.avoid_side_random_on_entry):
                                            side = -1 if (d.rng.random() < 0.5) else 1
                                        d.s.avoid_active = True
                                        d.s.avoid_entry_yaw = float(heading_yaw)
                                        d.s.avoid_side = int(side)
                                        d.s.avoid_start_x = float(d.s.x)
                                        d.s.avoid_start_y = float(d.s.y)
                                        d.s.avoid_start_t = float(t_ref)
                                        d.s.avoid_need_lateral_m = float(self.avoid_crab_offset_cells) * float(self.grid.cell_size_m)
                                        d.s.avoid_max_time_s = float(self.avoid_max_time_s)
                                        d.s.avoid_deposit_danger_amount = float(self.avoid_deposit_danger_amount)
                                        # reset timer so we don't spam
                                        d.s.last_progress_t = float(t_ref)

                        # If NOT blocked by a building, consider overflying static danger fields by climbing to their altitude.
                        try:
                            if (not bool(blocked_ahead)) and (danger_needed is not None) and (not bool(getattr(d.s, "overfly_active", False))):
                                desired = max(float(self.min_flight_altitude_m), float(danger_needed))
                                too_high = bool(desired > float(self.max_overfly_altitude_m) - 1e-6)
                                desired = min(float(desired), float(self.max_overfly_altitude_m))
                                # Decide: go around (horizontal) vs overfly (vertical cost).
                                dz_up = max(0.0, float(desired) - float(d.s.z))
                                dz_down = max(0.0, float(desired) - float(d.s.z_cruise))
                                horiz = float(max(0.0, fwd_clear))
                                overfly_cost = float(self.overfly_vertical_cost_mult) * (dz_up + float(self.overfly_descend_cost_factor) * dz_down) + horiz

                                # Around cost: local A* to a point ahead (or base when returning).
                                if d.s.mode == "RETURN":
                                    gx, gy = float(self.base_xy[0]), float(self.base_xy[1])
                                else:
                                    gx = float(d.s.x) + math.cos(heading_yaw) * float(probe_max)
                                    gy = float(d.s.y) + math.sin(heading_yaw) * float(probe_max)
                                    gx, gy = clamp_xy(gx, gy, self.map_bounds)
                                around_cost = d._local_a_star_cost(
                                    goal_xy=(gx, gy),
                                    mission_phase=str(self.mission_phase),
                                    building_index=self.building_index,
                                    safety_margin_z=self.safety_margin_z,
                                    map_bounds_m=self.map_bounds,
                                    plan_radius_m=float(self.local_plan_radius_m),
                                    inflate_cells=int(self.lidar_inflate_cells),
                                    max_nodes=600,
                                    unknown_penalty=1.0,
                                    recent_penalty=float(self.recent_cell_penalty),
                                )
                                want_overfly = not bool(too_high)
                                if around_cost is not None and float(around_cost) > 1e-6:
                                    frac = float(self.overfly_must_be_fraction_of_around)
                                    want_overfly = bool(overfly_cost <= frac * float(around_cost))
                                if want_overfly and (desired > float(d.s.z_cruise) + 0.5) and (desired > float(d.s.z) + 0.5):
                                    d.s.overfly_active = True
                                    d.s.z_target = float(desired)
                                    d.s.overfly_start_x = float(d.s.x)
                                    d.s.overfly_start_y = float(d.s.y)
                                    d.s.overfly_start_t = float(t_ref)
                                    d.s.last_progress_t = float(t_ref)
                        except Exception:
                            pass

                        # If we're already at max overfly altitude but the building ahead still needs more,
                        # abort overfly and switch to crab avoidance (explore around to discover empty space).
                        try:
                            if bool(getattr(d.s, "overfly_active", False)) and blocked_ahead:
                                if roof_needed is not None and float(roof_needed) > float(self.max_overfly_altitude_m) - 0.1 and float(d.s.z) >= float(self.max_overfly_altitude_m) - 0.6:
                                    # Mark blocked
                                    if hit_cell is not None:
                                        try:
                                            d.pher.deposit_danger(
                                                (int(hit_cell[0]), int(hit_cell[1])),
                                                amount=1.2,
                                                t=t_ref,
                                                conf=0.9,
                                                src="blocked",
                                                kind="blocked",
                                            )
                                        except Exception:
                                            pass
                                    d.s.overfly_active = False
                                    d.s.z_target = max(float(self.min_flight_altitude_m), float(getattr(d.s, "z_cruise", self.drone_altitude_m)))
                                    if bool(self.avoid_enabled) and not bool(getattr(d.s, "avoid_active", False)):
                                        side = int(getattr(d, "crab_side", 1))
                                        if bool(self.avoid_side_random_on_entry):
                                            side = -1 if (d.rng.random() < 0.5) else 1
                                        d.s.avoid_active = True
                                        d.s.avoid_entry_yaw = float(heading_yaw)
                                        d.s.avoid_side = int(side)
                                        d.s.avoid_start_x = float(d.s.x)
                                        d.s.avoid_start_y = float(d.s.y)
                                        d.s.avoid_start_t = float(t_ref)
                                        d.s.avoid_need_lateral_m = float(self.avoid_crab_offset_cells) * float(self.grid.cell_size_m)
                                        d.s.avoid_max_time_s = float(self.avoid_max_time_s)
                                        d.s.avoid_deposit_danger_amount = float(self.avoid_deposit_danger_amount)
                                        d.s.last_progress_t = float(t_ref)
                        except Exception:
                            pass

                        # If already overflying and forward is clear at cruise altitude, allow descent.
                        if bool(getattr(d.s, "overfly_active", False)):
                            cruise_z = float(d.s.z_cruise)
                            ok = True
                            for dist_m in (5.0, 10.0, 15.0):
                                px = d.s.x + math.cos(heading_yaw) * dist_m
                                py = d.s.y + math.sin(heading_yaw) * dist_m
                                if self.building_index.is_obstacle_xy(px, py, cruise_z, self.safety_margin_z):
                                    ok = False
                                    break
                            # Only end overfly after:
                            # - we actually reached the overfly altitude, AND
                            # - we have moved forward a bit while high (prevents "jumping"),
                            # - and the path at cruise altitude is now clear.
                            moved = math.hypot(float(d.s.x) - float(getattr(d.s, "overfly_start_x", d.s.x)), float(d.s.y) - float(getattr(d.s, "overfly_start_y", d.s.y)))
                            reached = float(d.s.z) >= float(d.s.z_target) - 0.6
                            moved_enough = moved >= max(6.0, float(self.grid.cell_size_m) * 1.2)
                            if ok and reached and moved_enough:
                                d.s.overfly_active = False
                                d.s.z_target = max(float(self.min_flight_altitude_m), float(d.s.z_cruise))
                except Exception:
                    pass

                with self._state_lock:
                    self.path_hist[d.s.drone_uid].append((d.s.x, d.s.y, d.s.z))

                if d.s.mode == "RECHARGE":
                    if bool(self.get_parameter("share_targets_at_base").value):
                        self._sync_targets_base_to_drone(d, t_ref=t_ref)
                    d.s.recharge_until_t = t_ref + 3.0
                    # Immediately sync pheromones on entering recharge.
                    if share_pher_base and math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1]) <= base_sync_r:
                        did_sync = False
                        if self._upload_drone_to_base(d) > 0:
                            did_sync = True
                        d.s.last_base_pher_upload_t = float(t_ref)
                        if download_pher_base:
                            if self._download_base_to_drone(d, since_t=float(d.s.last_base_pher_download_t), max_cells=base_dl_max) > 0:
                                did_sync = True
                            d.s.last_base_pher_download_t = float(t_ref)
                        if did_sync and self.base_comm_viz_enabled:
                            # Connect to the "base pillar" tip (same height as BASE pointer)
                            base_tip_z = float(self.get_parameter("drone_pointer_z").value)
                            with self._comm_lock:
                                self.base_comm_lines.append(
                                    (
                                        time.time() + float(self.base_comm_viz_expire_s),
                                        (d.s.x, d.s.y, d.s.z, self.base_xy[0], self.base_xy[1], base_tip_z - 1.0),
                                    )
                                )

            return any_found_local
            # (unreachable)

        # Substep integration to avoid huge per-tick jumps when speed multiplier is high.
        any_found = False
        t_local = float(self.t_sim)
        remaining = float(dt_sim_total + float(self._sim_debt))
        prev_debt = float(self._sim_debt)
        # Per-tick sensing cap bookkeeping (used when sense_period_mode == "sim")
        sense_counts: Dict[str, int] = {}
        t_simloop0 = time.perf_counter()

        # Compute a conservative dt_step so drones don't jump across many grid cells in one update.
        max_move = float(self.get_parameter("max_move_per_step_m").value)
        if max_move <= 1e-6:
            max_move = float(self.grid.cell_size_m) * 0.8
        max_speed = max(
            float(self.energy_model.normal_speed_mps),
            float(self.energy_model.low_energy_speed_mps),
            float(self.get_parameter("return_speed_mps").value),
        )
        dt_step = max(0.01, min(0.2, max_move / max(0.1, max_speed)))

        max_substeps = int(self.get_parameter("max_sim_substeps_per_tick").value)
        steps = 0
        while remaining > 1e-9 and steps < max_substeps:
            # Optional wall-time budget: stop sim work early to avoid starving ROS callbacks / GUI.
            try:
                wall_budget = float(self.get_parameter("sim_tick_wall_budget_s").value)
                if wall_budget > 1e-9 and (time.perf_counter() - t_simloop0) >= wall_budget:
                    break
            except Exception:
                pass
            dt_s = min(dt_step, remaining)
            t_local += dt_s
            t_ref = t_local
            if _sim_step(dt_s, t_ref):
                any_found = True
            remaining -= dt_s
            steps += 1
        self._perf_add("sim_step_total", time.perf_counter() - t_simloop0)
        # Carry over any unprocessed simulated time so we don't freeze the GUI on huge speed multipliers.
        self._sim_debt = max(0.0, remaining)
        self._perf_tick_n += 1
        # Amount of simulated time actually processed in this tick (includes carried debt).
        processed_sim = (float(dt_sim_total) + prev_debt) - float(remaining)
        self._perf_sim_advanced_s += float(processed_sim)
        self._perf_sim_debt_s = float(self._sim_debt)
        self._perf_substeps += int(steps)

        # Always publish /clock while sim is running (accelerated).
        self.t_sim = t_local
        clk = Clock()
        sec = int(self.t_sim)
        nsec = int((self.t_sim - sec) * 1e9)
        clk.clock.sec = sec
        clk.clock.nanosec = nsec
        self.pub_clock.publish(clk)

        t_ref = self.t_sim
        self._last_t_ref = float(t_ref)

        # comms: exchange pheromone diffs within 200m
        self._handle_comms(t_ref)

        # If all visible targets (inside exploration radius) are found -> request return.
        radius = float(getattr(self, "exploration_area_radius_m", 0.0))
        if radius > 1e-6:
            active_targets = [t for t in self.targets if math.hypot(float(t.x) - float(self.base_xy[0]), float(t.y) - float(self.base_xy[1])) <= radius + 1e-6]
        else:
            active_targets = list(self.targets)
        # IMPORTANT: no global "all targets found => everyone returns" logic.
        # Each drone decides to return based on its OWN communicated target knowledge.
        # (Inspector exception: it may recharge, but resumes inspection until trajectory is complete.)

        # Finalize when return requested and all are at base
        if self.running and self.returning:
            all_at_base = all(math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1]) <= self.grid.cell_size_m * 2.0 for d in self.drones)
            if all_at_base:
                self._upload_all_to_base(t_ref)
                self._persist_outputs()
                for d in self.drones:
                    d.s.mode = "IDLE"
                # Stop accelerated time after everyone is home (pause-independent)
                self.running = False
                self.returning = False
                # Freeze mission time (so user can snapshot/compare without evaporation drift).
                self.paused = True
                self.gui_log = "Explore->return complete: all drones at base; time frozen (paused)"

        # Exploration complete (distributed): if all targets are found and all drones are back at base/idle, freeze time.
        try:
            if self.running and (not self.returning):
                all_found = (len(active_targets) > 0) and all(bool(t.is_found()) for t in active_targets)
                all_at_base2 = all(
                    math.hypot(d.s.x - self.base_xy[0], d.s.y - self.base_xy[1]) <= self.grid.cell_size_m * 2.0 for d in self.drones
                )
                all_idle = all(str(getattr(d.s, "mode", "") or "") == "IDLE" for d in self.drones)
                if all_found and all_at_base2 and all_idle and str(getattr(self, "mission_phase", "") or "").upper() != "EXPLOIT":
                    self.running = False
                    self.paused = True
                    self.gui_log = "Explore complete: all targets found and all drones at base/idle; time frozen (paused)"
        except Exception:
            pass

        # Exploit-run completion: when all exploit drones reach the selected target, freeze time and emit stats.
        try:
            if (
                bool(getattr(self, "_exploit_active", False))
                and bool(getattr(self, "selected_target_id", ""))
                and (not bool(getattr(self, "returning", False)))
                and bool(getattr(self, "running", False))
                and str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT"
            ):
                tid = str(getattr(self, "selected_target_id", "") or "").strip()
                tgt0 = None
                for tt in self.targets:
                    if str(tt.target_id) == tid:
                        tgt0 = tt
                        break
                if tgt0 is not None:
                    # Stop exploit when drones are in the *target cell* and landed at (near) zero altitude.
                    tgt_cell = self.grid.world_to_cell(float(tgt0.x), float(tgt0.y))
                    land_z = float(self.get_parameter("exploit_land_z_done_m").value)
                    for d in self.drones:
                        uid = str(getattr(d.s, "drone_uid", "") or "")
                        if uid and uid in (getattr(self, "_exploit_active_uids", set()) or set()):
                            if (
                                tuple(self.grid.world_to_cell(float(d.s.x), float(d.s.y))) == tuple(tgt_cell)
                                and float(getattr(d.s, "z", 0.0)) <= float(land_z) + 1e-6
                            ):
                                if uid not in self._exploit_arrived_uids:
                                    # Record per-drone landing time/energy once.
                                    try:
                                        st = (getattr(self, "_exploit_stats_by_uid", {}) or {}).get(uid)
                                        if isinstance(st, dict) and st.get("land_t", None) is None:
                                            st["land_t"] = float(getattr(self, "t_sim", 0.0))
                                            st["land_energy"] = float(getattr(d.s, "energy_units", 0.0))
                                    except Exception:
                                        pass
                                self._exploit_arrived_uids.add(uid)
                                d.s.mode = "IDLE"
                    if self._exploit_active_uids and self._exploit_arrived_uids.issuperset(self._exploit_active_uids):
                        rows = []
                        for d in self.drones:
                            uid = str(getattr(d.s, "drone_uid", "") or "")
                            if uid and uid in self._exploit_active_uids:
                                st = (getattr(self, "_exploit_stats_by_uid", {}) or {}).get(uid, {}) if uid else {}
                                try:
                                    t0 = float(st.get("start_t", float(getattr(self, "_exploit_start_t", 0.0))))
                                except Exception:
                                    t0 = float(getattr(self, "_exploit_start_t", 0.0))
                                try:
                                    tl = st.get("land_t", None)
                                    t_land = (float(tl) - float(t0)) if tl is not None else None
                                except Exception:
                                    t_land = None
                                try:
                                    e0 = float(st.get("start_energy", float(self.energy_model.full_units)))
                                except Exception:
                                    e0 = float(self.energy_model.full_units)
                                try:
                                    el_raw = st.get("land_energy", None)
                                    e_land = float(el_raw) if el_raw is not None else float(getattr(d.s, "energy_units", 0.0))
                                except Exception:
                                    e_land = float(getattr(d.s, "energy_units", 0.0))
                                spent = max(0.0, float(e0) - float(e_land))
                                frac = (float(e_land) / float(self.energy_model.full_units)) if float(self.energy_model.full_units) > 1e-9 else 0.0
                                horiz = float(getattr(d.s, "total_dist_m", 0.0))
                                try:
                                    vert = float(st.get("vert_m", 0.0))
                                except Exception:
                                    vert = 0.0
                                hv = float(horiz) + float(vert)
                                t_txt = f"{t_land:.1f}s" if t_land is not None else "—"
                                rows.append(
                                    f"{uid}: time={t_txt}, dist(h+v)={hv:.1f}m (h={horiz:.1f}, v={vert:.1f}), battery_spent={spent:.2f}u, battery_left={100.0*frac:.1f}%"
                                )
                        self.gui_log = "Exploit complete:\n" + ("\n".join(rows) if rows else "—")
                        self._exploit_active = False
                        self.running = False
                        self.paused = True
        except Exception:
            pass

        # RViz publishing is throttled via a timer; do not publish every tick.
        self._perf_add("tick_total", time.perf_counter() - t_tick0)
        self._perf_maybe_log(time.time())

    def _sync_targets_base_to_drone(self, drone: PythonDrone, t_ref: float, force: bool = False):
        """
        Base -> drone download:
        - drones learn about targets (existence + location)
        - drones learn 'found' status if base knows it
        """
        # Only share targets inside the exploration area radius (if enabled).
        radius = float(getattr(self, "exploration_area_radius_m", 0.0))
        for tgt in self.targets:
            if radius > 1e-6:
                if math.hypot(float(tgt.x) - float(self.base_xy[0]), float(tgt.y) - float(self.base_xy[1])) > radius + 1e-6:
                    continue
            kt = drone.s.known_targets.get(tgt.target_id)
            if kt is None or force:
                nk = TargetKnowledge(
                    target_id=tgt.target_id,
                    x=float(tgt.x),
                    y=float(tgt.y),
                    z=float(tgt.z),
                    found=bool(tgt.is_found()),
                    found_by=tgt.found_by,
                    found_t=tgt.found_t,
                    updated_t=float(t_ref),
                )
                drone.s.known_targets[tgt.target_id] = nk
                drone.s.recent_target_updates.append(nk)
            else:
                # update found status if base has newer info
                if tgt.is_found() and not kt.found:
                    kt.found = True
                    kt.found_by = tgt.found_by
                    kt.found_t = tgt.found_t
                    kt.updated_t = float(t_ref)
                    drone.s.recent_target_updates.append(kt)

    def _publish_gui_status(self):
        with self._state_lock:
            return self._publish_gui_status_locked()

    def _publish_gui_status_locked(self):
        # Targets summary (only targets inside exploration area radius are visible/counted).
        radius = float(getattr(self, "exploration_area_radius_m", 0.0))
        if radius > 1e-6:
            visible_targets = [t for t in self.targets if math.hypot(float(t.x) - float(self.base_xy[0]), float(t.y) - float(self.base_xy[1])) <= radius + 1e-6]
        else:
            visible_targets = list(self.targets)
        total = len(visible_targets)
        found = sum(1 for t in visible_targets if t.is_found())
        unfound = total - found

        drone_rows = []
        all_ids = [t.target_id for t in visible_targets]
        t_sim_now = float(getattr(self, "t_sim", 0.0))
        for d in self.drones:
            known_ids = set(d.s.known_targets.keys())
            missing = [tid for tid in all_ids if tid not in known_ids]
            not_found_known = [tid for tid, kt in d.s.known_targets.items() if not kt.found and tid in all_ids]
            # Debug fields (helps explain "stuck" / A* overrides from the GUI).
            try:
                src = str(getattr(d, "last_move_source", "") or "")
            except Exception:
                src = ""
            try:
                plan_age = float(t_sim_now - float(getattr(d, "last_plan_t", -1e9)))
            except Exception:
                plan_age = 1e9
            try:
                aco_age = float(t_sim_now - float(getattr(d, "last_aco_choice_t", -1e9)))
            except Exception:
                aco_age = 1e9
            drone_rows.append(
                {
                    "uid": d.s.drone_uid,
                    "mode": d.s.mode,
                    "x": float(d.s.x),
                    "y": float(d.s.y),
                    "z": float(d.s.z),
                    "known_targets": len(known_ids),
                    "missing_targets": missing,
                    "not_found_known": not_found_known,
                    "last_move_source": src,
                    "avoid_active": bool(getattr(d.s, "avoid_active", False)),
                    "overfly_active": bool(getattr(d.s, "overfly_active", False)),
                    "hop_active": bool(getattr(d.s, "hop_active", False)),
                    "plan_age_s": float(plan_age),
                    "aco_age_s": float(aco_age),
                }
            )

        # Include a few runtime stats to make performance issues visible from the GUI.
        try:
            sim_debt = float(getattr(self, "_sim_debt", 0.0))
        except Exception:
            sim_debt = 0.0
        try:
            speed = float(self.get_parameter("speed").value)
        except Exception:
            speed = 0.0

        # Report speed limits for UI:
        # - horizontal max: normal cruise vs return speed
        # - vertical max: multiplier * horizontal max (when multiplier mode enabled)
        try:
            max_h = float(max(float(self.energy_model.normal_speed_mps), float(self.get_parameter("return_speed_mps").value)))
        except Exception:
            max_h = float(getattr(self.energy_model, "normal_speed_mps", 10.0))
        v_mult = float(getattr(self, "vertical_speed_mult", 0.30))
        v_mult_enabled = bool(getattr(self, "vertical_speed_mult_enabled", False))
        max_v = (float(clamp(v_mult, 0.1, 1.0)) * float(max_h)) if v_mult_enabled else float(
            max(getattr(self, "climb_rate_mps", 0.0), getattr(self, "descend_rate_mps", 0.0))
        )

        payload = {
            "targets_total": total,
            "targets_found": found,
            "targets_unfound": unfound,
            "targets": [{"id": str(t.target_id), "x": float(t.x), "y": float(t.y), "z": float(t.z), "found": bool(t.is_found())} for t in visible_targets],
            "selected_target_id": str(getattr(self, "selected_target_id", "") or ""),
            "sim_log": str(getattr(self, "gui_log", "") or ""),
            "drones": drone_rows,
            "sim_debt_s": sim_debt,
            "speed": speed,
            "paused": bool(getattr(self, "paused", False)),
            "running": bool(getattr(self, "running", False)),
            "max_horizontal_speed_mps": float(max_h),
            "vertical_speed_mult_enabled": bool(v_mult_enabled),
            "vertical_speed_mult": float(v_mult),
            "max_vertical_speed_mps": float(max_v),
        }
        # Exploit-run per-drone stats (only populated for GUI-initiated exploit runs).
        try:
            ex_active = bool(getattr(self, "_exploit_active", False))
        except Exception:
            ex_active = False
        try:
            ex_start = float(getattr(self, "_exploit_start_t", 0.0))
        except Exception:
            ex_start = 0.0
        ex_rows: List[dict] = []
        try:
            stats_by_uid = getattr(self, "_exploit_stats_by_uid", {}) or {}
            active_uids = getattr(self, "_exploit_active_uids", set()) or set()
            for d in self.drones:
                uid = str(getattr(d.s, "drone_uid", "") or "")
                if not uid:
                    continue
                if uid not in active_uids and uid not in (stats_by_uid.keys() if isinstance(stats_by_uid, dict) else []):
                    continue
                st = stats_by_uid.get(uid, {}) if isinstance(stats_by_uid, dict) else {}
                try:
                    t0 = float(st.get("start_t", ex_start))
                except Exception:
                    t0 = ex_start
                try:
                    tl = st.get("land_t", None)
                    t_land = (float(tl) - float(t0)) if tl is not None else None
                except Exception:
                    t_land = None
                try:
                    e0 = float(st.get("start_energy", float(self.energy_model.full_units)))
                except Exception:
                    e0 = float(self.energy_model.full_units)
                try:
                    el_raw = st.get("land_energy", None)
                    e_land = float(el_raw) if el_raw is not None else None
                except Exception:
                    e_land = None
                try:
                    vert = float(st.get("vert_m", 0.0))
                except Exception:
                    vert = 0.0
                horiz = float(getattr(d.s, "total_dist_m", 0.0))
                # If not landed yet, show live battery left; if landed, use recorded land_energy if present.
                e_now = float(getattr(d.s, "energy_units", 0.0))
                e_used = max(0.0, float(e0) - float(e_land if e_land is not None else e_now))
                ex_rows.append(
                    {
                        "uid": uid,
                        "landed": bool(t_land is not None),
                        "time_to_land_s": (float(t_land) if t_land is not None else None),
                        "battery_spent_units": float(e_used),
                        "battery_left_units": float(e_land if e_land is not None else e_now),
                        "dist_horizontal_m": float(horiz),
                        "dist_vertical_m": float(vert),
                        "dist_hv_m": float(horiz + vert),
                    }
                )
        except Exception:
            ex_rows = []
        payload["exploit_stats"] = {
            "active": bool(ex_active),
            "target_id": str(getattr(self, "selected_target_id", "") or ""),
            "start_t_sim": float(ex_start),
            "drones": ex_rows,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_gui_status.publish(msg)

    def _targets_file_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / str(self.get_parameter("targets_path").value)

    def _load_targets(self):
        path = self._targets_file_path()
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            loaded: List[Target] = []
            for item in data.get("targets", []):
                t = Target(
                    target_id=str(item.get("id", "")) or f"T{len(loaded)+1}",
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", 0.0)),
                    found_by=item.get("found_by"),
                    found_t=item.get("found_t"),
                )
                loaded.append(t)
            self.targets = loaded
            self._targets_dirty = False
            self.get_logger().info(f"Loaded {len(self.targets)} targets from {path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load targets from {path}: {e}")

    def _save_targets(self):
        if not bool(self.get_parameter("persist_targets").value):
            return
        path = self._targets_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "time": {"wall": time.time(), "t_sim": float(self.t_sim)},
            "targets": [
                {
                    "id": t.target_id,
                    "x": float(t.x),
                    "y": float(t.y),
                    "z": float(t.z),
                    "found_by": t.found_by,
                    "found_t": t.found_t,
                }
                for t in self.targets
            ],
        }
        tmp = str(path) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
            self._targets_dirty = False
        except Exception as e:
            self.get_logger().warn(f"Failed to save targets to {path}: {e}")

    def _autosave_targets(self):
        if self._targets_dirty:
            self._save_targets()

    def _handle_comms(self, t_ref: float):
        # For simplicity: when close, exchange recent updates since last exchange time.
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                a = self.drones[i]
                b = self.drones[j]
                dist = math.hypot(a.s.x - b.s.x, a.s.y - b.s.y)
                if dist > self.comm_radius:
                    continue
                peer_key_a = b.s.drone_uid
                peer_key_b = a.s.drone_uid
                last_a = a.s.last_comm_t.get(peer_key_a, -1.0)
                last_b = b.s.last_comm_t.get(peer_key_b, -1.0)
                since = max(last_a, last_b)

                patch_a_nav = a.pher.make_patch("nav", max_cells=2000, since_t=since)
                patch_a_danger = a.pher.make_patch("danger", max_cells=2000, since_t=since)
                patch_a_empty = a.pher.make_patch("empty", max_cells=2000, since_t=since)
                patch_a_explored = a.pher.make_patch("explored", max_cells=2000, since_t=since)
                patch_b_nav = b.pher.make_patch("nav", max_cells=2000, since_t=since)
                patch_b_danger = b.pher.make_patch("danger", max_cells=2000, since_t=since)
                patch_b_empty = b.pher.make_patch("empty", max_cells=2000, since_t=since)
                patch_b_explored = b.pher.make_patch("explored", max_cells=2000, since_t=since)

                changed = 0
                changed += b.pher.merge_patch("nav", patch_a_nav)
                changed += b.pher.merge_patch("danger", patch_a_danger)
                changed += b.pher.merge_patch("empty", patch_a_empty)
                changed += b.pher.merge_patch("explored", patch_a_explored)
                changed += a.pher.merge_patch("nav", patch_b_nav)
                changed += a.pher.merge_patch("danger", patch_b_danger)
                changed += a.pher.merge_patch("empty", patch_b_empty)
                changed += a.pher.merge_patch("explored", patch_b_explored)

                # Ingest dynamic kernel observations from received danger patches (for threat reasoning).
                try:
                    self._ingest_dynamic_from_danger_patch(b, patch_a_danger)
                    self._ingest_dynamic_from_danger_patch(a, patch_b_danger)
                except Exception:
                    pass

                a.s.last_comm_t[peer_key_a] = t_ref
                b.s.last_comm_t[peer_key_b] = t_ref

                if changed > 0:
                    # comm wire for visualization (expire quickly). Skip very near pairs (clutter/obvious).
                    if self.comm_viz_enabled and self.comm_viz_mode == "events" and dist >= float(self.comm_viz_min_dist_m):
                        with self._comm_lock:
                            # cap buffer size (deque already caps, but avoid churning)
                            if int(self.comm_viz_max_lines) > 0 and len(self.comm_lines) >= int(self.comm_viz_max_lines):
                                pass
                            else:
                                self.comm_lines.append(
                                    (time.time() + float(self.comm_viz_expire_s), (a.s.x, a.s.y, a.s.z, b.s.x, b.s.y, b.s.z))
                                )

                # Target knowledge exchange (IDs + locations + found flag)
                if bool(self.get_parameter("share_targets_comm").value) and dist <= float(self.get_parameter("targets_comm_radius").value):
                    self._exchange_targets(a, b, t_ref)
                # Exploration vector exchange (anti-crowding)
                try:
                    # Skip any explore-vector comm overhead when the feature isn't used in scoring.
                    if bool(getattr(self, "share_explore_vectors_comm", True)) and float(getattr(self, "explore_vector_avoid_weight", 0.0)) > 1e-9:
                        self._exchange_explore_vectors(a, b, t_ref)
                except Exception:
                    pass
                # Dynamic danger inspector conflict resolution (older timestamp wins).
                try:
                    aid = str(getattr(a.s, "dynamic_inspect_active_id", "") or "").strip()
                    bid = str(getattr(b.s, "dynamic_inspect_active_id", "") or "").strip()
                    if aid and (aid == bid):
                        at = float(getattr(a.s, "dynamic_inspect_active_t", -1e9))
                        bt = float(getattr(b.s, "dynamic_inspect_active_t", -1e9))
                        # newer stops + adds to skip set
                        if at > bt + 1e-6:
                            a.s.dynamic_inspect_active_id = ""
                            a.s.dynamic_inspect_skip_ids.add(str(aid))
                        elif bt > at + 1e-6:
                            b.s.dynamic_inspect_active_id = ""
                            b.s.dynamic_inspect_skip_ids.add(str(bid))
                        else:
                            # tie: lower seq wins (stable)
                            if int(a.s.seq) > int(b.s.seq):
                                a.s.dynamic_inspect_active_id = ""
                                a.s.dynamic_inspect_skip_ids.add(str(aid))
                            elif int(b.s.seq) > int(a.s.seq):
                                b.s.dynamic_inspect_active_id = ""
                                b.s.dynamic_inspect_skip_ids.add(str(bid))
                except Exception:
                    pass

    def _exchange_targets(self, a: PythonDrone, b: PythonDrone, t_ref: float):
        max_items = int(self.get_parameter("targets_comm_max").value)
        last_a = a.s.last_target_comm_t.get(b.s.drone_uid, -1.0)
        last_b = b.s.last_target_comm_t.get(a.s.drone_uid, -1.0)
        since = max(last_a, last_b)

        def patch(dr: PythonDrone):
            out = []
            for kt in reversed(dr.s.recent_target_updates):
                if kt.updated_t <= since + 1e-9:
                    break
                out.append(
                    {
                        "id": kt.target_id,
                        "x": kt.x,
                        "y": kt.y,
                        "z": kt.z,
                        "found": kt.found,
                        "found_by": kt.found_by,
                        "found_t": kt.found_t,
                        "t": kt.updated_t,
                    }
                )
                if len(out) >= max_items:
                    break
            return out

        pa = patch(a)
        pb = patch(b)

        def apply(dst: PythonDrone, items: List[dict]):
            changed = 0
            for it in items:
                tid = str(it["id"])
                t_upd = float(it.get("t", t_ref))
                cur = dst.s.known_targets.get(tid)
                if cur is None or t_upd > cur.updated_t + 1e-6:
                    nk = TargetKnowledge(
                        target_id=tid,
                        x=float(it["x"]),
                        y=float(it["y"]),
                        z=float(it.get("z", 0.0)),
                        found=bool(it.get("found", False)),
                        found_by=it.get("found_by"),
                        found_t=it.get("found_t"),
                        updated_t=t_upd,
                    )
                    dst.s.known_targets[tid] = nk
                    dst.s.recent_target_updates.append(nk)
                    changed += 1
                else:
                    # allow found flag to upgrade even if timestamps equal-ish
                    if bool(it.get("found", False)) and not cur.found:
                        cur.found = True
                        cur.found_by = it.get("found_by")
                        cur.found_t = it.get("found_t")
                        cur.updated_t = max(cur.updated_t, t_upd)
                        dst.s.recent_target_updates.append(cur)
                        changed += 1
            return changed

        changed = apply(a, pb) + apply(b, pa)
        a.s.last_target_comm_t[b.s.drone_uid] = t_ref
        b.s.last_target_comm_t[a.s.drone_uid] = t_ref
        return changed

    def _exchange_explore_vectors(self, a: PythonDrone, b: PythonDrone, t_ref: float):
        """
        Exchange recently updated exploration intent vectors between drones and allow forwarding.
        This helps drones avoid choosing the same direction as peers they didn't directly meet.
        """
        # Fast-path: if the feature isn't used, don't spend any time building/merging patches.
        try:
            if float(getattr(self, "explore_vector_avoid_weight", 0.0)) <= 1e-9:
                return 0
        except Exception:
            return 0
        max_items = int(getattr(self, "explore_vector_comm_max", 120))
        max_items = int(clamp(float(max_items), 0.0, 10000.0))
        last_a = a.s.last_explore_vector_comm_t.get(b.s.drone_uid, -1.0)
        last_b = b.s.last_explore_vector_comm_t.get(a.s.drone_uid, -1.0)
        since = max(last_a, last_b)

        def patch(dr: PythonDrone) -> List[dict]:
            out: List[dict] = []
            for ev in reversed(dr.s.recent_explore_vector_updates):
                if float(ev.t) <= float(since) + 1e-9:
                    break
                out.append({"origin": str(ev.origin_uid), "sx": float(ev.start_x), "sy": float(ev.start_y), "yaw": float(ev.yaw), "t": float(ev.t)})
                if max_items > 0 and len(out) >= max_items:
                    break
            return out

        pa = patch(a)
        pb = patch(b)

        def apply(dst: PythonDrone, items: List[dict]) -> int:
            changed = 0
            for it in items:
                ouid = str(it.get("origin", "")).strip()
                if not ouid:
                    continue
                t_msg = float(it.get("t", t_ref))
                cur = dst.s.known_explore_vectors.get(ouid)
                if cur is None or float(t_msg) > float(cur.t) + 1e-6:
                    ev = ExploreVector(
                        origin_uid=ouid,
                        start_x=float(it.get("sx", 0.0)),
                        start_y=float(it.get("sy", 0.0)),
                        yaw=float(it.get("yaw", 0.0)),
                        t=float(t_msg),
                    )
                    dst.s.known_explore_vectors[ouid] = ev
                    dst.s.recent_explore_vector_updates.append(ev)
                    changed += 1
            return changed

        changed = apply(a, pb) + apply(b, pa)
        a.s.last_explore_vector_comm_t[b.s.drone_uid] = float(t_ref)
        b.s.last_explore_vector_comm_t[a.s.drone_uid] = float(t_ref)
        return changed

    # (dynamic inspect claims removed; see comm-time conflict resolution)

    def _upload_all_to_base(self, t_ref: float):
        # Merge each drone's pheromone maps into base
        for d in self.drones:
            changed = 0
            # merge all cells (sparse dict); base prefers newer (t) and higher conf
            for c, v in d.pher.nav.v.items():
                meta = d.pher.nav.meta.get(c)
                if meta:
                    changed += self.base_map.nav.merge_cell(c, v, meta)
            for c, v in d.pher.danger.v.items():
                meta = d.pher.danger.meta.get(c)
                if meta:
                    changed += self.base_map.danger.merge_cell(c, v, meta)
            for c, v in d.pher.empty.v.items():
                meta = d.pher.empty.meta.get(c)
                if meta:
                    changed += self.base_map.empty.merge_cell(c, v, meta)
            for c, v in d.pher.explored.v.items():
                meta = d.pher.explored.meta.get(c)
                if meta:
                    changed += self.base_map.explored.merge_cell(c, v, meta)
            if changed > 0:
                d.s.base_uploads += 1

    def _upload_drone_to_base(self, d: PythonDrone) -> int:
        """Merge one drone's pheromones into base_map (returns changed count)."""
        changed = 0
        # Snapshot iteration to avoid concurrent mutation crashes.
        for c, v in list(d.pher.nav.v.items()):
            meta = d.pher.nav.meta.get(c)
            if meta:
                changed += 1 if self.base_map.nav.merge_cell(c, v, meta) else 0
        for c, v in list(d.pher.danger.v.items()):
            meta = d.pher.danger.meta.get(c)
            if meta:
                changed += 1 if self.base_map.danger.merge_cell(c, v, meta) else 0
        for c, v in list(d.pher.empty.v.items()):
            meta = d.pher.empty.meta.get(c)
            if meta:
                changed += 1 if self.base_map.empty.merge_cell(c, v, meta) else 0
        for c, v in list(d.pher.explored.v.items()):
            meta = d.pher.explored.meta.get(c)
            if meta:
                changed += 1 if self.base_map.explored.merge_cell(c, v, meta) else 0
        if changed > 0:
            d.s.base_uploads += 1
        return changed

    def _download_base_to_drone(self, d: PythonDrone, since_t: float, max_cells: int) -> int:
        """
        Download newest base pheromone cells newer than since_t into the drone.
        Uses a bounded heap to avoid sending huge updates.
        Returns merged cell count.
        """
        max_cells = int(clamp(float(max_cells), 0.0, 100000.0))
        if max_cells <= 0:
            return 0

        def _newest_patch(layer: SparseLayer) -> List[dict]:
            # keep a min-heap of (t, ...)
            heap: List[Tuple[float, Tuple[int, int], float, float, str, Optional[float], str, Optional[float]]] = []
            for c, meta in list(layer.meta.items()):
                if meta.t <= since_t + 1e-9:
                    continue
                v = float(layer.v.get(c, 0.0))
                if v <= 1e-9:
                    continue
                item = (
                    float(meta.t),
                    (int(c[0]), int(c[1])),
                    float(v),
                    float(meta.conf),
                    str(meta.src),
                    meta.alt_m,
                    str(meta.kind),
                    getattr(meta, "speed_s_per_cell", None),
                )
                if len(heap) < max_cells:
                    heapq.heappush(heap, item)
                else:
                    if item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            out = []
            # newest first
            heap.sort(reverse=True)
            for t, (x, y), v, conf, src, alt, kind, sp in heap:
                row = {"x": x, "y": y, "v": float(v), "t": float(t), "conf": float(conf), "src": src}
                if alt is not None:
                    row["alt"] = float(alt)
                if sp is not None:
                    try:
                        row["speed"] = float(sp)
                    except Exception:
                        pass
                if kind:
                    row["kind"] = str(kind)
                out.append(row)
            return out

        changed = 0
        changed += d.pher.merge_patch("nav", _newest_patch(self.base_map.nav))
        patch_danger = _newest_patch(self.base_map.danger)
        changed += d.pher.merge_patch("danger", patch_danger)
        changed += d.pher.merge_patch("empty", _newest_patch(self.base_map.empty))
        changed += d.pher.merge_patch("explored", _newest_patch(self.base_map.explored))
        try:
            self._ingest_dynamic_from_danger_patch(d, patch_danger)
        except Exception:
            pass
        return changed

    def _ingest_dynamic_from_danger_patch(self, d: PythonDrone, patch_cells: List[dict]):
        """
        Update per-drone dynamic danger kernel cache + known ids from a danger-layer patch.
        This is how a drone can reason about dynamic threats discovered by peers (via pheromone sharing).
        """
        if not patch_cells:
            return
        try:
            my_uid = str(d.s.drone_uid)
        except Exception:
            my_uid = ""
        for it in patch_cells:
            try:
                kind = str(it.get("kind", "") or "")
                if kind.startswith("danger_dyn_kernel_done"):
                    did = kind.split(":", 1)[1] if ":" in kind else ""
                    if did:
                        d.s.dynamic_inspect_skip_ids.add(str(did))
                        # If we were inspecting it, stop.
                        if str(getattr(d.s, "dynamic_inspect_active_id", "") or "") == str(did):
                            d.s.dynamic_inspect_active_id = ""
                    continue
                if not kind.startswith("danger_dyn_kernel"):
                    continue
                did = kind.split(":", 1)[1] if ":" in kind else ""
                if did:
                    try:
                        d.s.known_dynamic_danger_ids.add(str(did))
                    except Exception:
                        pass
                sp = it.get("speed", None)
                t = float(it.get("t", 0.0))
                cell = (int(it.get("x")), int(it.get("y")))
                speed_s = None
                try:
                    if sp is not None:
                        speed_s = float(sp)
                except Exception:
                    speed_s = None
                src = str(it.get("src", "")) or "unknown"
                # Record observation into drone's kernel path reconstruction.
                d._record_dyn_kernel_obs(str(did), cell, t, speed_s, src)
            except Exception:
                continue

    # -------- RViz publishing --------

    def _publish_rviz(self, t_ref: float):
        t0 = time.perf_counter()
        rviz_marker_count = 0
        rviz_point_count = 0
        # drones markers
        ma = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.header.frame_id = "world"
        ma.markers.append(delete_all)

        poses = PoseArray()
        poses.header.frame_id = "world"
        poses.header.stamp = self.get_clock().now().to_msg()

        for d in self.drones:
            # PoseArray
            p = Pose()
            p.position.x = float(d.s.x)
            p.position.y = float(d.s.y)
            p.position.z = float(d.s.z)
            p.orientation = yaw_to_quat(d.s.yaw)
            poses.poses.append(p)

            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "swarm_drones"
            m.id = int(1000 + d.s.seq)  # PY offset
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = float(d.s.x)
            m.pose.position.y = float(d.s.y)
            m.pose.position.z = float(d.s.z)
            m.pose.orientation = yaw_to_quat(d.s.yaw)
            scale = float(self.get_parameter("drone_marker_scale").value)
            m.scale.x = 2.0 * scale
            m.scale.y = 0.5 * scale
            m.scale.z = 0.5 * scale
            # Color: PY = cyan-ish; (REAL would be blue)
            m.color.r = 0.0
            m.color.g = 0.9
            m.color.b = 1.0
            m.color.a = 1.0
            ma.markers.append(m)

            # label
            t = Marker()
            t.header.frame_id = "world"
            t.header.stamp = m.header.stamp
            t.ns = "swarm_drone_labels"
            t.id = int(2000 + d.s.seq)
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(d.s.x)
            t.pose.position.y = float(d.s.y)
            t.pose.position.z = float(d.s.z + 2.0)
            t.scale.z = 1.5
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = f"{d.s.drone_uid} {d.s.mode} E:{d.s.energy_units:.0f}"
            ma.markers.append(t)

        # Drone pointers (big XY readout above drones)
        if bool(self.get_parameter("drone_pointer_enabled").value):
            from geometry_msgs.msg import Point
            pz = float(self.get_parameter("drone_pointer_z").value)
            ps = float(self.get_parameter("drone_pointer_scale").value)
            pa = float(self.get_parameter("drone_pointer_alpha").value)
            # If user is visualizing pheromone map for a specific drone, highlight that drone's pointer.
            sel_owner = str(self.get_parameter("pheromone_viz_owner").value).strip().lower()
            sel_seq = int(self.get_parameter("pheromone_viz_drone_seq").value)
            highlight_seq = sel_seq if sel_owner == "drone" else None

            # One LINE_LIST for vertical pointers
            lines = Marker()
            lines.header.frame_id = "world"
            lines.header.stamp = self.get_clock().now().to_msg()
            lines.ns = "swarm_drone_pointers"
            lines.id = 9000
            lines.type = Marker.LINE_LIST
            lines.action = Marker.ADD
            lines.scale.x = 0.4 * ps
            lines.color.r = 1.0
            lines.color.g = 1.0
            lines.color.b = 1.0
            lines.color.a = pa
            lines.points = []
            lines.colors = []

            for d in self.drones:
                p1 = Point()
                p1.x = float(d.s.x)
                p1.y = float(d.s.y)
                p1.z = float(0.0)
                p2 = Point()
                p2.x = float(d.s.x)
                p2.y = float(d.s.y)
                p2.z = float(pz)
                lines.points.append(p1)
                lines.points.append(p2)

                c = ColorRGBA()
                # Inspector drone (dynamic danger path discovery) is violet while active.
                is_inspector = bool(str(getattr(d.s, "dynamic_inspect_active_id", "") or "").strip())
                if is_inspector:
                    c.r = 0.65
                    c.g = 0.15
                    c.b = 0.85
                    c.a = float(pa)
                elif highlight_seq is not None and int(d.s.seq) == int(highlight_seq):
                    # green highlight for selected-drone pheromone view
                    c.r = 0.0
                    c.g = 1.0
                    c.b = 0.0
                    c.a = float(pa)
                else:
                    c.r = 1.0
                    c.g = 1.0
                    c.b = 1.0
                    c.a = float(pa)
                # LINE_LIST uses per-point colors (two points per segment).
                lines.colors.append(c)
                lines.colors.append(c)

                txt = Marker()
                txt.header.frame_id = "world"
                txt.header.stamp = lines.header.stamp
                txt.ns = "swarm_drone_xy"
                txt.id = 9100 + d.s.seq
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = float(d.s.x)
                txt.pose.position.y = float(d.s.y)
                txt.pose.position.z = float(pz + 0.5)
                txt.pose.orientation.w = 1.0
                txt.scale.z = float(2.5 * ps)
                if highlight_seq is not None and int(d.s.seq) == int(highlight_seq):
                    txt.color.r = 0.0
                    txt.color.g = 1.0
                    txt.color.b = 0.0
                    txt.color.a = pa
                else:
                    txt.color.r = 1.0
                    txt.color.g = 1.0
                    txt.color.b = 1.0
                    txt.color.a = pa
                txt.text = f"{d.s.drone_uid}  v={d.s.speed_mps:.1f}m/s\nx={d.s.x:.1f} y={d.s.y:.1f} z={d.s.z:.1f}"
                ma.markers.append(txt)

            ma.markers.append(lines)

            # Inspector realtime kernel beam (visible under /swarm/markers/drones as well).
            # Red line from inspector drone to LiDAR-seen kernel cell (fresh within TTL).
            try:
                ttl_rt = float(self.get_parameter("dyn_inspector_rt_ttl_s").value)
                t_now = float(getattr(self, "t_sim", 0.0))
                if ttl_rt > 1e-6:
                    ib = Marker()
                    ib.header.frame_id = "world"
                    ib.header.stamp = lines.header.stamp
                    ib.ns = "swarm_dyn_inspector_rt"
                    ib.id = 9200
                    ib.type = Marker.LINE_LIST
                    ib.action = Marker.ADD
                    ib.scale.x = float(0.22 * ps)
                    ib.color.r = 1.0
                    ib.color.g = 0.0
                    ib.color.b = 0.0
                    ib.color.a = 1.0
                    try:
                        ib.lifetime.sec = 1
                    except Exception:
                        pass
                    ib.points = []
                    ib.colors = []
                    for d in self.drones:
                        try:
                            did = str(getattr(d.s, "dynamic_inspect_active_id", "") or "").strip()
                            if not did:
                                continue
                            rt = (getattr(d, "_dyn_kernel_realtime", None) or {}).get(did)
                            if not rt:
                                continue
                            age = float(t_now) - float(rt.get("t", -1e9))
                            if age > ttl_rt + 1e-6:
                                continue
                            cc = tuple(rt.get("cell", ()))
                            if len(cc) != 2:
                                continue
                            wx, wy = self.grid.cell_to_world(int(cc[0]), int(cc[1]))
                            p1 = Point(); p1.x = float(d.s.x); p1.y = float(d.s.y); p1.z = float(max(0.2, pz))
                            p2 = Point(); p2.x = float(wx); p2.y = float(wy); p2.z = float(max(0.2, pz))
                            ib.points.append(p1); ib.points.append(p2)
                            c = ColorRGBA(); c.r = 1.0; c.g = 0.0; c.b = 0.0; c.a = float(pa)
                            ib.colors.append(c); ib.colors.append(c)
                        except Exception:
                            continue
                    if ib.points:
                        ma.markers.append(ib)
            except Exception:
                pass

            # Base pointer (green)
            base_lines = Marker()
            base_lines.header.frame_id = "world"
            base_lines.header.stamp = lines.header.stamp
            base_lines.ns = "swarm_base_pointer"
            base_lines.id = 9050
            base_lines.type = Marker.LINE_LIST
            base_lines.action = Marker.ADD
            base_lines.scale.x = 0.5 * ps
            base_lines.color.r = 0.0
            base_lines.color.g = 1.0
            base_lines.color.b = 0.0
            base_lines.color.a = pa
            base_lines.points = []
            b1 = Point(); b1.x = float(self.base_xy[0]); b1.y = float(self.base_xy[1]); b1.z = 0.0
            b2 = Point(); b2.x = float(self.base_xy[0]); b2.y = float(self.base_xy[1]); b2.z = float(pz)
            base_lines.points.extend([b1, b2])
            ma.markers.append(base_lines)

            base_txt = Marker()
            base_txt.header.frame_id = "world"
            base_txt.header.stamp = lines.header.stamp
            base_txt.ns = "swarm_base_text"
            base_txt.id = 9051
            base_txt.type = Marker.TEXT_VIEW_FACING
            base_txt.action = Marker.ADD
            base_txt.pose.position.x = float(self.base_xy[0])
            base_txt.pose.position.y = float(self.base_xy[1])
            base_txt.pose.position.z = float(pz + 0.8)
            base_txt.pose.orientation.w = 1.0
            base_txt.scale.z = float(3.0 * ps)
            base_txt.color.r = 0.0
            base_txt.color.g = 1.0
            base_txt.color.b = 0.0
            base_txt.color.a = pa
            base_txt.text = "BASE (0,0)"
            ma.markers.append(base_txt)

            # Current pheromone viz selection (small HUD text near base)
            try:
                hud = Marker()
                hud.header.frame_id = "world"
                hud.header.stamp = lines.header.stamp
                hud.ns = "swarm_pher_viz_state"
                hud.id = 9060
                hud.type = Marker.TEXT_VIEW_FACING
                hud.action = Marker.ADD
                hud.pose.position.x = float(self.base_xy[0] + 5.0)
                hud.pose.position.y = float(self.base_xy[1] + 5.0)
                hud.pose.position.z = float(pz + 1.8)
                hud.pose.orientation.w = 1.0
                hud.scale.z = float(2.0 * ps)
                hud.color.r = 1.0
                hud.color.g = 1.0
                hud.color.b = 1.0
                hud.color.a = float(min(1.0, max(0.2, pa)))
                hud.text = f"pher_viz: {sel_owner}/{str(self.get_parameter('pheromone_viz_layer').value)} #{sel_seq}"
                ma.markers.append(hud)
            except Exception:
                pass

        self.pub_drones.publish(ma)
        self.pub_poses.publish(poses)
        rviz_marker_count += len(ma.markers)
        # pointers are LINE_LIST with points, and we also have no points for arrows/text
        try:
            for mm in ma.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # paths
        pm = MarkerArray()
        delp = Marker()
        delp.action = Marker.DELETEALL
        delp.header.frame_id = "world"
        pm.markers.append(delp)
        now_msg = self.get_clock().now().to_msg()
        for d in self.drones:
            path = self.path_hist.get(d.s.drone_uid)
            if path is None:
                path = deque(maxlen=400)
                self.path_hist[d.s.drone_uid] = path
            if len(path) < 2:
                continue
            # Snapshot under a short lock to avoid "deque mutated during iteration" when the sim tick appends.
            with self._state_lock:
                path_pts = list(path)
            # Downsample aggressively to keep RViz fast even when history_len is large.
            max_pts = max(50, int(getattr(self, "path_viz_max_points", 350)))
            if len(path_pts) > max_pts:
                stride = max(1, int(len(path_pts) / max_pts))
                path_pts = path_pts[::stride]
                if path_pts and path_pts[-1] != list(path)[-1]:
                    # ensure last point is included
                    path_pts.append(list(path)[-1])
            line = Marker()
            line.header.frame_id = "world"
            line.header.stamp = now_msg
            line.ns = "swarm_paths"
            line.id = int(3000 + d.s.seq)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.2
            line.color.r = 0.0
            line.color.g = 0.7
            line.color.b = 1.0
            line.color.a = 0.8
            from geometry_msgs.msg import Point
            line.points = []
            for x, y, z in path_pts:
                pp = Point()
                pp.x = float(x)
                pp.y = float(y)
                pp.z = float(z + 0.5)
                line.points.append(pp)
            pm.markers.append(line)
        self.pub_paths.publish(pm)
        rviz_marker_count += len(pm.markers)
        try:
            for mm in pm.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # targets
        tm = MarkerArray()
        delt = Marker()
        delt.action = Marker.DELETEALL
        delt.header.frame_id = "world"
        tm.markers.append(delt)
        target_d = float(self.get_parameter("target_viz_diameter").value)
        target_a = float(self.get_parameter("target_viz_alpha").value)
        sel_tid = str(getattr(self, "selected_target_id", "") or "").strip()
        in_exploit = str(getattr(self, "mission_phase", "") or "").upper() == "EXPLOIT"
        # Only draw targets inside exploration area radius (if enabled).
        radius = float(getattr(self, "exploration_area_radius_m", 0.0))
        if radius > 1e-6:
            vis_targets = [t for t in self.targets if math.hypot(float(t.x) - float(self.base_xy[0]), float(t.y) - float(self.base_xy[1])) <= radius + 1e-6]
        else:
            vis_targets = list(self.targets)
        # Optionally hide all but the selected target (so the goal stands out).
        try:
            if bool(self.get_parameter("exploit_hide_other_targets").value):
                sel = str(getattr(self, "selected_target_id", "") or "").strip()
                if sel:
                    vis_targets = [t for t in vis_targets if str(getattr(t, "target_id", "")) == sel]
        except Exception:
            pass
        for idx, tgt in enumerate(vis_targets, 1):
            s = Marker()
            s.header.frame_id = "world"
            s.header.stamp = now_msg
            s.ns = "swarm_targets"
            s.id = 4000 + idx
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = float(tgt.x)
            s.pose.position.y = float(tgt.y)
            # EXPLOIT: show as a "dome" above ground (no geometry below z=0).
            # Approximate a half-sphere by using a flattened sphere (ellipsoid) whose bottom touches z=0.
            if in_exploit:
                # We'll set z after computing d0/scale.z.
                s.pose.position.z = 0.0
            else:
                s.pose.position.z = float(tgt.z)
            s.pose.orientation.w = 1.0
            d0 = float(target_d)
            a0 = float(target_a)
            if in_exploit:
                d0 = float(target_d) * 0.5
                a0 = float(target_a) * 0.5
            s.scale.x = float(d0)
            s.scale.y = float(d0)
            if in_exploit:
                # Flatten into a dome: height = d0/2, bottom at z=0.
                s.scale.z = float(max(0.01, float(d0) * 0.5))
                s.pose.position.z = float(s.scale.z) * 0.5
            else:
                s.scale.z = float(d0)

            # Color semantics:
            # - Explore: found=green, unfound=red
            # - Exploit: selected goal is always red (even if it was marked found in the saved targets file)
            if in_exploit and sel_tid and str(getattr(tgt, "target_id", "")) == sel_tid:
                s.color.r = 1.0
                s.color.g = 0.0
                s.color.b = 0.0
                s.color.a = float(a0)
            else:
                if tgt.is_found():
                    s.color.r = 0.0
                    s.color.g = 1.0
                    s.color.b = 0.0
                    s.color.a = float(a0)
                else:
                    s.color.r = 1.0
                    s.color.g = 0.0
                    s.color.b = 0.0
                    s.color.a = float(a0)
            tm.markers.append(s)
        self.pub_targets.publish(tm)
        rviz_marker_count += len(tm.markers)

        # comm wires (temporary)
        cm = MarkerArray()
        delc = Marker()
        delc.action = Marker.DELETEALL
        delc.header.frame_id = "world"
        cm.markers.append(delc)
        if self.comm_viz_enabled:
            # prune expired + snapshot under a dedicated comm lock (do not stall sim state).
            mode = str(self.comm_viz_mode).strip().lower()
            # Base sync comms (green, thicker) — shown in any mode if enabled.
            if bool(self.base_comm_viz_enabled):
                with self._comm_lock:
                    while self.base_comm_lines and self.base_comm_lines[0][0] < time.time():
                        self.base_comm_lines.popleft()
                    base_comm_snapshot = list(self.base_comm_lines)
                if base_comm_snapshot:
                    from geometry_msgs.msg import Point
                    bm = Marker()
                    bm.header.frame_id = "world"
                    bm.header.stamp = now_msg
                    bm.ns = "swarm_comm_base"
                    bm.id = 5002
                    bm.type = Marker.LINE_LIST
                    bm.action = Marker.ADD
                    bm.scale.x = float(max(0.01, float(self.base_comm_viz_line_width)))
                    bm.color.r = 0.0
                    bm.color.g = 1.0
                    bm.color.b = 0.0
                    bm.color.a = 0.95
                    bm.points = []
                    for _, (x1, y1, z1, x2, y2, z2) in base_comm_snapshot:
                        p1 = Point(); p1.x = float(x1); p1.y = float(y1); p1.z = float(z1 + 1.0)
                        p2 = Point(); p2.x = float(x2); p2.y = float(y2); p2.z = float(z2 + 1.0)
                        bm.points.append(p1)
                        bm.points.append(p2)
                    cm.markers.append(bm)
            if mode == "events":
                with self._comm_lock:
                    while self.comm_lines and self.comm_lines[0][0] < time.time():
                        self.comm_lines.popleft()
                    comm_snapshot = list(self.comm_lines)
                if comm_snapshot:
                    from geometry_msgs.msg import Point
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now_msg
                    m.ns = "swarm_comm_events"
                    m.id = 5000
                    m.type = Marker.LINE_LIST
                    m.action = Marker.ADD
                    m.scale.x = 0.4
                    m.color.r = 0.2
                    m.color.g = 0.6
                    m.color.b = 1.0
                    m.color.a = 0.8
                    m.points = []
                    for _, (x1, y1, z1, x2, y2, z2) in comm_snapshot:
                        p1 = Point()
                        p1.x = float(x1)
                        p1.y = float(y1)
                        p1.z = float(z1 + 1.0)
                        p2 = Point()
                        p2.x = float(x2)
                        p2.y = float(y2)
                        p2.z = float(z2 + 1.0)
                        m.points.append(p1)
                        m.points.append(p2)
                    cm.markers.append(m)
            else:
                # clusters mode: draw one "star" per connected component (centroid -> drones).
                # This is purely visual and much less cluttered than pairwise wires.
                from geometry_msgs.msg import Point
                # Snapshot positions quickly; no strict sync needed.
                drones_snapshot = [(d.s.x, d.s.y, d.s.z, d.s.seq) for d in self.drones]
                n = len(drones_snapshot)
                if n >= 2:
                    # Union-Find
                    parent = list(range(n))

                    def find(i: int) -> int:
                        while parent[i] != i:
                            parent[i] = parent[parent[i]]
                            i = parent[i]
                        return i

                    def union(i: int, j: int):
                        ri = find(i)
                        rj = find(j)
                        if ri != rj:
                            parent[rj] = ri

                    r2 = float(self.comm_radius) * float(self.comm_radius)
                    for i in range(n):
                        xi, yi, _, _ = drones_snapshot[i]
                        for j in range(i + 1, n):
                            xj, yj, _, _ = drones_snapshot[j]
                            dx = xi - xj
                            dy = yi - yj
                            if (dx * dx + dy * dy) <= r2:
                                union(i, j)

                    comps: Dict[int, List[int]] = {}
                    for i in range(n):
                        r = find(i)
                        comps.setdefault(r, []).append(i)

                    # One LINE_LIST for all cluster stars.
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now_msg
                    m.ns = "swarm_comm_clusters"
                    m.id = 5001
                    m.type = Marker.LINE_LIST
                    m.action = Marker.ADD
                    m.scale.x = float(max(0.01, float(self.comm_viz_cluster_line_width)))
                    m.color.r = 0.2
                    m.color.g = 0.6
                    m.color.b = 1.0
                    m.color.a = 0.7
                    m.points = []

                    min_d = float(self.comm_viz_min_dist_m)
                    for _, idxs in comps.items():
                        if len(idxs) < 2:
                            continue
                        cx = sum(drones_snapshot[i][0] for i in idxs) / len(idxs)
                        cy = sum(drones_snapshot[i][1] for i in idxs) / len(idxs)
                        cz = sum(drones_snapshot[i][2] for i in idxs) / len(idxs)
                        for i in idxs:
                            x, y, z, _ = drones_snapshot[i]
                            if math.hypot(x - cx, y - cy) < min_d:
                                continue
                            p1 = Point(); p1.x = float(cx); p1.y = float(cy); p1.z = float(cz + 1.0)
                            p2 = Point(); p2.x = float(x); p2.y = float(y); p2.z = float(z + 1.0)
                            m.points.append(p1)
                            m.points.append(p2)
                    if m.points:
                        cm.markers.append(m)

        # Dynamic-threat decision beams (drone -> predicted threat cell).
        # Yellow = we decided to cross the threat track; Red = we decided to avoid it.
        try:
            if bool(self.get_parameter("dynamic_threat_decision_enabled").value):
                from geometry_msgs.msg import Point
                ttl_s = 2.0
                t_now = float(getattr(self, "t_sim", 0.0))
                top_z = float(self.get_parameter("drone_pointer_z").value)

                bm = Marker()
                bm.header.frame_id = "world"
                bm.header.stamp = now_msg
                bm.ns = "swarm_threat_decision"
                bm.id = 5010
                bm.type = Marker.LINE_LIST
                bm.action = Marker.ADD
                bm.scale.x = 0.35
                # RViz safety: set a non-zero alpha even though we use per-point colors
                bm.color.r = 1.0
                bm.color.g = 1.0
                bm.color.b = 1.0
                bm.color.a = 1.0
                bm.points = []
                bm.colors = []

                for d in self.drones:
                    try:
                        cc = getattr(d.s, "threat_decision_cell", None)
                        mode = str(getattr(d.s, "threat_decision_mode", "") or "")
                        t0 = float(getattr(d.s, "threat_decision_t", -1e9))
                        if not cc or (t_now - t0) > ttl_s:
                            continue
                        wx, wy = self.grid.cell_to_world(int(cc[0]), int(cc[1]))
                        p1 = Point(); p1.x = float(d.s.x); p1.y = float(d.s.y); p1.z = float(max(0.2, top_z))
                        p2 = Point(); p2.x = float(wx); p2.y = float(wy); p2.z = float(max(0.2, top_z))
                        bm.points.append(p1)
                        bm.points.append(p2)
                        c = ColorRGBA()
                        if mode == "cross":
                            c.r = 1.0; c.g = 1.0; c.b = 0.0; c.a = 0.95
                        else:
                            c.r = 1.0; c.g = 0.0; c.b = 0.0; c.a = 0.95
                        bm.colors.append(c)
                        bm.colors.append(c)
                    except Exception:
                        continue

                if bm.points:
                    cm.markers.append(bm)
        except Exception:
            pass

        # Inspector real-time kernel "communication line" (drone -> LiDAR-seen kernel cell).
        # Always red. Shown only when the drone is the active inspector and has a fresh realtime sighting.
        try:
            from geometry_msgs.msg import Point
            ttl_rt = float(self.get_parameter("dyn_inspector_rt_ttl_s").value)
            t_now = float(getattr(self, "t_sim", 0.0))
            top_z = float(self.get_parameter("drone_pointer_z").value)
            if ttl_rt > 1e-6:
                rtm = Marker()
                rtm.header.frame_id = "world"
                rtm.header.stamp = now_msg
                rtm.ns = "swarm_dyn_inspector_rt"
                rtm.id = 5020
                rtm.type = Marker.LINE_LIST
                rtm.action = Marker.ADD
                rtm.scale.x = 0.25
                # RViz safety: set a non-zero alpha even though we use per-point colors
                rtm.color.r = 1.0
                rtm.color.g = 0.0
                rtm.color.b = 0.0
                rtm.color.a = 1.0
                # Avoid stale beam if ticks pause.
                try:
                    rtm.lifetime.sec = 1
                except Exception:
                    pass
                rtm.points = []
                rtm.colors = []

                for d in self.drones:
                    try:
                        did = str(getattr(d.s, "dynamic_inspect_active_id", "") or "").strip()
                        if not did:
                            continue
                        rt_map = getattr(d, "_dyn_kernel_realtime", None) or {}
                        rt = rt_map.get(did)
                        if not rt:
                            continue
                        age = float(t_now) - float(rt.get("t", -1e9))
                        if age > ttl_rt + 1e-6:
                            continue
                        cc = tuple(rt.get("cell", ()))
                        if len(cc) != 2:
                            continue
                        wx, wy = self.grid.cell_to_world(int(cc[0]), int(cc[1]))
                        p1 = Point(); p1.x = float(d.s.x); p1.y = float(d.s.y); p1.z = float(max(0.2, top_z))
                        p2 = Point(); p2.x = float(wx); p2.y = float(wy); p2.z = float(max(0.2, top_z))
                        rtm.points.append(p1)
                        rtm.points.append(p2)
                        c = ColorRGBA()
                        c.r = 1.0; c.g = 0.0; c.b = 0.0; c.a = 0.95
                        rtm.colors.append(c)
                        rtm.colors.append(c)
                    except Exception:
                        continue

                if rtm.points:
                    cm.markers.append(rtm)
        except Exception:
            pass
        self.pub_comm.publish(cm)
        rviz_marker_count += len(cm.markers)
        try:
            for mm in cm.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # Planned path visualization (what the drone planned, not what it already flew).
        pm2 = MarkerArray()
        delp2 = Marker()
        delp2.action = Marker.DELETEALL
        delp2.header.frame_id = "world"
        pm2.markers.append(delp2)
        if bool(self.get_parameter("plan_viz_enabled").value):
            alpha = float(self.get_parameter("plan_viz_alpha").value)
            w = float(self.get_parameter("plan_viz_line_width").value)
            seq = int(self.get_parameter("plan_viz_drone_seq").value)
            from geometry_msgs.msg import Point
            import colorsys
            ttl_s = float(self.get_parameter("plan_viz_ttl_s").value)
            # Use sim time for freshness checks (agent timestamps are in sim time).
            t_sim_now = float(getattr(self, "t_sim", 0.0))

            def _add_plan_for_agent(agent_i: int):
                # agent_i is 1-based drone seq
                try:
                    agent = self.drones[agent_i - 1]
                except Exception:
                    return
                # Avoid "always on": show only if plan is still active OR was created recently.
                try:
                    active = bool(getattr(agent, "_active_plan_world", []) or [])
                    fresh = (t_sim_now - float(getattr(agent, "last_plan_t", -1e9))) <= max(0.0, float(ttl_s))
                    if not (active or fresh):
                        return
                except Exception:
                    pass
                pts = list(getattr(agent, "last_plan_world", []) or [])
                if len(pts) < 2:
                    return
                z = float(agent.s.z)
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = now_msg
                m.ns = "swarm_plan"
                m.id = 8000 + int(agent_i)
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = float(max(0.01, w))
                # color by drone index for readability
                h = (float(agent_i - 1) / max(1.0, float(self.num_py)))
                r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0)
                m.color.r = float(r)
                m.color.g = float(g)
                m.color.b = float(b)
                m.color.a = float(clamp(alpha, 0.0, 1.0))
                m.points = []
                for x, y in pts:
                    p = Point()
                    p.x = float(x)
                    p.y = float(y)
                    p.z = float(z)
                    m.points.append(p)
                pm2.markers.append(m)

            if seq == 0:
                for agent_i in range(1, int(self.num_py) + 1):
                    _add_plan_for_agent(agent_i)
            else:
                seq = max(1, min(self.num_py, seq))
                _add_plan_for_agent(seq)
        self.pub_plan.publish(pm2)
        rviz_marker_count += len(pm2.markers)
        try:
            for mm in pm2.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # ACO decision visualization (chosen heading + candidate rays).
        am = MarkerArray()
        aco_enabled = bool(self.get_parameter("aco_viz_enabled").value)
        # If we just disabled ACO viz, delete all ACO markers immediately.
        if (not aco_enabled) and bool(getattr(self, "_aco_viz_prev_enabled", False)):
            dela = Marker()
            dela.action = Marker.DELETEALL
            dela.header.frame_id = "world"
            am.markers.append(dela)
        self._aco_viz_prev_enabled = bool(aco_enabled)

        if aco_enabled:
            seq = int(self.get_parameter("aco_viz_drone_seq").value)
            alpha = float(self.get_parameter("aco_viz_alpha").value)
            w = float(self.get_parameter("aco_viz_line_width").value)
            show_cands = bool(self.get_parameter("aco_viz_show_candidates").value)
            top_k = int(self.get_parameter("aco_viz_top_k").value)
            top_k = max(0, min(24, top_k))
            from geometry_msgs.msg import Point
            # Prevent stale arrows from "sticking" when ACO is not updating.
            ttl_s = float(max(0.05, float(self.get_parameter("aco_viz_ttl_s").value)))
            # Use sim time for freshness checks (agent timestamps are in sim time).
            t_sim_now = float(getattr(self, "t_sim", 0.0))
            z_off = float(self.get_parameter("aco_viz_z_offset_m").value)
            show_heat = bool(self.get_parameter("aco_viz_show_heatmap").value)
            heat_k = int(self.get_parameter("aco_viz_heatmap_top_k").value)
            heat_k = max(0, min(24, heat_k))
            heat_s = float(self.get_parameter("aco_viz_heatmap_scale_m").value)
            heat_s = clamp(float(heat_s), 0.25, 20.0)
            heat_hist_s = float(self.get_parameter("aco_viz_heatmap_history_s").value)
            heat_hist_s = float(max(0.0, heat_hist_s))
            show_pillar = bool(self.get_parameter("aco_viz_show_best_pillar").value)
            pillar_hist_s = float(self.get_parameter("aco_viz_best_pillar_history_s").value)
            pillar_hist_s = float(max(0.0, pillar_hist_s))
            pillar_w = float(self.get_parameter("aco_viz_best_pillar_line_width").value)
            pillar_w = float(max(0.01, pillar_w))
            # Reuse the same "pointer height" visual convention for pillar height.
            pillar_top_z = float(self.get_parameter("drone_pointer_z").value)
            arrow_len_mult = float(self.get_parameter("aco_viz_arrow_length_mult").value)
            arrow_len_mult = clamp(float(arrow_len_mult), 0.3, 10.0)
            arrow_w = float(self.get_parameter("aco_viz_arrow_width_m").value)
            arrow_w = clamp(float(arrow_w), 0.02, 5.0)

            def _add_aco_for_agent(agent_i: int):
                try:
                    agent = self.drones[agent_i - 1]
                except Exception:
                    return
                # Show ACO scoring/intent even when A* overrides (that helps explain why A* took over).
                choice = getattr(agent, "last_aco_choice_world", None)
                if not choice:
                    return
                try:
                    if (t_sim_now - float(getattr(agent, "last_aco_choice_t", -1e9))) > float(ttl_s):
                        return
                except Exception:
                    pass
                cx, cy, cyaw = float(choice[0]), float(choice[1]), float(choice[2])
                z = float(agent.s.z) + float(z_off)

                # Candidate rays: top-K by score
                cands = list(getattr(agent, "last_aco_candidates", []) or [])
                if show_cands and cands:
                    cands_sorted = list(cands)
                    cands_sorted.sort(key=lambda t: float(t[0]), reverse=True)
                    if top_k > 0:
                        cands_sorted = cands_sorted[:top_k]
                        rays = Marker()
                        rays.header.frame_id = "world"
                        rays.header.stamp = now_msg
                        rays.ns = "swarm_aco_candidates"
                        rays.id = 11000 + int(agent_i)
                        rays.type = Marker.LINE_LIST
                        rays.action = Marker.ADD
                        rays.scale.x = float(max(0.01, w))
                        rays.color.r = 0.85
                        rays.color.g = 0.85
                        rays.color.b = 0.85
                        rays.color.a = float(clamp(alpha * 0.55, 0.0, 1.0))
                        rays.points = []
                        p0 = Point()
                        p0.x = float(agent.s.x)
                        p0.y = float(agent.s.y)
                        p0.z = float(z)
                        for _score, (nx, ny, _yaw) in cands_sorted:
                            p1 = Point()
                            p1.x = float(nx)
                            p1.y = float(ny)
                            p1.z = float(z)
                            rays.points.append(p0)
                            rays.points.append(p1)
                        if rays.points:
                            # If the selection changes (single->all), let old rays expire.
                            rays.lifetime.sec = int(ttl_s)
                            rays.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
                            am.markers.append(rays)

                # Candidate heatmap: cube per candidate next-step cell, colored by score.
                if show_heat and cands:
                    import math as _math
                    # Normalize scores within the candidate set.
                    scores = [float(s) for (s, _v) in cands]
                    s_min = min(scores) if scores else 0.0
                    s_max = max(scores) if scores else 1.0
                    den = (s_max - s_min) if abs(s_max - s_min) > 1e-9 else 1.0
                    # Prefer showing best candidates to reduce clutter.
                    cands_sorted = list(cands)
                    cands_sorted.sort(key=lambda t: float(t[0]), reverse=True)
                    if heat_k > 0:
                        cands_sorted = cands_sorted[:heat_k]

                    cubes = Marker()
                    cubes.header.frame_id = "world"
                    cubes.header.stamp = now_msg
                    # History mode: use unique ids and per-drone namespaces so cubes persist for a few seconds.
                    if heat_hist_s > 1e-6:
                        cubes.ns = f"swarm_aco_heatmap_{int(agent_i)}"
                        cubes.id = 20000 + int(self._aco_heat_id % 10000)
                        self._aco_heat_id += 1
                        cubes.lifetime.sec = int(heat_hist_s)
                        cubes.lifetime.nanosec = int((heat_hist_s - int(heat_hist_s)) * 1e9)
                    else:
                        cubes.ns = "swarm_aco_heatmap"
                        cubes.id = 12000 + int(agent_i)
                        cubes.lifetime.sec = int(ttl_s)
                        cubes.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
                    cubes.type = Marker.CUBE_LIST
                    cubes.action = Marker.ADD
                    cubes.scale.x = float(heat_s)
                    cubes.scale.y = float(heat_s)
                    cubes.scale.z = float(max(0.15, heat_s * 0.35))
                    cubes.points = []
                    cubes.colors = []
                    for s, (nx, ny, _yaw) in cands_sorted:
                        nn = (float(s) - float(s_min)) / float(den)
                        nn = 0.0 if nn < 0.0 else (1.0 if nn > 1.0 else float(nn))
                        p = Point()
                        p.x = float(nx)
                        p.y = float(ny)
                        p.z = float(z)
                        cubes.points.append(p)
                        col = ColorRGBA()
                        # Orange heat: darker -> brighter, with alpha following global alpha.
                        col.r = 1.0
                        col.g = float(0.25 + 0.70 * nn)
                        col.b = 0.0
                        col.a = float(clamp(alpha * (0.35 + 0.55 * _math.sqrt(max(0.0, nn))), 0.0, 1.0))
                        cubes.colors.append(col)
                    if cubes.points:
                        am.markers.append(cubes)

                # Best-step pillar trail (orange): shows where the best ACO-scored step points.
                if show_pillar and pillar_hist_s > 1e-6:
                    pillar = Marker()
                    pillar.header.frame_id = "world"
                    pillar.header.stamp = now_msg
                    pillar.ns = f"swarm_aco_best_pillar_{int(agent_i)}"
                    pillar.id = 30000 + int(self._aco_pillar_id % 10000)
                    self._aco_pillar_id += 1
                    pillar.type = Marker.LINE_LIST
                    pillar.action = Marker.ADD
                    pillar.scale.x = float(pillar_w)
                    pillar.color.r = 1.0
                    pillar.color.g = 0.55
                    pillar.color.b = 0.0
                    pillar.color.a = float(clamp(alpha * 0.75, 0.0, 1.0))
                    pillar.lifetime.sec = int(pillar_hist_s)
                    pillar.lifetime.nanosec = int((pillar_hist_s - int(pillar_hist_s)) * 1e9)
                    p0 = Point()
                    p0.x = float(cx)
                    p0.y = float(cy)
                    p0.z = 0.05
                    p1 = Point()
                    p1.x = float(cx)
                    p1.y = float(cy)
                    p1.z = float(max(0.2, pillar_top_z))
                    pillar.points = [p0, p1]
                    am.markers.append(pillar)

                # Chosen arrow
                arr = Marker()
                arr.header.frame_id = "world"
                arr.header.stamp = now_msg
                arr.ns = "swarm_aco_choice"
                arr.id = 10000 + int(agent_i)
                arr.type = Marker.ARROW
                arr.action = Marker.ADD
                # Arrow defined by two points. Scale fields control shaft/head size.
                # Use explicit GUI-controlled width + derived head size.
                arr.scale.x = float(arrow_w)
                arr.scale.y = float(max(arrow_w * 1.8, 0.06))
                arr.scale.z = float(max(arrow_w * 2.2, 0.08))
                # Color indicates whether ACO was executed or overridden:
                # - green: executed by ACO
                # - orange: A* executed (ACO shown as "intent")
                # - gray: other
                src = str(getattr(agent, "last_move_source", "") or "")
                if src.startswith("ACO"):
                    arr.color.r = 0.0
                    arr.color.g = 1.0
                    arr.color.b = 0.2
                elif src == "A*":
                    arr.color.r = 1.0
                    arr.color.g = 0.55
                    arr.color.b = 0.0
                else:
                    arr.color.r = 0.6
                    arr.color.g = 0.6
                    arr.color.b = 0.6
                arr.color.a = float(clamp(alpha, 0.0, 1.0))
                # Let old arrows expire so the viz doesn't "stick" when drone selection changes.
                arr.lifetime.sec = int(ttl_s)
                arr.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
                p0 = Point()
                p0.x = float(agent.s.x)
                p0.y = float(agent.s.y)
                p0.z = float(z)
                p1 = Point()
                # Extend/shorten arrow for readability.
                dx = float(cx) - float(agent.s.x)
                dy = float(cy) - float(agent.s.y)
                base_len = math.hypot(dx, dy)
                if base_len <= 1e-9:
                    base_len = max(0.5, float(getattr(self.grid, "cell_size_m", 5.0)) * 0.25)
                    dx = math.cos(float(cyaw)) * base_len
                    dy = math.sin(float(cyaw)) * base_len
                scale_len = float(arrow_len_mult)
                p1.x = float(agent.s.x) + dx * scale_len
                p1.y = float(agent.s.y) + dy * scale_len
                p1.z = float(z)
                arr.points = [p0, p1]
                am.markers.append(arr)

            if seq == 0:
                for agent_i in range(1, int(self.num_py) + 1):
                    _add_aco_for_agent(agent_i)
            else:
                seq = max(1, min(self.num_py, seq))
                _add_aco_for_agent(seq)
        self.pub_aco.publish(am)
        rviz_marker_count += len(am.markers)
        try:
            for mm in am.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # Lidar scan visualization (what the drone "saw" as beams + hits)
        sm = MarkerArray()
        dels = Marker()
        dels.action = Marker.DELETEALL
        dels.header.frame_id = "world"
        sm.markers.append(dels)
        if bool(self.get_parameter("lidar_scan_viz_enabled").value):
            seq = int(self.get_parameter("lidar_scan_viz_drone_seq").value)
            seq = max(1, min(self.num_py, seq))
            agent = self.drones[seq - 1]
            beams = list(getattr(agent, "last_lidar_beams", []) or [])
            stride = max(1, int(self.get_parameter("lidar_scan_viz_beam_stride").value))
            alpha = float(self.get_parameter("lidar_scan_viz_alpha").value)
            w = float(self.get_parameter("lidar_scan_viz_line_width").value)
            ttl_s = float(self.get_parameter("lidar_scan_viz_ttl_s").value)
            from geometry_msgs.msg import Point

            lines = Marker()
            lines.header.frame_id = "world"
            lines.header.stamp = now_msg
            lines.ns = "swarm_lidar_scan"
            lines.id = 9000 + seq
            lines.type = Marker.LINE_LIST
            lines.action = Marker.ADD
            lines.scale.x = float(max(0.01, w))
            if ttl_s > 0.0:
                lines.lifetime.sec = int(ttl_s)
                lines.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
            lines.color.r = 1.0
            lines.color.g = 0.0
            lines.color.b = 0.0
            lines.color.a = float(clamp(alpha, 0.0, 1.0))
            lines.points = []

            hits = Marker()
            hits.header.frame_id = "world"
            hits.header.stamp = now_msg
            hits.ns = "swarm_lidar_hits"
            hits.id = 9100 + seq
            hits.type = Marker.POINTS
            hits.action = Marker.ADD
            hits.scale.x = float(max(0.05, self.grid.cell_size_m * 0.25))
            hits.scale.y = float(max(0.05, self.grid.cell_size_m * 0.25))
            if ttl_s > 0.0:
                hits.lifetime.sec = int(ttl_s)
                hits.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
            hits.color.r = 1.0
            hits.color.g = 0.2
            hits.color.b = 0.2
            hits.color.a = float(clamp(min(1.0, alpha + 0.2), 0.0, 1.0))
            hits.points = []

            sx, sy, sz = float(agent.s.x), float(agent.s.y), float(agent.s.z)
            for i, (ex, ey, hit) in enumerate(beams):
                if i % stride != 0:
                    continue
                p1 = Point(); p1.x = sx; p1.y = sy; p1.z = sz
                p2 = Point(); p2.x = float(ex); p2.y = float(ey); p2.z = sz
                lines.points.append(p1)
                lines.points.append(p2)
                if hit:
                    ph = Point(); ph.x = float(ex); ph.y = float(ey); ph.z = sz
                    hits.points.append(ph)

            if lines.points:
                sm.markers.append(lines)
            if hits.points:
                sm.markers.append(hits)

        self.pub_lidar_scan.publish(sm)
        rviz_marker_count += len(sm.markers)
        try:
            for mm in sm.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        # Lidar debug markers (walls as red line segments + corners as red dots), at drone altitude.
        lm = MarkerArray()
        delm = Marker()
        delm.action = Marker.DELETEALL
        delm.header.frame_id = "world"
        lm.markers.append(delm)

        if bool(self.get_parameter("lidar_viz_enabled").value):
            seq = int(self.get_parameter("lidar_viz_drone_seq").value)
            seq = max(1, min(self.num_py, seq))
            agent = self.drones[seq - 1]
            z = float(agent.s.z)
            alpha = float(self.get_parameter("lidar_viz_alpha").value)
            line_w = float(self.get_parameter("lidar_viz_line_width").value)
            corner_size = float(self.get_parameter("lidar_viz_corner_size").value)
            ttl_s = float(self.get_parameter("lidar_viz_ttl_s").value)

            # Snapshot known occupied cells (avoid mutation during iteration).
            occ_cells = list(agent.known_occ.keys())

            from geometry_msgs.msg import Point

            # Walls: connect adjacent occupied cells (4-neighborhood) -> line segments.
            occ_set = set(occ_cells)
            wall = Marker()
            wall.header.frame_id = "world"
            wall.header.stamp = now_msg
            wall.ns = "swarm_lidar_walls"
            wall.id = 6000 + seq
            wall.type = Marker.LINE_LIST
            wall.action = Marker.ADD
            wall.scale.x = float(max(0.01, line_w))
            if ttl_s > 0.0:
                wall.lifetime.sec = int(ttl_s)
                wall.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
            wall.color.r = 1.0
            wall.color.g = 0.0
            wall.color.b = 0.0
            wall.color.a = float(clamp(alpha, 0.0, 1.0))
            wall.points = []

            # Corners: occupied cells with orthogonal neighbors (turn) or endpoints.
            corners = Marker()
            corners.header.frame_id = "world"
            corners.header.stamp = now_msg
            corners.ns = "swarm_lidar_corners"
            corners.id = 7000 + seq
            corners.type = Marker.SPHERE_LIST
            corners.action = Marker.ADD
            corners.scale.x = float(max(0.05, corner_size))
            corners.scale.y = float(max(0.05, corner_size))
            corners.scale.z = float(max(0.05, corner_size))
            if ttl_s > 0.0:
                corners.lifetime.sec = int(ttl_s)
                corners.lifetime.nanosec = int((ttl_s - int(ttl_s)) * 1e9)
            corners.color.r = 1.0
            corners.color.g = 0.0
            corners.color.b = 0.0
            corners.color.a = float(clamp(alpha, 0.0, 1.0))
            corners.points = []

            for c in occ_cells:
                cx, cy = int(c[0]), int(c[1])
                # wall segments: to right and up to avoid duplicates
                for dx, dy in ((1, 0), (0, 1)):
                    nb = (cx + dx, cy + dy)
                    if nb in occ_set:
                        x1, y1 = self.grid.cell_to_world(cx, cy)
                        x2, y2 = self.grid.cell_to_world(nb[0], nb[1])
                        p1 = Point(); p1.x = float(x1); p1.y = float(y1); p1.z = float(z)
                        p2 = Point(); p2.x = float(x2); p2.y = float(y2); p2.z = float(z)
                        wall.points.append(p1)
                        wall.points.append(p2)

                left = (cx - 1, cy) in occ_set
                right = (cx + 1, cy) in occ_set
                down = (cx, cy - 1) in occ_set
                up = (cx, cy + 1) in occ_set
                deg = int(left) + int(right) + int(down) + int(up)
                is_turn = (left or right) and (up or down) and not ((left and right) or (up and down))
                is_endpoint = deg <= 1
                if is_turn or is_endpoint:
                    wx, wy = self.grid.cell_to_world(cx, cy)
                    p = Point(); p.x = float(wx); p.y = float(wy); p.z = float(z)
                    corners.points.append(p)

            if wall.points:
                lm.markers.append(wall)
            if corners.points:
                lm.markers.append(corners)

        self.pub_lidar.publish(lm)
        rviz_marker_count += len(lm.markers)
        try:
            for mm in lm.markers:
                rviz_point_count += len(getattr(mm, "points", []) or [])
        except Exception:
            pass

        self._last_rviz_marker_count = int(rviz_marker_count)
        self._last_rviz_point_count = int(rviz_point_count)
        self._perf_add("rviz_publish", time.perf_counter() - t0)

    # -------- persistence --------

    def _persist_outputs(self):
        # Merge current drone maps into base then write files.
        t_ref = self.t_sim
        self._upload_all_to_base(t_ref)

        workspace_root = Path(__file__).resolve().parent.parent
        base_full_path = workspace_root / str(self.get_parameter("base_full_path").value)
        compat_path = workspace_root / str(self.get_parameter("compat_export_path").value)
        stats_path = workspace_root / str(self.get_parameter("stats_path").value)
        base_full_path.parent.mkdir(parents=True, exist_ok=True)
        compat_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # full base map
        def layer_to_list(layer: SparseLayer):
            out = []
            for (x, y), v in layer.v.items():
                meta = layer.meta.get((x, y))
                row = {
                    "x": int(x),
                    "y": int(y),
                    "v": float(v),
                    "t": float(meta.t if meta else 0.0),
                    "conf": float(meta.conf if meta else 0.5),
                    "src": str(meta.src if meta else "unknown"),
                }
                if meta is not None and meta.alt_m is not None:
                    row["alt"] = float(meta.alt_m)
                if meta is not None and getattr(meta, "speed_s_per_cell", None) is not None:
                    try:
                        row["speed"] = float(getattr(meta, "speed_s_per_cell"))
                    except Exception:
                        pass
                if meta is not None and meta.kind:
                    row["kind"] = str(meta.kind)
                out.append(row)
            return out

        full = {
            "version": 5,
            "grid": {"grid_size": self.grid.grid_size_m, "cell_size": self.grid.cell_size_m},
            "base_uid": self.base_map.owner_uid,
            "time": {"t_sim": float(self.t_sim), "wall": time.time()},
            # Layers: nav + danger + empty + explored. `danger` carries optional per-cell kind/alt metadata.
            "layers": {
                "nav": layer_to_list(self.base_map.nav),
                "danger": layer_to_list(self.base_map.danger),
                "empty": layer_to_list(self.base_map.empty),
                "explored": layer_to_list(self.base_map.explored),
            },
        }
        tmp = str(base_full_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(full, f, indent=2)
        os.replace(tmp, base_full_path)

        # compatibility export: intensity bins 0..3 from danger layer
        compat_cells = []
        for (cx, cy), v in self.base_map.danger.v.items():
            # simple binning for existing scripts
            if v <= 0.5:
                intensity = 1
            elif v <= 1.5:
                intensity = 2
            else:
                intensity = 3
            compat_cells.append({"cell_x": int(cx), "cell_y": int(cy), "intensity": int(intensity)})
        tmp = str(compat_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(compat_cells, f, indent=2)
        os.replace(tmp, compat_path)

        # stats
        stats = {
            "t_sim": float(self.t_sim),
            "mission_phase": self.mission_phase,
            "num_py_drones": self.num_py,
            "targets_total": len(self.targets),
            "targets_found": sum(1 for t in self.targets if t.is_found()),
            "drones": [
                {
                    "uid": d.s.drone_uid,
                    "type": d.s.drone_type,
                    "seq": d.s.seq,
                    "total_dist_m": float(d.s.total_dist_m),
                    "energy_units": float(d.s.energy_units),
                    "encounters": int(d.s.encounters),
                    "base_uploads": int(d.s.base_uploads),
                }
                for d in self.drones
            ],
        }
        tmp = str(stats_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, stats_path)


def main(args=None):
    rclpy.init(args=args)
    node = PythonFastSim()
    try:
        # MultiThreadedExecutor helps keep subscriber callbacks (e.g. GUI commands) responsive
        # even when the sim tick is expensive. This does not bypass the GIL, but it reduces
        # starvation for short callbacks.
        from rclpy.executors import MultiThreadedExecutor

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Avoid double-shutdown errors on Ctrl-C in some environments
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()


