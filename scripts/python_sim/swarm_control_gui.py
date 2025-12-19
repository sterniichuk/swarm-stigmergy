#!/usr/bin/env python3
"""
Swarm Control GUI (Python, ROS2)

Buttons/controls for the Python fast sim (and future real-drones adapter) using ROS2:
- Start Python Drones (Trigger: /swarm/cmd/start_python)
- Stop Python Drones (Trigger: /swarm/cmd/stop_python)
- Speed slider (Float32 pub: /swarm/cmd/speed)
- Target add mode toggle (Bool pub: /swarm/cmd/target_add_mode)
- Clear targets (Trigger: /swarm/cmd/clear_targets)

Live status:
- Mission phase (/swarm/mission_phase)
- Sim time (/clock)

Note: This is a standalone Qt app (not an RViz panel plugin).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
import sys
import json
import time
import uuid
import os
import signal
import atexit
import threading
import traceback
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, Float32, String
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from std_srvs.srv import Trigger
from rosgraph_msgs.msg import Clock


def _format_sim_time(sec: int, nsec: int) -> str:
    # Human readable relative time (HH:MM:SS.mmm)
    total_ms = (sec * 1000) + (nsec // 1_000_000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_min = total_s // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _settings_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "gui_settings.json"


def _load_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_settings(data: dict):
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    Path(tmp).replace(p)


def _gui_lock_path() -> Path:
    # Per-user lock in /tmp so multiple checkouts don't fight each other.
    user = os.environ.get("USER") or str(os.getuid())
    return Path("/tmp") / f"stigmergy-lab-gui.{user}.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _find_other_gui_pids() -> list[int]:
    """
    Best-effort: find other running swarm_control_gui.py processes for the same user.
    We avoid external deps; on Linux we can scan /proc.
    """
    me = os.getpid()
    uid = os.getuid()
    out: list[int] = []
    proc_root = Path("/proc")
    try:
        for ent in proc_root.iterdir():
            if not ent.name.isdigit():
                continue
            pid = int(ent.name)
            if pid == me:
                continue
            try:
                st = (ent / "status").read_text(encoding="utf-8", errors="ignore")
                if f"Uid:\t{uid}\t" not in st and f"Uid:\t{uid} " not in st:
                    continue
            except Exception:
                continue
            try:
                cmd = (ent / "cmdline").read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ")
            except Exception:
                continue
            if "swarm_control_gui.py" in cmd:
                out.append(pid)
    except Exception:
        return []
    return sorted(set(out))


def _ensure_single_instance_or_prompt() -> Optional[int]:
    """
    Enforce single GUI instance per user.

    Returns an open file descriptor to the lock file if we are the active instance, else None (caller should exit).
    Prompts on the terminal (stdin) when another instance is detected.
    """
    lock_path = _gui_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_pid_from_lock() -> int:
        try:
            raw = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return -1
        try:
            return int(raw.splitlines()[0].strip()) if raw else -1
        except Exception:
            return -1

    # Use an atomic lock file create (O_EXCL). This is robust across environments and doesn't rely
    # on advisory file locks.
    for _attempt in range(3):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            os.fsync(fd)
            return fd
        except FileExistsError:
            other_pid = _read_pid_from_lock()
            other_pids = []
            if other_pid > 0:
                other_pids = [other_pid]
            else:
                # If PID is missing/invalid, try to discover other running GUIs to avoid
                # two instances fighting over marker topics.
                other_pids = _find_other_gui_pids()
                other_pid = other_pids[0] if other_pids else other_pid
            # Stale lock: remove and retry.
            if other_pid > 0 and not _pid_alive(other_pid) and not other_pids:
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            msg = (
                f"\nStigmergy Lab GUI appears to be already running (pid={other_pid}).\n"
                "Choose an action:\n"
                "  [k] Kill the old GUI and start this one\n"
                "  [q] Quit this instance\n"
                "Selection [q]: "
            )
            try:
                sys.stderr.write(msg)
                sys.stderr.flush()
            except Exception:
                pass

            if not sys.stdin or not sys.stdin.isatty():
                try:
                    sys.stderr.write("\nNo interactive stdin detected -> quitting this instance.\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                return None

            try:
                choice = (sys.stdin.readline() or "").strip().lower()
            except Exception:
                choice = ""
            if choice not in ("k", "kill"):
                return None

            # Kill old and remove lock; then retry.
            # If we couldn't read a PID, kill all discovered GUI PIDs (same user).
            kill_list = [p for p in other_pids if p > 0 and _pid_alive(p)]
            if other_pid > 0 and not kill_list:
                kill_list = [other_pid]
            if kill_list:
                for p in kill_list:
                    try:
                        os.kill(p, signal.SIGTERM)
                    except Exception:
                        pass
                deadline = time.time() + 2.0
                while time.time() < deadline and any(_pid_alive(p) for p in kill_list):
                    time.sleep(0.1)
                # Escalate
                for p in kill_list:
                    if _pid_alive(p):
                        try:
                            os.kill(p, signal.SIGKILL)
                        except Exception:
                            pass
                time.sleep(0.2)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        except Exception:
            # If we can't lock for any reason, fail safe by allowing to run.
            try:
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
                os.fsync(fd)
                return fd
            except Exception:
                return None

    return None


class ClickMode:
    NONE = "NONE"
    TARGETS = "TARGETS"
    DANGER_STATIC = "DANGER_STATIC"
    DANGER_DYNAMIC = "DANGER_DYNAMIC"


def _tooltip_text_start() -> str:
    return (
        "Start Python Drones:\n"
        "- Calls /swarm/cmd/start_python\n"
        "- Resets and starts the Python swarm exploration phase (time-accelerated)\n"
        "- Drones begin exploring with ACO-style motion"
    )


def _tooltip_text_stop() -> str:
    return (
        "Return to Base:\n"
        "- Calls /swarm/cmd/stop_python\n"
        "- Ends exploration, switches to RETURN-to-base, and persists results to disk\n"
        "- Does NOT close the GUI or stop other publishers"
    )


def _tooltip_text_pause() -> str:
    return (
        "Pause / Resume:\n"
        "- Publishes /swarm/cmd/pause_python\n"
        "- Pause freezes sim time (/clock stops advancing) and drones stop moving/sensing\n"
        "- Resume continues from the same sim time"
    )


def _tooltip_text_clear_targets() -> str:
    return (
        "Clear Targets:\n"
        "- Calls /swarm/cmd/clear_targets\n"
        "- Removes all targets (and saves targets.json via autosave)\n"
        "- Drones will stop searching for cleared targets"
    )


def _tooltip_text_clear_targets_found() -> str:
    return (
        "Clear FOUND Targets:\n"
        "- Calls /swarm/cmd/clear_targets_found\n"
        "- Removes ONLY targets already marked found\n"
        "- Saves targets.json via autosave"
    )


def _tooltip_text_clear_targets_unfound() -> str:
    return (
        "Clear UNFOUND Targets:\n"
        "- Calls /swarm/cmd/clear_targets_unfound\n"
        "- Removes ONLY targets NOT yet found\n"
        "- Saves targets.json via autosave"
    )


def _tooltip_text_set_all_targets_unfound() -> str:
    return (
        "Set ALL Targets UNFOUND:\n"
        "- Calls /swarm/cmd/set_all_targets_unfound\n"
        "- Keeps all targets, but clears found_by/found_t for every target\n"
        "- Saves targets.json via autosave\n"
        "- Forces drones to re-sync target found state"
    )


def _tooltip_text_delete_nearest_target() -> str:
    return (
        "Delete nearest target to last published pose:\n"
        "- Calls /swarm/cmd/delete_nearest_target\n"
        "- Uses last PoseStamped received by python_fast_sim (default topic: /move_base_simple/goal)\n"
        "- Useful with RViz publish_pose / 2D Nav Goal tools"
    )


class _TkTooltip:
    """Minimal Tk tooltip helper (no external deps)."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self._tip = None
        self._after = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _evt=None):
        # Delay a bit so tooltips don't flicker
        try:
            self._after = self.widget.after(400, self._show)
        except Exception:
            pass

    def _on_leave(self, _evt=None):
        try:
            if self._after is not None:
                self.widget.after_cancel(self._after)
        except Exception:
            pass
        self._after = None
        self._hide()

    def _show(self):
        if self._tip is not None:
            return
        try:
            import tkinter as tk
        except Exception:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip,
            text=self.text,
            justify="left",
            background="#111",
            foreground="#fff",
            relief="solid",
            borderwidth=1,
            font=("Sans", 9),
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
        self._tip = None


class SwarmControlNode(Node):
    def __init__(self):
        super().__init__("swarm_control_gui")

        self.pub_speed = self.create_publisher(Float32, "/swarm/cmd/speed", 10)
        self.pub_target_mode = self.create_publisher(Bool, "/swarm/cmd/target_add_mode", 10)
        self.pub_building_alpha = self.create_publisher(Float32, "/swarm/cmd/building_alpha", 10)
        self.pub_pher_viz_enable = self.create_publisher(Bool, "/swarm/cmd/pheromone_viz_enable", 10)
        self._Float64MultiArray = Float64MultiArray
        self.pub_pher_viz_params = self.create_publisher(Float64MultiArray, "/swarm/cmd/pheromone_viz_params", 10)
        self.pub_target_viz_params = self.create_publisher(Float64MultiArray, "/swarm/cmd/target_viz_params", 10)
        self.pub_pause = self.create_publisher(Bool, "/swarm/cmd/pause_python", 10)
        self.pub_drone_scale = self.create_publisher(Float32, "/swarm/cmd/drone_marker_scale", 10)
        self.pub_pher_select = self.create_publisher(String, "/swarm/cmd/pheromone_viz_select", 10)
        self.pub_pointer_params = self.create_publisher(Float64MultiArray, "/swarm/cmd/drone_pointer_params", 10)
        self.pub_return_speed = self.create_publisher(Float32, "/swarm/cmd/return_speed_mps", 10)
        self.pub_drone_altitude = self.create_publisher(Float32, "/swarm/cmd/drone_altitude_m", 10)
        # Optional sim control: tie vertical speed to current horizontal speed.
        self.pub_vertical_speed_mult = self.create_publisher(Float32, "/swarm/cmd/vertical_speed_mult", 10)
        self.pub_target_viz_params = self.create_publisher(Float64MultiArray, "/swarm/cmd/target_viz_params", 10)
        self.pub_lidar_viz_select = self.create_publisher(String, "/swarm/cmd/lidar_viz_select", 10)
        self.pub_plan_viz_select = self.create_publisher(String, "/swarm/cmd/plan_viz_select", 10)
        self.pub_aco_viz_select = self.create_publisher(String, "/swarm/cmd/aco_viz_select", 10)
        self.pub_lidar_scan_viz_select = self.create_publisher(String, "/swarm/cmd/lidar_scan_viz_select", 10)
        # Goal/selection helpers (used for real drones + optional sim goal-seeking)
        self.pub_selected_target = self.create_publisher(String, "/swarm/cmd/selected_target", 10)
        # Pheromone snapshot storage (python_fast_sim)
        self.pub_pheromone_map_storage = self.create_publisher(String, "/swarm/cmd/pheromone_map_storage", 10)
        # Start exploit run (python_fast_sim)
        self.pub_exploit_start = self.create_publisher(String, "/swarm/cmd/exploit_start", 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, "/move_base_simple/goal", 10)

        # Click routing: GUI is the single owner of /clicked_point (exclusive mode)
        self.sub_clicked = self.create_subscription(PointStamped, "/clicked_point", self._on_clicked_point, 10)
        self.pub_add_target = self.create_publisher(PointStamped, "/swarm/targets/add", 10)
        self.pub_danger_static = self.create_publisher(PointStamped, "/danger/add_static", 10)
        self.pub_danger_dyn_point = self.create_publisher(PointStamped, "/danger/add_dynamic_point", 10)
        self.pub_danger_dyn_finish = self.create_publisher(String, "/danger/finish_dynamic", 10)

        self.cli_start = self.create_client(Trigger, "/swarm/cmd/start_python")
        self.cli_stop = self.create_client(Trigger, "/swarm/cmd/stop_python")
        self.cli_clear = self.create_client(Trigger, "/swarm/cmd/clear_targets")
        self.cli_clear_found = self.create_client(Trigger, "/swarm/cmd/clear_targets_found")
        self.cli_clear_unfound = self.create_client(Trigger, "/swarm/cmd/clear_targets_unfound")
        self.cli_set_all_targets_unfound = self.create_client(Trigger, "/swarm/cmd/set_all_targets_unfound")
        self.cli_delete_nearest_target = self.create_client(Trigger, "/swarm/cmd/delete_nearest_target")

        self.sub_phase = self.create_subscription(String, "/swarm/mission_phase", self._on_phase, 10)
        self.sub_clock = self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.sub_gui_status = self.create_subscription(String, "/swarm/gui_status", self._on_gui_status, 10)

        self.latest_phase: str = "—"
        self.latest_clock: Optional[Clock] = None
        self.latest_log: str = ""
        self.gui_status: dict = {}

        # Click mode state
        self.click_mode: str = ClickMode.NONE
        self.dynamic_speed_sec_per_cell: float = 4.0
        self.danger_radius_cells: int = 0
        # Threat height above ground (meters). Used for static threats and as a default for dynamic threats.
        self.threat_height_m: float = 50.0
        self.dynamic_recording: bool = False
        self.dynamic_id: Optional[str] = None

    def _on_phase(self, msg: String):
        self.latest_phase = msg.data or "—"

    def _on_clock(self, msg: Clock):
        self.latest_clock = msg

    def _on_gui_status(self, msg: String):
        try:
            self.gui_status = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.gui_status = {}

    def set_speed(self, speed: float):
        m = Float32()
        m.data = float(speed)
        self.pub_speed.publish(m)

    def set_target_add_mode(self, enabled: bool):
        m = Bool()
        m.data = bool(enabled)
        self.pub_target_mode.publish(m)

    def set_pheromone_viz_enabled(self, enabled: bool):
        m = Bool()
        m.data = bool(enabled)
        self.pub_pher_viz_enable.publish(m)

    def set_pheromone_viz_params(self, display_size_m: float, z: float, alpha: float):
        msg = self._Float64MultiArray()
        msg.data = [float(display_size_m), float(z), float(alpha)]
        self.pub_pher_viz_params.publish(msg)

    def set_building_alpha(self, alpha: float):
        m = Float32()
        m.data = float(alpha)
        self.pub_building_alpha.publish(m)

    def set_target_viz_params(self, diameter_m: float, alpha: float):
        msg = self._Float64MultiArray()
        msg.data = [float(diameter_m), float(alpha)]
        self.pub_target_viz_params.publish(msg)

    def set_paused(self, paused: bool):
        m = Bool()
        m.data = bool(paused)
        self.pub_pause.publish(m)

    def set_drone_marker_scale(self, scale: float):
        m = Float32()
        m.data = float(scale)
        self.pub_drone_scale.publish(m)

    def set_pheromone_viz_select(self, owner: str, layer: str, drone_seq: int):
        payload = {"owner": owner, "layer": layer, "drone_seq": int(drone_seq)}
        m = String()
        m.data = json.dumps(payload)
        self.pub_pher_select.publish(m)

    def set_drone_pointer_params(self, enabled: bool, z: float, scale: float, alpha: float):
        msg = self._Float64MultiArray()
        msg.data = [1.0 if enabled else 0.0, float(z), float(scale), float(alpha)]
        self.pub_pointer_params.publish(msg)

    def set_return_speed_mps(self, v: float):
        m = Float32()
        m.data = float(v)
        self.pub_return_speed.publish(m)

    def set_drone_altitude_m(self, z: float):
        m = Float32()
        m.data = float(z)
        self.pub_drone_altitude.publish(m)

    def set_vertical_speed_mult(self, mult: float):
        """Set vertical speed multiplier (0.1..1.0). Used by python_fast_sim when enabled."""
        m = Float32()
        m.data = float(mult)
        self.pub_vertical_speed_mult.publish(m)

    def set_lidar_viz(self, enabled: bool, drone_seq: int):
        payload = {"enabled": bool(enabled), "drone_seq": int(drone_seq)}
        m = String()
        m.data = json.dumps(payload)
        self.pub_lidar_viz_select.publish(m)

    def set_plan_viz(self, enabled: bool, drone_seq: int):
        payload = {"enabled": bool(enabled), "drone_seq": int(drone_seq)}
        m = String()
        m.data = json.dumps(payload)
        self.pub_plan_viz_select.publish(m)

    def set_aco_viz(self, enabled: bool, drone_seq: int):
        payload = {"enabled": bool(enabled), "drone_seq": int(drone_seq)}
        m = String()
        m.data = json.dumps(payload)
        self.pub_aco_viz_select.publish(m)

    def set_lidar_scan_viz(self, enabled: bool, drone_seq: int):
        payload = {"enabled": bool(enabled), "drone_seq": int(drone_seq)}
        m = String()
        m.data = json.dumps(payload)
        self.pub_lidar_scan_viz_select.publish(m)

    def set_selected_target(self, target_id: str):
        """Select a target id for goal-seeking (String JSON payload). Empty string clears selection."""
        payload = {"id": str(target_id or "")}
        m = String()
        m.data = json.dumps(payload)
        self.pub_selected_target.publish(m)

    def save_pheromone_snapshot(self, name: str):
        payload = {"cmd": "save", "name": str(name or "")}
        m = String()
        m.data = json.dumps(payload)
        self.pub_pheromone_map_storage.publish(m)
        self.latest_log = f"Snapshot save requested: {payload['name'] or '—'}"

    def load_pheromone_snapshot(self, path: str):
        payload = {"cmd": "load", "path": str(path or "")}
        m = String()
        m.data = json.dumps(payload)
        self.pub_pheromone_map_storage.publish(m)
        self.latest_log = f"Snapshot load requested: {payload['path'] or '—'}"

    def load_compat_pheromone(self):
        """Ask python_fast_sim to load legacy compat pheromone file into base map."""
        payload = {"cmd": "load_compat"}
        m = String()
        m.data = json.dumps(payload)
        self.pub_pheromone_map_storage.publish(m)
        self.latest_log = "Compat pheromone load requested"

    def start_exploit(self, target_id: str, drone_count: int, dynamic_mode: str):
        payload = {"target_id": str(target_id or ""), "drone_count": int(drone_count), "dynamic_mode": str(dynamic_mode or "handled")}
        m = String()
        m.data = json.dumps(payload)
        self.pub_exploit_start.publish(m)
        self.latest_log = f"Exploit start requested: target={payload['target_id']}, drones={payload['drone_count']}, mode={payload['dynamic_mode']}"

    def send_goal_pose(self, x: float, y: float, z: float = 0.0, frame_id: str = "world"):
        """Publish a PoseStamped goal (compatible with /move_base_simple/goal consumers)."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(frame_id)
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        # identity orientation (yaw ignored by most consumers)
        msg.pose.orientation.w = 1.0
        self.pub_goal_pose.publish(msg)

    def set_click_mode(self, mode: str):
        self.click_mode = mode
        self.set_target_add_mode(mode == ClickMode.TARGETS)
        if mode != ClickMode.DANGER_DYNAMIC:
            self.dynamic_recording = False
            self.dynamic_id = None

    def start_dynamic_recording(self):
        self.dynamic_recording = True
        self.dynamic_id = str(uuid.uuid4())
        self.latest_log = f"Dynamic danger recording started: {self.dynamic_id[:8]}..."

    def stop_dynamic_recording(self):
        if not self.dynamic_recording or not self.dynamic_id:
            self.latest_log = "Dynamic danger recording not active"
            return
        payload = {
            "id": self.dynamic_id,
            "speed": float(self.dynamic_speed_sec_per_cell),
            "radius": int(self.danger_radius_cells),
            "height_m": float(self.threat_height_m),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_danger_dyn_finish.publish(msg)
        self.latest_log = f"Dynamic danger finish sent: {self.dynamic_id[:8]}... speed={payload['speed']:.2f}s/cell"
        self.dynamic_recording = False
        self.dynamic_id = None

    def _on_clicked_point(self, msg: PointStamped):
        if self.click_mode == ClickMode.TARGETS:
            self.pub_add_target.publish(msg)
            return
        if self.click_mode == ClickMode.DANGER_STATIC:
            # Encode static danger:
            # - integer part: radius (cells)
            # - fractional part: height_m (meters) * 0.001
            try:
                r = int(self.danger_radius_cells)
                h = float(self.threat_height_m)
                h = max(0.0, min(999.0, h))
                msg.point.z = float(r) + (float(h) * 0.001)
            except Exception:
                msg.point.z = 0.0
            self.pub_danger_static.publish(msg)
            return
        if self.click_mode == ClickMode.DANGER_DYNAMIC:
            if not self.dynamic_recording or not self.dynamic_id:
                return
            out = PointStamped()
            out.header = msg.header
            out.header.frame_id = f"danger_id:{self.dynamic_id}"
            out.point = msg.point
            self.pub_danger_dyn_point.publish(out)
            return

    def call_start(self):
        self._call_trigger(self.cli_start, "Start Python Drones")

    def call_stop(self):
        self._call_trigger(self.cli_stop, "Stop Python Drones")

    def call_clear_targets(self):
        self._call_trigger(self.cli_clear, "Clear targets")

    def call_clear_targets_found(self):
        self._call_trigger(self.cli_clear_found, "Clear FOUND targets")

    def call_clear_targets_unfound(self):
        self._call_trigger(self.cli_clear_unfound, "Clear UNFOUND targets")

    def call_set_all_targets_unfound(self):
        self._call_trigger(self.cli_set_all_targets_unfound, "Set ALL targets UNFOUND")

    def call_delete_nearest_target(self):
        self._call_trigger(self.cli_delete_nearest_target, "Delete nearest target (pose)")

    def _call_trigger(self, client, label: str):
        if not client.service_is_ready():
            self.latest_log = f"{label}: service not ready"
            return
        req = Trigger.Request()
        future = client.call_async(req)

        def _done(_fut):
            try:
                res = _fut.result()
                self.latest_log = f"{label}: {res.success} — {res.message}"
            except Exception as e:
                self.latest_log = f"{label}: failed — {e}"

        future.add_done_callback(_done)


def main(argv=None):
    argv = argv if argv is not None else sys.argv

    # Single-instance guard (before rclpy.init()).
    _lock_fd = _ensure_single_instance_or_prompt()
    if _lock_fd is None:
        return 0
    # Ensure we release lock on exit.
    def _release_lock():
        try:
            os.close(_lock_fd)
        except Exception:
            pass
        try:
            _gui_lock_path().unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_release_lock)

    # Ensure repo root is importable even when running from scripts/python_sim/
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Decide GUI backend based on availability, not runtime errors after ROS init.
    # If Qt is present, we run Qt. If Qt import is missing, we run Tk.
    try:
        from PyQt5 import QtCore, QtWidgets  # type: ignore
    except Exception:
        return _main_tk(argv)

    # Qt is available; do not fall back to Tk on runtime exceptions (it can cause double rclpy.init()).
    return _main_qt(argv, QtCore, QtWidgets)

def _main_qt(argv, QtCore, QtWidgets):
    rclpy.init(args=argv)
    # Run all nodes (GUI + fast sim + buildings + danger) in the SAME process.
    from scripts.python_sim.python_fast_sim import PythonFastSim
    from scripts.publishers.publish_gazebo_buildings import GazeboBuildingsPublisher
    from scripts.publishers.publish_ground_plane import GroundPlanePublisher
    from scripts.danger.danger_map_manager import DangerMapManager

    exec_ = MultiThreadedExecutor(num_threads=4)
    fast_sim = PythonFastSim()  # publishes /pheromone_heatmap immediately
    buildings = GazeboBuildingsPublisher()  # publishes /gazebo_buildings immediately
    ground = GroundPlanePublisher()  # publishes /ground_plane immediately
    danger = DangerMapManager()  # publishes /danger_map immediately
    node = SwarmControlNode()

    for n in (fast_sim, buildings, ground, danger, node):
        exec_.add_node(n)

    # Spin ROS executor in a background thread so the GUI never blocks.
    # IMPORTANT: use exec_.spin() (not spin_once) to avoid starving some nodes/timers.
    stop_spin = threading.Event()

    def _spin_loop():
        try:
            exec_.spin()
        except Exception as e:
            try:
                print("ROS executor thread crashed:", repr(e), file=sys.stderr)
                traceback.print_exc()
            except Exception:
                pass

    spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    spin_thread.start()

    # Qt UI
    app = QtWidgets.QApplication(argv)
    repo_root = Path(__file__).resolve().parents[2]
    logo_path = repo_root / "data" / "gui-app-logo.png"

    app.setApplicationName("Stigmergy Lab")
    try:
        app.setApplicationDisplayName("Stigmergy Lab")
    except Exception:
        pass

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Stigmergy Lab")
    # Use logo ONLY as window icon (Ubuntu titlebar). Do not render it inside the app.
    try:
        from PyQt5 import QtGui
        if logo_path.exists():
            win.setWindowIcon(QtGui.QIcon(str(logo_path)))
    except Exception:
        pass

    # Scrollable main panel (settings have grown a lot).
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    central = QtWidgets.QWidget()
    scroll.setWidget(central)
    win.setCentralWidget(scroll)

    layout = QtWidgets.QVBoxLayout(central)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(14)

    # Settings (used for initial UI state + later persistence)
    settings = _load_settings()
    # Remember last folder used when loading pheromone snapshots.
    # Stored as an absolute path in gui_settings.json.
    pher_snapshot_last_dir = {"v": str(settings.get("pheromone_snapshot_last_dir", "") or "")}
    # Remember last snapshot file path that the user loaded (for "Load latest" button).
    # Stored as an absolute path in gui_settings.json.
    pher_snapshot_last_path = {"v": str(settings.get("pheromone_snapshot_last_path", "") or "")}

    class _QtCollapsibleSection(QtWidgets.QWidget):
        """
        A simple collapsible section: a header button + a body widget.

        This avoids relying on QGroupBox "checkable" (which disables but doesn't
        actually collapse/hide content + reclaim space).
        """

        def __init__(self, title: str, collapsed_key: str, default_expanded: bool = True):
            super().__init__()
            self._collapsed_key = str(collapsed_key)
            try:
                collapsed = bool(settings.get(self._collapsed_key, not bool(default_expanded)))
            except Exception:
                collapsed = not bool(default_expanded)

            self.toggle_btn = QtWidgets.QToolButton()
            self.toggle_btn.setText(title)
            self.toggle_btn.setCheckable(True)
            self.toggle_btn.setChecked(not collapsed)
            self.toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
            self.toggle_btn.setStyleSheet("QToolButton { font-weight: 600; }")
            self.toggle_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

            # A thin separator line under the header helps readability in long UIs.
            self._sep = QtWidgets.QFrame()
            self._sep.setFrameShape(QtWidgets.QFrame.HLine)
            self._sep.setFrameShadow(QtWidgets.QFrame.Sunken)

            self.body = QtWidgets.QWidget()
            self.body.setVisible(not collapsed)

            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(6)
            outer.addWidget(self.toggle_btn)
            outer.addWidget(self._sep)
            outer.addWidget(self.body)

            self.toggle_btn.toggled.connect(self._on_toggled)

        def _on_toggled(self, expanded: bool):
            try:
                self.body.setVisible(bool(expanded))
                self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            except Exception:
                pass

        def is_collapsed(self) -> bool:
            try:
                return not bool(self.toggle_btn.isChecked())
            except Exception:
                return False

    # Clock
    lbl_clock = QtWidgets.QLabel("Sim time: —")
    lbl_clock.setStyleSheet("font-size: 18px;")
    layout.addWidget(lbl_clock)

    lbl_clock_small = QtWidgets.QLabel("Timestamp: —")
    lbl_clock_small.setStyleSheet("font-size: 11px; color: #888;")
    layout.addWidget(lbl_clock_small)

    # Phase
    lbl_phase = QtWidgets.QLabel("Phase: —")
    lbl_phase.setStyleSheet("font-size: 14px;")
    layout.addWidget(lbl_phase)
    layout.addSpacing(6)

    # Targets summary + drones needing info
    status_box = _QtCollapsibleSection("Mission status", "qt_section_mission_status_collapsed", default_expanded=True)
    status_layout = QtWidgets.QVBoxLayout(status_box.body)
    lbl_targets = QtWidgets.QLabel("Targets: —")
    lbl_targets.setStyleSheet("font-size: 13px; font-weight: 600;")
    status_layout.addWidget(lbl_targets)
    txt_drones = QtWidgets.QPlainTextEdit()
    txt_drones.setReadOnly(True)
    txt_drones.setPlaceholderText("Drones missing target knowledge / not found targets will appear here.")
    status_layout.addWidget(txt_drones)
    layout.addWidget(status_box)
    layout.addSpacing(10)

    # Buttons
    btn_start = QtWidgets.QPushButton("Start Python Drones")
    btn_stop = QtWidgets.QPushButton("Return to Base")
    btn_pause = QtWidgets.QPushButton("Pause")
    btn_clear_found = QtWidgets.QPushButton("Clear FOUND Targets")
    btn_clear_unfound = QtWidgets.QPushButton("Clear UNFOUND Targets")
    btn_set_unfound = QtWidgets.QPushButton("Set ALL Targets UNFOUND")
    btn_clear = QtWidgets.QPushButton("Clear ALL Targets")
    btn_del_nearest = QtWidgets.QPushButton("Delete nearest target (pose)")
    # Tooltips (top control section only)
    btn_start.setToolTip(_tooltip_text_start())
    btn_stop.setToolTip(_tooltip_text_stop())
    btn_pause.setToolTip(_tooltip_text_pause())
    btn_clear_found.setToolTip(_tooltip_text_clear_targets_found())
    btn_clear_unfound.setToolTip(_tooltip_text_clear_targets_unfound())
    btn_set_unfound.setToolTip(_tooltip_text_set_all_targets_unfound())
    btn_clear.setToolTip(_tooltip_text_clear_targets())
    btn_del_nearest.setToolTip(_tooltip_text_delete_nearest_target())
    row_main = QtWidgets.QHBoxLayout()
    row_main.addWidget(btn_start)
    row_main.addWidget(btn_stop)
    row_main.addWidget(btn_pause)
    layout.addLayout(row_main)

    # Group target maintenance buttons into one subsection (keeps top controls compact).
    tgt_btn_box = _QtCollapsibleSection("Targets", "qt_section_targets_controls_collapsed", default_expanded=True)
    tgt_btn_layout = QtWidgets.QGridLayout(tgt_btn_box.body)
    tgt_btn_layout.setHorizontalSpacing(8)
    tgt_btn_layout.setVerticalSpacing(6)
    tgt_btn_layout.addWidget(btn_clear_found, 0, 0)
    tgt_btn_layout.addWidget(btn_clear_unfound, 0, 1)
    tgt_btn_layout.addWidget(btn_set_unfound, 0, 2)
    tgt_btn_layout.addWidget(btn_clear, 1, 0)
    tgt_btn_layout.addWidget(btn_del_nearest, 1, 1, 1, 2)
    layout.addWidget(tgt_btn_box)
    layout.addSpacing(10)

    # Speed slider
    speed_box = _QtCollapsibleSection("Python speed (time multiplier)", "qt_section_python_speed_collapsed", default_expanded=True)
    speed_layout = QtWidgets.QVBoxLayout(speed_box.body)
    lbl_speed = QtWidgets.QLabel("Speed: 10.0×")
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    # map 1..2000 -> 0.1..200.0
    slider.setMinimum(1)
    slider.setMaximum(2000)
    slider.setValue(100)  # 10.0×
    speed_layout.addWidget(lbl_speed)
    speed_layout.addWidget(slider)

    # Return speed (m/s)
    lbl_return = QtWidgets.QLabel("Return speed: 10.0 m/s")
    slider_return = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_return.setMinimum(1)    # 0.1 m/s
    slider_return.setMaximum(500)  # 50.0 m/s
    slider_return.setValue(100)    # 10.0 m/s
    speed_layout.addWidget(lbl_return)
    speed_layout.addWidget(slider_return)

    # Speed limits (reported by python_fast_sim via /swarm/gui_status)
    lbl_speed_limits = QtWidgets.QLabel("Max speeds: horizontal — m/s, vertical — m/s")
    lbl_speed_limits.setStyleSheet("color: #666;")
    speed_layout.addWidget(lbl_speed_limits)

    # Vertical speed multiplier (0.1..1.0). When enabled in sim, climb/descend rate becomes:
    #   vertical_speed_mult * current_horizontal_speed_mps
    chk_vert_enabled = QtWidgets.QCheckBox("Enable vertical speed multiplier (tie climb/descend to horizontal speed)")
    chk_vert_enabled.setChecked(True)
    speed_layout.addWidget(chk_vert_enabled)
    lbl_vmult = QtWidgets.QLabel("Vertical speed mult: 0.30")
    slider_vmult = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_vmult.setMinimum(10)   # 0.10
    slider_vmult.setMaximum(100)  # 1.00
    slider_vmult.setValue(30)     # 0.30
    slider_vmult.setToolTip("Vertical speed multiplier (0.10..1.00). 1.0 means vertical speed can match horizontal.")
    speed_layout.addWidget(lbl_vmult)
    speed_layout.addWidget(slider_vmult)

    # ACO tuning (simulation behavior)
    aco_box = _QtCollapsibleSection("ACO (exploration tuning)", "qt_section_aco_collapsed", default_expanded=True)
    aco_layout = QtWidgets.QVBoxLayout(aco_box.body)
    lbl_aco = QtWidgets.QLabel(
        "These sliders control how exploration decisions are made (no physics / no A*).\n"
        "- If drones loiter near base: increase Min explore radius or Recent-cell penalty.\n"
        "- If they feel too forced outward: reduce Min-radius strength."
    )
    lbl_aco.setStyleSheet("color: #666;")
    aco_layout.addWidget(lbl_aco)

    def _set_fast_sim_param_double(name: str, value: float):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.DOUBLE, float(value))])
        except Exception:
            pass

    def _set_fast_sim_param_int(name: str, value: int):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.INTEGER, int(value))])
        except Exception:
            pass

    def _set_fast_sim_param_str(name: str, value: str):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.STRING, str(value))])
        except Exception:
            pass

    def _set_fast_sim_param_bool(name: str, value: bool):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.BOOL, bool(value))])
        except Exception:
            pass

    def _set_fast_sim_param_int(name: str, value: int):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.INTEGER, int(value))])
        except Exception:
            pass

    def _set_fast_sim_param_str(name: str, value: str):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.STRING, str(value))])
        except Exception:
            pass

    def _set_fast_sim_param_int(name: str, value: int):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.INTEGER, int(value))])
        except Exception:
            pass

    def _set_fast_sim_param_str(name: str, value: str):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.STRING, str(value))])
        except Exception:
            pass

    # aco_temperature
    lbl_temp = QtWidgets.QLabel("ACO temperature: 0.70")
    help_temp = QtWidgets.QLabel(
        "Randomness of direction choice. Lower = more greedy (repeatable), higher = more exploratory.\n"
        "Range: 0.01 .. 3.00"
    )
    help_temp.setStyleSheet("color: #666;")
    slider_temp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_temp.setMinimum(1)   # 0.01
    slider_temp.setMaximum(300) # 3.00
    slider_temp.setValue(70)    # 0.70
    slider_temp.setToolTip("ACO temperature (0.01..3.00). Lower=greedy, higher=more random exploration.")
    aco_layout.addWidget(lbl_temp)
    aco_layout.addWidget(help_temp)
    aco_layout.addWidget(slider_temp)

    # explore_min_radius_m
    lbl_min_r = QtWidgets.QLabel("Min explore radius: 200 m")
    help_min_r = QtWidgets.QLabel(
        "Until drones reach this distance from the base, they are rewarded for moves that increase distance.\n"
        "Use it to break the 'orbit around base' behavior. Range: 0 .. 5000 m"
    )
    help_min_r.setStyleSheet("color: #666;")
    slider_min_r = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_min_r.setMinimum(0)
    slider_min_r.setMaximum(5000)
    slider_min_r.setValue(200)
    slider_min_r.setToolTip("Min explore radius in meters (0..5000). Pushes drones outward until they reach it.")
    aco_layout.addWidget(lbl_min_r)
    aco_layout.addWidget(help_min_r)
    aco_layout.addWidget(slider_min_r)

    # explore_min_radius_strength
    lbl_min_s = QtWidgets.QLabel("Min-radius strength: 10.0")
    help_min_s = QtWidgets.QLabel(
        "How strongly the sim rewards increasing distance while inside Min explore radius.\n"
        "Higher = faster outward expansion but less 'natural' wandering. Range: 0.0 .. 30.0"
    )
    help_min_s.setStyleSheet("color: #666;")
    slider_min_s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_min_s.setMinimum(0)    # 0.0
    slider_min_s.setMaximum(300)  # 30.0
    slider_min_s.setValue(100)    # 10.0
    slider_min_s.setToolTip("Min-radius strength (0..30). Higher pushes outward more aggressively.")
    aco_layout.addWidget(lbl_min_s)
    aco_layout.addWidget(help_min_s)
    aco_layout.addWidget(slider_min_s)

    # recent_cell_penalty
    lbl_recent = QtWidgets.QLabel("Recent-cell penalty: 2.0")
    help_recent = QtWidgets.QLabel(
        "Penalty for stepping into a cell visited very recently (per drone). Prevents small loops.\n"
        "Range: 0.0 .. 10.0"
    )
    help_recent.setStyleSheet("color: #666;")
    slider_recent = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_recent.setMinimum(0)    # 0.0
    slider_recent.setMaximum(100)  # 10.0
    slider_recent.setValue(20)     # 2.0
    slider_recent.setToolTip("Recent-cell penalty (0..10). Higher breaks loops; too high can make motion jittery.")
    aco_layout.addWidget(lbl_recent)
    aco_layout.addWidget(help_recent)
    aco_layout.addWidget(slider_recent)

    # base push (related to exploration near base)
    row_basepush = QtWidgets.QHBoxLayout()
    row_basepush.addWidget(QtWidgets.QLabel("Base push radius (m):"))
    spin_basepush_r = QtWidgets.QDoubleSpinBox()
    spin_basepush_r.setMinimum(0.0)
    spin_basepush_r.setMaximum(2000.0)
    spin_basepush_r.setSingleStep(10.0)
    spin_basepush_r.setValue(60.0)
    spin_basepush_r.setToolTip("Within this radius, EXPLORE slightly prefers moves that increase distance from base.")
    row_basepush.addWidget(spin_basepush_r)
    row_basepush.addWidget(QtWidgets.QLabel("Base push strength:"))
    spin_basepush_s = QtWidgets.QDoubleSpinBox()
    spin_basepush_s.setMinimum(0.0)
    spin_basepush_s.setMaximum(200.0)
    spin_basepush_s.setSingleStep(0.5)
    spin_basepush_s.setValue(4.0)
    spin_basepush_s.setToolTip("Strength of the base push. Higher encourages leaving base area sooner.")
    row_basepush.addWidget(spin_basepush_s)
    aco_layout.addLayout(row_basepush)

    # Exploration area radius + exploration coordination (anti-crowding)
    row_explore_area = QtWidgets.QHBoxLayout()
    row_explore_area.addWidget(QtWidgets.QLabel("Exploration area radius (m):"))
    spin_explore_area_r = QtWidgets.QDoubleSpinBox()
    spin_explore_area_r.setMinimum(0.0)
    spin_explore_area_r.setMaximum(500000.0)
    spin_explore_area_r.setSingleStep(50.0)
    spin_explore_area_r.setValue(0.0)
    spin_explore_area_r.setToolTip("0 disables. Targets are visible/counted only inside this radius. Drones avoid moving further outward beyond it.")
    row_explore_area.addWidget(spin_explore_area_r)
    row_explore_area.addWidget(QtWidgets.QLabel("Edge margin (m):"))
    spin_explore_area_margin = QtWidgets.QDoubleSpinBox()
    spin_explore_area_margin.setMinimum(0.0)
    spin_explore_area_margin.setMaximum(100000.0)
    spin_explore_area_margin.setSingleStep(5.0)
    spin_explore_area_margin.setValue(30.0)
    spin_explore_area_margin.setToolTip("Within this distance from the boundary, outward-push heuristics are disabled.")
    row_explore_area.addWidget(spin_explore_area_margin)
    aco_layout.addLayout(row_explore_area)

    row_explore_coord = QtWidgets.QHBoxLayout()
    row_explore_coord.addWidget(QtWidgets.QLabel("Low-nav reward:"))
    spin_low_nav = QtWidgets.QDoubleSpinBox()
    spin_low_nav.setMinimum(0.0)
    spin_low_nav.setMaximum(50.0)
    spin_low_nav.setSingleStep(0.2)
    spin_low_nav.setValue(0.0)
    spin_low_nav.setToolTip("Rewards choosing directions with low navigation pheromone (less visited). 0 disables.")
    row_explore_coord.addWidget(spin_low_nav)
    row_explore_coord.addWidget(QtWidgets.QLabel("Avoid peer-vector weight:"))
    spin_vec_avoid = QtWidgets.QDoubleSpinBox()
    spin_vec_avoid.setMinimum(0.0)
    spin_vec_avoid.setMaximum(50.0)
    spin_vec_avoid.setSingleStep(0.2)
    spin_vec_avoid.setValue(0.0)
    spin_vec_avoid.setToolTip("Penalizes choosing a heading aligned with vectors shared by other drones. 0 disables.")
    row_explore_coord.addWidget(spin_vec_avoid)
    row_explore_coord.addWidget(QtWidgets.QLabel("Share every N cells:"))
    spin_vec_share_cells = QtWidgets.QSpinBox()
    spin_vec_share_cells.setMinimum(1)
    spin_vec_share_cells.setMaximum(100)
    spin_vec_share_cells.setValue(3)
    spin_vec_share_cells.setToolTip("How often each drone refreshes its exploration vector (in moved cells) for comm sharing.")
    row_explore_coord.addWidget(spin_vec_share_cells)
    aco_layout.addLayout(row_explore_coord)

    # Exploration reward (age + distance aware explored evidence)
    explr_box = _QtCollapsibleSection(
        "Exploration reward (age + distance)",
        "qt_section_exploration_reward_collapsed",
        default_expanded=False,
    )
    explr_box.setToolTip(
        "Rewards moving toward cells that are 'less explored' when taking into account:\n"
        "- freshness: older explored evidence becomes less trusted\n"
        "- quality: far observations count as weaker exploration evidence\n"
        "This helps when the map looks uniformly explored, but some areas are older / only observed from far away."
    )
    explr_layout = QtWidgets.QHBoxLayout(explr_box.body)
    explr_layout.addWidget(QtWidgets.QLabel("Reward weight:"))
    spin_explr_w = QtWidgets.QDoubleSpinBox()
    spin_explr_w.setDecimals(2)
    spin_explr_w.setMinimum(0.0)
    spin_explr_w.setMaximum(200.0)
    spin_explr_w.setSingleStep(0.5)
    spin_explr_w.setValue(0.0)
    spin_explr_w.setToolTip("Overall strength for preferring less-explored cells (age+distance aware). 0 disables.")
    explr_layout.addWidget(spin_explr_w)
    explr_layout.addWidget(QtWidgets.QLabel("Age weight:"))
    spin_explr_age = QtWidgets.QDoubleSpinBox()
    spin_explr_age.setDecimals(2)
    spin_explr_age.setMinimum(0.0)
    spin_explr_age.setMaximum(50.0)
    spin_explr_age.setSingleStep(0.25)
    spin_explr_age.setValue(0.0)
    spin_explr_age.setToolTip("Higher => explored evidence 'expires' faster (older areas become attractive again).")
    explr_layout.addWidget(spin_explr_age)
    explr_layout.addWidget(QtWidgets.QLabel("Distance weight:"))
    spin_explr_dist = QtWidgets.QDoubleSpinBox()
    spin_explr_dist.setDecimals(2)
    spin_explr_dist.setMinimum(0.0)
    spin_explr_dist.setMaximum(50.0)
    spin_explr_dist.setSingleStep(0.25)
    spin_explr_dist.setValue(0.0)
    spin_explr_dist.setToolTip("Higher => far observations count as weaker explored evidence (encourages close re-checks).")
    explr_layout.addWidget(spin_explr_dist)
    aco_layout.addWidget(explr_box)

    # Danger inspection curiosity (EXPLORE only)
    inspect_box = _QtCollapsibleSection(
        "Danger inspection curiosity (EXPLORE)",
        "qt_section_danger_inspection_collapsed",
        default_expanded=False,
    )
    inspect_box.setToolTip(
        "Rewards moves that would reveal unexplored cells adjacent to already-known danger (boundary tracing).\n"
        "This naturally turns off once the boundary band becomes explored."
    )
    inspect_layout = QtWidgets.QHBoxLayout(inspect_box.body)
    inspect_layout.addWidget(QtWidgets.QLabel("Weight:"))
    spin_ins_w = QtWidgets.QDoubleSpinBox()
    spin_ins_w.setMinimum(0.0)
    spin_ins_w.setMaximum(500.0)
    spin_ins_w.setSingleStep(0.1)
    spin_ins_w.setValue(0.0)
    spin_ins_w.setToolTip("0 disables. Higher = stronger pull to inspect danger boundary (still penalizes danger).")
    inspect_layout.addWidget(spin_ins_w)
    inspect_layout.addWidget(QtWidgets.QLabel("Kernel (cells):"))
    spin_ins_k = QtWidgets.QSpinBox()
    spin_ins_k.setMinimum(0)
    spin_ins_k.setMaximum(50)
    spin_ins_k.setValue(3)
    spin_ins_k.setToolTip("Neighborhood size around candidate cell for counting unexplored frontier cells.")
    inspect_layout.addWidget(spin_ins_k)
    inspect_layout.addWidget(QtWidgets.QLabel("Danger thr:"))
    spin_ins_thr = QtWidgets.QDoubleSpinBox()
    spin_ins_thr.setMinimum(0.0)
    spin_ins_thr.setMaximum(50.0)
    spin_ins_thr.setSingleStep(0.05)
    spin_ins_thr.setValue(0.35)
    spin_ins_thr.setToolTip("Only consider adjacent cells as 'danger' if danger pheromone > this threshold.")
    inspect_layout.addWidget(spin_ins_thr)
    inspect_layout.addWidget(QtWidgets.QLabel("Max cell danger:"))
    spin_ins_max = QtWidgets.QDoubleSpinBox()
    spin_ins_max.setMinimum(0.0)
    spin_ins_max.setMaximum(50.0)
    spin_ins_max.setSingleStep(0.05)
    spin_ins_max.setValue(0.6)
    spin_ins_max.setToolTip("Do not apply curiosity bonus when the candidate cell's danger pheromone exceeds this.")
    inspect_layout.addWidget(spin_ins_max)
    aco_layout.addWidget(inspect_box)

    # Far-ring low explored density shaping (optional)
    far_box = _QtCollapsibleSection(
        "Far-density kernel (ring probes)",
        "qt_section_far_density_collapsed",
        default_expanded=False,
    )
    far_box.setToolTip("EXPLORE: probe explored-density on a ring around the drone; bias headings toward the lowest density.")
    far_layout = QtWidgets.QHBoxLayout(far_box.body)
    far_layout.addWidget(QtWidgets.QLabel("Weight:"))
    spin_far_w = QtWidgets.QDoubleSpinBox()
    spin_far_w.setMinimum(0.0)
    spin_far_w.setMaximum(200.0)
    spin_far_w.setSingleStep(0.5)
    spin_far_w.setValue(0.0)
    spin_far_w.setToolTip("0 disables. Higher = stronger pull toward least-explored direction on the ring.")
    far_layout.addWidget(spin_far_w)
    far_layout.addWidget(QtWidgets.QLabel("Ring X (cells):"))
    spin_far_ring = QtWidgets.QSpinBox()
    spin_far_ring.setMinimum(0)
    spin_far_ring.setMaximum(200000)
    spin_far_ring.setValue(0)
    spin_far_ring.setToolTip("Probe distance around the drone (cells). 0 disables.")
    far_layout.addWidget(spin_far_ring)
    far_layout.addWidget(QtWidgets.QLabel("Kernel Y (cells):"))
    spin_far_kern = QtWidgets.QSpinBox()
    spin_far_kern.setMinimum(0)
    spin_far_kern.setMaximum(200000)
    spin_far_kern.setValue(3)
    spin_far_kern.setToolTip("Neighborhood radius around each probe cell. Square size = (2Y+1)^2.")
    far_layout.addWidget(spin_far_kern)
    far_layout.addWidget(QtWidgets.QLabel("Step (deg):"))
    spin_far_step = QtWidgets.QDoubleSpinBox()
    spin_far_step.setMinimum(1.0)
    spin_far_step.setMaximum(90.0)
    spin_far_step.setSingleStep(5.0)
    spin_far_step.setValue(30.0)
    spin_far_step.setToolTip("Probe spacing on the ring. 30° = 12 probes.")
    far_layout.addWidget(spin_far_step)
    chk_far_excl = QtWidgets.QCheckBox("Exclude inside ring")
    chk_far_excl.setChecked(True)
    chk_far_excl.setToolTip("Ignore kernel cells closer than ring X to the drone (frontier-only).")
    far_layout.addWidget(chk_far_excl)
    aco_layout.addWidget(far_box)

    # evaporation (pheromone behavior)
    row_evap = QtWidgets.QHBoxLayout()
    row_evap.addWidget(QtWidgets.QLabel("Evap nav:"))
    spin_evap_nav = QtWidgets.QDoubleSpinBox()
    spin_evap_nav.setDecimals(4)
    spin_evap_nav.setMinimum(0.0)
    spin_evap_nav.setMaximum(1.0)
    spin_evap_nav.setSingleStep(0.0005)
    spin_evap_nav.setValue(0.0020)
    spin_evap_nav.setToolTip("Navigation pheromone evaporation rate per second. Higher = pheromones fade faster.")
    row_evap.addWidget(spin_evap_nav)
    row_evap.addWidget(QtWidgets.QLabel("Evap danger:"))
    spin_evap_danger = QtWidgets.QDoubleSpinBox()
    spin_evap_danger.setDecimals(4)
    spin_evap_danger.setMinimum(0.0)
    spin_evap_danger.setMaximum(1.0)
    spin_evap_danger.setSingleStep(0.0005)
    spin_evap_danger.setValue(0.0010)
    spin_evap_danger.setToolTip("Danger pheromone evaporation rate per second. Higher = danger fades faster.")
    row_evap.addWidget(spin_evap_danger)
    aco_layout.addLayout(row_evap)

    # evaporation kind multipliers (per-kind scaling on top of evap_danger_rate)
    row_evap2 = QtWidgets.QHBoxLayout()
    row_evap2.addWidget(QtWidgets.QLabel("Danger mult static:"))
    spin_danger_mult_static = QtWidgets.QDoubleSpinBox()
    spin_danger_mult_static.setDecimals(3)
    spin_danger_mult_static.setMinimum(0.0)
    spin_danger_mult_static.setMaximum(10.0)
    spin_danger_mult_static.setSingleStep(0.05)
    spin_danger_mult_static.setValue(1.0)
    spin_danger_mult_static.setToolTip("Multiplier for static danger evaporation (1.0 = baseline).")
    row_evap2.addWidget(spin_danger_mult_static)
    row_evap2.addWidget(QtWidgets.QLabel("Danger mult dynamic:"))
    spin_danger_mult_dynamic = QtWidgets.QDoubleSpinBox()
    spin_danger_mult_dynamic.setDecimals(3)
    spin_danger_mult_dynamic.setMinimum(0.0)
    spin_danger_mult_dynamic.setMaximum(10.0)
    spin_danger_mult_dynamic.setSingleStep(0.05)
    spin_danger_mult_dynamic.setValue(1.25)
    spin_danger_mult_dynamic.setToolTip("Multiplier for dynamic danger evaporation (kernel+damage). >1 fades faster than static.")
    row_evap2.addWidget(spin_danger_mult_dynamic)
    row_evap2.addWidget(QtWidgets.QLabel("Wall mult:"))
    spin_danger_mult_wall = QtWidgets.QDoubleSpinBox()
    spin_danger_mult_wall.setDecimals(4)
    spin_danger_mult_wall.setMinimum(0.0)
    spin_danger_mult_wall.setMaximum(1.0)
    spin_danger_mult_wall.setSingleStep(0.005)
    spin_danger_mult_wall.setValue(0.02)
    spin_danger_mult_wall.setToolTip("Multiplier for nav_danger (walls). Smaller = longer-lived wall memory.")
    row_evap2.addWidget(spin_danger_mult_wall)
    aco_layout.addLayout(row_evap2)

    layout.addWidget(aco_box)

    # Drones (visualization)
    drones_box = _QtCollapsibleSection("Drones (visualization)", "qt_section_drones_viz_collapsed", default_expanded=True)
    drones_layout = QtWidgets.QVBoxLayout(drones_box.body)

    # Drones in mission (dynamic)
    row_n = QtWidgets.QHBoxLayout()
    row_n.addWidget(QtWidgets.QLabel("Python drones in mission:"))
    spin_n = QtWidgets.QSpinBox()
    spin_n.setMinimum(1)
    spin_n.setMaximum(200)
    spin_n.setValue(15)
    spin_n.setToolTip("Number of Python drones to simulate (can be changed live).")
    row_n.addWidget(spin_n)
    drones_layout.addLayout(row_n)
    lbl_drone_scale = QtWidgets.QLabel("Drone size: 1.00×")
    drones_layout.addWidget(lbl_drone_scale)
    slider_drone_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_drone_scale.setMinimum(10)   # 0.10x
    slider_drone_scale.setMaximum(300)  # 3.00x
    slider_drone_scale.setValue(100)
    drones_layout.addWidget(slider_drone_scale)
    # altitude (affects which buildings are obstacles)
    lbl_drone_alt = QtWidgets.QLabel("Altitude: 8.0 m")
    drones_layout.addWidget(lbl_drone_alt)
    slider_drone_alt = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_drone_alt.setMinimum(-500)   # -50.0m
    slider_drone_alt.setMaximum(5000)   # 500.0m
    slider_drone_alt.setValue(80)       # 8.0m
    drones_layout.addWidget(slider_drone_alt)

    # altitude safety (sim)
    row_alt2 = QtWidgets.QHBoxLayout()
    row_alt2.addWidget(QtWidgets.QLabel("Min altitude (m):"))
    spin_min_alt = QtWidgets.QDoubleSpinBox()
    spin_min_alt.setMinimum(0.0)
    spin_min_alt.setMaximum(200.0)
    spin_min_alt.setSingleStep(0.5)
    spin_min_alt.setValue(5.0)
    row_alt2.addWidget(spin_min_alt)
    row_alt2.addWidget(QtWidgets.QLabel("Roof margin (m):"))
    spin_roof_margin = QtWidgets.QDoubleSpinBox()
    spin_roof_margin.setMinimum(0.0)
    spin_roof_margin.setMaximum(50.0)
    spin_roof_margin.setSingleStep(0.5)
    spin_roof_margin.setValue(3.0)
    row_alt2.addWidget(spin_roof_margin)
    drones_layout.addLayout(row_alt2)

    # Overfly / vertical cost tuning (sim)
    row_overfly = QtWidgets.QHBoxLayout()
    row_overfly.addWidget(QtWidgets.QLabel("Vertical cost ×:"))
    spin_overfly_cost = QtWidgets.QDoubleSpinBox()
    spin_overfly_cost.setMinimum(0.1)
    spin_overfly_cost.setMaximum(50.0)
    spin_overfly_cost.setSingleStep(0.1)
    spin_overfly_cost.setValue(3.0)
    spin_overfly_cost.setToolTip("How expensive climbing/descending is vs horizontal path cost (A* overfly-vs-around).")
    row_overfly.addWidget(spin_overfly_cost)
    row_overfly.addWidget(QtWidgets.QLabel("Vertical battery ×:"))
    spin_vert_energy = QtWidgets.QDoubleSpinBox()
    spin_vert_energy.setMinimum(0.0)
    spin_vert_energy.setMaximum(50.0)
    spin_vert_energy.setSingleStep(0.1)
    spin_vert_energy.setValue(3.0)
    spin_vert_energy.setToolTip("Extra battery drain per vertical meter (multiplier vs horizontal energy per meter).")
    row_overfly.addWidget(spin_vert_energy)
    drones_layout.addLayout(row_overfly)

    # Static danger altitude violation penalty (sim)
    row_static_alt = QtWidgets.QHBoxLayout()
    row_static_alt.addWidget(QtWidgets.QLabel("Static danger alt penalty:"))
    spin_static_alt_pen = QtWidgets.QDoubleSpinBox()
    spin_static_alt_pen.setMinimum(0.0)
    spin_static_alt_pen.setMaximum(500.0)
    spin_static_alt_pen.setSingleStep(0.5)
    spin_static_alt_pen.setValue(6.0)
    spin_static_alt_pen.setToolTip(
        "Penalty weight applied when a drone is in a static-danger region below its required altitude.\n"
        "0 disables. Higher -> stronger push to climb/avoid when below the danger altitude."
    )
    row_static_alt.addWidget(spin_static_alt_pen)
    row_static_alt.addStretch(1)
    drones_layout.addLayout(row_static_alt)
    # lidar sense radius (affects mock lidar + local planning)
    lbl_sense = QtWidgets.QLabel("LiDAR sense radius: 75 m")
    drones_layout.addWidget(lbl_sense)
    slider_sense = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_sense.setMinimum(5)
    slider_sense.setMaximum(200)
    slider_sense.setValue(35)
    drones_layout.addWidget(slider_sense)
    # pointers
    row_ptr = QtWidgets.QHBoxLayout()
    chk_ptr = QtWidgets.QCheckBox("Show pointers")
    chk_ptr.setChecked(True)
    row_ptr.addWidget(chk_ptr)
    row_ptr.addWidget(QtWidgets.QLabel("Z:"))
    spin_ptr_z = QtWidgets.QDoubleSpinBox()
    spin_ptr_z.setMinimum(-50.0)
    spin_ptr_z.setMaximum(200.0)
    spin_ptr_z.setSingleStep(1.0)
    spin_ptr_z.setValue(8.0)
    row_ptr.addWidget(spin_ptr_z)
    row_ptr.addWidget(QtWidgets.QLabel("Size:"))
    spin_ptr_scale = QtWidgets.QDoubleSpinBox()
    spin_ptr_scale.setMinimum(0.1)
    spin_ptr_scale.setMaximum(10.0)
    spin_ptr_scale.setSingleStep(0.1)
    spin_ptr_scale.setValue(1.0)
    row_ptr.addWidget(spin_ptr_scale)
    row_ptr.addWidget(QtWidgets.QLabel("Alpha:"))
    spin_ptr_alpha = QtWidgets.QDoubleSpinBox()
    spin_ptr_alpha.setMinimum(0.0)
    spin_ptr_alpha.setMaximum(1.0)
    spin_ptr_alpha.setSingleStep(0.05)
    spin_ptr_alpha.setValue(0.9)
    row_ptr.addWidget(spin_ptr_alpha)
    drones_layout.addLayout(row_ptr)
    layout.addWidget(drones_box)
    layout.addSpacing(10)

    # Click mode (exclusive owner of RViz /clicked_point)
    click_box = _QtCollapsibleSection(
        "Click mode (uses RViz /clicked_point) — exclusive",
        "qt_section_click_mode_collapsed",
        default_expanded=True,
    )
    click_layout = QtWidgets.QVBoxLayout(click_box.body)

    row_click = QtWidgets.QHBoxLayout()
    row_click.addWidget(QtWidgets.QLabel("Mode:"))
    cmb_click = QtWidgets.QComboBox()
    cmb_click.addItems(
        [
            "None",
            "Targets",
            "Danger: static",
            "Danger: dynamic",
        ]
    )
    row_click.addWidget(cmb_click)
    click_layout.addLayout(row_click)

    # Danger radius (cells)
    row_dr = QtWidgets.QHBoxLayout()
    row_dr.addWidget(QtWidgets.QLabel("Danger radius (cells):"))
    spin_danger_r = QtWidgets.QSpinBox()
    spin_danger_r.setMinimum(0)
    spin_danger_r.setMaximum(50)
    spin_danger_r.setSingleStep(1)
    spin_danger_r.setValue(2)
    spin_danger_r.setToolTip(
        "Stored in danger_map.json.\n"
        "- Static threats: this is the *base* radius (cells). Drone discovers ONLY the center cell via lidar, then paints 3 rings:\n"
        "  r: altitude=H, 2r: altitude=H/2, 3r: altitude=H/4.\n"
        "- Dynamic threats: radius applies, but altitude does NOT fall (treated like other drones)."
    )
    row_dr.addWidget(spin_danger_r)
    click_layout.addLayout(row_dr)

    # Threat height (meters)
    row_th = QtWidgets.QHBoxLayout()
    row_th.addWidget(QtWidgets.QLabel("Threat height (m):"))
    spin_threat_h = QtWidgets.QDoubleSpinBox()
    spin_threat_h.setMinimum(0.0)
    spin_threat_h.setMaximum(200.0)
    spin_threat_h.setSingleStep(1.0)
    spin_threat_h.setValue(50.0)
    spin_threat_h.setToolTip("Height above ground for static threats (and default for dynamic).")
    row_th.addWidget(spin_threat_h)
    click_layout.addLayout(row_th)

    # dynamic controls
    row_dyn = QtWidgets.QHBoxLayout()
    btn_dyn_start = QtWidgets.QPushButton("Start recording")
    btn_dyn_stop = QtWidgets.QPushButton("Stop + create dynamic danger")
    row_dyn.addWidget(btn_dyn_start)
    row_dyn.addWidget(btn_dyn_stop)
    click_layout.addLayout(row_dyn)

    row_dyn2 = QtWidgets.QHBoxLayout()
    row_dyn2.addWidget(QtWidgets.QLabel("Dynamic speed (sec/cell):"))
    spin_dyn_speed = QtWidgets.QDoubleSpinBox()
    spin_dyn_speed.setMinimum(0.1)
    spin_dyn_speed.setMaximum(60.0)
    spin_dyn_speed.setSingleStep(0.5)
    spin_dyn_speed.setValue(4.0)
    row_dyn2.addWidget(spin_dyn_speed)
    click_layout.addLayout(row_dyn2)

    layout.addWidget(click_box)
    layout.addSpacing(10)

    # Targets section (visual-only)
    tgt_box = _QtCollapsibleSection("Targets (visualization)", "qt_section_targets_viz_collapsed", default_expanded=False)
    tgt_layout = QtWidgets.QVBoxLayout(tgt_box.body)
    lbl_tgt = QtWidgets.QLabel("Diameter: 10.0m, alpha=0.30")
    tgt_layout.addWidget(lbl_tgt)
    slider_tgt_d = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_tgt_d.setMinimum(1)      # 1m
    slider_tgt_d.setMaximum(200)    # 200m
    slider_tgt_d.setValue(10)
    tgt_layout.addWidget(QtWidgets.QLabel("Diameter (m)"))
    tgt_layout.addWidget(slider_tgt_d)
    slider_tgt_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_tgt_a.setMinimum(0)
    slider_tgt_a.setMaximum(100)
    slider_tgt_a.setValue(30)
    tgt_layout.addWidget(QtWidgets.QLabel("Alpha (0..1)"))
    tgt_layout.addWidget(slider_tgt_a)
    layout.addWidget(tgt_box)
    layout.addSpacing(10)

    # Exploit section (goal-seeking + comparison mode)
    exploit_box = _QtCollapsibleSection("Exploit mission", "qt_section_exploit_collapsed", default_expanded=False)
    exploit_layout = QtWidgets.QVBoxLayout(exploit_box.body)
    exploit_layout.addWidget(
        QtWidgets.QLabel(
            "Use a saved (or in-memory) pheromone map to run an exploit trial to a selected target.\n"
            "Other targets can be hidden automatically when a goal is selected."
        )
    )

    row_ex_tgt = QtWidgets.QHBoxLayout()
    row_ex_tgt.addWidget(QtWidgets.QLabel("Target goal:"))
    cmb_exploit_target = QtWidgets.QComboBox()
    cmb_exploit_target.addItems(["—"])
    row_ex_tgt.addWidget(cmb_exploit_target, 1)
    exploit_layout.addLayout(row_ex_tgt)

    row_ex_n = QtWidgets.QHBoxLayout()
    row_ex_n.addWidget(QtWidgets.QLabel("Exploit drones:"))
    spin_exploit_n = QtWidgets.QSpinBox()
    spin_exploit_n.setMinimum(1)
    spin_exploit_n.setMaximum(200)
    spin_exploit_n.setValue(3)
    row_ex_n.addWidget(spin_exploit_n)
    row_ex_n.addStretch(1)
    exploit_layout.addLayout(row_ex_n)

    row_ex_mode = QtWidgets.QHBoxLayout()
    row_ex_mode.addWidget(QtWidgets.QLabel("Dynamic danger mode:"))
    cmb_exploit_dyn = QtWidgets.QComboBox()
    cmb_exploit_dyn.addItems(["Handled dynamic danger", "Dynamic danger treated as static"])
    row_ex_mode.addWidget(cmb_exploit_dyn, 1)
    exploit_layout.addLayout(row_ex_mode)

    # Exploit: spacing + path adherence
    exploit_layout.addSpacing(6)
    exploit_layout.addWidget(QtWidgets.QLabel("Exploit spacing / path adherence"))

    row_ex_avoid = QtWidgets.QHBoxLayout()
    row_ex_avoid.addWidget(QtWidgets.QLabel("Avoid radius (m):"))
    spin_ex_avoid_r = QtWidgets.QDoubleSpinBox()
    spin_ex_avoid_r.setMinimum(0.0)
    spin_ex_avoid_r.setMaximum(500.0)
    spin_ex_avoid_r.setSingleStep(5.0)
    spin_ex_avoid_r.setValue(40.0)
    row_ex_avoid.addWidget(spin_ex_avoid_r)
    row_ex_avoid.addWidget(QtWidgets.QLabel("Avoid weight:"))
    spin_ex_avoid_w = QtWidgets.QDoubleSpinBox()
    spin_ex_avoid_w.setMinimum(0.0)
    spin_ex_avoid_w.setMaximum(20.0)
    spin_ex_avoid_w.setSingleStep(0.2)
    spin_ex_avoid_w.setValue(2.0)
    row_ex_avoid.addWidget(spin_ex_avoid_w)
    row_ex_avoid.addStretch(1)
    exploit_layout.addLayout(row_ex_avoid)

    row_ex_fade = QtWidgets.QHBoxLayout()
    row_ex_fade.addWidget(QtWidgets.QLabel("Fade start (m):"))
    spin_ex_fade0 = QtWidgets.QDoubleSpinBox()
    spin_ex_fade0.setMinimum(0.0)
    spin_ex_fade0.setMaximum(300.0)
    spin_ex_fade0.setSingleStep(2.0)
    spin_ex_fade0.setValue(25.0)
    row_ex_fade.addWidget(spin_ex_fade0)
    row_ex_fade.addWidget(QtWidgets.QLabel("Fade range (m):"))
    spin_ex_fade_rng = QtWidgets.QDoubleSpinBox()
    spin_ex_fade_rng.setMinimum(0.0)
    spin_ex_fade_rng.setMaximum(500.0)
    spin_ex_fade_rng.setSingleStep(5.0)
    spin_ex_fade_rng.setValue(50.0)
    row_ex_fade.addWidget(spin_ex_fade_rng)
    row_ex_fade.addStretch(1)
    exploit_layout.addLayout(row_ex_fade)

    row_ex_follow = QtWidgets.QHBoxLayout()
    row_ex_follow.addWidget(QtWidgets.QLabel("A* follow weight:"))
    spin_ex_follow_w = QtWidgets.QDoubleSpinBox()
    spin_ex_follow_w.setMinimum(0.05)
    spin_ex_follow_w.setMaximum(10.0)
    spin_ex_follow_w.setSingleStep(0.1)
    spin_ex_follow_w.setValue(1.0)
    spin_ex_follow_w.setToolTip("Higher -> stay closer to best A* direction (less deviation from peers avoidance).")
    row_ex_follow.addWidget(spin_ex_follow_w)
    row_ex_follow.addStretch(1)
    exploit_layout.addLayout(row_ex_follow)

    # Advanced: dynamic-trail overlay (comparison mode)
    exploit_overlay_box = _QtCollapsibleSection(
        "Advanced: dynamic trail overlay (treat-as-static)",
        "qt_section_exploit_dyn_overlay_collapsed",
        default_expanded=False,
    )
    exo_layout = QtWidgets.QVBoxLayout(exploit_overlay_box.body)
    exo_layout.addWidget(
        QtWidgets.QLabel(
            "Applies only when 'Dynamic danger treated as static' is selected.\n"
            "Higher strength/gamma makes A* avoid red (high) dynamic trail pheromone more aggressively."
        )
    )
    row_ex_overlay = QtWidgets.QHBoxLayout()
    row_ex_overlay.addWidget(QtWidgets.QLabel("Overlay strength:"))
    spin_ex_dyn_overlay_strength = QtWidgets.QDoubleSpinBox()
    spin_ex_dyn_overlay_strength.setMinimum(0.0)
    spin_ex_dyn_overlay_strength.setMaximum(50.0)
    spin_ex_dyn_overlay_strength.setSingleStep(0.5)
    spin_ex_dyn_overlay_strength.setValue(3.0)
    row_ex_overlay.addWidget(spin_ex_dyn_overlay_strength)
    row_ex_overlay.addWidget(QtWidgets.QLabel("Gamma:"))
    spin_ex_dyn_overlay_gamma = QtWidgets.QDoubleSpinBox()
    spin_ex_dyn_overlay_gamma.setMinimum(0.5)
    spin_ex_dyn_overlay_gamma.setMaximum(6.0)
    spin_ex_dyn_overlay_gamma.setSingleStep(0.1)
    spin_ex_dyn_overlay_gamma.setValue(1.8)
    row_ex_overlay.addWidget(spin_ex_dyn_overlay_gamma)
    row_ex_overlay.addStretch(1)
    exo_layout.addLayout(row_ex_overlay)
    exploit_layout.addWidget(exploit_overlay_box)

    btn_exploit_start = QtWidgets.QPushButton("Start exploit run")
    btn_exploit_start.setEnabled(False)  # enabled only when a real target is selected
    exploit_layout.addWidget(btn_exploit_start)

    exploit_layout.addSpacing(6)
    exploit_layout.addWidget(QtWidgets.QLabel("Exploit run statistics (per drone)"))
    txt_exploit_stats = QtWidgets.QPlainTextEdit()
    txt_exploit_stats.setReadOnly(True)
    txt_exploit_stats.setPlaceholderText("—")
    txt_exploit_stats.setMinimumHeight(110)
    exploit_layout.addWidget(txt_exploit_stats)

    layout.addWidget(exploit_box)
    layout.addSpacing(10)

    # Pheromone map section
    pher_box = _QtCollapsibleSection("Pheromone map (visualization)", "qt_section_pheromone_viz_collapsed", default_expanded=False)
    pher_layout = QtWidgets.QVBoxLayout(pher_box.body)

    chk_pher = QtWidgets.QCheckBox("Enable pheromone map (/pheromone_heatmap)")
    chk_pher.setChecked(True)  # auto-publish immediately
    pher_layout.addWidget(chk_pher)

    lbl_pher = QtWidgets.QLabel("Display size: 2000m, z=0.10, alpha=0.10")
    pher_layout.addWidget(lbl_pher)

    # display size slider (100..grid max-ish) as 100..11000
    slider_disp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_disp.setMinimum(100)
    slider_disp.setMaximum(11000)
    slider_disp.setValue(2000)
    pher_layout.addWidget(QtWidgets.QLabel("Display size (m)"))
    pher_layout.addWidget(slider_disp)

    slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_z.setMinimum(-50)  # -5.0m
    slider_z.setMaximum(2000)  # 200.0m
    slider_z.setValue(1)  # 0.1m
    pher_layout.addWidget(QtWidgets.QLabel("Z position (m)"))
    pher_layout.addWidget(slider_z)

    slider_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_a.setMinimum(0)
    slider_a.setMaximum(100)
    slider_a.setValue(10)
    pher_layout.addWidget(QtWidgets.QLabel("Alpha (0..1)"))
    pher_layout.addWidget(slider_a)

    # Source selection
    row_src = QtWidgets.QHBoxLayout()
    row_src.addWidget(QtWidgets.QLabel("Source:"))
    cmb_pher_source = QtWidgets.QComboBox()
    cmb_pher_source.addItems(["Base", "Combined", "Drone"])
    row_src.addWidget(cmb_pher_source)
    row_src.addWidget(QtWidgets.QLabel("Drone #:"))
    spin_pher_drone = QtWidgets.QSpinBox()
    spin_pher_drone.setMinimum(1)
    spin_pher_drone.setMaximum(200)
    spin_pher_drone.setValue(1)
    row_src.addWidget(spin_pher_drone)
    row_src.addWidget(QtWidgets.QLabel("Layer:"))
    cmb_pher_layer = QtWidgets.QComboBox()
    cmb_pher_layer.addItems(["Danger", "Nav", "Empty", "Explored"])
    # danger kind filter (what to show on the danger layer)
    row_dk = QtWidgets.QHBoxLayout()
    row_dk.addWidget(QtWidgets.QLabel("Danger kind:"))
    cmb_danger_kind = QtWidgets.QComboBox()
    cmb_danger_kind.addItems(
        [
            "All",
            "NavDanger (walls)",
            "Static danger",
            "Dynamic kernel",
            "Dynamic damage",
            "Dynamic (kernel+damage)",
        ]
    )
    row_dk.addWidget(cmb_danger_kind)
    pher_layout.addLayout(row_dk)
    row_src.addWidget(cmb_pher_layer)
    pher_layout.addLayout(row_src)

    # Snapshot storage (save/load full pheromone base map + timestamp)
    pher_layout.addSpacing(6)
    pher_layout.addWidget(QtWidgets.QLabel("Pheromone snapshots (disk)"))
    row_snap = QtWidgets.QHBoxLayout()
    row_snap.addWidget(QtWidgets.QLabel("Name:"))
    txt_snap_name = QtWidgets.QLineEdit()
    txt_snap_name.setPlaceholderText("e.g. explore_run_01")
    row_snap.addWidget(txt_snap_name, 1)
    btn_snap_save = QtWidgets.QPushButton("Save snapshot")
    btn_snap_load_latest = QtWidgets.QPushButton("Load latest")
    btn_snap_load = QtWidgets.QPushButton("Load snapshot…")
    row_snap.addWidget(btn_snap_save)
    row_snap.addWidget(btn_snap_load_latest)
    row_snap.addWidget(btn_snap_load)
    pher_layout.addLayout(row_snap)

    # LiDAR debug (per-drone)
    lidar_box = _QtCollapsibleSection("LiDAR debug (per drone)", "qt_section_lidar_debug_collapsed", default_expanded=False)
    lidar_layout = QtWidgets.QVBoxLayout(lidar_box.body)
    chk_lidar = QtWidgets.QCheckBox("Show LiDAR walls/corners (/swarm/markers/lidar)")
    chk_lidar.setChecked(False)
    lidar_layout.addWidget(chk_lidar)
    chk_lidar_scan = QtWidgets.QCheckBox("Show LiDAR scan beams (/swarm/markers/lidar_scan)")
    chk_lidar_scan.setChecked(False)
    lidar_layout.addWidget(chk_lidar_scan)
    chk_plan = QtWidgets.QCheckBox("Show planned path (A* local plan) (/swarm/markers/plan)")
    chk_plan.setChecked(False)
    lidar_layout.addWidget(chk_plan)
    chk_plan_all = QtWidgets.QCheckBox("Show planned paths (ALL drones) (/swarm/markers/plan)")
    chk_plan_all.setChecked(False)
    chk_plan_all.setToolTip("When enabled, publishes plan visualization for all drones at once.")
    lidar_layout.addWidget(chk_plan_all)
    chk_aco = QtWidgets.QCheckBox("Show ACO decision (chosen heading) (/swarm/markers/aco)")
    chk_aco.setChecked(False)
    lidar_layout.addWidget(chk_aco)
    chk_aco_all = QtWidgets.QCheckBox("Show ACO decisions (ALL drones) (/swarm/markers/aco)")
    chk_aco_all.setChecked(False)
    chk_aco_all.setToolTip("When enabled, shows ACO decisions for all drones at once.")
    lidar_layout.addWidget(chk_aco_all)
    # ACO arrow tuning
    row_aco_arrow = QtWidgets.QHBoxLayout()
    row_aco_arrow.addWidget(QtWidgets.QLabel("ACO arrow width (m):"))
    spin_aco_w = QtWidgets.QDoubleSpinBox()
    spin_aco_w.setMinimum(0.02)
    spin_aco_w.setMaximum(5.0)
    spin_aco_w.setSingleStep(0.05)
    spin_aco_w.setValue(0.35)
    spin_aco_w.setToolTip("Thickness of the ACO arrow (meters).")
    row_aco_arrow.addWidget(spin_aco_w)
    row_aco_arrow.addWidget(QtWidgets.QLabel("Length ×:"))
    spin_aco_len = QtWidgets.QDoubleSpinBox()
    spin_aco_len.setMinimum(0.3)
    spin_aco_len.setMaximum(10.0)
    spin_aco_len.setSingleStep(0.1)
    spin_aco_len.setValue(1.6)
    spin_aco_len.setToolTip("Multiplier applied to the chosen ACO step vector.")
    row_aco_arrow.addWidget(spin_aco_len)
    row_aco_arrow.addStretch(1)
    lidar_layout.addLayout(row_aco_arrow)

    # ACO commitment (reduce dithering)
    chk_aco_commit = QtWidgets.QCheckBox("ACO: stick to chosen next cell until new obstacle/danger is sensed")
    chk_aco_commit.setChecked(True)
    chk_aco_commit.setToolTip("Reduces indecision by keeping the chosen next cell until perception changes.")
    lidar_layout.addWidget(chk_aco_commit)

    # Corner backoff (shorter steps near corners)
    chk_corner_backoff = QtWidgets.QCheckBox("Corner backoff: allow shorter steps near building corners")
    chk_corner_backoff.setChecked(True)
    chk_corner_backoff.setToolTip(
        "When enabled, drones may use a shorter step if the full step would collide (helps escape tight corners)."
    )
    lidar_layout.addWidget(chk_corner_backoff)

    chk_unstick = QtWidgets.QCheckBox("Unstick escape: allow short 'jump away' move when boxed in")
    chk_unstick.setChecked(True)
    chk_unstick.setToolTip("When disabled, boxed-in drones won't do the escape hop; overfly/hop logic must resolve it.")
    lidar_layout.addWidget(chk_unstick)

    row_commit = QtWidgets.QHBoxLayout()
    row_commit.addWidget(QtWidgets.QLabel("Commit timeout (s):"))
    spin_commit = QtWidgets.QDoubleSpinBox()
    spin_commit.setMinimum(0.2)
    spin_commit.setMaximum(30.0)
    spin_commit.setSingleStep(0.5)
    spin_commit.setValue(5.0)
    row_commit.addWidget(spin_commit)
    row_commit.addStretch(1)
    lidar_layout.addLayout(row_commit)
    row_lidar = QtWidgets.QHBoxLayout()
    row_lidar.addWidget(QtWidgets.QLabel("Drone #:"))
    spin_lidar_drone = QtWidgets.QSpinBox()
    spin_lidar_drone.setMinimum(1)
    spin_lidar_drone.setMaximum(200)
    spin_lidar_drone.setValue(1)
    row_lidar.addWidget(spin_lidar_drone)
    row_lidar.addStretch(1)
    lidar_layout.addLayout(row_lidar)
    help_lidar = QtWidgets.QLabel(
        "Red lines: walls inferred from this drone's mock lidar hits. Red dots: corners/endpoints.\n"
        "Rendered at that drone's altitude."
    )
    help_lidar.setStyleSheet("color: #666;")
    help_lidar.setWordWrap(True)
    lidar_layout.addWidget(help_lidar)
    layout.addWidget(lidar_box)
    layout.addSpacing(10)

    # Buildings section (auto-starts publishing immediately)
    bld_box = _QtCollapsibleSection("Buildings (opacity)", "qt_section_buildings_collapsed", default_expanded=False)
    bld_layout = QtWidgets.QVBoxLayout(bld_box.body)

    lbl_bld = QtWidgets.QLabel("Opacity: 1.00")
    bld_layout.addWidget(lbl_bld)
    slider_bld_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_bld_a.setMinimum(0)
    slider_bld_a.setMaximum(100)
    slider_bld_a.setValue(100)
    bld_layout.addWidget(slider_bld_a)
    layout.addWidget(bld_box)
    layout.addSpacing(10)

    # Log line
    lbl_log = QtWidgets.QLabel("—")
    lbl_log.setStyleSheet("font-size: 12px; color: #333;")
    layout.addWidget(lbl_log)

    # Wiring
    btn_start.clicked.connect(node.call_start)
    btn_stop.clicked.connect(node.call_stop)
    btn_clear_found.clicked.connect(node.call_clear_targets_found)
    btn_clear_unfound.clicked.connect(node.call_clear_targets_unfound)
    btn_set_unfound.clicked.connect(node.call_set_all_targets_unfound)
    btn_clear.clicked.connect(node.call_clear_targets)
    btn_del_nearest.clicked.connect(node.call_delete_nearest_target)

    paused_state = {"v": False}

    def _toggle_pause():
        paused_state["v"] = not paused_state["v"]
        node.set_paused(paused_state["v"])
        btn_pause.setText("Resume" if paused_state["v"] else "Pause")

    btn_pause.clicked.connect(_toggle_pause)

    def _on_slider(v: int):
        speed = max(0.1, min(200.0, v / 10.0))
        lbl_speed.setText(f"Speed: {speed:.1f}×")
        node.set_speed(speed)

    slider.valueChanged.connect(_on_slider)

    def _on_aco_temp(_v: int):
        t = float(slider_temp.value()) / 100.0
        lbl_temp.setText(f"ACO temperature: {t:.2f}")
        _set_fast_sim_param_double("aco_temperature", t)

    slider_temp.valueChanged.connect(_on_aco_temp)

    def _on_min_r(_v: int):
        r = float(slider_min_r.value())
        lbl_min_r.setText(f"Min explore radius: {r:.0f} m")
        _set_fast_sim_param_double("explore_min_radius_m", r)

    slider_min_r.valueChanged.connect(_on_min_r)

    def _on_min_s(_v: int):
        s = float(slider_min_s.value()) / 10.0
        lbl_min_s.setText(f"Min-radius strength: {s:.1f}")
        _set_fast_sim_param_double("explore_min_radius_strength", s)

    slider_min_s.valueChanged.connect(_on_min_s)

    def _on_recent(_v: int):
        p = float(slider_recent.value()) / 10.0
        lbl_recent.setText(f"Recent-cell penalty: {p:.1f}")
        _set_fast_sim_param_double("recent_cell_penalty", p)

    slider_recent.valueChanged.connect(_on_recent)

    def _on_basepush(_v=None):
        _set_fast_sim_param_double("base_push_radius_m", float(spin_basepush_r.value()))
        _set_fast_sim_param_double("base_push_strength", float(spin_basepush_s.value()))

    spin_basepush_r.valueChanged.connect(lambda _v: _on_basepush())
    spin_basepush_s.valueChanged.connect(lambda _v: _on_basepush())

    def _on_explore_area(_v=None):
        _set_fast_sim_param_double("exploration_area_radius_m", float(spin_explore_area_r.value()))
        _set_fast_sim_param_double("exploration_radius_margin_m", float(spin_explore_area_margin.value()))
        _set_fast_sim_param_double("explore_low_nav_weight", float(spin_low_nav.value()))
        _set_fast_sim_param_double("explore_unexplored_reward_weight", float(spin_explr_w.value()))
        _set_fast_sim_param_double("explore_explored_age_weight", float(spin_explr_age.value()))
        _set_fast_sim_param_double("explore_explored_dist_weight", float(spin_explr_dist.value()))
        _set_fast_sim_param_double("explore_far_density_weight", float(spin_far_w.value()))
        _set_fast_sim_param_int("explore_far_density_ring_radius_cells", int(spin_far_ring.value()))
        _set_fast_sim_param_int("explore_far_density_kernel_radius_cells", int(spin_far_kern.value()))
        _set_fast_sim_param_double("explore_far_density_angle_step_deg", float(spin_far_step.value()))
        _set_fast_sim_param_bool("explore_far_density_exclude_inside_ring", bool(chk_far_excl.isChecked()))
        _set_fast_sim_param_double("explore_vector_avoid_weight", float(spin_vec_avoid.value()))
        _set_fast_sim_param_int("explore_vector_share_every_cells", int(spin_vec_share_cells.value()))

    spin_explore_area_r.valueChanged.connect(lambda _v: _on_explore_area())
    spin_explore_area_margin.valueChanged.connect(lambda _v: _on_explore_area())
    spin_low_nav.valueChanged.connect(lambda _v: _on_explore_area())
    spin_explr_w.valueChanged.connect(lambda _v: _on_explore_area())
    spin_explr_age.valueChanged.connect(lambda _v: _on_explore_area())
    spin_explr_dist.valueChanged.connect(lambda _v: _on_explore_area())
    spin_far_w.valueChanged.connect(lambda _v: _on_explore_area())
    spin_far_ring.valueChanged.connect(lambda _v: _on_explore_area())
    spin_far_kern.valueChanged.connect(lambda _v: _on_explore_area())
    spin_far_step.valueChanged.connect(lambda _v: _on_explore_area())
    chk_far_excl.stateChanged.connect(lambda _v: _on_explore_area())
    spin_vec_avoid.valueChanged.connect(lambda _v: _on_explore_area())
    spin_vec_share_cells.valueChanged.connect(lambda _v: _on_explore_area())

    def _on_danger_inspect(_v=None):
        _set_fast_sim_param_double("danger_inspect_weight", float(spin_ins_w.value()))
        _set_fast_sim_param_int("danger_inspect_kernel_cells", int(spin_ins_k.value()))
        _set_fast_sim_param_double("danger_inspect_danger_thr", float(spin_ins_thr.value()))
        _set_fast_sim_param_double("danger_inspect_max_cell_danger", float(spin_ins_max.value()))

    spin_ins_w.valueChanged.connect(lambda _v: _on_danger_inspect())
    spin_ins_k.valueChanged.connect(lambda _v: _on_danger_inspect())
    spin_ins_thr.valueChanged.connect(lambda _v: _on_danger_inspect())
    spin_ins_max.valueChanged.connect(lambda _v: _on_danger_inspect())

    # Inspector (dynamic danger)
    insp_box = _QtCollapsibleSection("Inspector (dynamic danger)", "qt_section_inspector_collapsed", default_expanded=False)
    insp_layout = QtWidgets.QVBoxLayout(insp_box.body)
    insp_help = QtWidgets.QLabel(
        "Purpose: only one drone (the inspector) follows a moving threat’s <b>kernel</b> to reconstruct its full path.<br>"
        "While inspecting, the drone ignores normal exploration incentives (unknown/unexplored space, nav pheromones) and instead:<br>"
        "- is attracted toward the latest known kernel cell<br>"
        "- is penalized for stepping into the threat <b>damage radius</b> (violet cells) when the damage value is high enough.<br><br>"
        "<b>Weights (how the inspector scores candidate moves):</b><br>"
        "- <b>Kernel follow (historical)</b>: attraction to last-known kernel from pheromone sharing (works even if kernel is not in LiDAR now).<br>"
        "- <b>Kernel follow (realtime)</b>: LiDAR-based reward around the currently seen kernel (short-lived; prefers a standoff ring).<br>"
        "- <b>TTL</b>: how long the realtime reward stays active after the last LiDAR kernel sighting.<br>"
        "- <b>Standoff</b>: preferred offset outside the estimated damage radius (grid cells).<br>"
        "- <b>Avoid damage weight</b>: penalty for stepping into dynamic damage cells while inspecting.<br>"
        "- <b>Damage thr</b>: ignore weak damage; apply avoid-damage only when danger(cell) ≥ threshold."
    )
    insp_help.setWordWrap(True)
    insp_help.setStyleSheet("color: #666;")
    insp_layout.addWidget(insp_help)

    # Inspector ACO formula (high-level, matches python_fast_sim scoring terms)
    insp_formula = QtWidgets.QLabel(
        "<b>Inspector ACO score (for each candidate next cell <code>c</code>):</b><br>"
        "<code>score(c) = base_ACO(c)</code><br>"
        "<code>+ w_hist · (1 − dist(c, kernel_hist)/sense_radius)</code><br>"
        "<code>+ w_rt · (1 − |dist(c, kernel_rt) − d_target| / d_target)</code> &nbsp; (only if seen in LiDAR within TTL)<br>"
        "<code>− w_avoid</code> &nbsp; if <code>danger(c) ≥ thr</code> and <code>c</code> is inside dynamic damage<br><br>"
        "Where:<br>"
        "- <code>kernel_hist</code>: last known kernel (pheromone/shared)<br>"
        "- <code>kernel_rt</code>: LiDAR-seen kernel (realtime)<br>"
        "- <code>d_target = (damage_radius_cells + standoff_cells) · cell_size</code><br>"
        "- <code>w_hist</code>=Kernel follow (historical), <code>w_rt</code>=Kernel follow (realtime), <code>w_avoid</code>=Avoid damage weight, <code>thr</code>=Damage thr"
    )
    insp_formula.setWordWrap(True)
    insp_formula.setStyleSheet("color: #666;")
    insp_layout.addWidget(insp_formula)

    row_insp1 = QtWidgets.QHBoxLayout()
    row_insp1.addWidget(QtWidgets.QLabel("Kernel follow (historical):"))
    spin_dyn_ins_w = QtWidgets.QDoubleSpinBox()
    spin_dyn_ins_w.setDecimals(2)
    spin_dyn_ins_w.setMinimum(0.0)
    spin_dyn_ins_w.setMaximum(500.0)
    spin_dyn_ins_w.setSingleStep(0.5)
    spin_dyn_ins_w.setValue(8.0)
    spin_dyn_ins_w.setToolTip(
        "Historical kernel-follow weight (pheromone-based).\n"
        "Uses last known kernel location from pheromone sharing.\n"
        "Higher -> inspector follows the kernel more aggressively.\n"
        "0 disables historical kernel-follow."
    )
    row_insp1.addWidget(spin_dyn_ins_w)
    insp_layout.addLayout(row_insp1)

    row_insp_rt = QtWidgets.QHBoxLayout()
    row_insp_rt.addWidget(QtWidgets.QLabel("Kernel follow (realtime):"))
    spin_insp_rt_w = QtWidgets.QDoubleSpinBox()
    spin_insp_rt_w.setDecimals(2)
    spin_insp_rt_w.setMinimum(0.0)
    spin_insp_rt_w.setMaximum(500.0)
    spin_insp_rt_w.setSingleStep(1.0)
    spin_insp_rt_w.setValue(12.0)
    spin_insp_rt_w.setToolTip(
        "Realtime kernel-follow weight (LiDAR-based).\n"
        "Active only for a short TTL after the drone sees the dynamic kernel in LiDAR.\n"
        "It targets the 'sweet spot': near the kernel but outside the damage radius."
    )
    row_insp_rt.addWidget(spin_insp_rt_w)
    row_insp_rt.addWidget(QtWidgets.QLabel("TTL (s):"))
    spin_insp_rt_ttl = QtWidgets.QDoubleSpinBox()
    spin_insp_rt_ttl.setDecimals(2)
    spin_insp_rt_ttl.setMinimum(0.0)
    spin_insp_rt_ttl.setMaximum(60.0)
    spin_insp_rt_ttl.setSingleStep(0.25)
    spin_insp_rt_ttl.setValue(2.0)
    spin_insp_rt_ttl.setToolTip("How long the realtime kernel-follow reward stays active after a LiDAR sighting.")
    row_insp_rt.addWidget(spin_insp_rt_ttl)
    row_insp_rt.addWidget(QtWidgets.QLabel("Standoff (cells):"))
    spin_insp_standoff = QtWidgets.QSpinBox()
    spin_insp_standoff.setMinimum(0)
    spin_insp_standoff.setMaximum(50)
    spin_insp_standoff.setValue(1)
    spin_insp_standoff.setToolTip("Desired offset outside the estimated damage radius (in grid cells).")
    row_insp_rt.addWidget(spin_insp_standoff)
    insp_layout.addLayout(row_insp_rt)

    row_insp2 = QtWidgets.QHBoxLayout()
    row_insp2.addWidget(QtWidgets.QLabel("Avoid damage weight:"))
    spin_insp_avoid_w = QtWidgets.QDoubleSpinBox()
    spin_insp_avoid_w.setDecimals(2)
    spin_insp_avoid_w.setMinimum(0.0)
    spin_insp_avoid_w.setMaximum(500.0)
    spin_insp_avoid_w.setSingleStep(1.0)
    spin_insp_avoid_w.setValue(25.0)
    spin_insp_avoid_w.setToolTip(
        "Extra penalty applied to the INSPECTOR when the candidate move would step into\n"
        "a dynamic danger DAMAGE cell (violet), provided that cell’s danger value >= Damage thr.\n"
        "Higher -> inspector stays outside the damage radius more strongly."
    )
    row_insp2.addWidget(spin_insp_avoid_w)
    row_insp2.addWidget(QtWidgets.QLabel("Damage thr:"))
    spin_insp_thr = QtWidgets.QDoubleSpinBox()
    spin_insp_thr.setDecimals(2)
    spin_insp_thr.setMinimum(0.0)
    spin_insp_thr.setMaximum(50.0)
    spin_insp_thr.setSingleStep(0.05)
    spin_insp_thr.setValue(0.05)
    spin_insp_thr.setToolTip(
        "Threshold on the cell’s stored danger value for applying the inspector’s avoid-damage penalty.\n"
        "Lower -> avoid even weak damage traces.\n"
        "Higher -> tolerate lightly-painted damage cells; avoid only strong damage."
    )
    row_insp2.addWidget(spin_insp_thr)
    insp_layout.addLayout(row_insp2)
    # Place tuning blocks near the top (requested order):
    # ACO -> Inspector -> Pheromone map -> Python speed
    try:
        insert_at = int(layout.indexOf(aco_box)) + 1
        if insert_at < 0:
            insert_at = 0
        layout.insertWidget(insert_at, insp_box)
        layout.insertSpacing(insert_at + 1, 10)
        layout.insertWidget(insert_at + 2, pher_box)
        layout.insertSpacing(insert_at + 3, 10)
        layout.insertWidget(insert_at + 4, speed_box)
        layout.insertSpacing(insert_at + 5, 10)
    except Exception:
        layout.addWidget(insp_box)
        layout.addSpacing(10)
        layout.addWidget(pher_box)
        layout.addSpacing(10)
        layout.addWidget(speed_box)
        layout.addSpacing(10)

    def _on_inspector(_v=None):
        _set_fast_sim_param_double("dyn_danger_inspect_weight", float(spin_dyn_ins_w.value()))
        _set_fast_sim_param_double("dyn_inspector_rt_weight", float(spin_insp_rt_w.value()))
        _set_fast_sim_param_double("dyn_inspector_rt_ttl_s", float(spin_insp_rt_ttl.value()))
        _set_fast_sim_param_int("dyn_inspector_rt_standoff_cells", int(spin_insp_standoff.value()))
        _set_fast_sim_param_double("dyn_inspector_avoid_damage_weight", float(spin_insp_avoid_w.value()))
        _set_fast_sim_param_double("dyn_inspector_avoid_damage_thr", float(spin_insp_thr.value()))

    spin_dyn_ins_w.valueChanged.connect(lambda _v: _on_inspector())
    spin_insp_rt_w.valueChanged.connect(lambda _v: _on_inspector())
    spin_insp_rt_ttl.valueChanged.connect(lambda _v: _on_inspector())
    spin_insp_standoff.valueChanged.connect(lambda _v: _on_inspector())
    spin_insp_avoid_w.valueChanged.connect(lambda _v: _on_inspector())
    spin_insp_thr.valueChanged.connect(lambda _v: _on_inspector())

    def _on_evap(_v=None):
        _set_fast_sim_param_double("evap_nav_rate", float(spin_evap_nav.value()))
        _set_fast_sim_param_double("evap_danger_rate", float(spin_evap_danger.value()))
        _set_fast_sim_param_double("danger_evap_mult_static", float(spin_danger_mult_static.value()))
        _set_fast_sim_param_double("danger_evap_mult_dynamic", float(spin_danger_mult_dynamic.value()))
        _set_fast_sim_param_double("wall_danger_evap_mult", float(spin_danger_mult_wall.value()))

    spin_evap_nav.valueChanged.connect(lambda _v: _on_evap())
    spin_evap_danger.valueChanged.connect(lambda _v: _on_evap())
    spin_danger_mult_static.valueChanged.connect(lambda _v: _on_evap())
    spin_danger_mult_dynamic.valueChanged.connect(lambda _v: _on_evap())
    spin_danger_mult_wall.valueChanged.connect(lambda _v: _on_evap())

    def _on_return_speed(_v: int):
        v = float(slider_return.value()) / 10.0
        lbl_return.setText(f"Return speed: {v:.1f} m/s")
        node.set_return_speed_mps(v)

    slider_return.valueChanged.connect(_on_return_speed)

    def _on_vmult(_v: int):
        mult = float(slider_vmult.value()) / 100.0
        mult = max(0.1, min(1.0, float(mult)))
        lbl_vmult.setText(f"Vertical speed mult: {mult:.2f}")
        # Publish for robustness (works even if param set fails in other setups).
        node.set_vertical_speed_mult(mult)
        # Keep param in sync for persistence / introspection.
        _set_fast_sim_param_double("vertical_speed_mult", mult)

    def _on_vmult_enabled(_v: int):
        enabled = bool(chk_vert_enabled.isChecked())
        _set_fast_sim_param_bool("vertical_speed_mult_enabled", enabled)
        _on_vmult(0)

    chk_vert_enabled.stateChanged.connect(_on_vmult_enabled)
    slider_vmult.valueChanged.connect(_on_vmult)

    def _on_drone_scale(_v: int):
        scale = float(slider_drone_scale.value()) / 100.0
        lbl_drone_scale.setText(f"Drone size: {scale:.2f}×")
        node.set_drone_marker_scale(scale)

    slider_drone_scale.valueChanged.connect(_on_drone_scale)

    def _on_drone_alt(_v: int):
        z = float(slider_drone_alt.value()) / 10.0
        lbl_drone_alt.setText(f"Altitude: {z:.1f} m")
        node.set_drone_altitude_m(z)

    slider_drone_alt.valueChanged.connect(_on_drone_alt)

    def _on_alt_safety(_v=None):
        try:
            _set_fast_sim_param_double("min_flight_altitude_m", float(spin_min_alt.value()))
            _set_fast_sim_param_double("roof_clearance_margin_m", float(spin_roof_margin.value()))
        except Exception:
            pass

    spin_min_alt.valueChanged.connect(lambda _v: _on_alt_safety())
    spin_roof_margin.valueChanged.connect(lambda _v: _on_alt_safety())

    # Overfly / vertical cost tuning
    def _on_overfly_cost(_v=None):
        try:
            _set_fast_sim_param_double("overfly_vertical_cost_mult", float(spin_overfly_cost.value()))
            _set_fast_sim_param_double("vertical_energy_cost_mult", float(spin_vert_energy.value()))
        except Exception:
            pass

    spin_overfly_cost.valueChanged.connect(lambda _v: _on_overfly_cost())
    spin_vert_energy.valueChanged.connect(lambda _v: _on_overfly_cost())

    def _on_static_alt_penalty(_v=None):
        try:
            _set_fast_sim_param_double("static_danger_altitude_violation_weight", float(spin_static_alt_pen.value()))
        except Exception:
            pass

    spin_static_alt_pen.valueChanged.connect(lambda _v: _on_static_alt_penalty())

    def _on_sense(_v: int):
        r = float(slider_sense.value())
        lbl_sense.setText(f"LiDAR sense radius: {r:.0f} m")
        _set_fast_sim_param_double("sense_radius", r)
        _set_fast_sim_param_double("local_plan_radius_m", r)

    slider_sense.valueChanged.connect(_on_sense)

    def _on_num_drones(_v: int):
        n = int(spin_n.value())
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter("num_py_drones", rclpy.Parameter.Type.INTEGER, n)])
        except Exception:
            pass

    spin_n.valueChanged.connect(_on_num_drones)

    def _publish_ptr():
        node.set_drone_pointer_params(
            enabled=bool(chk_ptr.isChecked()),
            z=float(spin_ptr_z.value()),
            scale=float(spin_ptr_scale.value()),
            alpha=float(spin_ptr_alpha.value()),
        )

    chk_ptr.stateChanged.connect(lambda _v: _publish_ptr())
    spin_ptr_z.valueChanged.connect(lambda _v: _publish_ptr())
    spin_ptr_scale.valueChanged.connect(lambda _v: _publish_ptr())
    spin_ptr_alpha.valueChanged.connect(lambda _v: _publish_ptr())

    def _apply_click_mode():
        idx = int(cmb_click.currentIndex())
        if idx == 0:
            node.set_click_mode(ClickMode.NONE)
        elif idx == 1:
            node.set_click_mode(ClickMode.TARGETS)
        elif idx == 2:
            node.set_click_mode(ClickMode.DANGER_STATIC)
        else:
            node.set_click_mode(ClickMode.DANGER_DYNAMIC)

    cmb_click.currentIndexChanged.connect(lambda _i: _apply_click_mode())
    spin_dyn_speed.valueChanged.connect(lambda v: setattr(node, "dynamic_speed_sec_per_cell", float(v)))
    spin_danger_r.valueChanged.connect(lambda v: setattr(node, "danger_radius_cells", int(v)))
    spin_threat_h.valueChanged.connect(lambda v: setattr(node, "threat_height_m", float(v)))
    btn_dyn_start.clicked.connect(node.start_dynamic_recording)
    btn_dyn_stop.clicked.connect(node.stop_dynamic_recording)

    def _publish_pher_params():
        disp = float(slider_disp.value())
        z = float(slider_z.value()) / 10.0
        a = float(slider_a.value()) / 100.0
        lbl_pher.setText(f"Display size: {disp:.0f}m, z={z:.2f}, alpha={a:.2f}")
        node.set_pheromone_viz_params(disp, z, a)

        # Batched publish (visual perf): publish one batch per tick, refresh full snapshot over ~snapshot_period_s.
        # Keep the relationship publish_hz ≈ batch_count / snapshot_period_s.
        try:
            bc = int(spin_pher_batches.value())
            sp = float(spin_pher_snapshot.value())
            _set_fast_sim_param_int("pheromone_viz_batch_count", bc)
            _set_fast_sim_param_double("pheromone_viz_snapshot_period_s", sp)
            _set_fast_sim_param_double("pheromone_viz_publish_hz", max(0.2, float(bc) / max(0.1, sp)))
        except Exception:
            pass

    def _on_pher_enable(state: int):
        enabled = state != 0
        node.set_pheromone_viz_enabled(enabled)
        _publish_pher_params()

    chk_pher.stateChanged.connect(_on_pher_enable)
    slider_disp.valueChanged.connect(lambda _v: _publish_pher_params())
    slider_z.valueChanged.connect(lambda _v: _publish_pher_params())
    slider_a.valueChanged.connect(lambda _v: _publish_pher_params())

    # Pheromone batching controls (performance)
    row_pher_batch = QtWidgets.QHBoxLayout()
    row_pher_batch.addWidget(QtWidgets.QLabel("Pheromone batching:"))
    spin_pher_batches = QtWidgets.QSpinBox()
    spin_pher_batches.setMinimum(1)
    spin_pher_batches.setMaximum(16)
    spin_pher_batches.setValue(1)
    spin_pher_batches.setToolTip("Split pheromone cells marker into N batches (publish one batch per tick).")
    row_pher_batch.addWidget(QtWidgets.QLabel("batches"))
    row_pher_batch.addWidget(spin_pher_batches)
    spin_pher_snapshot = QtWidgets.QDoubleSpinBox()
    spin_pher_snapshot.setMinimum(0.2)
    spin_pher_snapshot.setMaximum(10.0)
    spin_pher_snapshot.setSingleStep(0.2)
    spin_pher_snapshot.setValue(1.0)
    spin_pher_snapshot.setToolTip("How often to rebuild the snapshot of pheromone cells (seconds).")
    row_pher_batch.addWidget(QtWidgets.QLabel("snapshot s"))
    row_pher_batch.addWidget(spin_pher_snapshot)
    pher_layout.addLayout(row_pher_batch)
    spin_pher_batches.valueChanged.connect(lambda _v: _publish_pher_params())
    spin_pher_snapshot.valueChanged.connect(lambda _v: _publish_pher_params())

    def _publish_pher_select():
        owner = "base" if cmb_pher_source.currentIndex() == 0 else "combined" if cmb_pher_source.currentIndex() == 1 else "drone"
        idx = int(cmb_pher_layer.currentIndex())
        layer = "danger" if idx == 0 else ("nav" if idx == 1 else ("empty" if idx == 2 else "explored"))
        seq = int(spin_pher_drone.value())
        # Publish for external tooling, but ALSO apply directly via parameters so it can't be
        # delayed/starved by executor timing when the sim loop is heavy.
        node.set_pheromone_viz_select(owner, layer, seq)
        _set_fast_sim_param_str("pheromone_viz_owner", owner)
        _set_fast_sim_param_str("pheromone_viz_layer", layer)
        # Apply danger kind filter as a param (only used when layer==danger).
        # Can be a single kind or a comma-separated list.
        idxk = int(cmb_danger_kind.currentIndex())
        if idxk == 0:
            dk = "all"
        elif idxk == 1:
            dk = "nav_danger"
        elif idxk == 2:
            dk = "danger_static"
        elif idxk == 3:
            dk = "danger_dyn_kernel"
        elif idxk == 4:
            dk = "danger_dyn_damage"
        else:
            dk = "danger_dyn_kernel,danger_dyn_damage"
        _set_fast_sim_param_str("pheromone_viz_danger_kind", dk)
        _set_fast_sim_param_int("pheromone_viz_drone_seq", seq)
        try:
            node.latest_log = f"pheromone_viz_select -> owner={owner}, layer={layer}, drone_seq={seq}"
        except Exception:
            pass

    cmb_pher_source.currentIndexChanged.connect(lambda _v: _publish_pher_select())
    cmb_pher_layer.currentIndexChanged.connect(lambda _v: _publish_pher_select())
    cmb_danger_kind.currentIndexChanged.connect(lambda _v: _publish_pher_select())
    spin_pher_drone.valueChanged.connect(lambda _v: _publish_pher_select())

    # Snapshot save/load actions
    def _on_snapshot_save():
        node.save_pheromone_snapshot(str(txt_snap_name.text() or ""))

    def _on_snapshot_load():
        try:
            path, _flt = QtWidgets.QFileDialog.getOpenFileName(
                win,
                "Load pheromone snapshot",
                str(pher_snapshot_last_dir.get("v", "") or ""),
                "JSON (*.json);;All files (*)",
            )
        except Exception:
            path = ""
        if path:
            # Persist last-used directory to simplify next file selection.
            try:
                from pathlib import Path

                p = Path(str(path))
                if p.exists():
                    pher_snapshot_last_dir["v"] = str(p.parent.resolve())
                    pher_snapshot_last_path["v"] = str(p.resolve())
                    _mark_dirty()
            except Exception:
                pass
            node.load_pheromone_snapshot(str(path))

    btn_snap_save.clicked.connect(_on_snapshot_save)
    def _on_snapshot_load_latest():
        try:
            from pathlib import Path

            # "Load latest" means: load the last snapshot FILE the user previously loaded.
            last = str(pher_snapshot_last_path.get("v", "") or "")
            if not last:
                node.latest_log = "Load latest: no previously loaded snapshot yet (use 'Load snapshot…' first)"
                return
            p = Path(last)
            if not p.exists():
                node.latest_log = f"Load latest: file not found: {p}"
                return
            pher_snapshot_last_dir["v"] = str(p.parent.resolve())
            pher_snapshot_last_path["v"] = str(p.resolve())
            _mark_dirty()
            node.load_pheromone_snapshot(str(p))
            node.latest_log = f"Loaded last-used snapshot: {p.name}"
        except Exception as e:
            try:
                node.latest_log = f"Load latest snapshot failed: {e}"
            except Exception:
                pass

    btn_snap_load.clicked.connect(_on_snapshot_load)
    btn_snap_load_latest.clicked.connect(_on_snapshot_load_latest)

    # Exploit actions
    def _get_exploit_target_id() -> str:
        try:
            tid = cmb_exploit_target.currentData()
            if tid is None:
                return ""
            return str(tid)
        except Exception:
            # fallback: parse label prefix "T1 ..."
            try:
                txt = str(cmb_exploit_target.currentText() or "").strip()
                return txt.split(" ", 1)[0]
            except Exception:
                return ""

    def _update_exploit_start_enabled():
        tid = _get_exploit_target_id()
        btn_exploit_start.setEnabled(bool(tid and tid != "—"))

    def _on_exploit_target_changed(_v=None):
        tid = _get_exploit_target_id()
        node.set_selected_target(tid)
        _update_exploit_start_enabled()

    cmb_exploit_target.currentIndexChanged.connect(_on_exploit_target_changed)

    # Exploit preflight: if base pheromone map is empty, auto-load latest snapshot/compat before starting.
    pending_exploit = {"active": False, "t0": 0.0, "tid": "", "n": 0, "mode": "handled"}

    def _base_map_total_cells() -> int:
        try:
            bm = getattr(fast_sim, "base_map", None)
            if bm is None:
                return 0
            return int(len(bm.nav.v) + len(bm.danger.v) + len(bm.empty.v) + len(bm.explored.v))
        except Exception:
            return 0

    def _find_latest_snapshot_path():
        try:
            from pathlib import Path

            snap_dir = fast_sim._pheromone_snapshot_dir_path()
            if snap_dir is None:
                return None
            snap_dir = Path(str(snap_dir))
            if not snap_dir.exists():
                return None
            cands = [p for p in snap_dir.glob("*.json") if p.is_file()]
            if not cands:
                return None
            cands.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
            return cands[0]
        except Exception:
            return None

    exploit_autoload_timer = QtCore.QTimer()
    exploit_autoload_timer.setInterval(120)

    def _poll_exploit_autoload():
        if not bool(pending_exploit.get("active", False)):
            try:
                exploit_autoload_timer.stop()
            except Exception:
                pass
            return

        # Wait until base_map has something meaningful.
        if _base_map_total_cells() > 0:
            pending_exploit["active"] = False
            try:
                exploit_autoload_timer.stop()
            except Exception:
                pass
            try:
                node.latest_log = "Exploit: pheromone map loaded; starting exploit run…"
            except Exception:
                pass
            # Re-enable based on current selection.
            _update_exploit_start_enabled()
            node.start_exploit(str(pending_exploit.get("tid", "") or ""), int(pending_exploit.get("n", 3) or 3), str(pending_exploit.get("mode", "handled") or "handled"))
            return

        # Timeout protection (avoid hanging forever).
        try:
            t0 = float(pending_exploit.get("t0", 0.0) or 0.0)
        except Exception:
            t0 = 0.0
        if t0 > 0.0 and (time.time() - t0) > 6.0:
            pending_exploit["active"] = False
            try:
                exploit_autoload_timer.stop()
            except Exception:
                pass
            _update_exploit_start_enabled()
            try:
                node.latest_log = "Exploit start aborted: auto-load pheromone map timed out (no snapshot/compat loaded)"
            except Exception:
                pass

    exploit_autoload_timer.timeout.connect(_poll_exploit_autoload)

    def _on_exploit_start_clicked():
        tid = _get_exploit_target_id()
        if not tid or tid == "—":
            node.latest_log = "Exploit start: select a target goal first"
            return
        n = int(spin_exploit_n.value())
        mode_static = bool(cmb_exploit_dyn.currentIndex() == 1)
        # Keep sim params in sync (so mode is visible via `ros2 param get`)
        _set_fast_sim_param_bool("exploit_dynamic_danger_as_static", bool(mode_static))
        _set_fast_sim_param_int("num_py_drones", int(n))
        # Exploit spacing params
        try:
            _set_fast_sim_param_double("exploit_peer_avoid_radius_m", float(spin_ex_avoid_r.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_weight", float(spin_ex_avoid_w.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_fade_start_m", float(spin_ex_fade0.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_fade_range_m", float(spin_ex_fade_rng.value()))
            _set_fast_sim_param_double("exploit_peer_path_follow_weight", float(spin_ex_follow_w.value()))
            _set_fast_sim_param_double("exploit_dyn_trail_overlay_strength", float(spin_ex_dyn_overlay_strength.value()))
            _set_fast_sim_param_double("exploit_dyn_trail_overlay_gamma", float(spin_ex_dyn_overlay_gamma.value()))
        except Exception:
            pass
        dynamic_mode = ("static" if mode_static else "handled")

        # If pheromone base map is empty (no exploration and no loaded snapshot), auto-load latest map first.
        if _base_map_total_cells() <= 0:
            if bool(pending_exploit.get("active", False)):
                try:
                    node.latest_log = "Exploit: auto-loading pheromone map already in progress…"
                except Exception:
                    pass
                return

            # Prefer latest snapshot from configured snapshot directory; fall back to compat file.
            p = _find_latest_snapshot_path()
            if p is not None:
                try:
                    node.latest_log = f"Exploit: pheromone map empty; auto-loading latest snapshot: {p.name}"
                except Exception:
                    pass
                node.load_pheromone_snapshot(str(p))
            else:
                # Compat fallback (data/pheromone_data.json)
                try:
                    from pathlib import Path

                    repo_root2 = Path(__file__).resolve().parents[2]
                    compat_rel = str(fast_sim.get_parameter("compat_export_path").value)
                    compat_path = (repo_root2 / compat_rel) if compat_rel else (repo_root2 / "data" / "pheromone_data.json")
                    if compat_path.exists():
                        try:
                            node.latest_log = f"Exploit: pheromone map empty; auto-loading compat pheromone: {compat_path.name}"
                        except Exception:
                            pass
                        node.load_compat_pheromone()
                    else:
                        try:
                            node.latest_log = "Exploit start aborted: pheromone map is empty and no snapshots/compat file found"
                        except Exception:
                            pass
                        return
                except Exception:
                    try:
                        node.latest_log = "Exploit start aborted: pheromone map is empty and auto-load failed"
                    except Exception:
                        pass
                    return

            # Defer exploit start until load completes (base_map non-empty).
            pending_exploit["active"] = True
            pending_exploit["t0"] = time.time()
            pending_exploit["tid"] = str(tid)
            pending_exploit["n"] = int(n)
            pending_exploit["mode"] = str(dynamic_mode)
            try:
                btn_exploit_start.setEnabled(False)
            except Exception:
                pass
            try:
                exploit_autoload_timer.start()
            except Exception:
                pass
            return

        node.start_exploit(tid, n, dynamic_mode)

    btn_exploit_start.clicked.connect(_on_exploit_start_clicked)
    # Ensure correct state after wiring.
    _update_exploit_start_enabled()

    def _publish_lidar_viz():
        node.set_lidar_viz(bool(chk_lidar.isChecked()), int(spin_lidar_drone.value()))
        node.set_lidar_scan_viz(bool(chk_lidar_scan.isChecked()), int(spin_lidar_drone.value()))
        if bool(chk_plan_all.isChecked()):
            node.set_plan_viz(True, 0)  # 0 => ALL drones
        else:
            node.set_plan_viz(bool(chk_plan.isChecked()), int(spin_lidar_drone.value()))
        if bool(chk_aco_all.isChecked()):
            node.set_aco_viz(True, 0)  # 0 => ALL drones
        else:
            node.set_aco_viz(bool(chk_aco.isChecked()), int(spin_lidar_drone.value()))
        # ACO arrow params are parameters on the sim node (same process), so set directly.
        try:
            _set_fast_sim_param_double("aco_viz_arrow_width_m", float(spin_aco_w.value()))
            _set_fast_sim_param_double("aco_viz_arrow_length_mult", float(spin_aco_len.value()))
            _set_fast_sim_param_bool("aco_commit_enabled", bool(chk_aco_commit.isChecked()))
            _set_fast_sim_param_bool("corner_backoff_enabled", bool(chk_corner_backoff.isChecked()))
            _set_fast_sim_param_bool("unstick_move_enabled", bool(chk_unstick.isChecked()))
            _set_fast_sim_param_double("aco_commit_timeout_s", float(spin_commit.value()))
        except Exception:
            pass

    def _on_exploit_spacing(_v=None):
        try:
            _set_fast_sim_param_double("exploit_peer_avoid_radius_m", float(spin_ex_avoid_r.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_weight", float(spin_ex_avoid_w.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_fade_start_m", float(spin_ex_fade0.value()))
            _set_fast_sim_param_double("exploit_peer_avoid_fade_range_m", float(spin_ex_fade_rng.value()))
            _set_fast_sim_param_double("exploit_peer_path_follow_weight", float(spin_ex_follow_w.value()))
            _set_fast_sim_param_double("exploit_dyn_trail_overlay_strength", float(spin_ex_dyn_overlay_strength.value()))
            _set_fast_sim_param_double("exploit_dyn_trail_overlay_gamma", float(spin_ex_dyn_overlay_gamma.value()))
        except Exception:
            pass

    spin_ex_avoid_r.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_avoid_w.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_fade0.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_fade_rng.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_follow_w.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_dyn_overlay_strength.valueChanged.connect(lambda _v: _on_exploit_spacing())
    spin_ex_dyn_overlay_gamma.valueChanged.connect(lambda _v: _on_exploit_spacing())

    chk_lidar.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_lidar_scan.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_plan.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_plan_all.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_aco.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_aco_all.stateChanged.connect(lambda _v: _publish_lidar_viz())
    spin_aco_w.valueChanged.connect(lambda _v: _publish_lidar_viz())
    spin_aco_len.valueChanged.connect(lambda _v: _publish_lidar_viz())
    chk_aco_commit.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_corner_backoff.stateChanged.connect(lambda _v: _publish_lidar_viz())
    chk_unstick.stateChanged.connect(lambda _v: _publish_lidar_viz())
    spin_commit.valueChanged.connect(lambda _v: _publish_lidar_viz())
    spin_lidar_drone.valueChanged.connect(lambda _v: _publish_lidar_viz())

    # UI update timer (no ROS spinning here)
    last_clock_ui_update = {"t": 0.0}
    shutting_down = {"v": False}
    exploit_stats_state = {
        "last_run_key": "",  # changes when a new exploit run starts
        "last_text": None,  # last text we put into the widget
        "last_update_t": 0.0,  # throttle live updates
        "frozen": False,  # freeze after run completes so user can copy
    }

    def _tick_ui():
        if shutting_down["v"]:
            try:
                timer.stop()
            except Exception:
                pass
            return
        _maybe_save_settings()

        # update labels
        lbl_phase.setText(f"Phase: {node.latest_phase}")

        # Update clock text at 0.5 Hz (every 2 seconds) to reduce UI churn
        now = time.time()
        if now - last_clock_ui_update["t"] >= 2.0:
            last_clock_ui_update["t"] = now
            if node.latest_clock is not None:
                sec = int(node.latest_clock.clock.sec)
                nsec = int(node.latest_clock.clock.nanosec)
                lbl_clock.setText(f"Sim time: {_format_sim_time(sec, nsec)}")
                # "absolute" timestamp (for copy/paste) = seconds.nanoseconds
                lbl_clock_small.setText(f"Timestamp: {sec}.{nsec:09d}")
            else:
                lbl_clock.setText("Sim time: —")
                lbl_clock_small.setText("Timestamp: —")

        # targets summary + drone lists
        st = node.gui_status or {}
        if st:
            # Prefer simulator-provided event log, fallback to GUI-local log.
            try:
                lbl_log.setText(str(st.get("sim_log") or node.latest_log or "—"))
            except Exception:
                lbl_log.setText(node.latest_log or "—")

            tf = int(st.get("targets_found", 0))
            tu = int(st.get("targets_unfound", 0))
            tt = int(st.get("targets_total", tf + tu))
            lbl_targets.setText(f"Targets: found {tf} / {tt}   (unfound: {tu})")

            # Refresh exploit target list from gui_status targets.
            try:
                tgt_list = st.get("targets", []) or []
                # Avoid flicker: only rebuild the combo box when the set/order of target ids changes.
                sig = []
                for trow in tgt_list:
                    tid = str(trow.get("id", "") or "")
                    if tid:
                        sig.append(tid)
                sig_key = "|".join(sig)
                if not hasattr(_tick_ui, "_exploit_targets_sig"):
                    _tick_ui._exploit_targets_sig = ""  # type: ignore[attr-defined]
                old_sig = str(getattr(_tick_ui, "_exploit_targets_sig", "") or "")  # type: ignore[attr-defined]

                current_id = _get_exploit_target_id()

                if sig_key != old_sig:
                    setattr(_tick_ui, "_exploit_targets_sig", sig_key)  # type: ignore[attr-defined]
                    cmb_exploit_target.blockSignals(True)
                    cmb_exploit_target.setUpdatesEnabled(False)
                    cmb_exploit_target.clear()
                    cmb_exploit_target.addItem("—", "")
                    for trow in tgt_list:
                        tid = str(trow.get("id", "") or "")
                        if not tid:
                            continue
                        found0 = bool(trow.get("found", False))
                        cmb_exploit_target.addItem(f"{tid}  ({'found' if found0 else 'unfound'})", tid)
                    # Restore selection if possible
                    if current_id:
                        idx = cmb_exploit_target.findData(current_id)
                        if idx >= 0:
                            cmb_exploit_target.setCurrentIndex(idx)
                    cmb_exploit_target.setUpdatesEnabled(True)
                    cmb_exploit_target.blockSignals(False)
                else:
                    # Same ids -> update labels in-place without clearing (no blink).
                    for trow in tgt_list:
                        tid = str(trow.get("id", "") or "")
                        if not tid:
                            continue
                        found0 = bool(trow.get("found", False))
                        idx = cmb_exploit_target.findData(tid)
                        if idx >= 0:
                            cmb_exploit_target.setItemText(idx, f"{tid}  ({'found' if found0 else 'unfound'})")

                _update_exploit_start_enabled()
            except Exception:
                try:
                    cmb_exploit_target.blockSignals(False)
                except Exception:
                    pass

            lines = []
            for drow in st.get("drones", []):
                missing = drow.get("missing_targets", []) or []
                not_found = drow.get("not_found_known", []) or []
                if missing or not_found:
                    lines.append(f'{drow.get("uid")} [{drow.get("mode")}]: missing={len(missing)} known_unfound={len(not_found)}')
            txt_drones.setPlainText("\n".join(lines) if lines else "All drones know all targets and have no known-unfound targets.")

            # Speed limits display
            try:
                mh = float(st.get("max_horizontal_speed_mps", 0.0))
                mv = float(st.get("max_vertical_speed_mps", 0.0))
                if mh > 1e-6 and mv > 1e-6:
                    lbl_speed_limits.setText(f"Max speeds: horizontal {mh:.1f} m/s, vertical {mv:.1f} m/s")
                else:
                    lbl_speed_limits.setText("Max speeds: horizontal — m/s, vertical — m/s")
            except Exception:
                lbl_speed_limits.setText("Max speeds: horizontal — m/s, vertical — m/s")

            # Exploit stats (if available)
            try:
                running = bool(st.get("running", False))
                ex = st.get("exploit_stats", {}) or {}
                ex_active = bool(ex.get("active", False))
                ex_target = str(ex.get("target_id", "") or "")
                try:
                    ex_start = float(ex.get("start_t_sim", 0.0))
                except Exception:
                    ex_start = 0.0
                run_key = f"{ex_target}|{ex_start:.3f}"
                # New run => unfreeze so text can update.
                if run_key.strip("|") and run_key != str(exploit_stats_state.get("last_run_key", "")):
                    exploit_stats_state["last_run_key"] = run_key
                    exploit_stats_state["frozen"] = False
                    exploit_stats_state["last_text"] = None

                rows = ex.get("drones", []) or []
                if not rows:
                    text = "—"
                else:
                    lines = []
                    for r in rows:
                        uid = str(r.get("uid", "") or "")
                        t = r.get("time_to_land_s", None)
                        ttxt = f"{float(t):.1f}s" if t is not None else "—"
                        hv = float(r.get("dist_hv_m", 0.0))
                        hh = float(r.get("dist_horizontal_m", 0.0))
                        vv = float(r.get("dist_vertical_m", 0.0))
                        spent = float(r.get("battery_spent_units", 0.0))
                        left_u = float(r.get("battery_left_units", 0.0))
                        # Battery percent left is meaningful for current energy model where full=100.
                        lines.append(
                            f"{uid}: time={ttxt}, dist(h+v)={hv:.1f}m (h={hh:.1f}, v={vv:.1f}), battery_spent={spent:.2f}u, battery_left={left_u:.2f}u"
                        )
                    text = "\n".join(lines) if lines else "—"

                # Freeze behavior:
                # - while exploit is active, update at a low rate to avoid constantly resetting selection
                # - when exploit becomes inactive and we have any stats, write once and freeze (copy-friendly)
                # - when sim isn't running, don't churn this section at all
                if ex_active:
                    exploit_stats_state["frozen"] = False

                if (not ex_active) and (text != "—"):
                    # Final stats available => write once (if needed) and freeze.
                    if text != exploit_stats_state.get("last_text"):
                        txt_exploit_stats.setPlainText(text)
                        exploit_stats_state["last_text"] = text
                    exploit_stats_state["frozen"] = True
                else:
                    if exploit_stats_state.get("frozen"):
                        pass
                    else:
                        if (not running) and (not ex_active):
                            # Nothing is running, and no final stats to freeze -> leave as-is to allow copy.
                            pass
                        else:
                            now2 = time.time()
                            if now2 - float(exploit_stats_state.get("last_update_t", 0.0)) >= 0.5:
                                exploit_stats_state["last_update_t"] = now2
                                if text != exploit_stats_state.get("last_text"):
                                    txt_exploit_stats.setPlainText(text)
                                    exploit_stats_state["last_text"] = text
            except Exception:
                try:
                    if not bool(exploit_stats_state.get("frozen")):
                        txt_exploit_stats.setPlainText("—")
                except Exception:
                    pass

        else:
            lbl_targets.setText("Targets: —")
            lbl_log.setText(node.latest_log or "—")
            try:
                if not bool(exploit_stats_state.get("frozen")):
                    txt_exploit_stats.setPlainText("—")
            except Exception:
                pass

    timer = QtCore.QTimer()
    timer.timeout.connect(_tick_ui)
    timer.start(33)  # ~30Hz UI refresh

    # Ensure we stop UI timer before ROS shutdown.
    try:
        def _on_quit():
            shutting_down["v"] = True
            try:
                timer.stop()
            except Exception:
                pass
            stop_spin.set()
            try:
                # Don't hang forever on shutdown if a callback is still running.
                exec_.shutdown(timeout_sec=0.2)
            except Exception:
                pass
            try:
                spin_thread.join(timeout=1.0)
            except Exception:
                pass

        app.aboutToQuit.connect(_on_quit)
    except Exception:
        pass

    # Apply initial state (auto publishing)
    node.set_pheromone_viz_enabled(True)
    _publish_pher_params()
    _publish_pher_select()
    _publish_lidar_viz()
    node.set_building_alpha(1.0)
    _on_drone_scale(0)
    _on_drone_alt(0)
    _on_sense(0)
    _publish_ptr()
    _on_return_speed(0)
    _on_vmult_enabled(0)
    _on_vmult(0)
    # ACO defaults -> fast sim params
    _on_aco_temp(0)
    _on_min_r(0)
    _on_min_s(0)
    _on_recent(0)
    _on_basepush(0)
    _on_explore_area(0)
    _on_danger_inspect(0)
    try:
        _on_inspector(0)
    except Exception:
        pass
    _on_evap(0)
    try:
        _on_static_alt_penalty(0)
    except Exception:
        pass
    # Targets
    def _publish_tgt_params():
        d = float(slider_tgt_d.value())
        a = float(slider_tgt_a.value()) / 100.0
        lbl_tgt.setText(f"Diameter: {d:.1f}m, alpha={a:.2f}")
        node.set_target_viz_params(d, a)

    slider_tgt_d.valueChanged.connect(lambda _v: _publish_tgt_params())
    slider_tgt_a.valueChanged.connect(lambda _v: _publish_tgt_params())
    _publish_tgt_params()

    win.resize(820, 680)
    win.show()

    def _on_bld_alpha(_v: int):
        alpha = float(slider_bld_a.value()) / 100.0
        lbl_bld.setText(f"Opacity: {alpha:.2f}")
        node.set_building_alpha(alpha)

    slider_bld_a.valueChanged.connect(_on_bld_alpha)

    # Settings persistence (Qt)
    try:
        slider.setValue(int(settings.get("speed_slider", slider.value())))
        slider_disp.setValue(int(settings.get("pher_display_size", slider_disp.value())))
        slider_z.setValue(int(settings.get("pher_z_slider", slider_z.value())))
        slider_a.setValue(int(settings.get("pher_alpha_slider", slider_a.value())))
        spin_pher_batches.setValue(int(settings.get("pher_batch_count", spin_pher_batches.value())))
        spin_pher_snapshot.setValue(float(settings.get("pher_snapshot_period_s", spin_pher_snapshot.value())))
        slider_bld_a.setValue(int(settings.get("building_alpha_slider", slider_bld_a.value())))
        slider_tgt_d.setValue(int(settings.get("target_diameter_slider", slider_tgt_d.value())))
        slider_tgt_a.setValue(int(settings.get("target_alpha_slider", slider_tgt_a.value())))
        slider_drone_scale.setValue(int(settings.get("drone_marker_scale_slider", slider_drone_scale.value())))
        slider_return.setValue(int(settings.get("return_speed_slider", slider_return.value())))
        slider_drone_alt.setValue(int(settings.get("drone_altitude_slider", slider_drone_alt.value())))
        chk_vert_enabled.setChecked(bool(settings.get("vertical_speed_mult_enabled", chk_vert_enabled.isChecked())))
        slider_vmult.setValue(int(settings.get("vertical_speed_mult_slider", slider_vmult.value())))
        slider_sense.setValue(int(settings.get("sense_radius_m", slider_sense.value())))
        spin_n.setValue(int(settings.get("num_py_drones", spin_n.value())))
        spin_min_alt.setValue(float(settings.get("min_flight_altitude_m", spin_min_alt.value())))
        spin_roof_margin.setValue(float(settings.get("roof_clearance_margin_m", spin_roof_margin.value())))
        spin_overfly_cost.setValue(float(settings.get("overfly_vertical_cost_mult", spin_overfly_cost.value())))
        spin_vert_energy.setValue(float(settings.get("vertical_energy_cost_mult", spin_vert_energy.value())))
        spin_static_alt_pen.setValue(
            float(settings.get("static_danger_altitude_violation_weight", spin_static_alt_pen.value()))
        )
        slider_temp.setValue(int(settings.get("aco_temperature_slider", slider_temp.value())))
        slider_min_r.setValue(int(settings.get("explore_min_radius_slider", slider_min_r.value())))
        slider_min_s.setValue(int(settings.get("explore_min_strength_slider", slider_min_s.value())))
        slider_recent.setValue(int(settings.get("recent_cell_penalty_slider", slider_recent.value())))
        spin_basepush_r.setValue(float(settings.get("base_push_radius_m", spin_basepush_r.value())))
        spin_basepush_s.setValue(float(settings.get("base_push_strength", spin_basepush_s.value())))
        spin_explore_area_r.setValue(float(settings.get("exploration_area_radius_m", spin_explore_area_r.value())))
        spin_explore_area_margin.setValue(float(settings.get("exploration_radius_margin_m", spin_explore_area_margin.value())))
        spin_low_nav.setValue(float(settings.get("explore_low_nav_weight", spin_low_nav.value())))
        spin_explr_w.setValue(float(settings.get("explore_unexplored_reward_weight", spin_explr_w.value())))
        spin_explr_age.setValue(float(settings.get("explore_explored_age_weight", spin_explr_age.value())))
        spin_explr_dist.setValue(float(settings.get("explore_explored_dist_weight", spin_explr_dist.value())))
        spin_ins_w.setValue(float(settings.get("danger_inspect_weight", spin_ins_w.value())))
        spin_ins_k.setValue(int(settings.get("danger_inspect_kernel_cells", spin_ins_k.value())))
        spin_ins_thr.setValue(float(settings.get("danger_inspect_danger_thr", spin_ins_thr.value())))
        spin_ins_max.setValue(float(settings.get("danger_inspect_max_cell_danger", spin_ins_max.value())))
        # Inspector (dynamic danger)
        try:
            spin_dyn_ins_w.setValue(float(settings.get("dyn_danger_inspect_weight", spin_dyn_ins_w.value())))
            spin_insp_rt_w.setValue(float(settings.get("dyn_inspector_rt_weight", spin_insp_rt_w.value())))
            spin_insp_rt_ttl.setValue(float(settings.get("dyn_inspector_rt_ttl_s", spin_insp_rt_ttl.value())))
            spin_insp_standoff.setValue(int(settings.get("dyn_inspector_rt_standoff_cells", spin_insp_standoff.value())))
            spin_insp_avoid_w.setValue(float(settings.get("dyn_inspector_avoid_damage_weight", spin_insp_avoid_w.value())))
            spin_insp_thr.setValue(float(settings.get("dyn_inspector_avoid_damage_thr", spin_insp_thr.value())))
        except Exception:
            pass
        spin_far_w.setValue(float(settings.get("explore_far_density_weight", spin_far_w.value())))
        spin_far_ring.setValue(int(settings.get("explore_far_density_ring_radius_cells", spin_far_ring.value())))
        spin_far_kern.setValue(int(settings.get("explore_far_density_kernel_radius_cells", spin_far_kern.value())))
        spin_far_step.setValue(float(settings.get("explore_far_density_angle_step_deg", spin_far_step.value())))
        chk_far_excl.setChecked(bool(settings.get("explore_far_density_exclude_inside_ring", chk_far_excl.isChecked())))
        spin_vec_avoid.setValue(float(settings.get("explore_vector_avoid_weight", spin_vec_avoid.value())))
        spin_vec_share_cells.setValue(int(settings.get("explore_vector_share_every_cells", spin_vec_share_cells.value())))
        spin_evap_nav.setValue(float(settings.get("evap_nav_rate", spin_evap_nav.value())))
        spin_evap_danger.setValue(float(settings.get("evap_danger_rate", spin_evap_danger.value())))
        spin_danger_mult_static.setValue(float(settings.get("danger_evap_mult_static", spin_danger_mult_static.value())))
        spin_danger_mult_dynamic.setValue(float(settings.get("danger_evap_mult_dynamic", spin_danger_mult_dynamic.value())))
        spin_danger_mult_wall.setValue(float(settings.get("wall_danger_evap_mult", spin_danger_mult_wall.value())))
        cmb_click.setCurrentIndex(int(settings.get("click_mode_index", cmb_click.currentIndex())))
        spin_dyn_speed.setValue(float(settings.get("dynamic_speed_sec_per_cell", spin_dyn_speed.value())))
        spin_danger_r.setValue(int(settings.get("danger_radius_cells", spin_danger_r.value())))
        spin_threat_h.setValue(float(settings.get("threat_height_m", spin_threat_h.value())))
        cmb_pher_source.setCurrentIndex(int(settings.get("pher_source_index", cmb_pher_source.currentIndex())))
        cmb_pher_layer.setCurrentIndex(int(settings.get("pher_layer_index", cmb_pher_layer.currentIndex())))
        cmb_danger_kind.setCurrentIndex(int(settings.get("pher_danger_kind_index", cmb_danger_kind.currentIndex())))
        spin_pher_drone.setValue(int(settings.get("pher_drone_seq", spin_pher_drone.value())))
        chk_lidar.setChecked(bool(settings.get("lidar_viz_enabled", chk_lidar.isChecked())))
        spin_lidar_drone.setValue(int(settings.get("lidar_viz_drone_seq", spin_lidar_drone.value())))
        chk_lidar_scan.setChecked(bool(settings.get("lidar_scan_viz_enabled", chk_lidar_scan.isChecked())))
        chk_plan.setChecked(bool(settings.get("plan_viz_enabled", chk_plan.isChecked())))
        chk_plan_all.setChecked(bool(settings.get("plan_viz_all_enabled", chk_plan_all.isChecked())))
        chk_aco.setChecked(bool(settings.get("aco_viz_enabled", chk_aco.isChecked())))
        chk_aco_all.setChecked(bool(settings.get("aco_viz_all_enabled", chk_aco_all.isChecked())))
        spin_aco_w.setValue(float(settings.get("aco_viz_arrow_width_m", spin_aco_w.value())))
        spin_aco_len.setValue(float(settings.get("aco_viz_arrow_length_mult", spin_aco_len.value())))
        chk_aco_commit.setChecked(bool(settings.get("aco_commit_enabled", chk_aco_commit.isChecked())))
        chk_corner_backoff.setChecked(bool(settings.get("corner_backoff_enabled", chk_corner_backoff.isChecked())))
        chk_unstick.setChecked(bool(settings.get("unstick_move_enabled", chk_unstick.isChecked())))
        spin_commit.setValue(float(settings.get("aco_commit_timeout_s", spin_commit.value())))
        chk_ptr.setChecked(bool(settings.get("pointer_enabled", chk_ptr.isChecked())))
        spin_ptr_z.setValue(float(settings.get("pointer_z", spin_ptr_z.value())))
        spin_ptr_scale.setValue(float(settings.get("pointer_scale", spin_ptr_scale.value())))
        spin_ptr_alpha.setValue(float(settings.get("pointer_alpha", spin_ptr_alpha.value())))
        # Exploit spacing / path adherence
        try:
            spin_ex_avoid_r.setValue(float(settings.get("exploit_peer_avoid_radius_m", spin_ex_avoid_r.value())))
            spin_ex_avoid_w.setValue(float(settings.get("exploit_peer_avoid_weight", spin_ex_avoid_w.value())))
            spin_ex_fade0.setValue(float(settings.get("exploit_peer_avoid_fade_start_m", spin_ex_fade0.value())))
            spin_ex_fade_rng.setValue(float(settings.get("exploit_peer_avoid_fade_range_m", spin_ex_fade_rng.value())))
            spin_ex_follow_w.setValue(float(settings.get("exploit_peer_path_follow_weight", spin_ex_follow_w.value())))
            spin_ex_dyn_overlay_strength.setValue(
                float(settings.get("exploit_dyn_trail_overlay_strength", spin_ex_dyn_overlay_strength.value()))
            )
            spin_ex_dyn_overlay_gamma.setValue(float(settings.get("exploit_dyn_trail_overlay_gamma", spin_ex_dyn_overlay_gamma.value())))
        except Exception:
            pass
    except Exception:
        pass
    _apply_click_mode()
    _publish_ptr()
    # Some Qt versions don't emit currentIndexChanged when setCurrentIndex is called with the same value.
    # Always apply pheromone selection after loading settings.
    _publish_pher_select()

    dirty = {"v": False}
    last_save = {"t": 0.0}

    def _mark_dirty():
        dirty["v"] = True

    # Mark dirty on section expand/collapse (persist in settings).
    for sec in (
        status_box,
        tgt_btn_box,
        speed_box,
        aco_box,
        inspect_box,
        far_box,
        drones_box,
        click_box,
        tgt_box,
        pher_box,
        lidar_box,
        bld_box,
        insp_box,
    ):
        try:
            sec.toggle_btn.toggled.connect(lambda _v: _mark_dirty())
        except Exception:
            pass

    for wdg in (slider, slider_disp, slider_z, slider_a, slider_bld_a, slider_tgt_d, slider_tgt_a):
        wdg.valueChanged.connect(lambda _v: _mark_dirty())
    spin_n.valueChanged.connect(lambda _v: _mark_dirty())
    slider_drone_scale.valueChanged.connect(lambda _v: _mark_dirty())
    slider_drone_alt.valueChanged.connect(lambda _v: _mark_dirty())
    slider_sense.valueChanged.connect(lambda _v: _mark_dirty())
    spin_min_alt.valueChanged.connect(lambda _v: _mark_dirty())
    spin_roof_margin.valueChanged.connect(lambda _v: _mark_dirty())
    spin_overfly_cost.valueChanged.connect(lambda _v: _mark_dirty())
    spin_vert_energy.valueChanged.connect(lambda _v: _mark_dirty())
    spin_static_alt_pen.valueChanged.connect(lambda _v: _mark_dirty())
    for wdg in (slider_temp, slider_min_r, slider_min_s, slider_recent):
        wdg.valueChanged.connect(lambda _v: _mark_dirty())
    spin_basepush_r.valueChanged.connect(lambda _v: _mark_dirty())
    spin_basepush_s.valueChanged.connect(lambda _v: _mark_dirty())
    spin_explore_area_r.valueChanged.connect(lambda _v: _mark_dirty())
    spin_explore_area_margin.valueChanged.connect(lambda _v: _mark_dirty())
    spin_low_nav.valueChanged.connect(lambda _v: _mark_dirty())
    spin_explr_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_explr_age.valueChanged.connect(lambda _v: _mark_dirty())
    spin_explr_dist.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ins_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ins_k.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ins_thr.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ins_max.valueChanged.connect(lambda _v: _mark_dirty())
    # Inspector (dynamic danger)
    spin_dyn_ins_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_insp_rt_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_insp_rt_ttl.valueChanged.connect(lambda _v: _mark_dirty())
    spin_insp_standoff.valueChanged.connect(lambda _v: _mark_dirty())
    spin_insp_avoid_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_insp_thr.valueChanged.connect(lambda _v: _mark_dirty())
    # Far-density kernel (ring probes)
    spin_far_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_far_ring.valueChanged.connect(lambda _v: _mark_dirty())
    spin_far_kern.valueChanged.connect(lambda _v: _mark_dirty())
    spin_far_step.valueChanged.connect(lambda _v: _mark_dirty())
    chk_far_excl.stateChanged.connect(lambda _v: _mark_dirty())
    spin_vec_avoid.valueChanged.connect(lambda _v: _mark_dirty())
    spin_vec_share_cells.valueChanged.connect(lambda _v: _mark_dirty())
    spin_evap_nav.valueChanged.connect(lambda _v: _mark_dirty())
    spin_evap_danger.valueChanged.connect(lambda _v: _mark_dirty())
    spin_danger_mult_static.valueChanged.connect(lambda _v: _mark_dirty())
    spin_danger_mult_dynamic.valueChanged.connect(lambda _v: _mark_dirty())
    spin_danger_mult_wall.valueChanged.connect(lambda _v: _mark_dirty())
    cmb_click.currentIndexChanged.connect(lambda _v: _mark_dirty())
    spin_dyn_speed.valueChanged.connect(lambda _v: _mark_dirty())
    spin_danger_r.valueChanged.connect(lambda _v: _mark_dirty())
    spin_threat_h.valueChanged.connect(lambda _v: _mark_dirty())
    cmb_pher_source.currentIndexChanged.connect(lambda _v: _mark_dirty())
    cmb_pher_layer.currentIndexChanged.connect(lambda _v: _mark_dirty())
    cmb_danger_kind.currentIndexChanged.connect(lambda _v: _mark_dirty())
    spin_pher_drone.valueChanged.connect(lambda _v: _mark_dirty())
    chk_lidar.stateChanged.connect(lambda _v: _mark_dirty())
    chk_lidar_scan.stateChanged.connect(lambda _v: _mark_dirty())
    chk_plan.stateChanged.connect(lambda _v: _mark_dirty())
    chk_plan_all.stateChanged.connect(lambda _v: _mark_dirty())
    chk_aco.stateChanged.connect(lambda _v: _mark_dirty())
    chk_aco_all.stateChanged.connect(lambda _v: _mark_dirty())
    spin_aco_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_aco_len.valueChanged.connect(lambda _v: _mark_dirty())
    chk_aco_commit.stateChanged.connect(lambda _v: _mark_dirty())
    chk_corner_backoff.stateChanged.connect(lambda _v: _mark_dirty())
    chk_unstick.stateChanged.connect(lambda _v: _mark_dirty())
    spin_commit.valueChanged.connect(lambda _v: _mark_dirty())
    spin_lidar_drone.valueChanged.connect(lambda _v: _mark_dirty())
    slider_return.valueChanged.connect(lambda _v: _mark_dirty())
    chk_vert_enabled.stateChanged.connect(lambda _v: _mark_dirty())
    slider_vmult.valueChanged.connect(lambda _v: _mark_dirty())
    chk_ptr.stateChanged.connect(lambda _v: _mark_dirty())
    spin_ptr_z.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ptr_scale.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ptr_alpha.valueChanged.connect(lambda _v: _mark_dirty())
    # Exploit spacing / path adherence
    spin_ex_avoid_r.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_avoid_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_fade0.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_fade_rng.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_follow_w.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_dyn_overlay_strength.valueChanged.connect(lambda _v: _mark_dirty())
    spin_ex_dyn_overlay_gamma.valueChanged.connect(lambda _v: _mark_dirty())

    def _maybe_save_settings():
        now = time.time()
        if not dirty["v"] or (now - last_save["t"] < 2.0):
            return
        payload = {
            "speed_slider": int(slider.value()),
            "pher_display_size": int(slider_disp.value()),
            "pher_z_slider": int(slider_z.value()),
            "pher_alpha_slider": int(slider_a.value()),
            "pher_batch_count": int(spin_pher_batches.value()),
            "pher_snapshot_period_s": float(spin_pher_snapshot.value()),
            "building_alpha_slider": int(slider_bld_a.value()),
            "target_diameter_slider": int(slider_tgt_d.value()),
            "target_alpha_slider": int(slider_tgt_a.value()),
            "drone_marker_scale_slider": int(slider_drone_scale.value()),
            "return_speed_slider": int(slider_return.value()),
            "drone_altitude_slider": int(slider_drone_alt.value()),
            "vertical_speed_mult_enabled": bool(chk_vert_enabled.isChecked()),
            "vertical_speed_mult_slider": int(slider_vmult.value()),
            "sense_radius_m": int(slider_sense.value()),
            "num_py_drones": int(spin_n.value()),
            "min_flight_altitude_m": float(spin_min_alt.value()),
            "roof_clearance_margin_m": float(spin_roof_margin.value()),
            "overfly_vertical_cost_mult": float(spin_overfly_cost.value()),
            "vertical_energy_cost_mult": float(spin_vert_energy.value()),
            "static_danger_altitude_violation_weight": float(spin_static_alt_pen.value()),
            "aco_temperature_slider": int(slider_temp.value()),
            "explore_min_radius_slider": int(slider_min_r.value()),
            "explore_min_strength_slider": int(slider_min_s.value()),
            "recent_cell_penalty_slider": int(slider_recent.value()),
            "base_push_radius_m": float(spin_basepush_r.value()),
            "base_push_strength": float(spin_basepush_s.value()),
            "exploration_area_radius_m": float(spin_explore_area_r.value()),
            "exploration_radius_margin_m": float(spin_explore_area_margin.value()),
            "explore_low_nav_weight": float(spin_low_nav.value()),
            "explore_unexplored_reward_weight": float(spin_explr_w.value()),
            "explore_explored_age_weight": float(spin_explr_age.value()),
            "explore_explored_dist_weight": float(spin_explr_dist.value()),
            "danger_inspect_weight": float(spin_ins_w.value()),
            "danger_inspect_kernel_cells": int(spin_ins_k.value()),
            "danger_inspect_danger_thr": float(spin_ins_thr.value()),
            "danger_inspect_max_cell_danger": float(spin_ins_max.value()),
            "dyn_danger_inspect_weight": float(spin_dyn_ins_w.value()),
            "dyn_inspector_rt_weight": float(spin_insp_rt_w.value()),
            "dyn_inspector_rt_ttl_s": float(spin_insp_rt_ttl.value()),
            "dyn_inspector_rt_standoff_cells": int(spin_insp_standoff.value()),
            "dyn_inspector_avoid_damage_weight": float(spin_insp_avoid_w.value()),
            "dyn_inspector_avoid_damage_thr": float(spin_insp_thr.value()),
            "explore_far_density_weight": float(spin_far_w.value()),
            "explore_far_density_ring_radius_cells": int(spin_far_ring.value()),
            "explore_far_density_kernel_radius_cells": int(spin_far_kern.value()),
            "explore_far_density_angle_step_deg": float(spin_far_step.value()),
            "explore_far_density_exclude_inside_ring": bool(chk_far_excl.isChecked()),
            "explore_vector_avoid_weight": float(spin_vec_avoid.value()),
            "explore_vector_share_every_cells": int(spin_vec_share_cells.value()),
            "evap_nav_rate": float(spin_evap_nav.value()),
            "evap_danger_rate": float(spin_evap_danger.value()),
            "danger_evap_mult_static": float(spin_danger_mult_static.value()),
            "danger_evap_mult_dynamic": float(spin_danger_mult_dynamic.value()),
            "wall_danger_evap_mult": float(spin_danger_mult_wall.value()),
            "click_mode_index": int(cmb_click.currentIndex()),
            "dynamic_speed_sec_per_cell": float(spin_dyn_speed.value()),
            "danger_radius_cells": int(spin_danger_r.value()),
            "threat_height_m": float(spin_threat_h.value()),
            "pher_source_index": int(cmb_pher_source.currentIndex()),
            "pher_layer_index": int(cmb_pher_layer.currentIndex()),
            "pher_danger_kind_index": int(cmb_danger_kind.currentIndex()),
            "pher_drone_seq": int(spin_pher_drone.value()),
            "lidar_viz_enabled": bool(chk_lidar.isChecked()),
            "lidar_viz_drone_seq": int(spin_lidar_drone.value()),
            "lidar_scan_viz_enabled": bool(chk_lidar_scan.isChecked()),
            "plan_viz_enabled": bool(chk_plan.isChecked()),
            "plan_viz_all_enabled": bool(chk_plan_all.isChecked()),
            "aco_viz_enabled": bool(chk_aco.isChecked()),
            "aco_viz_all_enabled": bool(chk_aco_all.isChecked()),
            "aco_viz_arrow_width_m": float(spin_aco_w.value()),
            "aco_viz_arrow_length_mult": float(spin_aco_len.value()),
            "aco_commit_enabled": bool(chk_aco_commit.isChecked()),
            "corner_backoff_enabled": bool(chk_corner_backoff.isChecked()),
            "unstick_move_enabled": bool(chk_unstick.isChecked()),
            "aco_commit_timeout_s": float(spin_commit.value()),
            "pointer_enabled": bool(chk_ptr.isChecked()),
            "pointer_z": float(spin_ptr_z.value()),
            "pointer_scale": float(spin_ptr_scale.value()),
            "pointer_alpha": float(spin_ptr_alpha.value()),
            # Exploit spacing / path adherence
            "exploit_peer_avoid_radius_m": float(spin_ex_avoid_r.value()),
            "exploit_peer_avoid_weight": float(spin_ex_avoid_w.value()),
            "exploit_peer_avoid_fade_start_m": float(spin_ex_fade0.value()),
            "exploit_peer_avoid_fade_range_m": float(spin_ex_fade_rng.value()),
            "exploit_peer_path_follow_weight": float(spin_ex_follow_w.value()),
            "exploit_dyn_trail_overlay_strength": float(spin_ex_dyn_overlay_strength.value()),
            "exploit_dyn_trail_overlay_gamma": float(spin_ex_dyn_overlay_gamma.value()),
            "pheromone_snapshot_last_dir": str(pher_snapshot_last_dir.get("v", "") or ""),
            "pheromone_snapshot_last_path": str(pher_snapshot_last_path.get("v", "") or ""),
            # UI sections (collapsed state)
            "qt_section_mission_status_collapsed": bool(status_box.is_collapsed()),
            "qt_section_targets_controls_collapsed": bool(tgt_btn_box.is_collapsed()),
            "qt_section_python_speed_collapsed": bool(speed_box.is_collapsed()),
            "qt_section_aco_collapsed": bool(aco_box.is_collapsed()),
            "qt_section_danger_inspection_collapsed": bool(inspect_box.is_collapsed()),
            "qt_section_far_density_collapsed": bool(far_box.is_collapsed()),
            "qt_section_inspector_collapsed": bool(insp_box.is_collapsed()),
            "qt_section_pheromone_viz_collapsed": bool(pher_box.is_collapsed()),
            "qt_section_drones_viz_collapsed": bool(drones_box.is_collapsed()),
            "qt_section_click_mode_collapsed": bool(click_box.is_collapsed()),
            "qt_section_targets_viz_collapsed": bool(tgt_box.is_collapsed()),
            "qt_section_exploit_collapsed": bool(exploit_box.is_collapsed()),
            "qt_section_exploit_dyn_overlay_collapsed": bool(exploit_overlay_box.is_collapsed()),
            "qt_section_lidar_debug_collapsed": bool(lidar_box.is_collapsed()),
            "qt_section_buildings_collapsed": bool(bld_box.is_collapsed()),
            "saved_at_wall": now,
        }
        try:
            _save_settings(payload)
            dirty["v"] = False
            last_save["t"] = now
        except Exception:
            pass

    rc = app.exec_()
    stop_spin.set()
    try:
        exec_.shutdown(timeout_sec=0.2)
    except Exception:
        pass
    try:
        spin_thread.join(timeout=1.0)
    except Exception:
        pass
    for n in (node, danger, ground, buildings, fast_sim):
        try:
            exec_.remove_node(n)
        except Exception:
            pass
        try:
            n.destroy_node()
        except Exception:
            pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    return int(rc)


def _main_tk(argv):
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        print("Neither PyQt5 nor Tkinter is available, cannot start GUI.")
        print(f"Tkinter import error: {e}")
        return 2

    rclpy.init(args=argv)
    # Same-process nodes
    from scripts.python_sim.python_fast_sim import PythonFastSim
    from scripts.publishers.publish_gazebo_buildings import GazeboBuildingsPublisher
    from scripts.publishers.publish_ground_plane import GroundPlanePublisher
    from scripts.danger.danger_map_manager import DangerMapManager

    exec_ = MultiThreadedExecutor(num_threads=4)
    fast_sim = PythonFastSim()
    buildings = GazeboBuildingsPublisher()
    ground = GroundPlanePublisher()
    danger = DangerMapManager()
    node = SwarmControlNode()
    for n in (fast_sim, buildings, ground, danger, node):
        exec_.add_node(n)

    # Tk needs these helpers early (pheromone selector uses them before the ACO section).
    def _set_fast_sim_param_double(name: str, value: float):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.DOUBLE, float(value))])
        except Exception:
            pass

    def _set_fast_sim_param_int(name: str, value: int):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.INTEGER, int(value))])
        except Exception:
            pass

    def _set_fast_sim_param_str(name: str, value: str):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.STRING, str(value))])
        except Exception:
            pass

    # Spin ROS executor in a background thread so Tk UI stays responsive.
    # IMPORTANT: use exec_.spin() (not spin_once) to avoid starving nodes/timers.
    stop_spin = threading.Event()

    def _spin_loop():
        try:
            exec_.spin()
        except Exception as e:
            try:
                print("ROS executor thread crashed:", repr(e), file=sys.stderr)
                traceback.print_exc()
            except Exception:
                pass

    spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title("Stigmergy Lab")
    # Avoid Ubuntu hover showing "tk" by setting a meaningful WM_CLASS / icon name
    try:
        root.wm_class("Stigmergy Lab", "Stigmergy Lab")
    except Exception:
        pass
    try:
        root.iconname("Stigmergy Lab")
    except Exception:
        pass
    # Window icon (logo). Do not render the logo inside the app (it would take too much space).
    try:
        repo_root = Path(__file__).resolve().parents[2]
        logo_path = repo_root / "data" / "gui-app-logo.png"
        if logo_path.exists():
            img = tk.PhotoImage(file=str(logo_path))
            root.iconphoto(True, img)
            # keep reference
            root._swarm_logo = img
    except Exception:
        pass
    base_row = 0

    # Scrollable content (settings have grown a lot)
    window = root
    outer = ttk.Frame(window)
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, highlightthickness=0)
    vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)
    vbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    content = ttk.Frame(canvas)
    content_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _on_content_configure(_evt=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(evt):
        # Keep inner frame width synced to canvas width
        try:
            canvas.itemconfigure(content_id, width=evt.width)
        except Exception:
            pass

    content.bind("<Configure>", _on_content_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(evt):
        # Windows / macOS
        try:
            if evt.delta:
                canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        except Exception:
            pass

    def _on_mousewheel_linux_up(_evt):
        canvas.yview_scroll(-3, "units")

    def _on_mousewheel_linux_down(_evt):
        canvas.yview_scroll(3, "units")

    # Mouse wheel support (cross-platform)
    try:
        window.bind_all("<MouseWheel>", _on_mousewheel)
        window.bind_all("<Button-4>", _on_mousewheel_linux_up)
        window.bind_all("<Button-5>", _on_mousewheel_linux_down)
    except Exception:
        pass

    # From this point on, build UI into the scrollable frame.
    root = content

    # Top status
    lbl_clock = ttk.Label(root, text="Sim time: —", font=("Sans", 14))
    lbl_clock.grid(row=base_row + 0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 2))

    lbl_clock_small = ttk.Label(root, text="Timestamp: —", font=("Sans", 9))
    lbl_clock_small.grid(row=base_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))

    lbl_phase = ttk.Label(root, text="Phase: —", font=("Sans", 11))
    lbl_phase.grid(row=base_row + 2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 10))

    # Mission status (targets + drones needing info)
    ttk.Label(root, text="Mission status", font=("Sans", 11, "bold")).grid(row=base_row + 3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 4))
    lbl_targets = ttk.Label(root, text="Targets: —")
    lbl_targets.grid(row=base_row + 4, column=0, columnspan=3, sticky="w", padx=10)
    txt_drones = tk.Text(root, height=6, width=60)
    txt_drones.grid(row=base_row + 5, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 8))
    txt_drones.insert("1.0", "Drones missing target knowledge / not found targets will appear here.")
    txt_drones.configure(state="disabled")

    # Buttons
    btn_start = ttk.Button(root, text="Start Python Drones", command=node.call_start)
    btn_stop = ttk.Button(root, text="Return to Base", command=node.call_stop)
    paused_state = {"v": False}

    def _toggle_pause():
        paused_state["v"] = not paused_state["v"]
        node.set_paused(paused_state["v"])
        btn_pause.configure(text="Resume" if paused_state["v"] else "Pause")

    btn_pause = ttk.Button(root, text="Pause", command=_toggle_pause)
    # Target maintenance helpers
    tgt_box = ttk.LabelFrame(root, text="Targets")
    tgt_btn_row = ttk.Frame(tgt_box)
    btn_clear_found = ttk.Button(tgt_btn_row, text="Clear FOUND Targets", command=node.call_clear_targets_found)
    btn_clear_unfound = ttk.Button(tgt_btn_row, text="Clear UNFOUND Targets", command=node.call_clear_targets_unfound)
    btn_set_unfound = ttk.Button(tgt_btn_row, text="Set ALL Targets UNFOUND", command=node.call_set_all_targets_unfound)
    btn_clear = ttk.Button(tgt_btn_row, text="Clear ALL Targets", command=node.call_clear_targets)
    btn_del_nearest = ttk.Button(tgt_btn_row, text="Delete nearest target (pose)", command=node.call_delete_nearest_target)
    # Tooltips (top control section only)
    _TkTooltip(btn_start, _tooltip_text_start())
    _TkTooltip(btn_stop, _tooltip_text_stop())
    _TkTooltip(btn_pause, _tooltip_text_pause())
    _TkTooltip(btn_clear_found, _tooltip_text_clear_targets_found())
    _TkTooltip(btn_clear_unfound, _tooltip_text_clear_targets_unfound())
    _TkTooltip(btn_set_unfound, _tooltip_text_set_all_targets_unfound())
    _TkTooltip(btn_clear, _tooltip_text_clear_targets())
    _TkTooltip(btn_del_nearest, _tooltip_text_delete_nearest_target())
    btn_start.grid(row=base_row + 6, column=0, padx=10, pady=5, sticky="ew")
    btn_stop.grid(row=base_row + 6, column=1, padx=10, pady=5, sticky="ew")
    btn_pause.grid(row=base_row + 6, column=2, padx=10, pady=5, sticky="ew")
    tgt_box.grid(row=base_row + 7, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
    tgt_btn_row.pack(fill="x", expand=True, padx=6, pady=6)
    for b in (btn_clear_found, btn_clear_unfound, btn_set_unfound, btn_clear, btn_del_nearest):
        try:
            b.pack(side="left", fill="x", expand=True, padx=3)
        except Exception:
            pass

    # Speed + drone scale
    ttk.Separator(root, orient="horizontal").grid(row=base_row + 8, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 10))
    ttk.Label(root, text="Python speed (time multiplier)").grid(row=base_row + 9, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    speed_var = tk.DoubleVar(value=10.0)
    lbl_speed = ttk.Label(root, text="Speed: 10.0×")
    lbl_speed.grid(row=base_row + 10, column=0, sticky="w", padx=10)

    def _on_speed_change(_v=None):
        v = float(speed_var.get())
        v = max(0.1, min(200.0, v))
        lbl_speed.configure(text=f"Speed: {v:.1f}×")
        node.set_speed(v)

    # scale (0.1 .. 200.0)
    speed_scale = ttk.Scale(root, from_=0.1, to=200.0, orient="horizontal", variable=speed_var, command=_on_speed_change)
    speed_scale.grid(row=base_row + 10, column=1, columnspan=2, sticky="ew", padx=10)

    # Return speed (m/s)
    ttk.Label(root, text="Return speed (m/s)").grid(row=base_row + 11, column=0, sticky="w", padx=10)
    return_speed = tk.DoubleVar(value=10.0)
    lbl_return = ttk.Label(root, text="10.0 m/s")
    lbl_return.grid(row=base_row + 11, column=1, sticky="w")

    def _on_return_speed(_v=None):
        v = float(return_speed.get())
        v = max(0.1, min(50.0, v))
        lbl_return.configure(text=f"{v:.1f} m/s")
        node.set_return_speed_mps(v)

    ttk.Scale(root, from_=0.1, to=50.0, orient="horizontal", variable=return_speed, command=_on_return_speed).grid(row=base_row + 11, column=2, sticky="ew", padx=10)

    # Drone altitude (m)
    ttk.Label(root, text="Drone altitude (m)").grid(row=base_row + 12, column=0, sticky="w", padx=10)
    drone_alt = tk.DoubleVar(value=8.0)
    lbl_alt = ttk.Label(root, text="8.0 m")
    lbl_alt.grid(row=base_row + 12, column=1, sticky="w")

    def _on_drone_alt(_v=None):
        z = float(drone_alt.get())
        z = max(-50.0, min(500.0, z))
        lbl_alt.configure(text=f"{z:.1f} m")
        node.set_drone_altitude_m(z)

    ttk.Scale(root, from_=-50.0, to=500.0, orient="horizontal", variable=drone_alt, command=_on_drone_alt).grid(row=base_row + 12, column=2, sticky="ew", padx=10)

    # LiDAR sense radius (m) (affects mock lidar + local planning)
    ttk.Label(root, text="LiDAR sense radius (m)").grid(row=base_row + 12 + 1, column=0, sticky="w", padx=10)
    sense_radius = tk.DoubleVar(value=75.0)
    lbl_sense = ttk.Label(root, text="75 m")
    lbl_sense.grid(row=base_row + 12 + 1, column=1, sticky="w")

    def _on_sense(_v=None):
        r = float(sense_radius.get())
        r = max(5.0, min(200.0, r))
        lbl_sense.configure(text=f"{r:.0f} m")
        _set_fast_sim_param_double("sense_radius", r)
        _set_fast_sim_param_double("local_plan_radius_m", r)

    ttk.Scale(root, from_=5.0, to=200.0, orient="horizontal", variable=sense_radius, command=_on_sense).grid(
        row=base_row + 12 + 1, column=2, sticky="ew", padx=10
    )

    ttk.Label(root, text="Drone size").grid(row=base_row + 13, column=0, sticky="w", padx=10)
    drone_scale = tk.DoubleVar(value=1.0)
    lbl_drone = ttk.Label(root, text="1.00×")
    lbl_drone.grid(row=base_row + 13, column=1, sticky="w")

    def _on_drone_scale(_v=None):
        s = float(drone_scale.get())
        s = max(0.1, min(3.0, s))
        lbl_drone.configure(text=f"{s:.2f}×")
        node.set_drone_marker_scale(s)

    ttk.Scale(root, from_=0.1, to=3.0, orient="horizontal", variable=drone_scale, command=_on_drone_scale).grid(row=base_row + 13, column=2, sticky="ew", padx=10)

    # ACO tuning (simulation behavior)
    ttk.Separator(root, orient="horizontal").grid(row=base_row + 13 + 1, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="ACO (exploration tuning)", font=("Sans", 11, "bold")).grid(row=base_row + 13 + 2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    ttk.Label(
        root,
        text=(
            "These sliders control how exploration decisions are made (no physics / no A*).\n"
            "- If drones loiter near base: increase Min explore radius or Recent-cell penalty.\n"
            "- If they feel too forced outward: reduce Min-radius strength."
        ),
        foreground="#666",
        wraplength=640,
    ).grid(row=base_row + 13 + 3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

    def _set_fast_sim_param_double(name: str, value: float):
        try:
            fast_sim.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.DOUBLE, float(value))])
        except Exception:
            pass

    # aco_temperature
    aco_temp = tk.DoubleVar(value=0.70)
    ttk.Label(root, text="ACO temperature").grid(row=base_row + 13 + 4, column=0, sticky="w", padx=10)
    lbl_aco_temp = ttk.Label(root, text="0.70")
    lbl_aco_temp.grid(row=base_row + 13 + 4, column=1, sticky="w")

    def _on_aco_temp(_v=None):
        t = float(aco_temp.get())
        t = max(0.01, min(3.0, t))
        lbl_aco_temp.configure(text=f"{t:.2f}")
        _set_fast_sim_param_double("aco_temperature", t)

    ttk.Scale(root, from_=0.01, to=3.0, orient="horizontal", variable=aco_temp, command=_on_aco_temp).grid(row=base_row + 13 + 4, column=2, sticky="ew", padx=10)

    # explore min radius
    min_r = tk.DoubleVar(value=200.0)
    ttk.Label(root, text="Min explore radius (m)").grid(row=base_row + 13 + 5, column=0, sticky="w", padx=10)
    lbl_min_r = ttk.Label(root, text="200")
    lbl_min_r.grid(row=base_row + 13 + 5, column=1, sticky="w")

    def _on_min_r(_v=None):
        r = float(min_r.get())
        r = max(0.0, min(5000.0, r))
        lbl_min_r.configure(text=f"{r:.0f}")
        _set_fast_sim_param_double("explore_min_radius_m", r)

    ttk.Scale(root, from_=0.0, to=5000.0, orient="horizontal", variable=min_r, command=_on_min_r).grid(row=base_row + 13 + 5, column=2, sticky="ew", padx=10)

    # explore min radius strength
    min_s = tk.DoubleVar(value=10.0)
    ttk.Label(root, text="Min-radius strength").grid(row=base_row + 13 + 6, column=0, sticky="w", padx=10)
    lbl_min_s = ttk.Label(root, text="10.0")
    lbl_min_s.grid(row=base_row + 13 + 6, column=1, sticky="w")

    def _on_min_s(_v=None):
        s = float(min_s.get())
        s = max(0.0, min(30.0, s))
        lbl_min_s.configure(text=f"{s:.1f}")
        _set_fast_sim_param_double("explore_min_radius_strength", s)

    ttk.Scale(root, from_=0.0, to=30.0, orient="horizontal", variable=min_s, command=_on_min_s).grid(row=base_row + 13 + 6, column=2, sticky="ew", padx=10)

    # recent cell penalty
    recent_pen = tk.DoubleVar(value=2.0)
    ttk.Label(root, text="Recent-cell penalty").grid(row=base_row + 13 + 7, column=0, sticky="w", padx=10)
    lbl_recent_pen = ttk.Label(root, text="2.0")
    lbl_recent_pen.grid(row=base_row + 13 + 7, column=1, sticky="w")

    def _on_recent_pen(_v=None):
        p = float(recent_pen.get())
        p = max(0.0, min(10.0, p))
        lbl_recent_pen.configure(text=f"{p:.1f}")
        _set_fast_sim_param_double("recent_cell_penalty", p)

    ttk.Scale(root, from_=0.0, to=10.0, orient="horizontal", variable=recent_pen, command=_on_recent_pen).grid(row=base_row + 13 + 7, column=2, sticky="ew", padx=10)

    # base push knobs
    basepush_r = tk.DoubleVar(value=60.0)
    basepush_s = tk.DoubleVar(value=4.0)
    ttk.Label(root, text="Base push radius (m)").grid(row=base_row + 13 + 8, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=500.0, orient="horizontal", variable=basepush_r, command=lambda _v: _set_fast_sim_param_double("base_push_radius_m", float(basepush_r.get()))).grid(row=base_row + 13 + 8, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Base push strength").grid(row=base_row + 13 + 9, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=20.0, orient="horizontal", variable=basepush_s, command=lambda _v: _set_fast_sim_param_double("base_push_strength", float(basepush_s.get()))).grid(row=base_row + 13 + 9, column=1, columnspan=2, sticky="ew", padx=10)

    # evaporation knobs
    evap_nav = tk.DoubleVar(value=0.002)
    evap_danger = tk.DoubleVar(value=0.001)
    danger_mult_static = tk.DoubleVar(value=1.0)
    danger_mult_dynamic = tk.DoubleVar(value=1.25)
    wall_mult = tk.DoubleVar(value=0.02)
    ttk.Label(root, text="Evap nav rate").grid(row=base_row + 13 + 10, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=0.02, orient="horizontal", variable=evap_nav, command=lambda _v: _set_fast_sim_param_double("evap_nav_rate", float(evap_nav.get()))).grid(row=base_row + 13 + 10, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Evap danger rate").grid(row=base_row + 13 + 11, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=0.02, orient="horizontal", variable=evap_danger, command=lambda _v: _set_fast_sim_param_double("evap_danger_rate", float(evap_danger.get()))).grid(row=base_row + 13 + 11, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Danger mult static").grid(row=base_row + 13 + 12, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=5.0, orient="horizontal", variable=danger_mult_static, command=lambda _v: _set_fast_sim_param_double("danger_evap_mult_static", float(danger_mult_static.get()))).grid(row=base_row + 13 + 12, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Danger mult dynamic").grid(row=base_row + 13 + 13, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=5.0, orient="horizontal", variable=danger_mult_dynamic, command=lambda _v: _set_fast_sim_param_double("danger_evap_mult_dynamic", float(danger_mult_dynamic.get()))).grid(row=base_row + 13 + 13, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Wall mult (nav_danger)").grid(row=base_row + 13 + 14, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=wall_mult, command=lambda _v: _set_fast_sim_param_double("wall_danger_evap_mult", float(wall_mult.get()))).grid(row=base_row + 13 + 14, column=1, columnspan=2, sticky="ew", padx=10)

    # Danger inspection curiosity (EXPLORE only)
    ttk.Label(root, text="Danger inspection curiosity (EXPLORE)", font=("Sans", 10, "bold")).grid(
        row=base_row + 13 + 12, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 2)
    )
    ttk.Label(
        root,
        text="Rewards revealing unexplored cells adjacent to known danger (boundary tracing). Turns off once explored.",
        foreground="#666",
        wraplength=640,
    ).grid(row=base_row + 13 + 13, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

    danger_ins_w = tk.DoubleVar(value=0.0)
    danger_ins_k = tk.IntVar(value=3)
    danger_ins_thr = tk.DoubleVar(value=0.35)
    danger_ins_max = tk.DoubleVar(value=0.6)
    # Inspector (dynamic danger)
    insp_follow_w = tk.DoubleVar(value=8.0)          # historical kernel-follow weight
    insp_rt_w = tk.DoubleVar(value=12.0)             # realtime kernel-follow (LiDAR) weight
    insp_rt_ttl = tk.DoubleVar(value=2.0)            # seconds
    insp_rt_standoff = tk.IntVar(value=1)            # cells
    insp_avoid_w = tk.DoubleVar(value=25.0)
    insp_avoid_thr = tk.DoubleVar(value=0.05)

    def _on_danger_inspect(_v=None):
        _set_fast_sim_param_double("danger_inspect_weight", float(danger_ins_w.get()))
        _set_fast_sim_param_int("danger_inspect_kernel_cells", int(danger_ins_k.get()))
        _set_fast_sim_param_double("danger_inspect_danger_thr", float(danger_ins_thr.get()))
        _set_fast_sim_param_double("danger_inspect_max_cell_danger", float(danger_ins_max.get()))

    ttk.Label(root, text="Weight (0 disables)").grid(row=base_row + 13 + 14, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=20.0, orient="horizontal", variable=danger_ins_w, command=_on_danger_inspect).grid(
        row=base_row + 13 + 14, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Kernel (cells) / Danger thr").grid(row=base_row + 13 + 15, column=0, sticky="w", padx=10)
    ttk.Spinbox(root, from_=0, to=50, textvariable=danger_ins_k, width=6, command=_on_danger_inspect).grid(
        row=base_row + 13 + 15, column=1, sticky="w", padx=(10, 4)
    )
    ttk.Scale(root, from_=0.0, to=2.0, orient="horizontal", variable=danger_ins_thr, command=_on_danger_inspect).grid(
        row=base_row + 13 + 15, column=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Max cell danger (apply bonus only below)").grid(row=base_row + 13 + 16, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=2.0, orient="horizontal", variable=danger_ins_max, command=_on_danger_inspect).grid(
        row=base_row + 13 + 16, column=1, columnspan=2, sticky="ew", padx=10
    )

    ttk.Label(root, text="Inspector (dynamic danger)", font=("Sans", 10, "bold")).grid(
        row=base_row + 13 + 17, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 2)
    )
    ttk.Label(
        root,
        text=(
            "Only ONE drone acts as inspector per dynamic danger id.\n"
            "Inspector behavior:\n"
            "- follows the dynamic danger KERNEL to reconstruct its full loop (fills holes)\n"
            "- ignores normal exploration incentives (unknown/unexplored + nav pheromones)\n"
            "- avoids stepping into the DAMAGE radius when damage is strong enough (thresholded)\n"
            "\n"
            "Weights:\n"
            "- Kernel follow (historical): attraction to last-known kernel from pheromone sharing\n"
            "- Kernel follow (realtime): LiDAR-based reward around currently seen kernel (standoff ring)\n"
            "- Realtime TTL: how long the realtime reward stays active after last LiDAR kernel sighting\n"
            "- Realtime standoff (cells): preferred offset outside estimated damage radius\n"
            "- Avoid damage weight: penalty for stepping into damage while inspecting\n"
            "- Damage threshold: ignore weak damage; apply avoid only when danger(cell) ≥ thr\n"
            "\n"
            "Inspector ACO score (per candidate next cell c):\n"
            "score(c) = base_ACO(c)\n"
            "  + w_hist * (1 - dist(c, kernel_hist)/sense_radius)\n"
            "  + w_rt   * (1 - |dist(c, kernel_rt) - d_target|/d_target)   [only if LiDAR-seen within TTL]\n"
            "  - w_avoid  if danger(c) >= thr and c is inside dynamic damage\n"
            "\n"
            "d_target = (damage_radius_cells + standoff_cells) * cell_size\n"
        ),
        foreground="#666",
        wraplength=640,
    ).grid(row=base_row + 13 + 18, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

    ttk.Label(root, text="Kernel follow (historical)").grid(row=base_row + 13 + 19, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=100.0, orient="horizontal", variable=insp_follow_w, command=lambda _v: _set_fast_sim_param_double("dyn_danger_inspect_weight", float(insp_follow_w.get()))).grid(
        row=base_row + 13 + 19, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Kernel follow (realtime)").grid(row=base_row + 13 + 20, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=100.0, orient="horizontal", variable=insp_rt_w, command=lambda _v: _set_fast_sim_param_double("dyn_inspector_rt_weight", float(insp_rt_w.get()))).grid(
        row=base_row + 13 + 20, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Realtime TTL (s)").grid(row=base_row + 13 + 21, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=10.0, orient="horizontal", variable=insp_rt_ttl, command=lambda _v: _set_fast_sim_param_double("dyn_inspector_rt_ttl_s", float(insp_rt_ttl.get()))).grid(
        row=base_row + 13 + 21, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Realtime standoff (cells)").grid(row=base_row + 13 + 22, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=10.0, orient="horizontal", variable=insp_rt_standoff, command=lambda _v: _set_fast_sim_param_int("dyn_inspector_rt_standoff_cells", int(insp_rt_standoff.get()))).grid(
        row=base_row + 13 + 22, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Avoid damage weight").grid(row=base_row + 13 + 23, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=100.0, orient="horizontal", variable=insp_avoid_w, command=lambda _v: _set_fast_sim_param_double("dyn_inspector_avoid_damage_weight", float(insp_avoid_w.get()))).grid(
        row=base_row + 13 + 23, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Damage threshold").grid(row=base_row + 13 + 24, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=2.0, orient="horizontal", variable=insp_avoid_thr, command=lambda _v: _set_fast_sim_param_double("dyn_inspector_avoid_damage_thr", float(insp_avoid_thr.get()))).grid(
        row=base_row + 13 + 24, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(
        root,
        text="Tip: avoid-damage triggers only when danger(cell) ≥ threshold.",
        foreground="#666",
    ).grid(row=base_row + 13 + 25, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 0))

    # Drone pointers
    ptr_enabled = tk.BooleanVar(value=True)
    ptr_z = tk.DoubleVar(value=8.0)
    ptr_scale = tk.DoubleVar(value=1.0)
    ptr_alpha = tk.DoubleVar(value=0.9)

    def _publish_ptr(_v=None):
        node.set_drone_pointer_params(bool(ptr_enabled.get()), float(ptr_z.get()), float(ptr_scale.get()), float(ptr_alpha.get()))

    ttk.Checkbutton(root, text="Show pointers", variable=ptr_enabled, command=_publish_ptr).grid(row=base_row + 14, column=0, sticky="w", padx=10)
    ttk.Label(root, text="Z").grid(row=base_row + 14, column=1, sticky="e")
    ttk.Scale(root, from_=-50.0, to=200.0, orient="horizontal", variable=ptr_z, command=_publish_ptr).grid(row=base_row + 14, column=2, sticky="ew", padx=10)
    ttk.Label(root, text="Size").grid(row=base_row + 15, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.1, to=5.0, orient="horizontal", variable=ptr_scale, command=_publish_ptr).grid(row=base_row + 15, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Alpha").grid(row=base_row + 16, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=ptr_alpha, command=_publish_ptr).grid(row=base_row + 16, column=1, columnspan=2, sticky="ew", padx=10)

    # Exploration area (radius-limited visibility + coordinated exploration)
    ttk.Separator(root, orient="horizontal").grid(row=base_row + 13 + 17, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 8))
    ttk.Label(root, text="Exploration area (radius)", font=("Sans", 11, "bold")).grid(row=base_row + 13 + 18, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    explore_area_r = tk.DoubleVar(value=0.0)
    explore_area_margin = tk.DoubleVar(value=30.0)
    low_nav_w = tk.DoubleVar(value=0.0)
    vec_avoid_w = tk.DoubleVar(value=0.0)
    vec_share_cells = tk.IntVar(value=3)

    def _on_explore_area(_v=None):
        _set_fast_sim_param_double("exploration_area_radius_m", float(explore_area_r.get()))
        _set_fast_sim_param_double("exploration_radius_margin_m", float(explore_area_margin.get()))
        _set_fast_sim_param_double("explore_low_nav_weight", float(low_nav_w.get()))
        _set_fast_sim_param_double("explore_vector_avoid_weight", float(vec_avoid_w.get()))
        _set_fast_sim_param_int("explore_vector_share_every_cells", int(vec_share_cells.get()))

    ttk.Label(root, text="Area radius (m) (0 disables)").grid(row=base_row + 13 + 19, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=20000.0, orient="horizontal", variable=explore_area_r, command=_on_explore_area).grid(row=base_row + 13 + 19, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Edge margin (m)").grid(row=base_row + 13 + 20, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=500.0, orient="horizontal", variable=explore_area_margin, command=_on_explore_area).grid(row=base_row + 13 + 20, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Low-nav reward").grid(row=base_row + 13 + 21, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=10.0, orient="horizontal", variable=low_nav_w, command=_on_explore_area).grid(row=base_row + 13 + 21, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Avoid peer-vector weight / share every N cells").grid(row=base_row + 13 + 22, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=10.0, orient="horizontal", variable=vec_avoid_w, command=_on_explore_area).grid(row=base_row + 13 + 22, column=1, sticky="ew", padx=(10, 4))
    ttk.Spinbox(root, from_=1, to=100, textvariable=vec_share_cells, width=6, command=_on_explore_area).grid(row=base_row + 13 + 22, column=2, sticky="e", padx=10)

    # Click mode (exclusive owner of RViz /clicked_point)
    # Click mode (exclusive owner of RViz /clicked_point)
    click_sep_row = base_row + 13 + 23
    ttk.Separator(root, orient="horizontal").grid(row=click_sep_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="Click mode (uses RViz /clicked_point) — exclusive").grid(row=click_sep_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    click_mode = tk.StringVar(value=ClickMode.NONE)

    def _apply_click_mode_tk():
        node.set_click_mode(click_mode.get())

    ttk.Radiobutton(root, text="None", value=ClickMode.NONE, variable=click_mode, command=_apply_click_mode_tk).grid(row=click_sep_row + 2, column=0, sticky="w", padx=10)
    ttk.Radiobutton(root, text="Targets", value=ClickMode.TARGETS, variable=click_mode, command=_apply_click_mode_tk).grid(row=click_sep_row + 2, column=1, sticky="w", padx=10)
    ttk.Radiobutton(root, text="Danger static", value=ClickMode.DANGER_STATIC, variable=click_mode, command=_apply_click_mode_tk).grid(row=click_sep_row + 2, column=2, sticky="w", padx=10)
    ttk.Radiobutton(root, text="Danger dynamic", value=ClickMode.DANGER_DYNAMIC, variable=click_mode, command=_apply_click_mode_tk).grid(row=click_sep_row + 3, column=0, sticky="w", padx=10)

    ttk.Label(root, text="Dynamic speed (sec/cell)").grid(row=click_sep_row + 4, column=0, sticky="w", padx=10)
    dyn_speed = tk.DoubleVar(value=4.0)
    ttk.Scale(root, from_=0.1, to=60.0, orient="horizontal", variable=dyn_speed, command=lambda _v: setattr(node, "dynamic_speed_sec_per_cell", float(dyn_speed.get()))).grid(row=click_sep_row + 4, column=1, columnspan=2, sticky="ew", padx=10)
    ttk.Label(root, text="Danger radius (cells)").grid(row=click_sep_row + 5, column=0, sticky="w", padx=10)
    danger_r = tk.IntVar(value=0)
    ttk.Scale(root, from_=0, to=50, orient="horizontal", variable=danger_r, command=lambda _v: setattr(node, "danger_radius_cells", int(danger_r.get()))).grid(
        row=click_sep_row + 5, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Label(root, text="Threat height (m)").grid(row=click_sep_row + 6, column=0, sticky="w", padx=10)
    threat_h = tk.DoubleVar(value=50.0)
    ttk.Scale(root, from_=0.0, to=200.0, orient="horizontal", variable=threat_h, command=lambda _v: setattr(node, "threat_height_m", float(threat_h.get()))).grid(
        row=click_sep_row + 6, column=1, columnspan=2, sticky="ew", padx=10
    )
    ttk.Button(root, text="Start recording", command=node.start_dynamic_recording).grid(row=click_sep_row + 7, column=0, padx=10, pady=2, sticky="ew")
    ttk.Button(root, text="Stop + create dynamic danger", command=node.stop_dynamic_recording).grid(row=click_sep_row + 7, column=1, columnspan=2, padx=10, pady=2, sticky="ew")

    # Targets viz (visual-only)
    targets_sep_row = click_sep_row + 8
    ttk.Separator(root, orient="horizontal").grid(row=targets_sep_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="Targets (visualization)").grid(row=targets_sep_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    tgt_d = tk.DoubleVar(value=10.0)
    tgt_a = tk.DoubleVar(value=0.30)
    lbl_tgt = ttk.Label(root, text="Diameter: 10.0m, alpha=0.30")
    lbl_tgt.grid(row=targets_sep_row + 2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

    def _publish_tgt_params(_v=None):
        d = float(tgt_d.get())
        a = float(tgt_a.get())
        d = max(0.5, min(200.0, d))
        a = max(0.0, min(1.0, a))
        lbl_tgt.configure(text=f"Diameter: {d:.1f}m, alpha={a:.2f}")
        node.set_target_viz_params(d, a)

    ttk.Label(root, text="Diameter (m)").grid(row=targets_sep_row + 3, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.5, to=200.0, orient="horizontal", variable=tgt_d, command=_publish_tgt_params).grid(row=targets_sep_row + 3, column=1, columnspan=2, sticky="ew", padx=10)

    ttk.Label(root, text="Alpha").grid(row=targets_sep_row + 4, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=tgt_a, command=_publish_tgt_params).grid(row=targets_sep_row + 4, column=1, columnspan=2, sticky="ew", padx=10)

    # Pheromone map section
    pher_sep_row = targets_sep_row + 5
    ttk.Separator(root, orient="horizontal").grid(row=pher_sep_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="Pheromone map (visualization)").grid(row=pher_sep_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))

    pher_enabled = tk.BooleanVar(value=True)

    def _on_pher_toggle():
        node.set_pheromone_viz_enabled(bool(pher_enabled.get()))
        _publish_pher_params()

    chk_pher = ttk.Checkbutton(root, text="Enable pheromone map (/pheromone_heatmap)", variable=pher_enabled, command=_on_pher_toggle)
    chk_pher.grid(row=pher_sep_row + 2, column=0, columnspan=2, sticky="w", padx=10, pady=2)

    ttk.Label(root, text="(auto-publishes on GUI start)").grid(row=pher_sep_row + 2, column=2, padx=10, pady=2, sticky="w")

    pher_disp = tk.DoubleVar(value=2000.0)
    pher_z = tk.DoubleVar(value=0.10)
    pher_a = tk.DoubleVar(value=0.10)

    lbl_pher = ttk.Label(root, text="Display size: 2000m, z=0.10, alpha=0.10")
    lbl_pher.grid(row=pher_sep_row + 3, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 6))

    def _publish_pher_params(_v=None):
        disp = float(pher_disp.get())
        z = float(pher_z.get())
        a = float(pher_a.get())
        lbl_pher.configure(text=f"Display size: {disp:.0f}m, z={z:.2f}, alpha={a:.2f}")
        node.set_pheromone_viz_params(disp, z, a)

    ttk.Label(root, text="Display size (m)").grid(row=pher_sep_row + 4, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=50.0, to=11000.0, orient="horizontal", variable=pher_disp, command=_publish_pher_params).grid(row=pher_sep_row + 4, column=1, columnspan=2, sticky="ew", padx=10)

    ttk.Label(root, text="Z (m)").grid(row=pher_sep_row + 5, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=-50.0, to=200.0, orient="horizontal", variable=pher_z, command=_publish_pher_params).grid(row=pher_sep_row + 5, column=1, columnspan=2, sticky="ew", padx=10)

    ttk.Label(root, text="Alpha").grid(row=pher_sep_row + 6, column=0, sticky="w", padx=10)
    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=pher_a, command=_publish_pher_params).grid(row=pher_sep_row + 6, column=1, columnspan=2, sticky="ew", padx=10)

    # Pheromone selection (what we publish to RViz) — Base / Combined / Drone and layer
    ttk.Label(root, text="Source / Layer").grid(row=pher_sep_row + 7, column=0, sticky="w", padx=10)
    pher_owner = tk.StringVar(value="base")
    pher_layer = tk.StringVar(value="danger")
    danger_kind = tk.StringVar(value="all")
    pher_drone_seq = tk.IntVar(value=1)

    def _publish_pher_select(_v=None):
        owner = pher_owner.get()
        layer = pher_layer.get()
        dk = str(danger_kind.get()).strip().lower() or "all"
        seq = int(pher_drone_seq.get())
        if owner not in ("base", "combined", "drone"):
            owner = "base"
        if layer not in ("danger", "nav", "empty", "explored"):
            layer = "danger"
        seq = max(1, min(200, seq))
        # Publish for external tooling, but ALSO apply directly via parameters so it can't be delayed
        # by executor timing when the sim loop is heavy.
        node.set_pheromone_viz_select(owner, layer, seq)
        _set_fast_sim_param_str("pheromone_viz_owner", owner)
        _set_fast_sim_param_str("pheromone_viz_layer", layer)
        _set_fast_sim_param_str("pheromone_viz_danger_kind", dk)
        _set_fast_sim_param_int("pheromone_viz_drone_seq", seq)
        try:
            node.latest_log = f"pheromone_viz_select -> owner={owner}, layer={layer}, drone_seq={seq}"
        except Exception:
            pass

    ttk.OptionMenu(root, pher_owner, pher_owner.get(), "base", "combined", "drone", command=_publish_pher_select).grid(row=pher_sep_row + 7, column=1, sticky="ew", padx=10)
    ttk.OptionMenu(root, pher_layer, pher_layer.get(), "danger", "nav", "empty", "explored", command=_publish_pher_select).grid(row=pher_sep_row + 7, column=2, sticky="ew", padx=10)
    ttk.OptionMenu(
        root,
        danger_kind,
        danger_kind.get(),
        "all",
        "nav_danger",
        "danger_static",
        "danger_dyn_kernel",
        "danger_dyn_damage",
        "danger_dyn_kernel,danger_dyn_damage",
        command=_publish_pher_select,
    ).grid(row=pher_sep_row + 8, column=2, sticky="ew", padx=10)
    ttk.Label(root, text="Drone #").grid(row=pher_sep_row + 8, column=0, sticky="w", padx=10)
    ttk.Spinbox(root, from_=1, to=200, textvariable=pher_drone_seq, width=6, command=_publish_pher_select).grid(row=pher_sep_row + 8, column=1, sticky="w", padx=10)
    pher_drone_seq.trace_add("write", lambda *_a: _publish_pher_select())

    # Altitude safety params (sim): inserted here to avoid renumbering the whole base section.
    ttk.Label(root, text="Min flight altitude (m)").grid(row=pher_sep_row + 9, column=0, sticky="w", padx=10)
    min_alt = tk.DoubleVar(value=5.0)
    ttk.Scale(root, from_=0.0, to=200.0, orient="horizontal", variable=min_alt, command=lambda _v=None: _set_fast_sim_param_double("min_flight_altitude_m", float(min_alt.get()))).grid(
        row=pher_sep_row + 9, column=1, columnspan=2, sticky="ew", padx=10
    )

    ttk.Label(root, text="Roof clearance margin (m)").grid(row=pher_sep_row + 10, column=0, sticky="w", padx=10)
    roof_margin = tk.DoubleVar(value=3.0)
    ttk.Scale(root, from_=0.0, to=50.0, orient="horizontal", variable=roof_margin, command=lambda _v=None: _set_fast_sim_param_double("roof_clearance_margin_m", float(roof_margin.get()))).grid(
        row=pher_sep_row + 10, column=1, columnspan=2, sticky="ew", padx=10
    )

    # Overfly / vertical cost multipliers
    ttk.Label(root, text="Vertical cost × (A* overfly)").grid(row=pher_sep_row + 11, column=0, sticky="w", padx=10)
    overfly_cost = tk.DoubleVar(value=3.0)
    ttk.Scale(root, from_=0.1, to=50.0, orient="horizontal", variable=overfly_cost, command=lambda _v=None: _set_fast_sim_param_double("overfly_vertical_cost_mult", float(overfly_cost.get()))).grid(
        row=pher_sep_row + 11, column=1, columnspan=2, sticky="ew", padx=10
    )

    ttk.Label(root, text="Vertical battery ×").grid(row=pher_sep_row + 12, column=0, sticky="w", padx=10)
    vert_energy = tk.DoubleVar(value=3.0)
    ttk.Scale(root, from_=0.0, to=50.0, orient="horizontal", variable=vert_energy, command=lambda _v=None: _set_fast_sim_param_double("vertical_energy_cost_mult", float(vert_energy.get()))).grid(
        row=pher_sep_row + 12, column=1, columnspan=2, sticky="ew", padx=10
    )

    # Static danger altitude violation penalty
    ttk.Label(root, text="Static danger alt penalty").grid(row=pher_sep_row + 13, column=0, sticky="w", padx=10)
    static_alt_pen_w = tk.DoubleVar(value=6.0)
    ttk.Scale(
        root,
        from_=0.0,
        to=500.0,
        orient="horizontal",
        variable=static_alt_pen_w,
        command=lambda _v=None: _set_fast_sim_param_double("static_danger_altitude_violation_weight", float(static_alt_pen_w.get())),
    ).grid(row=pher_sep_row + 13, column=1, columnspan=2, sticky="ew", padx=10)

    # Buildings section
    bld_sep_row = pher_sep_row + 14
    ttk.Separator(root, orient="horizontal").grid(row=bld_sep_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="Buildings (opacity)").grid(row=bld_sep_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    bld_alpha = tk.DoubleVar(value=1.0)

    lbl_bld = ttk.Label(root, text="Opacity: 1.00 (auto-publishes on GUI start)")
    lbl_bld.grid(row=bld_sep_row + 2, column=0, columnspan=3, padx=10, pady=2, sticky="w")

    def _bld_alpha_change(_v=None):
        alpha = float(bld_alpha.get())
        alpha = max(0.0, min(1.0, alpha))
        lbl_bld.configure(text=f"Opacity: {alpha:.2f} (auto-publishes on GUI start)")
        node.set_building_alpha(alpha)

    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=bld_alpha, command=_bld_alpha_change).grid(row=bld_sep_row + 3, column=0, columnspan=3, sticky="ew", padx=10)

    # LiDAR debug (per-drone)
    lidar_sep_row = bld_sep_row + 5
    ttk.Separator(root, orient="horizontal").grid(row=lidar_sep_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(12, 10))
    ttk.Label(root, text="LiDAR debug (per drone)").grid(row=lidar_sep_row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
    lidar_enabled = tk.BooleanVar(value=False)
    lidar_scan_enabled = tk.BooleanVar(value=False)
    plan_enabled = tk.BooleanVar(value=False)
    lidar_drone = tk.IntVar(value=1)

    def _publish_lidar_viz(_v=None):
        node.set_lidar_viz(bool(lidar_enabled.get()), int(lidar_drone.get()))
        node.set_lidar_scan_viz(bool(lidar_scan_enabled.get()), int(lidar_drone.get()))
        node.set_plan_viz(bool(plan_enabled.get()), int(lidar_drone.get()))

    ttk.Checkbutton(root, text="Show LiDAR walls/corners (/swarm/markers/lidar)", variable=lidar_enabled, command=_publish_lidar_viz).grid(
        row=lidar_sep_row + 2, column=0, columnspan=2, sticky="w", padx=10, pady=2
    )
    ttk.Checkbutton(root, text="Show LiDAR scan beams (/swarm/markers/lidar_scan)", variable=lidar_scan_enabled, command=_publish_lidar_viz).grid(
        row=lidar_sep_row + 3, column=0, columnspan=2, sticky="w", padx=10, pady=2
    )
    ttk.Checkbutton(root, text="Show planned path (/swarm/markers/plan)", variable=plan_enabled, command=_publish_lidar_viz).grid(
        row=lidar_sep_row + 4, column=0, columnspan=2, sticky="w", padx=10, pady=2
    )
    ttk.Label(root, text="Drone #").grid(row=lidar_sep_row + 2, column=2, sticky="e", padx=10)
    ttk.Spinbox(root, from_=1, to=200, textvariable=lidar_drone, width=6, command=_publish_lidar_viz).grid(
        row=lidar_sep_row + 3, column=2, sticky="e", padx=10
    )
    lidar_drone.trace_add("write", lambda *_a: _publish_lidar_viz())
    ttk.Label(
        root,
        text="Red lines: walls inferred from this drone's mock lidar hits. Red dots: corners/endpoints. Rendered at that drone's altitude.",
        foreground="#666",
        wraplength=640,
    ).grid(row=lidar_sep_row + 5, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

    # Log
    lbl_log = ttk.Label(root, text="—")
    lbl_log.grid(row=lidar_sep_row + 6, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 10))

    # Layout column weights
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)

    last_clock_ui_update = {"t": 0.0}
    shutting_down = {"v": False}

    def _tick():
        if shutting_down["v"]:
            return
        _maybe_save()

        # phase
        lbl_phase.configure(text=f"Phase: {node.latest_phase}")

        # mission status
        st = getattr(node, "gui_status", None) or {}
        if st:
            tf = int(st.get("targets_found", 0))
            tu = int(st.get("targets_unfound", 0))
            tt = int(st.get("targets_total", tf + tu))
            lbl_targets.configure(text=f"Targets: found {tf}/{tt} (unfound {tu})")
            lines = []
            for drow in st.get("drones", []):
                missing = drow.get("missing_targets", []) or []
                not_found = drow.get("not_found_known", []) or []
                if missing or not_found:
                    lines.append(f'{drow.get("uid")} [{drow.get("mode")}]: missing={len(missing)} known_unfound={len(not_found)}')
            txt = "\n".join(lines) if lines else "All drones know all targets and have no known-unfound targets."
            txt_drones.configure(state="normal")
            txt_drones.delete("1.0", "end")
            txt_drones.insert("1.0", txt)
            txt_drones.configure(state="disabled")

        # clock
        now = time.time()
        if now - last_clock_ui_update["t"] >= 2.0:
            last_clock_ui_update["t"] = now
            if node.latest_clock is not None:
                sec = int(node.latest_clock.clock.sec)
                nsec = int(node.latest_clock.clock.nanosec)
                lbl_clock.configure(text=f"Sim time: {_format_sim_time(sec, nsec)}")
                lbl_clock_small.configure(text=f"Timestamp: {sec}.{nsec:09d}")
            else:
                lbl_clock.configure(text="Sim time: —")
                lbl_clock_small.configure(text="Timestamp: —")

        try:
            sim_log = str((st or {}).get("sim_log") or "")
        except Exception:
            sim_log = ""
        lbl_log.configure(text=(sim_log or node.latest_log or "—"))

        # reschedule
        root.after(33, _tick)

    root.after(33, _tick)
    # apply initial state
    node.set_pheromone_viz_enabled(True)
    _publish_pher_params()
    node.set_building_alpha(1.0)
    _publish_tgt_params()
    _apply_click_mode_tk()
    _on_drone_scale()
    _publish_ptr()
    _publish_pher_select()
    _publish_lidar_viz()
    # ACO defaults -> fast sim params
    _on_aco_temp()
    _on_min_r()
    _on_min_s()
    _on_recent_pen()
    _set_fast_sim_param_double("base_push_radius_m", float(basepush_r.get()))
    _set_fast_sim_param_double("base_push_strength", float(basepush_s.get()))
    _set_fast_sim_param_double("evap_nav_rate", float(evap_nav.get()))
    _set_fast_sim_param_double("evap_danger_rate", float(evap_danger.get()))
    _set_fast_sim_param_double("danger_evap_mult_static", float(danger_mult_static.get()))
    _set_fast_sim_param_double("danger_evap_mult_dynamic", float(danger_mult_dynamic.get()))
    _set_fast_sim_param_double("wall_danger_evap_mult", float(wall_mult.get()))
    try:
        _set_fast_sim_param_double("static_danger_altitude_violation_weight", float(static_alt_pen_w.get()))
    except Exception:
        pass
    try:
        _on_danger_inspect()
    except Exception:
        pass
    try:
        _on_explore_area()
    except Exception:
        pass

    # Settings persistence (Tk)
    settings = _load_settings()
    try:
        speed_var.set(float(settings.get("speed", speed_var.get())))
        return_speed.set(float(settings.get("return_speed_mps", return_speed.get())))
        drone_alt.set(float(settings.get("drone_altitude_m", drone_alt.get())))
        sense_radius.set(float(settings.get("sense_radius_m", sense_radius.get())))
        drone_scale.set(float(settings.get("drone_marker_scale", drone_scale.get())))
        pher_disp.set(float(settings.get("pher_display_size_m", pher_disp.get())))
        pher_z.set(float(settings.get("pher_z_m", pher_z.get())))
        pher_a.set(float(settings.get("pher_alpha", pher_a.get())))
        pher_owner.set(str(settings.get("pher_owner", pher_owner.get())))
        pher_layer.set(str(settings.get("pher_layer", pher_layer.get())))
        try:
            min_alt.set(float(settings.get("min_flight_altitude_m", float(min_alt.get()))))
            roof_margin.set(float(settings.get("roof_clearance_margin_m", float(roof_margin.get()))))
            overfly_cost.set(float(settings.get("overfly_vertical_cost_mult", float(overfly_cost.get()))))
            vert_energy.set(float(settings.get("vertical_energy_cost_mult", float(vert_energy.get()))))
            static_alt_pen_w.set(float(settings.get("static_danger_altitude_violation_weight", float(static_alt_pen_w.get()))))
        except Exception:
            pass
        pher_drone_seq.set(int(settings.get("pher_drone_seq", pher_drone_seq.get())))
        bld_alpha.set(float(settings.get("building_alpha", bld_alpha.get())))
        tgt_d.set(float(settings.get("target_diameter_m", tgt_d.get())))
        tgt_a.set(float(settings.get("target_alpha", tgt_a.get())))
        dyn_speed.set(float(settings.get("dynamic_speed_sec_per_cell", dyn_speed.get())))
        danger_r.set(int(settings.get("danger_radius_cells", danger_r.get())))
        try:
            threat_h.set(float(settings.get("threat_height_m", threat_h.get())))
        except Exception:
            pass
        click_mode.set(str(settings.get("click_mode", click_mode.get())))
        ptr_enabled.set(bool(settings.get("pointer_enabled", ptr_enabled.get())))
        ptr_z.set(float(settings.get("pointer_z", ptr_z.get())))
        ptr_scale.set(float(settings.get("pointer_scale", ptr_scale.get())))
        ptr_alpha.set(float(settings.get("pointer_alpha", ptr_alpha.get())))
        lidar_enabled.set(bool(settings.get("lidar_viz_enabled", lidar_enabled.get())))
        lidar_drone.set(int(settings.get("lidar_viz_drone_seq", lidar_drone.get())))
        lidar_scan_enabled.set(bool(settings.get("lidar_scan_viz_enabled", lidar_scan_enabled.get())))
        plan_enabled.set(bool(settings.get("plan_viz_enabled", plan_enabled.get())))
        aco_temp.set(float(settings.get("aco_temperature", aco_temp.get())))
        min_r.set(float(settings.get("explore_min_radius_m", min_r.get())))
        min_s.set(float(settings.get("explore_min_radius_strength", min_s.get())))
        recent_pen.set(float(settings.get("recent_cell_penalty", recent_pen.get())))
        basepush_r.set(float(settings.get("base_push_radius_m", basepush_r.get())))
        basepush_s.set(float(settings.get("base_push_strength", basepush_s.get())))
        explore_area_r.set(float(settings.get("exploration_area_radius_m", explore_area_r.get())))
        explore_area_margin.set(float(settings.get("exploration_radius_margin_m", explore_area_margin.get())))
        low_nav_w.set(float(settings.get("explore_low_nav_weight", low_nav_w.get())))
        vec_avoid_w.set(float(settings.get("explore_vector_avoid_weight", vec_avoid_w.get())))
        vec_share_cells.set(int(settings.get("explore_vector_share_every_cells", vec_share_cells.get())))
        evap_nav.set(float(settings.get("evap_nav_rate", evap_nav.get())))
        evap_danger.set(float(settings.get("evap_danger_rate", evap_danger.get())))
        danger_mult_static.set(float(settings.get("danger_evap_mult_static", danger_mult_static.get())))
        danger_mult_dynamic.set(float(settings.get("danger_evap_mult_dynamic", danger_mult_dynamic.get())))
        wall_mult.set(float(settings.get("wall_danger_evap_mult", wall_mult.get())))
        danger_ins_w.set(float(settings.get("danger_inspect_weight", danger_ins_w.get())))
        danger_ins_k.set(int(settings.get("danger_inspect_kernel_cells", danger_ins_k.get())))
        danger_ins_thr.set(float(settings.get("danger_inspect_danger_thr", danger_ins_thr.get())))
        danger_ins_max.set(float(settings.get("danger_inspect_max_cell_danger", danger_ins_max.get())))
        # Inspector (dynamic danger)
        insp_follow_w.set(float(settings.get("dyn_danger_inspect_weight", insp_follow_w.get())))
        insp_rt_w.set(float(settings.get("dyn_inspector_rt_weight", insp_rt_w.get())))
        insp_rt_ttl.set(float(settings.get("dyn_inspector_rt_ttl_s", insp_rt_ttl.get())))
        insp_rt_standoff.set(int(settings.get("dyn_inspector_rt_standoff_cells", insp_rt_standoff.get())))
        insp_avoid_w.set(float(settings.get("dyn_inspector_avoid_damage_weight", insp_avoid_w.get())))
        insp_avoid_thr.set(float(settings.get("dyn_inspector_avoid_damage_thr", insp_avoid_thr.get())))
    except Exception:
        pass

    _on_speed_change()
    _on_return_speed()
    _on_drone_alt()
    _on_sense()
    _on_drone_scale()
    _publish_ptr()
    _publish_pher_params()
    _publish_pher_select()
    _publish_lidar_viz()
    _bld_alpha_change()
    _publish_tgt_params()
    _apply_click_mode_tk()
    _on_aco_temp()
    _on_min_r()
    _on_min_s()
    _on_recent_pen()
    _set_fast_sim_param_double("base_push_radius_m", float(basepush_r.get()))
    _set_fast_sim_param_double("base_push_strength", float(basepush_s.get()))
    _set_fast_sim_param_double("evap_nav_rate", float(evap_nav.get()))
    _set_fast_sim_param_double("evap_danger_rate", float(evap_danger.get()))
    _set_fast_sim_param_double("danger_evap_mult_static", float(danger_mult_static.get()))
    _set_fast_sim_param_double("danger_evap_mult_dynamic", float(danger_mult_dynamic.get()))
    _set_fast_sim_param_double("wall_danger_evap_mult", float(wall_mult.get()))
    try:
        _set_fast_sim_param_double("static_danger_altitude_violation_weight", float(static_alt_pen_w.get()))
    except Exception:
        pass
    try:
        _on_danger_inspect()
    except Exception:
        pass
    try:
        _on_explore_area()
    except Exception:
        pass

    dirty = {"v": False}
    last_save = {"t": 0.0}

    def _mark_dirty(_v=None):
        dirty["v"] = True

    for var in (
        speed_var,
        return_speed,
        drone_alt,
        sense_radius,
        drone_scale,
        pher_disp,
        pher_z,
        pher_a,
        bld_alpha,
        tgt_d,
        tgt_a,
        dyn_speed,
        danger_r,
        pher_drone_seq,
        ptr_z,
        ptr_scale,
        ptr_alpha,
        aco_temp,
        min_r,
        min_s,
        recent_pen,
        basepush_r,
        basepush_s,
        explore_area_r,
        explore_area_margin,
        low_nav_w,
        vec_avoid_w,
        vec_share_cells,
        evap_nav,
        evap_danger,
        danger_mult_static,
        danger_mult_dynamic,
        wall_mult,
        danger_ins_w,
        danger_ins_k,
        danger_ins_thr,
        danger_ins_max,
        overfly_cost,
        vert_energy,
        static_alt_pen_w,
    ):
        var.trace_add("write", lambda *_a: _mark_dirty())
    click_mode.trace_add("write", lambda *_a: _mark_dirty())
    pher_owner.trace_add("write", lambda *_a: _mark_dirty())
    pher_layer.trace_add("write", lambda *_a: _mark_dirty())
    ptr_enabled.trace_add("write", lambda *_a: _mark_dirty())
    lidar_enabled.trace_add("write", lambda *_a: _mark_dirty())
    lidar_drone.trace_add("write", lambda *_a: _mark_dirty())
    lidar_scan_enabled.trace_add("write", lambda *_a: _mark_dirty())
    plan_enabled.trace_add("write", lambda *_a: _mark_dirty())

    def _maybe_save():
        now = time.time()
        if not dirty["v"] or (now - last_save["t"] < 2.0):
            return
        payload = {
            "speed": float(speed_var.get()),
            "return_speed_mps": float(return_speed.get()),
            "drone_altitude_m": float(drone_alt.get()),
            "sense_radius_m": float(sense_radius.get()),
            "drone_marker_scale": float(drone_scale.get()),
            "pher_display_size_m": float(pher_disp.get()),
            "pher_z_m": float(pher_z.get()),
            "pher_alpha": float(pher_a.get()),
            "pher_owner": str(pher_owner.get()),
            "pher_layer": str(pher_layer.get()),
            "min_flight_altitude_m": float(min_alt.get()),
            "roof_clearance_margin_m": float(roof_margin.get()),
            "overfly_vertical_cost_mult": float(overfly_cost.get()),
            "vertical_energy_cost_mult": float(vert_energy.get()),
            "static_danger_altitude_violation_weight": float(static_alt_pen_w.get()),
            "pher_drone_seq": int(pher_drone_seq.get()),
            "building_alpha": float(bld_alpha.get()),
            "target_diameter_m": float(tgt_d.get()),
            "target_alpha": float(tgt_a.get()),
            "dynamic_speed_sec_per_cell": float(dyn_speed.get()),
            "danger_radius_cells": int(danger_r.get()),
            "threat_height_m": float(threat_h.get()),
            "click_mode": str(click_mode.get()),
            "pointer_enabled": bool(ptr_enabled.get()),
            "pointer_z": float(ptr_z.get()),
            "pointer_scale": float(ptr_scale.get()),
            "pointer_alpha": float(ptr_alpha.get()),
            "lidar_viz_enabled": bool(lidar_enabled.get()),
            "lidar_viz_drone_seq": int(lidar_drone.get()),
            "lidar_scan_viz_enabled": bool(lidar_scan_enabled.get()),
            "plan_viz_enabled": bool(plan_enabled.get()),
            "aco_temperature": float(aco_temp.get()),
            "explore_min_radius_m": float(min_r.get()),
            "explore_min_radius_strength": float(min_s.get()),
            "recent_cell_penalty": float(recent_pen.get()),
            "base_push_radius_m": float(basepush_r.get()),
            "base_push_strength": float(basepush_s.get()),
            "exploration_area_radius_m": float(explore_area_r.get()),
            "exploration_radius_margin_m": float(explore_area_margin.get()),
            "explore_low_nav_weight": float(low_nav_w.get()),
            "explore_vector_avoid_weight": float(vec_avoid_w.get()),
            "explore_vector_share_every_cells": int(vec_share_cells.get()),
            "evap_nav_rate": float(evap_nav.get()),
            "evap_danger_rate": float(evap_danger.get()),
            "danger_evap_mult_static": float(danger_mult_static.get()),
            "danger_evap_mult_dynamic": float(danger_mult_dynamic.get()),
            "wall_danger_evap_mult": float(wall_mult.get()),
            "danger_inspect_weight": float(danger_ins_w.get()),
            "danger_inspect_kernel_cells": int(danger_ins_k.get()),
            "danger_inspect_danger_thr": float(danger_ins_thr.get()),
            "danger_inspect_max_cell_danger": float(danger_ins_max.get()),
            # Inspector (dynamic danger)
            "dyn_danger_inspect_weight": float(insp_follow_w.get()),
            "dyn_inspector_rt_weight": float(insp_rt_w.get()),
            "dyn_inspector_rt_ttl_s": float(insp_rt_ttl.get()),
            "dyn_inspector_rt_standoff_cells": int(insp_rt_standoff.get()),
            "dyn_inspector_avoid_damage_weight": float(insp_avoid_w.get()),
            "dyn_inspector_avoid_damage_thr": float(insp_avoid_thr.get()),
            "saved_at_wall": now,
        }
        try:
            _save_settings(payload)
            dirty["v"] = False
            last_save["t"] = now
        except Exception:
            pass

    try:
        window.mainloop()
        rc = 0
    except KeyboardInterrupt:
        rc = 0
    finally:
        shutting_down["v"] = True
        stop_spin.set()
        try:
            exec_.shutdown(timeout_sec=0.2)
        except Exception:
            pass
        try:
            spin_thread.join(timeout=1.0)
        except Exception:
            pass
        for n in (node, danger, ground, buildings, fast_sim):
            try:
                exec_.remove_node(n)
            except Exception:
                pass
            try:
                n.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    raise SystemExit(main())


