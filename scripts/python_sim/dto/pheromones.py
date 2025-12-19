from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


GridCell = Tuple[int, int]


@dataclass
class CellMeta:
    t: float
    conf: float
    src: str
    # Optional altitude metadata (meters).
    # Meaning depends on usage:
    # - nav: minimum successful traversal altitude (min over time)
    # - danger(kind="nav_danger"): maximum observed blocking altitude (max over time)
    alt_m: Optional[float] = None
    # Optional kind for the danger layer (e.g. "nav_danger" for walls/corridors).
    kind: str = "generic"
    # Optional dynamic danger speed metadata (seconds per cell), for shared threat reasoning.
    speed_s_per_cell: Optional[float] = None
    # Optional observation distance metadata (meters) for "evidence quality".
    # Intended usage:
    # - explored layer: distance from observer to the cell center at the time of observation
    #   (smaller = better, i.e. closer/stronger evidence).
    obs_dist_m: Optional[float] = None


@dataclass
class SparseLayer:
    """Sparse pheromone layer: only stores cells where v > 0."""

    v: Dict[GridCell, float] = field(default_factory=dict)
    meta: Dict[GridCell, CellMeta] = field(default_factory=dict)

    # How to merge altitude metadata when a cell already exists:
    # - "none": ignore alt_m
    # - "min": keep min alt_m across updates
    # - "max": keep max alt_m across updates
    alt_merge: str = "none"

    def get(self, c: GridCell) -> float:
        return float(self.v.get(c, 0.0))

    def set(self, c: GridCell, value: float, meta: CellMeta):
        if value <= 1e-9:
            self.v.pop(c, None)
            self.meta.pop(c, None)
            return
        self.v[c] = float(value)
        self.meta[c] = meta

    def evaporate(self, dt: float, rate_per_s: float, kind_rate_mult: Optional[Dict[str, float]] = None):
        if dt <= 0 or rate_per_s <= 0:
            return
        to_del: List[GridCell] = []
        for c, val in self.v.items():
            rr = float(rate_per_s)
            if kind_rate_mult:
                mk = self.meta.get(c)
                if mk is not None and mk.kind:
                    k = str(mk.kind)
                    mult = kind_rate_mult.get(k)
                    # Common: kind strings include an id suffix (e.g. "danger_dyn_kernel:<uuid>").
                    # Fall back to the base kind prefix before ':' for multiplier lookup.
                    if mult is None and ":" in k:
                        mult = kind_rate_mult.get(k.split(":", 1)[0])
                    rr *= float(mult if mult is not None else 1.0)
            decay = math.exp(-rr * dt)
            nv = val * decay
            if nv <= 1e-9:
                to_del.append(c)
            else:
                self.v[c] = nv
        for c in to_del:
            self.v.pop(c, None)
            self.meta.pop(c, None)

    def merge_cell(self, c: GridCell, value: float, meta: CellMeta) -> bool:
        old_meta = self.meta.get(c)
        if old_meta is None:
            self.set(c, value, meta)
            return True
        # Altitude metadata merge is independent from value merge.
        if self.alt_merge in ("min", "max") and meta.alt_m is not None:
            old_alt = old_meta.alt_m
            new_alt = float(meta.alt_m)
            if old_alt is None:
                old_meta.alt_m = new_alt
            else:
                old_meta.alt_m = min(float(old_alt), new_alt) if self.alt_merge == "min" else max(float(old_alt), new_alt)
        # prefer newer, then higher confidence
        if meta.t > old_meta.t + 1e-6 or (abs(meta.t - old_meta.t) <= 1e-6 and meta.conf > old_meta.conf):
            self.set(c, value, meta)
            return True
        return False


@dataclass
class PheromoneMap:
    owner_uid: str

    # Layers:
    # - nav: positive navigation pheromone, with alt_m = min successful traversal altitude.
    # - danger: negative pheromone, with alt_m used for nav_danger (walls) as max blocking altitude.
    # - empty: persistent "known empty (safe goal)" space for A* to aim at (does not evaporate).
    # - explored: persistent "seen by lidar" coverage map (does not evaporate).
    nav: SparseLayer = field(default_factory=lambda: SparseLayer(alt_merge="min"))
    danger: SparseLayer = field(default_factory=lambda: SparseLayer(alt_merge="max"))
    empty: SparseLayer = field(default_factory=lambda: SparseLayer(alt_merge="none"))
    explored: SparseLayer = field(default_factory=lambda: SparseLayer(alt_merge="none"))

    # Recently changed cells for communication
    recent_nav: Deque[Tuple[GridCell, float, CellMeta]] = field(default_factory=lambda: deque(maxlen=20000))
    recent_danger: Deque[Tuple[GridCell, float, CellMeta]] = field(default_factory=lambda: deque(maxlen=20000))
    recent_empty: Deque[Tuple[GridCell, float, CellMeta]] = field(default_factory=lambda: deque(maxlen=40000))
    recent_explored: Deque[Tuple[GridCell, float, CellMeta]] = field(default_factory=lambda: deque(maxlen=40000))

    def evaporate(self, dt: float, nav_rate: float, danger_rate: float, danger_kind_rate_mult: Optional[Dict[str, float]] = None):
        self.nav.evaporate(dt, nav_rate)
        self.danger.evaporate(dt, danger_rate, kind_rate_mult=danger_kind_rate_mult)
        # empty/explored are persistent (no evaporation)

    def deposit_nav(self, c: GridCell, amount: float, t: float, conf: float, src: str, alt_m: Optional[float] = None):
        cur = self.nav.get(c)
        nv = max(0.0, cur + amount)
        # Altitude: keep min successful altitude.
        old = self.nav.meta.get(c)
        if alt_m is not None and old is not None and old.alt_m is not None:
            alt_m = min(float(old.alt_m), float(alt_m))
        meta = CellMeta(t=t, conf=conf, src=src, alt_m=(float(alt_m) if alt_m is not None else None))
        self.nav.set(c, nv, meta)
        self.recent_nav.append((c, nv, meta))

    def deposit_danger(
        self,
        c: GridCell,
        amount: float,
        t: float,
        conf: float,
        src: str,
        kind: str = "generic",
        alt_m: Optional[float] = None,
        speed_s_per_cell: Optional[float] = None,
    ):
        cur = self.danger.get(c)
        nv = max(0.0, cur + amount)
        # Altitude: for nav_danger, keep max observed blocking altitude.
        old = self.danger.meta.get(c)
        if alt_m is not None and old is not None and old.alt_m is not None:
            alt_m = max(float(old.alt_m), float(alt_m))
        meta = CellMeta(
            t=t,
            conf=conf,
            src=src,
            alt_m=(float(alt_m) if alt_m is not None else None),
            kind=str(kind or "generic"),
            speed_s_per_cell=(float(speed_s_per_cell) if speed_s_per_cell is not None else None),
        )
        self.danger.set(c, nv, meta)
        self.recent_danger.append((c, nv, meta))

    def deposit_empty(self, c: GridCell, t: float, conf: float, src: str):
        # Persistent occupancy-free evidence. Store as binary (1.0).
        if self.empty.get(c) > 1e-6:
            return
        meta = CellMeta(t=t, conf=conf, src=src, kind="empty")
        self.empty.set(c, 1.0, meta)
        self.recent_empty.append((c, 1.0, meta))

    def deposit_explored(self, c: GridCell, t: float, conf: float, src: str, obs_dist_m: Optional[float] = None):
        # Persistent "seen" evidence. Store as binary (1.0) but keep metadata so we can reason about:
        # - freshness (t): newer sightings are stronger
        # - quality (obs_dist_m): closer sightings are stronger
        old = self.explored.meta.get(c)
        best_dist = None
        try:
            if old is not None and old.obs_dist_m is not None:
                best_dist = float(old.obs_dist_m)
        except Exception:
            best_dist = None
        try:
            if obs_dist_m is not None:
                d = float(obs_dist_m)
                if best_dist is None:
                    best_dist = d
                else:
                    best_dist = min(float(best_dist), d)
        except Exception:
            pass

        meta = CellMeta(t=t, conf=conf, src=src, kind="explored", obs_dist_m=(float(best_dist) if best_dist is not None else None))
        self.explored.set(c, 1.0, meta)
        # Communication throttling:
        # - Always publish first discovery.
        # - For already-known cells, only publish if the sighting is "meaningfully newer" or closer.
        if old is None:
            self.recent_explored.append((c, 1.0, meta))
        else:
            try:
                dt = float(t) - float(old.t)
            except Exception:
                dt = 0.0
            try:
                old_d = float(old.obs_dist_m) if old.obs_dist_m is not None else None
            except Exception:
                old_d = None
            try:
                new_d = float(best_dist) if best_dist is not None else None
            except Exception:
                new_d = None
            publish = False
            # Newer by at least 5 simulated seconds.
            if dt >= 5.0:
                publish = True
            # Or improved observation distance by >= 20%.
            if (old_d is not None) and (new_d is not None) and new_d <= 0.8 * old_d:
                publish = True
            if publish:
                self.recent_explored.append((c, 1.0, meta))

    def deposit_nav_danger(self, c: GridCell, amount: float, t: float, conf: float, src: str, alt_m: Optional[float]):
        # Convenience wrapper: walls/corridors as a specific danger kind.
        self.deposit_danger(c, amount=amount, t=t, conf=conf, src=src, kind="nav_danger", alt_m=alt_m)

    def merge_patch(self, layer: str, cells: List[dict]) -> int:
        changed = 0
        # Backward-compatible: accept nav_safe/nav_danger, but store into 2-layer model.
        if layer == "nav_safe":
            layer = "nav"
        if layer == "nav_danger":
            layer = "danger"
        if layer == "empty":
            target = self.empty
        elif layer == "explored":
            target = self.explored
        else:
            target = self.nav if layer == "nav" else self.danger

        for cell in cells:
            c = (int(cell["x"]), int(cell["y"]))
            v = float(cell["v"])
            alt = cell.get("alt", None)
            kind = str(cell.get("kind", "generic"))
            sp = cell.get("speed", None)
            od = cell.get("obs_dist_m", cell.get("obs_dist", None))
            meta = CellMeta(
                t=float(cell["t"]),
                conf=float(cell.get("conf", 0.5)),
                src=str(cell.get("src", "unknown")),
                alt_m=(float(alt) if alt is not None else None),
                kind=kind,
                speed_s_per_cell=(float(sp) if sp is not None else None),
                obs_dist_m=(float(od) if od is not None else None),
            )
            if target.merge_cell(c, v, meta):
                changed += 1
        return changed

    def make_patch(self, layer: str, max_cells: int, since_t: float) -> List[dict]:
        out: List[dict] = []
        if layer == "empty":
            buf = self.recent_empty
        elif layer == "explored":
            buf = self.recent_explored
        else:
            buf = self.recent_nav if layer in ("nav", "nav_safe") else self.recent_danger

        # iterate from newest backward (deque supports reversed)
        for c, v, meta in reversed(buf):
            if meta.t <= since_t + 1e-9:
                break
            row = {"x": c[0], "y": c[1], "v": float(v), "t": float(meta.t), "conf": float(meta.conf), "src": meta.src}
            if meta.alt_m is not None:
                row["alt"] = float(meta.alt_m)
            if meta.speed_s_per_cell is not None:
                row["speed"] = float(meta.speed_s_per_cell)
            if meta.obs_dist_m is not None:
                row["obs_dist_m"] = float(meta.obs_dist_m)
            if layer not in ("nav", "nav_safe") and meta.kind:
                row["kind"] = str(meta.kind)
            out.append(row)
            if len(out) >= max_cells:
                break
        return out

    def clear(self):
        self.nav.v.clear()
        self.nav.meta.clear()
        self.danger.v.clear()
        self.danger.meta.clear()
        self.empty.v.clear()
        self.empty.meta.clear()
        self.explored.v.clear()
        self.explored.meta.clear()
        self.recent_nav.clear()
        self.recent_danger.clear()
        self.recent_empty.clear()
        self.recent_explored.clear()

