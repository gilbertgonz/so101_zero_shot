"""Scene understanding: every object in view, numbered, measured in 3D, remembered across views.

One look (Scene.observe):
  1. the VLM lists every object (box + short label) and, separately, every human hand
  2. SAM masks each one, one multi-view depth pass measures them all (Locator.measure_many)
  3. results merge into the scene memory by 3D position, so an object keeps its number as the camera moves
  4. a set-of-marks image (mask outlines + numbers) is drawn for the agent and the UI

Facts are computed here, in geometry, so the planner only reasons over them: footprint size, height,
left/right, distance, which objects a hand covers. Hands are replaced on every look (people move);
objects stay in memory until the robot moves them or they are seen elsewhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .locate import Locator, ObjectEstimate, OtherViews, PerceptionError
from .pointer import DETECT_HANDS, DETECT_OBJECTS, Detection

logger = logging.getLogger(__name__)

MERGE_DIST_M = 0.035  # same kind, centres closer than this (or inside each other's footprint) = same thing
MAX_RANGE_M = 0.60  # measured centres further than this from the base are depth errors / background
HAND_COVER_GROW_M = 0.01  # a hand's footprint is grown by this when checking what it covers
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "black", "white", "gray", "grey",
          "brown", "silver", "gold", "beige", "tan", "cyan", "teal", "navy", "maroon"}
PALETTE = [(66, 135, 245), (245, 66, 66), (66, 245, 135), (245, 200, 66), (200, 66, 245), (66, 230, 245),
           (245, 130, 66), (160, 245, 66), (245, 66, 180), (120, 120, 255)]


def main_color(label: str) -> str | None:
    """First color word of a label ("black and white marker" -> black)."""
    for w in label.lower().replace("-", " ").split():
        if w in COLORS:
            return "gray" if w == "grey" else w
    return None


def same_kind_of_thing(a: str, b: str) -> bool:
    """Could two labels name the same object? Only a clash of main colors says no (labels vary a lot:
    "red cube" / "red block")."""
    ca, cb = main_color(a), main_color(b)
    return ca is None or cb is None or ca == cb


def footprint(points_base: np.ndarray) -> tuple[float, float, np.ndarray | None]:
    """Length and width (m) of the top-view footprint along its principal axes, and the long axis (xy)."""
    xy = points_base[:, :2]
    if len(xy) < 10:
        return 0.0, 0.0, None
    c = xy - np.median(xy, axis=0)
    w, v = np.linalg.eigh(np.cov(c.T))
    axis = v[:, 1]
    along, across = c @ axis, c @ v[:, 0]
    length = float(np.percentile(along, 97) - np.percentile(along, 3))
    width = float(np.percentile(across, 97) - np.percentile(across, 3))
    return length, width, axis if length > 1.5 * max(width, 1e-6) else None


@dataclass
class SceneObject:
    id: int
    label: str
    kind: str  # object | hand
    center_m: np.ndarray  # top-surface centre (base frame)
    top_z_m: float
    length_m: float
    width_m: float
    axis_xy: np.ndarray | None
    est: ObjectEstimate  # latest measurement; grasps start from it
    joints_deg: np.ndarray  # arm pose it was last seen from
    cam_dist_m: float  # camera distance of that view (closer views replace farther ones)
    seen: int = 1
    last_seen: float = field(default_factory=time.time)
    view: int = 0  # number of the look it was last seen in
    held: bool = False  # in the gripper
    covered_by_hand: float = 0.0  # share of its footprint under a hand (objects only)
    reachable: bool | None = None

    @property
    def name(self) -> str:
        return ("H" if self.kind == "hand" else "") + str(self.id)

    def describe(self) -> str:
        c = self.center_m * 100
        side = "left" if c[1] > 2 else "right" if c[1] < -2 else "centre"
        dist = float(np.hypot(c[0], c[1]))
        parts = [f'[{self.name}] "{self.label}"', f"at x={c[0]:.0f} y={c[1]:+.0f} cm ({side}, {dist:.0f} cm from the base)"]
        if self.kind == "hand":
            parts.append(f"{self.top_z_m * 100:.0f} cm above the table")
        else:
            parts.append(f"{self.length_m * 100:.0f} x {self.width_m * 100:.0f} cm footprint, "
                         f"about {self.top_z_m * 100:.0f} cm tall")
        if self.held:
            parts.append("IN THE GRIPPER")
        if self.covered_by_hand > 0.05:
            parts.append(f"{self.covered_by_hand:.0%} under a hand")
        if self.reachable is False:
            parts.append("out of reach")
        return ", ".join(parts)

    def summary(self) -> dict:
        return {"id": self.name, "label": self.label, "kind": self.kind,
                "center_cm": [round(float(v) * 100, 1) for v in self.center_m],
                "top_cm": round(self.top_z_m * 100, 1),
                "size_cm": [round(self.length_m * 100, 1), round(self.width_m * 100, 1)],
                "seen": self.seen, "held": self.held, "covered_by_hand": round(self.covered_by_hand, 2),
                "reachable": self.reachable}


@dataclass
class View:
    """One look: what was found and the images behind it."""
    number: int
    joints_deg: np.ndarray
    rect_bgr: np.ndarray
    objects: list[SceneObject]  # seen in this view
    masks: list[np.ndarray]  # each one's mask in this view (the memory may keep an older, closer measurement)
    failures: list[str]
    detections: list[str] = field(default_factory=list)  # what the VLM listed, for debugging
    marked_bgr: np.ndarray | None = None


class Scene:
    """Scene memory plus the perception that fills it."""

    def __init__(self, locator: Locator, workspace_check=None):
        """workspace_check(point_m) raises ValueError when a grasp point is out of the robot's workspace."""
        self.locator = locator
        self.workspace_check = workspace_check
        self.objects: dict[int, SceneObject] = {}
        self.hands: dict[int, SceneObject] = {}
        self._next_id = 1
        self._next_hand = 1
        self.views = 0
        self.last_view: View | None = None

    # ------------------------------------------------------------------ memory
    def clear(self) -> None:
        self.objects.clear()
        self.hands.clear()
        self._next_id = self._next_hand = 1
        self.views = 0
        self.last_view = None

    def get(self, ref) -> SceneObject:
        """Look up by number (3, "3", "[3]") or hand name ("H1"); KeyError with the known names."""
        s = str(ref).strip().strip("[]#").upper()
        try:
            if s.startswith("H"):
                return self.hands[int(s[1:])]
            return self.objects[int(s)]
        except (KeyError, ValueError):
            known = [o.name for o in self.all()]
            raise KeyError(f"no object {ref!r} in the scene (known: {', '.join(known) or 'none'})") from None

    def all(self) -> list[SceneObject]:
        return list(self.objects.values()) + list(self.hands.values())

    def forget(self, ref) -> None:
        o = self.get(ref)
        (self.hands if o.kind == "hand" else self.objects).pop(o.id, None)

    def held_object(self) -> SceneObject | None:
        return next((o for o in self.objects.values() if o.held), None)

    def set_held(self, ref, held: bool) -> None:
        for o in self.objects.values():
            o.held = False
        if held:
            self.get(ref).held = True

    def moved(self, ref, center_m: np.ndarray) -> None:
        """The robot moved an object: update where it is (until it is seen again)."""
        o = self.get(ref)
        o.center_m = np.asarray(center_m, dtype=float).copy()
        o.top_z_m = float(o.center_m[2])  # higher than its own height when it was stacked on something
        o.held = False

    def describe(self) -> str:
        """The numbered scene list, objects ordered closest to the robot first, with the comparisons a
        small model gets wrong (closest, furthest, tallest) worked out here."""
        objs = sorted(self.objects.values(), key=lambda o: float(np.hypot(*o.center_m[:2])))
        on_table = [o for o in objs if not o.held]
        notes: dict[int, list[str]] = {o.id: [] for o in objs}
        if len(on_table) > 1:
            notes[on_table[0].id].append("CLOSEST to the robot")
            notes[on_table[-1].id].append("FURTHEST from the robot")
            tallest = max(on_table, key=lambda o: o.top_z_m)
            notes[tallest.id].append("TALLEST")
        lines = [o.describe() + "".join(f", {n}" for n in notes[o.id]) for o in objs]
        if lines:
            lines.insert(0, "Objects, closest to the robot first:")
        lines += [h.describe() for h in self.hands.values()]
        return "\n".join(lines) if lines else "(nothing seen yet)"

    def _projects_into(self, o: SceneObject, est: ObjectEstimate, grow_px: int = 12) -> bool:
        """Does the remembered centre land on this view's mask? Depth errors run mostly along the camera
        ray, so this matches the same object even when the two 3D estimates disagree."""
        c = np.linalg.inv(est.T_base_cam) @ np.append(o.center_m, 1.0)
        if c[2] <= 0:
            return False
        K = self.locator.rect.K
        u, v = (K[:2, :2] @ (c[:2] / c[2]) + K[:2, 2]).round().astype(int)
        h, w = est.mask.shape
        if not (0 <= u < w and 0 <= v < h):
            return False
        y0, y1, x0, x1 = max(0, v - grow_px), min(h, v + grow_px + 1), max(0, u - grow_px), min(w, u + grow_px + 1)
        return bool(est.mask[y0:y1, x0:x1].any())

    def _merge(self, est: ObjectEstimate, label: str, kind: str, q: np.ndarray, cam_dist: float,
               taken: set[int] | None = None) -> SceneObject:
        """Match to a remembered object (same kind, close, compatible label, not already matched in this
        view) and update it, or remember a new one."""
        length, width, axis = footprint(est.points_base)
        store = self.hands if kind == "hand" else self.objects
        best, best_d = None, np.inf
        for o in store.values():
            if o.held or (taken is not None and o.id in taken):
                continue
            if kind == "object" and not same_kind_of_thing(o.label, label):
                continue
            d = float(np.linalg.norm(o.center_m[:2] - est.center_base[:2]))
            lim = max(MERGE_DIST_M, 0.5 * min(max(o.length_m, length), 0.12))
            if d >= lim and (kind != "object" or d > 0.10 or not self._projects_into(o, est)):
                continue
            if d < best_d:
                best, best_d = o, d
        if best is None:
            if kind == "hand":
                oid, self._next_hand = self._next_hand, self._next_hand + 1
            else:
                oid, self._next_id = self._next_id, self._next_id + 1
            best = SceneObject(oid, label, kind, est.center_base.copy(), est.top_z_m, length, width, axis, est,
                               np.asarray(q).copy(), cam_dist)
            store[oid] = best
        else:
            best.seen += 1
            if cam_dist <= 1.2 * best.cam_dist_m:  # a closer (more accurate) or similar view: take it
                best.center_m, best.top_z_m = est.center_base.copy(), est.top_z_m
                best.length_m, best.width_m, best.axis_xy = length, width, axis
                best.est, best.joints_deg, best.cam_dist_m = est, np.asarray(q).copy(), cam_dist
                if cam_dist < best.cam_dist_m * 0.8 or best.label in ("object", "unknown object"):
                    best.label = label
        best.last_seen = time.time()
        best.view = self.views
        if self.workspace_check is not None and kind == "object":
            try:
                self.workspace_check(np.array([best.center_m[0], best.center_m[1], 0.01]))
                best.reachable = True
            except ValueError:
                best.reachable = False
        return best

    def _hand_cover(self) -> None:
        """Share of each object's footprint (its measured points, top view) under a hand's footprint."""
        hulls = []
        for h in self.hands.values():
            xy = (h.est.points_base[:, :2] * 1000).astype(np.float32)  # mm
            if len(xy) >= 3:
                hulls.append(cv2.convexHull(xy))
        for o in self.objects.values():
            if not hulls:
                o.covered_by_hand = 0.0
                continue
            pts = o.est.points_base[:, :2] * 1000
            pts = pts[:: max(1, len(pts) // 200)]
            covered = np.zeros(len(pts), bool)
            for hull in hulls:
                covered |= np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) > -HAND_COVER_GROW_M * 1000
                                     for x, y in pts])
            o.covered_by_hand = float(covered.mean()) if len(pts) else 0.0

    # ------------------------------------------------------------------ perception
    def observe(self, fisheye_bgr: np.ndarray, joints_deg: np.ndarray, gripper_pct: float = 0.0,
                hands: bool = True, views: OtherViews | None = None) -> View:
        """Find, measure and remember everything in this photo (`views`: other photos or frames of the scene
        for metric depth). Hands seen before are replaced."""
        self.views += 1
        rect_bgr = self.locator.rect.rectify(fisheye_bgr)
        rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        pointer = self.locator.pointer
        dets: list[tuple[str, Detection]] = [("object", d) for d in pointer.detect(rgb, DETECT_OBJECTS)]
        if hands:
            dets += [("hand", d) for d in pointer.detect(rgb, DETECT_HANDS)]
            # The object list sometimes includes the hand too: drop object boxes that are mostly a hand box.
            hand_boxes = [d.box_px for k, d in dets if k == "hand"]
            dets = [(k, d) for k, d in dets if k == "hand" or not any(_inside(d.box_px, hb) > 0.7 for hb in hand_boxes)]
        failures: list[str] = []
        seen: list[SceneObject] = []
        masks: list[np.ndarray] = []
        if hands:
            self.hands.clear()
        taken: set[int] = set()
        if dets:
            ests, dbg = self.locator.measure_many(fisheye_bgr, joints_deg, [(d.label, d.as_pointing()) for _, d in dets],
                                                  gripper_pct=gripper_pct, views=views)
            T = dbg["T_base_cam"]
            for (kind, d), est in zip(dets, ests):
                if isinstance(est, PerceptionError):
                    failures.append(f"{d.label}: {est}")
                    continue
                if np.hypot(*est.center_base[:2]) > MAX_RANGE_M or est.center_base[0] < 0.0:
                    failures.append(f"{d.label}: measured {np.round(est.center_base * 100)} cm, beyond the table")
                    continue
                cam_dist = float(np.linalg.norm(est.center_base - T[:3, 3]))
                o = self._merge(est, d.label, kind, joints_deg, cam_dist, taken if kind == "object" else None)
                if kind == "object":
                    taken.add(o.id)
                if o not in seen:
                    seen.append(o)
                    masks.append(est.mask)
        self._hand_cover()
        view = View(self.views, np.asarray(joints_deg).copy(), rect_bgr, seen, masks, failures,
                    [f"{k}: {d.label}" for k, d in dets])
        view.marked_bgr = draw_marks(rect_bgr, seen, masks)
        self.last_view = view
        for f in failures:
            logger.info("scene: %s", f)
        return view


def _inside(a: np.ndarray, b: np.ndarray) -> float:
    """Share of box a's area inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return float(ix * iy / area)


def draw_marks(rect_bgr: np.ndarray, objects: list[SceneObject], masks: list[np.ndarray]) -> np.ndarray:
    """Set-of-marks image: each object's mask (in this view) outlined, with its number on a filled tag."""
    img = rect_bgr.copy()
    overlay = img.copy()
    for o, mask in zip(objects, masks):
        color = (40, 200, 255) if o.kind == "hand" else PALETTE[(o.id - 1) % len(PALETTE)]
        overlay[mask] = color
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for o, mask in zip(objects, masks):
        color = (40, 200, 255) if o.kind == "hand" else PALETTE[(o.id - 1) % len(PALETTE)]
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, color, 2)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        # Tag just above the mask so it does not hide the object.
        x, y = int(np.median(xs)), int(max(ys.min() - 18, 18))
        text = o.name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x - tw // 2 - 6, y - th // 2 - 6), (x + tw // 2 + 6, y + th // 2 + 6), (0, 0, 0), -1)
        cv2.rectangle(img, (x - tw // 2 - 6, y - th // 2 - 6), (x + tw // 2 + 6, y + th // 2 + 6), color, 2)
        cv2.putText(img, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img
