"""The agent's tools: perception and skills behind a JSON function-calling interface.

Each tool call is one step in the UI plan; tools made of several motions (pick_up) show their parts as
sub-steps ("t3.descend", ...). Expected failures (not found, unreachable, missed, declined) come back to
the model as a failed result with the reason, so it can try something else; Stop / E-stop and crashes
propagate to the session, which ends the run and returns the arm to rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..grasping.runner import STEPS_CLICK, GraspRunner
from ..perception.locate import PerceptionError
from ..perception.scene import Scene, SceneObject
from ..skills import Session, SkillFailed, Skills, add_step

logger = logging.getLogger(__name__)

SUBSTEP_LABELS = {**dict(STEPS_CLICK), "look": "Take a photo and find objects", "move": "Move the gripper"}
# Centre first (most accurate; it sets the numbers), then left and right (-pan turns toward +y = left).
LOOK_AROUND = (("centre", 0.0), ("left", -30.0), ("right", 30.0))
# Look-around views tilt the wrist this much further down than the scan pose: at the scan pose the gripper
# jaw hides the table closest to the robot (x below about 20 cm), where the closest object usually is.
LOOK_TILT_DEG = 12.0
MAX_LOOK_AROUNDS = 3  # including the one before planning; more finds nothing new in a still scene
MAX_WAYPOINTS = 12
PATH_SHORT_M = 0.015  # ending further than this from the last point = blocked on the way
HAND_NEAR_M = 0.12  # a gripper path this close to a seen hand asks the person first


def _fn(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


ID = {"type": "string", "description": 'object number from the scene list, e.g. "3", or a hand, e.g. "H1"'}
TOOL_SPECS = [
    _fn("look_around", "Scan the table from three camera angles (left, centre, right) and refresh the numbered "
        "scene list. Use it when something you need is not in the list or things may have moved."),
    _fn("look_at", "Move the camera close to one object or hand and measure it again (more accurate position, "
        "size and name).", {"id": ID}, ["id"]),
    _fn("find_hands", "Look for human hands in front of the robot (from the scan pose)."),
    _fn("pick_up", "Grasp an object from above and lift it; the robot keeps holding it. One object at a time.",
        {"id": ID}, ["id"]),
    _fn("place_at", "Put the held object down at a point (cm, base frame): on the table, or on top of whatever "
        "stands there.",
        {"x_cm": {"type": "number"}, "y_cm": {"type": "number"}}, ["x_cm", "y_cm"]),
    _fn("move_gripper", "Move the gripper's fingertips through a list of points, in straight lines from one point to "
        "the next (it first travels high to the first point). Points are [x, y, z] in cm in the base frame, z = "
        "height above the table. Optionally open or close the gripper at the end; opening while holding an "
        "object lets go of it. Use it for any motion the other tools do not cover.",
        {"waypoints": {"type": "array", "description": "e.g. [[20, 0, 10], [25, 0, 10]]",
                       "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
         "gripper": {"type": "string", "enum": ["open", "close"]}}, ["waypoints"]),
    _fn("go_home", "Move the arm back to its rest pose."),
    _fn("finish", "End the task: say whether it was done and give a short message for the person.",
        {"success": {"type": "boolean"}, "message": {"type": "string"}}, ["success", "message"]),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SPECS]


@dataclass
class ToolResult:
    ok: bool
    text: str  # what the model reads
    image: Path | None = None  # a new marked camera view to show the model
    done: bool = False  # finish was called
    detail: dict = field(default_factory=dict)  # shown in the UI step


class _Prefixed:
    """Reporter view for one tool call: sub-step ids get a "<tool step>." prefix and appear on first use."""

    def __init__(self, rep, prefix: str):
        self.rep, self.prefix = rep, prefix
        self._added: set[str] = set()
        self.open: set[str] = set()  # sub-steps started and not finished (closed as failed if the tool fails)

    def _id(self, step_id):
        sid = f"{self.prefix}.{step_id}"
        if sid not in self._added:
            self._added.add(sid)
            add_step(self.rep, sid, "    " + SUBSTEP_LABELS.get(step_id, step_id))
        return sid

    def add_step(self, step_id, label):
        add_step(self.rep, step_id, label)

    def step(self, step_id, status, message="", detail=None, images=None):
        sid = self._id(step_id)
        (self.open.add if status == "active" else self.open.discard)(sid)
        self.rep.step(sid, status, message, detail, images)

    def __getattr__(self, name):  # log, confirm, choose, pick, event
        return getattr(self.rep, name)


class Toolbox:
    def __init__(self, runner: GraspRunner, scene: Scene, session: Session, skills: Skills,
                 approve_hand: bool = True, approve_grasp: bool = True):
        self.runner, self.scene, self.s, self.sk = runner, scene, session, skills
        self.approve_hand, self.approve_grasp = approve_hand, approve_grasp
        self.picked_id: str | None = None  # scene id of the held object
        self._views = 0
        self.look_arounds = 0

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict, step_id: str) -> ToolResult:
        fn = getattr(self, f"t_{name}", None)
        if name not in TOOL_NAMES or fn is None:
            return ToolResult(False, f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}")
        rep = self.s.rep
        sub = self.s.rep = _Prefixed(rep, step_id)
        try:
            return fn(**(args or {}))
        except TypeError as e:  # wrong / missing arguments
            return ToolResult(False, f"bad arguments for {name}: {e}")
        except (KeyError, SkillFailed, PerceptionError) as e:
            for sid in sub.open:  # the part that was running when it failed
                rep.step(sid, "failed", str(e).strip('"')[:200])
            return ToolResult(False, str(e).strip('"'))
        finally:
            self.s.rep = rep
            self.s.current = step_id
            self.s.home_via_scan = True  # the arm may be anywhere: go home through the scan pose

    def _obj(self, ref) -> SceneObject:
        return self.scene.get(ref)

    def _scene_text(self) -> str:
        return "Scene now:\n" + self.scene.describe()

    # ------------------------------------------------------------------ perception
    def _look(self, q_cmd: np.ndarray | None, tag: str, hands: bool = True):
        self._views += 1
        self.s.begin("look", q_cmd)
        img, q, views = self.sk.look(q_cmd, f"{tag}{self._views}")
        self.sk.save_photo(img, f"{tag}{self._views}")
        view = self.scene.observe(img, q, self.sk.gripper_pct, hands=hands, views=views)
        path = self.s.out / f"{tag}{self._views}_marks.jpg"
        cv2.imwrite(str(path), view.marked_bgr)
        self.s.log.setdefault("views", []).append({"tag": f"{tag}{self._views}", "joints_deg": q,
                                                   "detected": view.detections, "failures": view.failures,
                                                   "seen": [o.name for o in view.objects]})
        self.s.rep.step("look", "done", f"{len(view.objects)} found",
                        {"seen": [o.name for o in view.objects], "detected": view.detections,
                         "not_measured": view.failures}, [path])
        return view, path

    def t_look_around(self) -> ToolResult:
        if self.look_arounds >= MAX_LOOK_AROUNDS:
            return ToolResult(False, f"Already looked around {self.look_arounds} times; another look will not find "
                                     "anything new. Whatever is still not in the scene list is not on the table: "
                                     "call finish with success false and say so.")
        self.look_arounds += 1
        held = self.scene.held_object()
        panels, seen, failures = {}, [], []
        for side, pan in LOOK_AROUND:
            view, _ = self._look(self.sk.scan + np.array([pan, 0, 0, LOOK_TILT_DEG, 0]), "look")
            panels[side] = cv2.resize(view.marked_bgr, (640, 480))
            seen += [o.name for o in view.objects if o.name not in seen]
            failures += view.failures
        if held is not None:
            held.held = True
        mosaic = np.hstack([panels["left"], panels["centre"], panels["right"]])
        for i, t in enumerate(("left", "centre", "right")):
            cv2.putText(mosaic, t, (i * 640 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        path = self.s.out / f"look_around{self._views}.jpg"
        cv2.imwrite(str(path), mosaic)
        return ToolResult(True, f"Looked left, centre and right; saw {', '.join(seen) or 'nothing'}.\n"
                          + self._scene_text(), path, detail={"seen": seen})

    def t_look_at(self, id) -> ToolResult:
        o = self._obj(id)
        q = self.sk.plan_relook(o.est, self.sk.arm.read_joints())
        view, path = self._look(q, "lookat")
        found = o.name in [v.name for v in view.objects]
        text = (f"Looked closely at {o.name}." if found else
                f"Looked where {o.name} was but did not see it there; it may have moved.") + "\n" + self._scene_text()
        return ToolResult(found, text, path)

    def t_find_hands(self) -> ToolResult:
        view, path = self._look(self.sk.scan, "hands")
        hands = [o for o in view.objects if o.kind == "hand"]
        text = (f"Found {len(hands)} hand(s): " + "; ".join(h.describe() for h in hands)) if hands else "No hands seen."
        return ToolResult(bool(hands), text + "\n" + self._scene_text(), path)

    # ------------------------------------------------------------------ manipulation
    def t_pick_up(self, id) -> ToolResult:
        o = self._obj(id)
        if o.kind == "hand":
            return ToolResult(False, "that is a person's hand, not an object")
        if self.sk.held is not None:
            return ToolResult(False, f"already holding {self.picked_id}; put it down first")
        if o.reachable is False:
            return ToolResult(False, f"{o.name} is out of reach")
        held = self.runner.grasp_steps(self.s, self.sk, o.est, o.label, self.sk.arm.read_joints(), keep=True,
                                       approve=self.approve_grasp, place_back=False)
        if not held:
            return ToolResult(False, f"missed {o.name}: the gripper closed on nothing.")
        self.picked_id = o.name
        self.scene.set_held(o.name, True)
        return ToolResult(True, f"Holding {o.name} ({o.label}).")

    def _held_obj(self) -> SceneObject | None:
        try:
            return self.scene.get(self.picked_id) if self.picked_id else None
        except KeyError:
            return None

    def _after_release(self, where_m: np.ndarray | None) -> None:
        if self.picked_id is not None:
            try:
                if where_m is not None:
                    self.scene.moved(self.picked_id, where_m)
                else:
                    self.scene.forget(self.picked_id)
            except KeyError:
                pass
        self.scene.set_held(None, False)
        self.picked_id = None

    def t_place_at(self, x_cm, y_cm) -> ToolResult:
        if self.sk.held is None:
            return ToolResult(False, "not holding anything")
        xy = np.array([float(x_cm), float(y_cm)]) / 100.0
        h = self._held_obj()
        held_r = 0.5 * max(h.length_m, h.width_m, 0.03) if h is not None else 0.025
        on_z, under = self._surface_under(xy, held_r)
        self.s.begin("place")
        where = self.sk.place_at(xy, on_z)
        at = f"x={float(x_cm):.0f} y={float(y_cm):.0f} cm"
        on = f", on top of {under} ({on_z * 100:.0f} cm up)" if under else ""
        self.s.rep.step("place", "done", f"at ({float(x_cm):.0f}, {float(y_cm):.0f}) cm{on}")
        name = self.picked_id
        self._after_release(where)
        return ToolResult(True, f"Put {name} down at {at}{on}.")

    def _surface_under(self, xy: np.ndarray, held_r: float) -> tuple[float, str]:
        """Height (m) of what the held object would come down on at xy, and what that is ("" = the table):
        the highest listed object under its middle, or else anything the 3D map saw sticking up there (things
        the detector did not list). place_at sets the object down on it instead of pushing into it."""
        top, what, ignore = 0.0, "", []
        for o in self.scene.objects.values():
            r = 0.5 * max(o.length_m, o.width_m, 0.02)
            if o.name == self.picked_id or o.held or o.kind == "hand":
                ignore.append((o.center_m[:2], r + 0.01))  # where the held one stood, for the map below
                continue
            if np.linalg.norm(o.center_m[:2] - xy) < r + 0.5 * held_r and o.top_z_m > top:
                top, what = float(o.top_z_m), f"{o.name} ({o.label})"
        scene_map = self.sk.scene_map
        if not what and scene_map is not None and scene_map.views:
            clear, high = scene_map.spot_clear(np.asarray(xy, float), held_r, ignore)
            if clear is False:
                top, what = high, "something the camera saw there"
        return top, what

    def _check_hands(self, pts: list[np.ndarray]) -> None:
        """Ask the person before a path passes close to one of their hands."""
        for hand in self.scene.hands.values():
            d = min(float(np.linalg.norm(p - hand.center_m)) for p in pts)
            if d < HAND_NEAR_M and self.approve_hand and not self.s.rep.confirm(
                    "hand", f"Move the gripper to within {d * 100:.0f} cm of the hand {hand.name}?",
                    [self.scene_image()] if self.scene_image() else None):
                raise SkillFailed("the person declined the move toward their hand")

    def scene_image(self) -> Path | None:
        imgs = sorted(self.s.out.glob("*_marks.jpg"), key=lambda p: p.stat().st_mtime)
        return imgs[-1] if imgs else None

    def t_move_gripper(self, waypoints, gripper=None) -> ToolResult:
        if isinstance(waypoints, dict) or (isinstance(waypoints, list) and waypoints
                                           and isinstance(waypoints[0], (int, float))):
            waypoints = [waypoints]  # a single point
        if not isinstance(waypoints, list) or not waypoints:
            return ToolResult(False, "waypoints must be a list of [x, y, z] points in cm, e.g. [[20, 0, 10]]")
        if len(waypoints) > MAX_WAYPOINTS:
            return ToolResult(False, f"at most {MAX_WAYPOINTS} waypoints per call")
        pts = []
        for i, w in enumerate(waypoints, 1):
            if isinstance(w, dict):
                w = [w.get(k, w.get(f"{k}_cm")) for k in "xyz"]
            try:
                p = np.array([float(v) for v in w]) / 100.0
            except (TypeError, ValueError):
                return ToolResult(False, f"waypoint {i} must be three numbers [x, y, z] in cm")
            if p.shape != (3,):
                return ToolResult(False, f"waypoint {i} must be three numbers [x, y, z] in cm")
            pts.append(p)
        gripper = None if gripper in (None, "", "none") else str(gripper).lower()
        self._check_hands(pts)
        held = self.picked_id
        self.s.begin("move")
        self.sk.move_path(pts, gripper)
        self.s.rep.step("move", "done", f"{len(pts)} points" + (f", gripper {gripper}" if gripper else ""))
        tip = self.sk.tip()
        text = f"Moved the gripper through {len(pts)} point(s); it is now at " + \
               "[{:.0f}, {:.0f}, {:.0f}] cm".format(*(tip * 100)) + "."
        short = float(np.linalg.norm(tip - pts[-1]))
        if short > PATH_SHORT_M:
            text += (f" It stopped {short * 100:.0f} cm short of the last point: something was in the way "
                     "(it pressed against it).")
        if gripper == "open" and held is not None:
            self._after_release(None)
            text += f" Opened the gripper and let go of {held}; look around to see where it landed."
        elif gripper:
            text += f" Gripper {'opened' if gripper == 'open' else 'closed'}."
        return ToolResult(True, text)

    def t_go_home(self) -> ToolResult:
        self.s.begin("home", self.sk.rest)
        self.sk.return_home(via_scan=True)
        self.s.rep.step("home", "done")
        return ToolResult(True, "At the rest pose.")

    def t_finish(self, success, message="") -> ToolResult:
        ok = success if isinstance(success, bool) else str(success).lower() in ("true", "yes", "1")
        return ToolResult(ok, str(message), done=True)
