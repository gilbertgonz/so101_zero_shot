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
from ..hardware.so101 import ARM_JOINTS
from ..perception.locate import PerceptionError
from ..perception.scene import Scene, SceneObject
from ..skills import Session, SkillFailed, Skills, add_step

logger = logging.getLogger(__name__)

SUBSTEP_LABELS = {**dict(STEPS_CLICK), "look": "Take a photo and find objects", "gesture": "Move the joints"}
# Centre first (most accurate; it sets the numbers), then left and right (-pan turns toward +y = left).
LOOK_AROUND = (("centre", 0.0), ("left", -30.0), ("right", 30.0))
SIDES = {"left": (0.0, 1.0), "right": (0.0, -1.0), "front": (-1.0, 0.0), "behind": (1.0, 0.0)}
PLACE_GAP_M = 0.03
# move_joints (gestures): per-move and per-call caps, and how close the gripper may come to the table.
GESTURE_MAX_DEG = 40.0
GESTURE_MAX_MOVES = 8
GESTURE_MIN_TIP_Z_M = 0.06
GESTURE_START_Z_M = 0.10  # below this (e.g. at rest) a gesture starts from the raised scan pose


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
    _fn("place_at", "Put the held object down at a point on the table (cm, base frame).",
        {"x_cm": {"type": "number"}, "y_cm": {"type": "number"}}, ["x_cm", "y_cm"]),
    _fn("place_next_to", "Put the held object down next to another object. Sides are seen from the robot: "
        "front = closer to the robot, behind = further away.",
        {"id": ID, "side": {"type": "string", "enum": list(SIDES)}}, ["id", "side"]),
    _fn("put_back", "Put the held object back exactly where it was picked up."),
    _fn("give_to_hand", "Hold the held object just above a person's hand and let go, so they get it.",
        {"id": ID}, ["id"]),
    _fn("touch", "Touch an object or a hand gently with the closed gripper, then back off. A high five is a "
        "touch on a hand.", {"id": ID}, ["id"]),
    _fn("move_above", "Hover the gripper above an object or hand (e.g. to point at it).",
        {"id": ID, "height_cm": {"type": "number", "description": "above its top, default 8"}}, ["id"]),
    _fn("move_joints", "Move the arm's own joints by relative angles, one move after another, for gestures and "
        "body language. Joints: shoulder_pan (+ turns the arm right, - left), shoulder_lift (+ leans forward and "
        "down), elbow_flex (+ bends down), wrist_flex (+ tilts the gripper down, - up), wrist_roll (twists the "
        f"gripper). Each value is degrees added to that joint: at most {GESTURE_MAX_DEG:.0f} per move, at most "
        f"{GESTURE_MAX_MOVES} moves. If the arm is low it first rises to a raised pose.",
        {"moves": {"type": "array", "description": 'e.g. [{"wrist_flex": 20}, {"wrist_flex": -20}]',
                   "items": {"type": "object", "properties": {j: {"type": "number"} for j in ARM_JOINTS}}}},
        ["moves"]),
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

    def _id(self, step_id):
        sid = f"{self.prefix}.{step_id}"
        if sid not in self._added:
            self._added.add(sid)
            add_step(self.rep, sid, "    " + SUBSTEP_LABELS.get(step_id, step_id))
        return sid

    def add_step(self, step_id, label):
        add_step(self.rep, step_id, label)

    def step(self, step_id, status, message="", detail=None, images=None):
        self.rep.step(self._id(step_id), status, message, detail, images)

    def __getattr__(self, name):  # log, confirm, choose, pick, event
        return getattr(self.rep, name)


class Toolbox:
    def __init__(self, runner: GraspRunner, scene: Scene, session: Session, skills: Skills,
                 approve_hand: bool = True, approve_grasp: bool = True):
        self.runner, self.scene, self.s, self.sk = runner, scene, session, skills
        self.approve_hand, self.approve_grasp = approve_hand, approve_grasp
        self.picked_id: str | None = None  # scene id of the held object
        self._views = 0

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict, step_id: str) -> ToolResult:
        fn = getattr(self, f"t_{name}", None)
        if name not in TOOL_NAMES or fn is None:
            return ToolResult(False, f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}")
        rep = self.s.rep
        self.s.rep = _Prefixed(rep, step_id)
        try:
            return fn(**(args or {}))
        except TypeError as e:  # wrong / missing arguments
            return ToolResult(False, f"bad arguments for {name}: {e}")
        except (KeyError, SkillFailed, PerceptionError) as e:
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
        held = self.scene.held_object()
        panels, seen, failures = {}, [], []
        for side, pan in LOOK_AROUND:
            view, _ = self._look(self.sk.scan + np.array([pan, 0, 0, 0, 0]), "look")
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
        if not self._spot_is_free(xy, held_r, exclude={self.picked_id}):
            return ToolResult(False, f"x={float(x_cm):.0f} y={float(y_cm):.0f} cm is taken by another object; "
                                     "choose a free spot")
        self.s.begin("place")
        where = self.sk.place_at(xy)
        self.s.rep.step("place", "done", f"at ({float(x_cm):.0f}, {float(y_cm):.0f}) cm")
        name = self.picked_id
        self._after_release(where)
        return ToolResult(True, f"Put {name} down at x={float(x_cm):.0f} y={float(y_cm):.0f} cm.")

    def t_place_next_to(self, id, side) -> ToolResult:
        if self.sk.held is None:
            return ToolResult(False, "not holding anything")
        side = str(side).lower()
        if side not in SIDES:
            return ToolResult(False, f"side must be one of {list(SIDES)}")
        o = self._obj(id)
        h = self._held_obj()
        held_r = 0.5 * max(h.length_m, h.width_m, 0.03) if h is not None else 0.025
        base = np.array(SIDES[side])
        # Try the requested side first, then turned up to 40 deg either way and a little further out,
        # until the spot is clear of every other known object and inside the workspace.
        for extra in (0.0, 0.015, 0.03):
            for turn in (0, 20, -20, 40, -40):
                c, s_ = np.cos(np.radians(turn)), np.sin(np.radians(turn))
                direction = np.array([c * base[0] - s_ * base[1], s_ * base[0] + c * base[1]])
                d = 0.5 * max(o.length_m, o.width_m, 0.02) + held_r + PLACE_GAP_M + extra
                xy = o.center_m[:2] + d * direction
                if self._spot_is_free(xy, held_r, exclude={o.name, self.picked_id}):
                    try:
                        self.runner.cfg.check_in_workspace(np.array([xy[0], xy[1], 0.02]))
                    except ValueError:
                        continue
                    return self.t_place_at(xy[0] * 100, xy[1] * 100)
        return ToolResult(False, f"no free spot on the {side} of {o.name} (other objects or the workspace edge "
                                 "are in the way)")

    def _spot_is_free(self, xy: np.ndarray, radius: float, exclude: set) -> bool:
        """Clear of every remembered object (footprints) and, where the 3D map has seen it, of anything else
        sticking up from the table (things the detector did not list). `exclude`: object ids to ignore (the
        held one, the one to place next to)."""
        ignore = []
        for other in self.scene.objects.values():
            r = 0.5 * max(other.length_m, other.width_m, 0.02)
            if other.name in exclude or other.held:
                ignore.append((other.center_m[:2], r + 0.01))  # where it stands (or stood) in the map too
                continue
            if np.linalg.norm(other.center_m[:2] - xy) < r + radius + 0.01:
                return False
        scene_map = self.sk.scene_map
        if scene_map is not None and scene_map.views:
            clear, top = scene_map.spot_clear(np.asarray(xy, float), radius + 0.005, ignore)
            if clear is False:
                logger.info("spot (%.0f, %.0f) cm: the 3D map has something %.1f cm tall there",
                            xy[0] * 100, xy[1] * 100, top * 100)
                return False
        return True

    def t_put_back(self) -> ToolResult:
        if self.sk.held is None:
            return ToolResult(False, "not holding anything")
        h = self.sk.held
        self.s.begin("place", h.plan.grasp_deg)
        self.sk.travel_to(h.plan.lift_deg)
        self.sk.place_back(h.plan)
        self.s.rep.step("place", "done")
        name = self.picked_id
        self._after_release(h.picked_from_m)
        return ToolResult(True, f"Put {name} back where it was.")

    def _approve_hand(self, what: str, hand: SceneObject) -> None:
        if self.approve_hand and hand.kind == "hand" and not self.s.rep.confirm(
                "hand", f"Move toward the hand {hand.name} to {what}?",
                [self.scene_image()] if self.scene_image() else None):
            raise SkillFailed("the person declined the move toward their hand")

    def scene_image(self) -> Path | None:
        imgs = sorted(self.s.out.glob("*_marks.jpg"), key=lambda p: p.stat().st_mtime)
        return imgs[-1] if imgs else None

    def t_give_to_hand(self, id) -> ToolResult:
        if self.sk.held is None:
            return ToolResult(False, "not holding anything; pick something up first")
        hand = self._obj(id)
        if hand.kind != "hand":
            return ToolResult(False, f"{hand.name} is not a hand")
        self._approve_hand("hand over the object", hand)
        self.s.begin("give")
        self.sk.release_above(np.array([hand.center_m[0], hand.center_m[1], hand.top_z_m]))
        self.s.rep.step("give", "done")
        name = self.picked_id
        self._after_release(None)
        return ToolResult(True, f"Released {name} above {hand.name}.")

    def _hand_point(self, o: SceneObject) -> np.ndarray:
        return np.median(o.est.points_base, axis=0)  # a hand's middle, not its highest finger

    def t_touch(self, id) -> ToolResult:
        o = self._obj(id)
        self._approve_hand("touch it", o)
        if o.kind == "hand":
            point, push = self._hand_point(o), 0.04
        else:
            # Heights are rough: keep feeling downward until the table if needed.
            point = np.array([o.center_m[0], o.center_m[1], o.top_z_m])
            push = max(0.02, o.top_z_m - 0.005)
        self.s.begin("touch")
        r = self.sk.touch(point, "auto", max_push_m=push)
        msg = (f"felt contact {r['travel_m'] * 100:+.1f} cm from the estimated surface" if r["contact"]
               else f"felt nothing ({r['signal']})")
        if r["contact"] and o.kind == "object" and r["approach"] == "down":
            o.felt_top_z_m = max(0.005, float(r["tip_m"][2]))
            msg += f"; its top is {o.felt_top_z_m * 100:.1f} cm above the table"
        self.s.rep.step("touch", "done" if r["contact"] else "failed", msg,
                        {"approach": r["approach"], "signal": r["signal"]})
        return ToolResult(r["contact"], f"Touch {o.name}: {msg}.", detail={"approach": r["approach"]})

    def t_move_above(self, id, height_cm=8.0) -> ToolResult:
        o = self._obj(id)
        self._approve_hand("hover above it", o)
        top = o.top_z_m if o.kind != "hand" else float(np.percentile(o.est.points_base[:, 2], 90))
        p = np.array([o.center_m[0], o.center_m[1], top + float(height_cm) / 100.0])
        self.s.begin("move")
        used = self.sk.move_to(p, "auto")
        self.s.rep.step("move", "done", f"approach {used}")
        return ToolResult(True, f"Gripper is {float(height_cm):.0f} cm above {o.name}.")

    def t_move_joints(self, moves) -> ToolResult:
        if isinstance(moves, dict):
            moves = [moves]
        if not isinstance(moves, list) or not moves:
            return ToolResult(False, "moves must be a list like [{\"wrist_flex\": 20}, {\"wrist_flex\": -20}]")
        if len(moves) > GESTURE_MAX_MOVES:
            return ToolResult(False, f"at most {GESTURE_MAX_MOVES} moves per call")
        deltas = []
        for i, m in enumerate(moves, 1):
            if not isinstance(m, dict) or not m:
                return ToolResult(False, f"move {i} must be an object like {{\"wrist_flex\": 20}}")
            bad = sorted(set(m) - set(ARM_JOINTS))
            if bad:
                return ToolResult(False, f"move {i}: unknown joint(s) {bad}; joints: {', '.join(ARM_JOINTS)}")
            try:
                d = np.array([float(m.get(j, 0.0)) for j in ARM_JOINTS])
            except (TypeError, ValueError):
                return ToolResult(False, f"move {i}: values must be numbers of degrees")
            if np.abs(d).max() > GESTURE_MAX_DEG:
                return ToolResult(False, f"move {i}: at most {GESTURE_MAX_DEG:.0f} degrees per joint per move")
            deltas.append(d)
        self.s.begin("gesture")
        q = self.sk.arm.read_joints()
        if self.sk.kin.fk(q)[2, 3] < GESTURE_START_Z_M:
            q = np.asarray(self.sk.scan, dtype=float)
            self.sk.travel_to(q)
        limits = getattr(self.sk.arm, "joint_limits", {}) or {}
        targets = []
        for i, d in enumerate(deltas, 1):  # check the whole gesture before moving
            q = q + d
            for j, v in zip(ARM_JOINTS, q):
                lo, hi = limits.get(j, (-np.inf, np.inf))
                if not lo <= v <= hi:
                    raise SkillFailed(f"move {i} takes {j} to {v:.0f} deg, outside its range {lo:.0f}..{hi:.0f}")
            if self.sk.kin.fk(q)[2, 3] < GESTURE_MIN_TIP_Z_M:
                raise SkillFailed(f"move {i} would bring the gripper within {GESTURE_MIN_TIP_Z_M * 100:.0f} cm of "
                                  "the table; use smaller or different moves")
            targets.append(q)
        for q in targets:
            self.sk.move_joints(q)
        self.s.rep.step("gesture", "done", f"{len(targets)} moves")
        return ToolResult(True, f"Did {len(targets)} joint moves.")

    def t_go_home(self) -> ToolResult:
        self.s.begin("home", self.sk.rest)
        self.sk.return_home(via_scan=True)
        self.s.rep.step("home", "done")
        return ToolResult(True, "At the rest pose.")

    def t_finish(self, success, message="") -> ToolResult:
        ok = success if isinstance(success, bool) else str(success).lower() in ("true", "yes", "1")
        return ToolResult(ok, str(message), done=True)
