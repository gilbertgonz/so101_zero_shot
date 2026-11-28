"""Robot skills: the building blocks the grasp runner and the agent are made of.

Each skill does one thing (take a photo, find an object, plan a grasp, descend, lift, ...) on the real arm
or the MuJoCo sim and either returns what it measured or raises SkillFailed with a reason a caller (a
person or the agent) can act on: "unreachable", "lost the object", ... Motion can also raise
MotionStopped (Stop / E-stop pressed); callers decide what happens next. Skills do not report UI steps;
the caller does, so the same skill can be one step of the grasp plan or one tool call of the agent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..calibration.viewpoints import min_link_height, path_is_safe
from ..config import RobotConfig
from ..grasping.feedback import is_holding, read_grip
from ..grasping.topdown import (
    GraspPlan,
    GraspPlanningError,
    TopDownPlanner,
    grasp_height,
    object_axis_xy,
)
from ..kinematics import SO101Kinematics
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import (
    Locator,
    ObjectEstimate,
    OtherViews,
    PerceptionError,
    draw_debug,
)
from ..perception.scene_map import SceneMap
from .motion_frames import (
    MIN_BASELINE_M,
    Frame,
    LookSettings,
    MotionFrames,
    spread_frames,
)

logger = logging.getLogger(__name__)

RELOCATE_MAX_JUMP_M = 0.04  # closer look: a VLM answer further than this from the first estimate is another object
OPEN_PCT = 100.0  # = the user's max safe opening (gripper calibration range_min)
DESCEND_SPEED = 15.0
DOWN = np.array([0.0, 0.0, -1.0])
APPROACHES = ("down", "forward", "auto")
# look(): base turn for a photo taken where the camera already was (~3 cm of camera travel at the scan pose)
NUDGE_PAN_DEG = 10.0
TRANSIT_Z_M = 0.13  # fingertips this high when travelling between spots (a held object hangs ~3-8 cm below)
# move_path(): straight lines in steps this long; the fingertips stay this high; a step that needs a bigger
# joint change than this is a wrist flip (the 5-joint arm switching between reaching down and forward).
PATH_STEP_M = 0.015
PATH_MIN_Z_M = 0.01
PATH_MAX_JUMP_DEG = 25.0
# place_at: (above the drop point before, after) in m. The pick's own hover heights first; then lower ones,
# which a high drop (on top of something) needs: fingers pointing straight down only reach ~9 cm up.
PLACE_HOVERS_M = ((0.05, 0.07), (0.03, 0.03))
PLACE_SCAN_X_M = np.arange(0.10, 0.33, 0.02)  # where a failed place_at looks for x values that do work
PLACE_CLEAR_M = 0.04  # after letting go, fingertips rise straight up to this far above the placed object's top


class SkillFailed(Exception):
    """An expected failure (not found, unreachable, missed, ...); the message says why."""


@dataclass
class Held:
    """What the gripper holds after a successful lift (for putting it back or placing it elsewhere)."""
    name: str
    plan: GraspPlan  # the grasp it was picked with: plan.target_m is the fingertip point on the object
    picked_from_m: np.ndarray  # object top-centre where it was picked
    top_z_m: float  # its height

    @property
    def hang_m(self) -> float:
        """How far its bottom hangs below the fingertips (it stood on the table when it was picked)."""
        return float(self.plan.target_m[2])


@dataclass
class Grip:
    open_pct: float
    load: float

    @property
    def squeezing(self) -> bool:
        return is_holding(self.open_pct, self.load)


def est_detail(est: ObjectEstimate) -> dict:
    c = est.center_base * 100
    return {"centre_cm": [round(float(c[0]), 1), round(float(c[1]), 1)], "top_cm": round(est.top_z_m * 100, 1),
            "depth_views": est.n_views, "warnings": est.warnings}


def deg_list(q: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in q]


def overlay_grasp(img: np.ndarray, est: ObjectEstimate, rect: Rectifier, target: np.ndarray,
                  closing_xy: np.ndarray | None, axis_xy: np.ndarray | None) -> np.ndarray:
    """Planned fingertip point (yellow), jaw closing line (red) and object long axis (blue)."""
    Tinv = np.linalg.inv(est.T_base_cam)

    def px(p):
        c = Tinv @ np.append(p, 1.0)
        return tuple(int(v) for v in rect.K[:2, :2] @ (c[:2] / c[2]) + rect.K[:2, 2])

    for vec, color, half in ((axis_xy, (255, 120, 0), 0.04), (closing_xy, (0, 0, 255), 0.025)):
        if vec is not None:
            d = np.append(vec, 0.0) * half
            cv2.line(img, px(target - d), px(target + d), color, 3)
    cv2.circle(img, px(target), 6, (0, 255, 255), -1)
    return img


@dataclass
class Skills:
    """Skills bound to one arm + camera and one run folder (`out`, where photos and debug images go).

    `log` is the run's JSON log; skills add the joints/gripper of every photo to it."""
    cfg: RobotConfig
    kin: SO101Kinematics
    cam_model: FisheyeCamera
    rect: Rectifier
    planner: TopDownPlanner
    locator: Locator
    arm: object
    camera: object
    out: Path
    log: dict = field(default_factory=dict)
    speed: float = 30.0
    gripper_pct: float = 0.0  # opening when the last photo was taken (the gripper's outline depends on it)
    held: Held | None = None
    look_settings: LookSettings = field(default_factory=LookSettings)
    # Frames from the last move_collecting() (for a photo at the pose it ended in); cleared by any other move.
    _move_frames: list[Frame] = field(default_factory=list, repr=False)
    last_photo: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)  # (fisheye BGR, joints)
    # 3D map of the workspace for this run: every photo's depth is fused into it (perception/scene_map.py).
    scene_map: SceneMap | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.scene_map is None:
            self.scene_map = SceneMap(table_z_m=self.locator.table_z_m if self.locator is not None else 0.0)
        if self.locator is not None:
            self.locator.scene_map = self.scene_map  # one run at a time uses the locator

    def save_map(self) -> list[Path]:
        """Write the run's scene map (scene_map.ply, scene_map_height.png) to the run folder."""
        if self.scene_map is None or self.scene_map.views == 0:
            return []
        paths = self.scene_map.save(self.out)
        self.log["scene_map"] = {"views": self.scene_map.views, "files": [p.name for p in paths]}
        return paths

    @property
    def rest(self) -> np.ndarray:
        return self.cfg.poses["rest"]

    @property
    def scan(self) -> np.ndarray:
        return self.cfg.poses["scan"]

    # ------------------------------------------------------------------ motion primitives
    def _drive(self, q: np.ndarray, gripper: float | None = None, speed: float | None = None,
               settle_timeout_s: float = 3.0, compensate: bool = True) -> None:
        """Send the arm to joint target `q` and wait for it (every motion below ends here). Camera frames
        from an earlier move no longer match where the arm is, so they are dropped."""
        self._move_frames = []
        self.arm.move_to(q, gripper=gripper, speed_deg_s=speed or self.speed, settle_timeout_s=settle_timeout_s,
                         compensate=compensate)

    def home(self) -> None:
        """Home = the scan pose (above the table, camera on it), gripper closed: where a run starts and what
        go_home returns to. The folded rest pose is only for parking at the end, before torque goes off: on
        its stops it does not fall when the motors go limp (the scan pose would)."""
        self.arm.move_to(self.scan, gripper=0.0, speed_deg_s=self.speed)

    def go_home(self) -> None:
        """Back to home (the scan pose) at <= 30 deg/s, from wherever the arm is."""
        self._drive(self.scan, speed=min(self.speed, 30.0))

    def go_scan(self) -> None:
        """To the scan pose, keeping frames from the way (multi-view depth for a photo taken there)."""
        self.move_collecting(self.scan)

    def _camera_center(self, q: np.ndarray) -> np.ndarray:
        return self.cam_model.pose_in_base(self.kin, q)[:3, 3]

    def move_collecting(self, q: np.ndarray) -> None:
        """Move to `q`, keeping camera frames from the way (with their joints): the other views a photo
        taken at `q` is measured with (look)."""
        ls = self.look_settings
        with MotionFrames(self.arm, self.camera, self._camera_center, min_step_m=ls.min_step_m,
                          camera_latency_s=ls.camera_latency_s) as mf:
            self._drive(q)
        self._move_frames = mf.frames()

    def return_home(self, via_scan: bool) -> None:
        """Back to rest at <= 30 deg/s, first up through the scan pose if the arm may be low or tangled."""
        speed = min(self.speed, 30.0)
        if via_scan:
            self._drive(self.scan, speed=speed)
        self._drive(self.rest, speed=speed)

    def tip(self) -> np.ndarray:
        return self.kin.fk(self.arm.read_joints())[:3, 3]

    # ------------------------------------------------------------------ perception
    def photo(self, q_cmd: np.ndarray | None, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Move to `q_cmd` (if given), let exposure settle, take a fresh photo. Returns (fisheye BGR, joints)."""
        if q_cmd is not None:
            self._drive(q_cmd)
            time.sleep(1.0)  # auto exposure settles after the move
        img = self.camera.read_fresh(0.3)
        q = self.arm.read_joints()
        self.gripper_pct = float(self.arm.read_gripper())
        self.log[f"{tag}_joints_deg"] = q
        self.log[f"{tag}_gripper_pct"] = self.gripper_pct
        self.last_photo = (img, q)
        return img, q

    def look(self, q_cmd: np.ndarray | None, tag: str,
             prior: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, OtherViews]:
        """A photo plus the other views its depth is measured with, at no extra motion: the frames the
        camera passed on its way to `q_cmd` (or on the last move_collecting / go_scan if q_cmd is None),
        spread along the path, and `prior`, an earlier photo (default: the last photo before this move).
        Views closer than MIN_BASELINE_M to this photo's camera are left out (no parallax). If that leaves
        none (the camera has not moved since the last photo), the base turns NUDGE_PAN_DEG and back to get
        some. Returns (fisheye BGR, joints, views) and saves <tag>_v<i>.jpg for each view."""
        if q_cmd is not None:
            if prior is None:
                prior = self.last_photo
            self.move_collecting(q_cmd)
            time.sleep(1.0)  # auto exposure settles after the move
        frames = self._move_frames
        if self.locator is not None:
            self.locator.holding = self.held is not None  # keep the held object out of the scene map
        img, q = self.photo(None, tag)
        here = self._camera_center(q)
        prior_ok = prior is not None and np.linalg.norm(self._camera_center(prior[1]) - here) >= MIN_BASELINE_M
        picked = spread_frames(frames, self._camera_center, self.look_settings.frames, exclude_near=here)
        if not picked and not prior_ok:
            frames = self._parallax_nudge(q)
            picked = spread_frames(frames, self._camera_center, self.look_settings.frames, exclude_near=here)
            self.log[f"{tag}_nudged"] = True
        views = [(f.bgr, f.joints_deg) for f in picked]
        if prior_ok:
            views.insert(0, prior)
        for i, (v_img, _) in enumerate(views):
            self.save_photo(v_img, f"{tag}_v{i}")
        self.log[f"{tag}_views_joints_deg"] = [v_q for _, v_q in views]
        if not views:
            logger.warning("%s: no other views to measure depth with", tag)
        return img, q, views

    def _parallax_nudge(self, q: np.ndarray) -> list:
        """Turn the base NUDGE_PAN_DEG (away from its nearer limit) and back, collecting frames both ways:
        the views for a photo taken where the camera already was. Returns the frames."""
        logger.info("no other views from here: turning the base %.0f deg and back", NUDGE_PAN_DEG)
        lo, hi = (getattr(self.arm, "joint_limits", {}) or {}).get("shoulder_pan", (-np.inf, np.inf))
        sign = 1.0 if q[0] + NUDGE_PAN_DEG <= hi else -1.0
        if not lo <= q[0] + sign * NUDGE_PAN_DEG <= hi:
            return []
        out = np.array(q, dtype=float)
        out[0] += sign * NUDGE_PAN_DEG
        self.move_collecting(out)
        frames = self._move_frames
        self.move_collecting(np.asarray(q, dtype=float))
        frames = frames + self._move_frames
        self._move_frames = frames
        return frames

    def save_photo(self, img: np.ndarray, tag: str) -> Path:
        path = self.out / f"{tag}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def locate(self, img: np.ndarray, q: np.ndarray, query: str, tag: str,
               views: OtherViews | None = None) -> tuple[ObjectEstimate, dict]:
        """VLM points at `query`, SAM masks it, depth measures it. Saves <tag>.jpg and <tag>_locate.png."""
        self.save_photo(img, tag)
        try:
            est, dbg = self.locator.locate(img, q, query, gripper_pct=self.gripper_pct, views=views)
        except PerceptionError as e:
            raise SkillFailed(str(e)) from e
        cv2.imwrite(str(self.out / f"{tag}_locate.png"), draw_debug(est, dbg))
        return est, dbg

    def locate_at(self, img: np.ndarray, q: np.ndarray, point_px: np.ndarray, box_px: np.ndarray | None = None,
                  label: str = "selected object", views: OtherViews | None = None) -> tuple[ObjectEstimate, dict]:
        """Mask and measure whatever is under `point_px` (rectified pixels). Raises PerceptionError."""
        return self.locator.locate_at(img, q, point_px, box_px, label=label, gripper_pct=self.gripper_pct,
                                      views=views)

    def project(self, est: ObjectEstimate, T_base_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Previous estimate's top-surface centre and padded point-cloud box, in a new rectified view."""
        Tinv = np.linalg.inv(T_base_cam)
        P = np.c_[est.points_base, np.ones(len(est.points_base))] @ Tinv.T
        c = Tinv @ np.append(est.center_base, 1.0)
        if c[2] <= 0 or np.any(P[:, 2] <= 0):
            return None
        uv = P[:, :2] / P[:, 2:] @ self.rect.K[:2, :2].T + self.rect.K[:2, 2]
        pt = self.rect.K[:2, :2] @ (c[:2] / c[2]) + self.rect.K[:2, 2]
        lo, hi = uv.min(0), uv.max(0)
        pad = 0.25 * (hi - lo) + 10
        w, h = self.rect.size
        box = np.clip(np.r_[lo - pad, hi + pad], 0, [w - 1, h - 1, w - 1, h - 1])
        return pt, box

    def relocate(self, img: np.ndarray, q: np.ndarray, name: str, prev: ObjectEstimate, tag: str,
                 views: OtherViews | None = None):
        """Find the same object again from a new pose: VLM by name, accepted only if it lands near the
        previous 3D estimate; otherwise SAM at the previous estimate projected into this view.
        Returns (estimate, debug, how it was found)."""
        self.save_photo(img, tag)
        how = "VLM"
        try:
            est, dbg = self.locator.locate(img, q, name, gripper_pct=self.gripper_pct, views=views)
            jump = float(np.linalg.norm(est.center_base[:2] - prev.center_base[:2]))
            if jump > RELOCATE_MAX_JUMP_M:
                raise PerceptionError(f"VLM answer is {jump * 100:.0f} cm from the first estimate")
        except PerceptionError as e:
            proj = self.project(prev, self.cam_model.pose_in_base(self.kin, q))
            if proj is None:
                raise SkillFailed(f"lost the object: {e}") from e
            try:
                est, dbg = self.locator.locate_at(img, q, proj[0], proj[1], label=name, gripper_pct=self.gripper_pct,
                                                   views=views)
            except PerceptionError as e2:
                raise SkillFailed(f"lost the object: {e}; projection: {e2}") from e2
            how = f"projected first estimate ({e})"
        cv2.imwrite(str(self.out / f"{tag}_locate.png"), draw_debug(est, dbg))
        return est, dbg, how

    # ------------------------------------------------------------------ planning
    def plan_relook(self, est: ObjectEstimate, from_q: np.ndarray) -> np.ndarray:
        """Joints that put the camera 12-22 cm from the object, looking at it."""
        try:
            return self.planner.plan_relook(est.center_base * [1, 1, 0], from_q)
        except GraspPlanningError as e:
            raise SkillFailed(str(e)) from e

    def look_closer(self, est: ObjectEstimate, name: str, from_q: np.ndarray, tag: str = "relook",
                    prior: tuple[np.ndarray, np.ndarray] | None = None):
        """Move the camera close to the object and measure it again (`prior`: the photo `est` came from,
        another view for depth). Returns (estimate, debug, how, joints)."""
        relook = self.plan_relook(est, from_q)
        img, q, views = self.look(relook, tag, prior)
        est2, dbg, how = self.relocate(img, q, name, est, tag, views)
        return est2, dbg, how, q

    def plan_grasp(self, est: ObjectEstimate, from_q: np.ndarray, dbg: dict, tag: str) -> tuple[GraspPlan, dict]:
        """Top-down grasp of the measured object; saves <tag>_grasp_overlay.jpg. Returns the plan and
        {"detail": for display, "raw": for the log}."""
        top = est.points_base[est.points_base[:, 2] > est.top_z_m - 0.01]
        axis = object_axis_xy(top) if len(top) else None
        target = np.array([est.center_base[0], est.center_base[1], grasp_height(est.top_z_m)])
        try:
            self.cfg.check_in_workspace(target)
            plan = self.planner.plan_grasp(target, axis, from_q)
        except (ValueError, GraspPlanningError) as e:
            raise SkillFailed(f"cannot grasp: {e}") from e
        cv2.imwrite(str(self.out / f"{tag}_grasp_overlay.jpg"),
                    overlay_grasp(dbg["rect_bgr"].copy(), est, self.rect, target, plan.closing_xy, axis))
        detail = {
            "fingertips_cm": [round(float(v) * 100, 1) for v in target],
            "jaw_alignment": "object not elongated, any wrist angle" if axis is None
            else f"{plan.misalignment_deg:.0f} deg from across the long axis",
            "wrist_roll_deg": round(float(plan.grasp_deg[4]), 0),
            "pregrasp_deg": deg_list(plan.pregrasp_deg),
            "grasp_deg": deg_list(plan.grasp_deg),
            "lift_deg": deg_list(plan.lift_deg),
        }
        raw = {"target_m": target, "object_axis_xy": axis, "closing_xy": plan.closing_xy,
               "misalignment_deg": plan.misalignment_deg, "pregrasp_deg": plan.pregrasp_deg,
               "grasp_deg": plan.grasp_deg, "lift_deg": plan.lift_deg}
        return plan, {"detail": detail, "raw": raw}

    # ------------------------------------------------------------------ grasp motions
    def pregrasp(self, plan: GraspPlan) -> None:
        """Open the gripper on the way to the pose above the object."""
        self._drive(plan.pregrasp_deg, gripper=OPEN_PCT)

    def descend(self, plan: GraspPlan) -> np.ndarray:
        """Slowly down to the grasp pose. Returns the measured fingertip point."""
        self._drive(plan.grasp_deg, speed=DESCEND_SPEED)
        return self.tip()

    def close(self) -> Grip:
        self.arm.set_gripper(0.0, duration_s=1.0)
        return Grip(*read_grip(self.arm))

    def grip(self) -> Grip:
        """The gripper's opening and squeeze force right now (is anything between the fingers?)."""
        return Grip(*read_grip(self.arm, seconds=0.3))

    def lift(self, plan: GraspPlan) -> Grip:
        """Up to the lift pose; the returned grip says whether something is held."""
        self._drive(plan.lift_deg, speed=DESCEND_SPEED)
        return Grip(*read_grip(self.arm))

    def place_back(self, plan: GraspPlan) -> None:
        """Put the held object back where it was picked, open, back off, close."""
        self._drive(plan.grasp_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
        self._drive(plan.lift_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(0.0, duration_s=0.6)
        self.held = None

    # ------------------------------------------------------------------ general motions (agent)
    def _approach_axis(self, point_m: np.ndarray, approach: str) -> np.ndarray:
        if approach == "down":
            return DOWN
        radial = np.append(point_m[:2] - [0.0388, 0.0], 0.0)
        return radial / max(np.linalg.norm(radial), 1e-9)

    def solve(self, point_m: np.ndarray, approach: str = "auto", from_q: np.ndarray | None = None,
              axis_tol_deg: float = 15.0) -> tuple[np.ndarray, str]:
        """Joints that put the fingertips at `point_m` coming from `approach` (down | forward | auto: down if
        possible, else forward, else tilted in between). Checked against the table, self-collision and
        the soft limits. Raises SkillFailed saying why not."""
        point_m = np.asarray(point_m, dtype=float)
        if approach not in APPROACHES:
            raise SkillFailed(f"approach must be one of {APPROACHES}")
        if point_m[2] < 0.0:
            raise SkillFailed(f"point {np.round(point_m * 100, 1)} cm is below the table")
        seed = self.arm.read_joints() if from_q is None else from_q
        tries = [approach] if approach != "auto" else ["down", "forward"]
        reasons = []
        for a in tries:
            axis = self._approach_axis(point_m, a)
            res = self.kin.ik(point_m, seed, approach_axis=axis, axis_tol_deg=axis_tol_deg)
            if not res.success:
                reasons.append(f"{a}: {res.position_error_m * 100:.1f} cm short / {res.axis_error_deg:.0f} deg off")
                continue
            q = res.joints_deg
            if min_link_height(self.kin, q) < 0.0 or self.planner.cc.in_collision(q):
                reasons.append(f"{a}: the arm would hit the table or itself")
                continue
            return q, a
        if approach == "auto":
            # 5 joints: many points are only reachable tilted. Accept any axis between down and forward.
            res = self.kin.ik(point_m, seed, approach_axis=None)
            if res.success and min_link_height(self.kin, res.joints_deg) >= 0.0 \
                    and not self.planner.cc.in_collision(res.joints_deg):
                return res.joints_deg, "tilted"
            reasons.append(f"any angle: {res.position_error_m * 100:.1f} cm short")
        raise SkillFailed(f"cannot reach {np.round(point_m * 100, 1)} cm ({'; '.join(reasons)})")

    def _transit_q(self, xy: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        res = self.kin.ik(np.array([xy[0], xy[1], TRANSIT_Z_M]), seed, approach_axis=DOWN, axis_tol_deg=40.0)
        return res.joints_deg if res.success else None

    def travel_to(self, q_goal: np.ndarray, gripper: float | None = None, speed: float | None = None) -> None:
        """Go to `q_goal` without sweeping across the table at object height: straight up to TRANSIT_Z_M,
        across at that height, then down to the goal. Moves that stay high anyway go direct."""
        here = self.arm.read_joints()
        tip_here, tip_goal = self.kin.fk(here)[:3, 3], self.kin.fk(q_goal)[:3, 3]
        far = np.linalg.norm(tip_here[:2] - tip_goal[:2]) > 0.03
        low = min(tip_here[2], tip_goal[2]) < TRANSIT_Z_M - 0.01
        if far and low:
            up = self._transit_q(tip_here[:2], here) if tip_here[2] < TRANSIT_Z_M - 0.01 else None
            over = self._transit_q(tip_goal[:2], up if up is not None else here)
            for q in (up, over):
                if q is not None and path_is_safe(self.kin, self.planner.cc, self.arm.read_joints(), q, 0.0):
                    self._drive(q)
        if not path_is_safe(self.kin, self.planner.cc, self.arm.read_joints(), q_goal, 0.0):
            # Straight joint-space path dips: go up through the scan pose first.
            if not path_is_safe(self.kin, self.planner.cc, self.scan, q_goal, 0.0):
                raise SkillFailed("no safe path there")
            self._drive(self.scan)
        self._drive(q_goal, gripper=gripper, speed=speed)

    def move_to(self, point_m: np.ndarray, approach: str = "auto", speed: float | None = None,
                gripper: float | None = None, axis_tol_deg: float = 15.0) -> str:
        """Fingertips to `point_m`, travelling high (travel_to). Returns the approach used."""
        q, used = self.solve(point_m, approach, axis_tol_deg=axis_tol_deg)
        self.travel_to(q, gripper=gripper, speed=speed)
        return used

    def place_at(self, xy_m: np.ndarray, on_z_m: float = 0.0) -> np.ndarray:
        """Set the held object down with its centre at `xy_m`, its bottom at height `on_z_m` (0 = the table;
        the top of something to stack it on): fingertips as high above that as when it was picked, open, back
        up, close. Returns the placed position (x, y, top z)."""
        if self.held is None:
            raise SkillFailed("not holding anything")
        h = self.held
        target = np.array([xy_m[0], xy_m[1], on_z_m + h.hang_m + 0.005])
        q = self.arm.read_joints()
        where = f"x={xy_m[0] * 100:.0f} y={xy_m[1] * 100:.0f} cm" + (f", {on_z_m * 100:.0f} cm up" if on_z_m >= 0.005 else "")
        try:
            self.cfg.check_in_workspace(target)
        except ValueError as e:
            raise SkillFailed(f"cannot set it down at {where}: that is outside the arm's reach") from e
        plan = self._place_plan(target, q)
        if plan is None:
            ok_x = [x for x in PLACE_SCAN_X_M if self._place_plan(np.array([x, target[1], target[2]]), q) is not None]
            hint = (f"at y={xy_m[1] * 100:.0f} cm and this height it can set things down from x={ok_x[0] * 100:.0f} "
                    f"to {ok_x[-1] * 100:.0f} cm" if ok_x else "nowhere along that y at this height")
            raise SkillFailed(f"cannot set it down at {where}: the arm cannot hold it straight above that spot "
                              f"({hint})")
        self.travel_to(plan.pregrasp_deg)
        self._drive(plan.grasp_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
        self._drive(plan.lift_deg, speed=DESCEND_SPEED)
        self.held = None
        top = on_z_m + h.top_z_m
        self._rise(top + PLACE_CLEAR_M)  # clear of it before closing and moving on, or the fingers knock it over
        self.arm.set_gripper(0.0, duration_s=0.6)
        return np.array([xy_m[0], xy_m[1], top])

    def _rise(self, to_z_m: float) -> None:
        """Fingertips straight up to height to_z_m (in PATH_STEP_M steps), as far as the arm can go."""
        q = self.arm.read_joints()
        tip = self.kin.fk(q)[:3, 3]
        n = int(np.ceil((to_z_m - tip[2]) / PATH_STEP_M))
        for z in np.linspace(tip[2], to_z_m, n + 1)[1:] if n > 0 else []:
            try:
                q_next, _ = self._solve_near(np.array([tip[0], tip[1], z]), q)
            except SkillFailed:
                break
            if np.max(np.abs(q_next - q)) > PATH_MAX_JUMP_DEG:
                break
            self._drive(q_next, speed=DESCEND_SPEED)
            q = q_next

    def _place_plan(self, target_m: np.ndarray, q: np.ndarray) -> GraspPlan | None:
        """A top-down plan to bring the held object's fingertip point to target_m, trying PLACE_HOVERS_M in
        order; None if none works."""
        for pre, lift in PLACE_HOVERS_M:
            try:
                return self.planner.plan_grasp(target_m, None, q, pregrasp_height_m=pre, lift_height_m=lift)
            except GraspPlanningError as e:
                logger.debug("place plan at %s (hover %.0f/%.0f cm): %s", np.round(target_m * 100, 1),
                             pre * 100, lift * 100, e)
        return None

    def _solve_near(self, point_m: np.ndarray, q_prev: np.ndarray) -> tuple[np.ndarray, str]:
        """Like solve(), but of the valid solutions (down, forward, any angle) the one closest to q_prev, so a
        path keeps its hand orientation instead of switching approach at every step."""
        found, reasons = [], []
        for a in ("down", "forward", "tilted"):
            if a == "tilted":
                res = self.kin.ik(point_m, q_prev, approach_axis=None)
                ok = res.success and min_link_height(self.kin, res.joints_deg) >= 0.0 \
                    and not self.planner.cc.in_collision(res.joints_deg)
                if ok:
                    found.append((res.joints_deg, a))
                continue
            try:
                found.append(self.solve(point_m, a, from_q=q_prev))
            except SkillFailed as e:
                reasons.append(str(e))
        if not found:
            raise SkillFailed(reasons[0] if reasons else f"cannot reach {np.round(point_m * 100, 1)} cm")
        return min(found, key=lambda qa: float(np.max(np.abs(qa[0] - q_prev))))

    def move_path(self, points_m, gripper: str | None = None) -> list[str]:
        """Fingertips through `points_m` (base frame) in order: to the first by the usual high travel
        (travel_to), then in straight lines from point to point, in PATH_STEP_M steps. The whole path is
        solved and checked (table, self-collision, soft limits, no sudden wrist flips) before anything
        moves; a SkillFailed says which point is the problem. gripper: "open" / "close" after the last point
        (opening while holding lets go of the object). Returns the approach used at each point."""
        pts = [np.asarray(p, dtype=float) for p in points_m]
        if not pts:
            raise SkillFailed("no points to move through")
        for i, p in enumerate(pts, 1):
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                raise SkillFailed(f"point {i} must be three numbers (x, y, z)")
            if p[2] < PATH_MIN_Z_M:
                raise SkillFailed(f"point {i} is {p[2] * 100:.1f} cm high; the fingertips must stay at least "
                                  f"{PATH_MIN_Z_M * 100:.0f} cm above the table")
        if gripper not in (None, "open", "close"):
            raise SkillFailed('gripper must be "open", "close" or left out')
        q_first, a = self.solve(pts[0], "auto")
        q_prev, used, steps = q_first, [a], []
        for i in range(1, len(pts)):
            seg = pts[i] - pts[i - 1]
            n = max(1, int(np.ceil(np.linalg.norm(seg) / PATH_STEP_M)))
            for k in range(1, n + 1):
                p = pts[i - 1] + seg * (k / n)
                try:
                    q, a = self._solve_near(p, q_prev)
                except SkillFailed as e:
                    raise SkillFailed(f"on the way to point {i + 1}: {e}") from None
                if np.max(np.abs(q - q_prev)) > PATH_MAX_JUMP_DEG:
                    raise SkillFailed(f"on the way to point {i + 1} the arm would have to flip its wrist "
                                      f"(near {np.round(p * 100, 1)} cm); use other points")
                steps.append(q)
                q_prev = q
            used.append(a)
        self.travel_to(q_first)
        for q in steps:
            self._drive(q, settle_timeout_s=1.0)
        if gripper == "open":
            self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
            self.held = None
        elif gripper == "close":
            self.arm.set_gripper(0.0, duration_s=0.6)
        return used
