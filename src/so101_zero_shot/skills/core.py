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
from ..grasping.topdown import GraspPlan, GraspPlanningError, TopDownPlanner, grasp_height, object_axis_xy
from ..kinematics import SO101Kinematics
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import Locator, ObjectEstimate, OtherViews, PerceptionError, draw_debug
from .motion_frames import MIN_BASELINE_M, Frame, LookSettings, MotionFrames, spread_frames

logger = logging.getLogger(__name__)

RELOCATE_MAX_JUMP_M = 0.04  # closer look: a VLM answer further than this from the first estimate is another object
OPEN_PCT = 100.0  # = the user's max safe opening (gripper calibration range_min)
DESCEND_SPEED = 15.0
DOWN = np.array([0.0, 0.0, -1.0])
APPROACHES = ("down", "forward", "auto")
# touch(): creep toward the target in steps this long, at this speed, and call it contact when a joint
# lags its command by this much or a load (0.1 % of max torque) rises this far above the free-space value.
# Measured in the MuJoCo sim (see tests/test_skills.py); to be measured on the real arm like the gripper.
TOUCH_STEP_M = 0.004
TOUCH_SPEED = 8.0
TOUCH_LAG_DEG = 1.5
TOUCH_LOAD_RISE = 150.0
TRANSIT_Z_M = 0.13  # fingertips this high when travelling between spots (a held object hangs ~3-8 cm below)
TOUCH_STANDOFF_M = 0.04
TOUCH_REF_STEPS = 3  # 12 mm of free-air creep before the target can be reached
TOUCH_AXIS_TOL_DEG = 30.0  # 5 joints: straight down / straight forward are only reachable in parts of the workspace


class SkillFailed(Exception):
    """An expected failure (not found, unreachable, missed, ...); the message says why."""


@dataclass
class Held:
    """What the gripper holds after a successful lift (for putting it back or placing it elsewhere)."""
    name: str
    plan: GraspPlan  # the grasp it was picked with: plan.target_m is the fingertip point on the object
    picked_from_m: np.ndarray  # object top-centre where it was picked
    top_z_m: float  # its height


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

    @property
    def rest(self) -> np.ndarray:
        return self.cfg.poses["rest"]

    @property
    def scan(self) -> np.ndarray:
        return self.cfg.poses["scan"]

    # ------------------------------------------------------------------ motion primitives
    def move_joints(self, q: np.ndarray, gripper: float | None = None, speed: float | None = None,
                    settle_timeout_s: float = 3.0, compensate: bool = True) -> None:
        self._move_frames = []
        self.arm.move_to(q, gripper=gripper, speed_deg_s=speed or self.speed, settle_timeout_s=settle_timeout_s,
                         compensate=compensate)

    def home(self) -> None:
        """Rest pose, gripper closed (start of a run: no settle wait, like before)."""
        self.arm.move_to(self.rest, gripper=0.0, speed_deg_s=self.speed)

    def go_scan(self) -> None:
        """To the scan pose, keeping frames from the way (multi-view depth for a photo taken there)."""
        self.move_collecting(self.scan)

    def _camera_center(self, q: np.ndarray) -> np.ndarray:
        return self.cam_model.pose_in_base(self.kin, q)[:3, 3]

    def move_collecting(self, q: np.ndarray) -> None:
        """move_joints, keeping camera frames from the way (with their joints): the other views a photo
        taken at `q` is measured with (look)."""
        ls = self.look_settings
        with MotionFrames(self.arm, self.camera, self._camera_center, min_step_m=ls.min_step_m,
                          camera_latency_s=ls.camera_latency_s) as mf:
            self.move_joints(q)
        self._move_frames = mf.frames()

    def return_home(self, via_scan: bool) -> None:
        """Back to rest at <= 30 deg/s, first up through the scan pose if the arm may be low or tangled."""
        speed = min(self.speed, 30.0)
        if via_scan:
            self.move_joints(self.scan, speed=speed)
        self.move_joints(self.rest, speed=speed)

    def tip(self) -> np.ndarray:
        return self.kin.fk(self.arm.read_joints())[:3, 3]

    # ------------------------------------------------------------------ perception
    def photo(self, q_cmd: np.ndarray | None, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Move to `q_cmd` (if given), let exposure settle, take a fresh photo. Returns (fisheye BGR, joints)."""
        if q_cmd is not None:
            self.move_joints(q_cmd)
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
        Views closer than MIN_BASELINE_M to this photo's camera are left out (no parallax). Returns
        (fisheye BGR, joints, views) and saves <tag>_v<i>.jpg for each view."""
        if q_cmd is not None:
            if prior is None:
                prior = self.last_photo
            self.move_collecting(q_cmd)
            time.sleep(1.0)  # auto exposure settles after the move
        frames = self._move_frames
        img, q = self.photo(None, tag)
        here = self._camera_center(q)
        picked = spread_frames(frames, self._camera_center, self.look_settings.frames, exclude_near=here)
        views = [(f.bgr, f.joints_deg) for f in picked]
        if prior is not None and np.linalg.norm(self._camera_center(prior[1]) - here) >= MIN_BASELINE_M:
            views.insert(0, prior)
        for i, (v_img, _) in enumerate(views):
            self.save_photo(v_img, f"{tag}_v{i}")
        self.log[f"{tag}_views_joints_deg"] = [v_q for _, v_q in views]
        if not views:
            logger.warning("%s: no other views to measure depth with", tag)
        return img, q, views

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
        self.move_joints(plan.pregrasp_deg, gripper=OPEN_PCT)

    def descend(self, plan: GraspPlan) -> np.ndarray:
        """Slowly down to the grasp pose. Returns the measured fingertip point."""
        self.move_joints(plan.grasp_deg, speed=DESCEND_SPEED)
        return self.tip()

    def close(self) -> Grip:
        self.arm.set_gripper(0.0, duration_s=1.0)
        return Grip(*read_grip(self.arm))

    def lift(self, plan: GraspPlan) -> Grip:
        """Up to the lift pose; the returned grip says whether something is held."""
        self.move_joints(plan.lift_deg, speed=DESCEND_SPEED)
        return Grip(*read_grip(self.arm))

    def place_back(self, plan: GraspPlan) -> None:
        """Put the held object back where it was picked, open, back off, close."""
        self.move_joints(plan.grasp_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
        self.move_joints(plan.lift_deg, speed=DESCEND_SPEED)
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
                    self.move_joints(q)
        if not path_is_safe(self.kin, self.planner.cc, self.arm.read_joints(), q_goal, 0.0):
            # Straight joint-space path dips: go up through the scan pose first.
            if not path_is_safe(self.kin, self.planner.cc, self.scan, q_goal, 0.0):
                raise SkillFailed("no safe path there")
            self.move_joints(self.scan)
        self.move_joints(q_goal, gripper=gripper, speed=speed)

    def move_to(self, point_m: np.ndarray, approach: str = "auto", speed: float | None = None,
                gripper: float | None = None, axis_tol_deg: float = 15.0) -> str:
        """Fingertips to `point_m`, travelling high (travel_to). Returns the approach used."""
        q, used = self.solve(point_m, approach, axis_tol_deg=axis_tol_deg)
        self.travel_to(q, gripper=gripper, speed=speed)
        return used

    def place_at(self, xy_m: np.ndarray) -> np.ndarray:
        """Set the held object down with its centre at `xy_m` (table): same fingertip height as it was
        picked at, open, back up, close. Returns the placed position (x, y, top z)."""
        if self.held is None:
            raise SkillFailed("not holding anything")
        h = self.held
        target = np.array([xy_m[0], xy_m[1], h.plan.target_m[2] + 0.005])
        try:
            self.cfg.check_in_workspace(target)
            plan = self.planner.plan_grasp(target, None, self.arm.read_joints())
        except (ValueError, GraspPlanningError) as e:
            raise SkillFailed(f"cannot place there: {e}") from e
        self.travel_to(plan.pregrasp_deg)
        self.move_joints(plan.grasp_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
        self.move_joints(plan.lift_deg, speed=DESCEND_SPEED)
        self.arm.set_gripper(0.0, duration_s=0.6)
        self.held = None
        return np.array([xy_m[0], xy_m[1], h.top_z_m])

    def release_above(self, point_m: np.ndarray, height_m: float = 0.06) -> None:
        """Hold the object `height_m` above `point_m` (e.g. a hand) and let go."""
        if self.held is None:
            raise SkillFailed("not holding anything")
        self.move_to(np.asarray(point_m, dtype=float) + [0, 0, height_m], approach="auto")
        self.arm.set_gripper(OPEN_PCT, duration_s=0.8)
        time.sleep(0.5)
        self.arm.set_gripper(0.0, duration_s=0.6)
        self.held = None

    def _steady_state(self, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Median (position, load) over a few reads after a short pause: right after a move the load
        reading is dominated by the motion itself (measured in sim: free-air spikes of +150 to +300)."""
        time.sleep(0.15)
        reads = [self.arm.read_state(num_retry=2) for _ in range(n)]
        return np.median([r[0] for r in reads], axis=0), np.median([r[1] for r in reads], axis=0)

    def touch(self, point_m: np.ndarray, approach: str = "auto", max_push_m: float = 0.03) -> dict:
        """Touch `point_m` gently with the closed gripper: line up TOUCH_STANDOFF_M before it, creep along the
        approach in small slow steps, stop at the first sign of contact or `max_push_m` past the point, then
        back off to the standoff. Contact = a joint lagging its command, or a joint load, more than the
        free-space creep did (the first TOUCH_REF_STEPS steps are in free air and set the reference).
        Returns {"contact", "tip_m" (where it stopped), "travel_m" (past the point; < 0 = before it),
        "approach", "signal", "trace" (per step: lag deg, load rise)}."""
        point_m = np.asarray(point_m, dtype=float)
        self.arm.set_gripper(0.0, duration_s=0.4)
        if approach == "auto":
            # Prefer coming from the robot's side for something standing up (a hand), from above otherwise.
            tries = ["forward", "down"] if point_m[2] > 0.06 else ["down", "forward"]
        else:
            tries = [approach]
        reasons = []
        for a in tries:
            axis = self._approach_axis(point_m, a)
            try:
                q_stand, _ = self.solve(point_m - TOUCH_STANDOFF_M * axis, a, axis_tol_deg=TOUCH_AXIS_TOL_DEG)
                break
            except SkillFailed as e:
                reasons.append(str(e))
        else:
            raise SkillFailed(f"cannot line up to touch it: {'; '.join(reasons)}")
        self.travel_to(q_stand)
        total = TOUCH_STANDOFF_M + max_push_m
        n = int(np.ceil(total / TOUCH_STEP_M))
        q, signal, travel = q_stand, None, 0.0
        trace: list[tuple[float, float]] = []
        ref_lag, ref_load = None, None
        for i in range(1, n + 1):
            d = min(i * TOUCH_STEP_M, total)
            res = self.kin.ik(point_m + (d - TOUCH_STANDOFF_M) * axis, q, approach_axis=axis,
                              axis_tol_deg=TOUCH_AXIS_TOL_DEG)
            if not res.success:
                signal = "out of reach"
                break
            q = res.joints_deg
            self.arm.move_to(q, speed_deg_s=TOUCH_SPEED, settle_timeout_s=0.4, compensate=False)
            pos, load = self._steady_state()
            lag = np.abs(q - pos[:5])
            travel = d
            if i <= TOUCH_REF_STEPS:  # free air: how much lag/load plain creeping causes here
                ref_lag = lag if ref_lag is None else np.maximum(ref_lag, lag)
                ref_load = load[:5] if ref_load is None else ref_load
                trace.append((round(float(lag.max()), 2), 0.0))
                continue
            lag_rise = float(np.max(lag - ref_lag))
            load_rise = float(np.max(np.abs(load[:5] - ref_load)))
            trace.append((round(lag_rise, 2), round(load_rise, 1)))
            if lag_rise > TOUCH_LAG_DEG or load_rise > TOUCH_LOAD_RISE:
                signal = f"lag +{lag_rise:.1f} deg, load +{load_rise:.0f}"
                break
        tip = self.tip()
        contact = signal is not None and signal != "out of reach"
        self.arm.move_to(q_stand, speed_deg_s=TOUCH_SPEED * 2, settle_timeout_s=1.0, compensate=False)
        return {"contact": contact, "tip_m": tip, "travel_m": travel - TOUCH_STANDOFF_M, "approach": a,
                "signal": signal or "nothing felt", "trace": trace}
