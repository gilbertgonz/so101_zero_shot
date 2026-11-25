"""Top-down grasp planning: re-look pose over a target, then pregrasp / grasp / lift joints.

Closing direction: the custom parallel gripper's fingers sit left and right in the wrist camera image,
so the jaws close along the camera x axis. That axis is known in the hand frame from the camera
calibration; the MuJoCo parallel gripper's clamp axis agrees with it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.viewpoints import gripper_radius, min_link_height, path_is_safe
from ..kinematics import TIP_FRAME, SO101Kinematics
from ..kinematics.collision import CollisionChecker
from ..perception.geometry import FisheyeCamera

DOWN = np.array([0.0, 0.0, -1.0])
# Bodies allowed to come close to the table during the final descent (the fingers must).
_GRIPPER_BODIES = {"gripper", "clamp_right", "clamp_left"}


class GraspPlanningError(RuntimeError):
    pass


def closing_axis_hand(cam: FisheyeCamera, kin: SO101Kinematics) -> np.ndarray:
    """Unit jaw-closing direction in cam.hand_frame (camera x, projected perpendicular to the approach)."""
    q0 = np.zeros(5)
    approach = (np.linalg.inv(kin.fk(q0, cam.hand_frame)) @ kin.fk(q0, TIP_FRAME))[:3, 2]
    c = cam.T_hand_camera[:3, 0]
    c = c - (c @ approach) * approach
    return c / np.linalg.norm(c)


def object_axis_xy(points_base: np.ndarray, min_elongation: float = 1.5) -> np.ndarray | None:
    """Long axis (unit xy) of the object's top-view footprint, or None if it is not clearly elongated."""
    xy = points_base[:, :2] - np.median(points_base[:, :2], axis=0)
    if len(xy) < 10:
        return None
    w, v = np.linalg.eigh(np.cov(xy.T))
    if w[0] <= 0 or np.sqrt(w[1] / w[0]) < min_elongation:
        return None
    return v[:, 1]


@dataclass
class GraspPlan:
    target_m: np.ndarray  # fingertip point at the grasp
    pregrasp_deg: np.ndarray
    grasp_deg: np.ndarray
    lift_deg: np.ndarray
    closing_xy: np.ndarray  # jaw closing direction in the base xy plane
    misalignment_deg: float  # angle between closing direction and the object's short axis (0 = ideal)


class TopDownPlanner:
    def __init__(
        self,
        kin: SO101Kinematics,
        cc: CollisionChecker,
        cam: FisheyeCamera,
        rest_deg: np.ndarray,
        clearance_m: float = 0.03,
        roll_limits_deg: tuple[float, float] = (-150.0, 150.0),
    ):
        self.kin, self.cc, self.cam = kin, cc, cam
        self.rest = np.asarray(rest_deg, dtype=float)
        self.clearance = clearance_m
        self.roll_limits = roll_limits_deg
        self.closing_hand = closing_axis_hand(cam, kin)

    # ------------------------------------------------------------------ checks
    def _self_collision_only(self, q: np.ndarray) -> bool:
        """Collisions other than the gripper/jaw touching the table margin (expected when grasping low)."""
        return any(
            not ({a, b} & {"world"} and {a, b} & _GRIPPER_BODIES)
            for a, b, _ in self.cc.contacts(q)
        )

    def _safe_pose(self, q: np.ndarray) -> bool:
        return (min_link_height(self.kin, q) >= self.clearance and gripper_radius(self.kin, q) >= 0.08
                and not self.cc.in_collision(q))

    def _closing_xy(self, q: np.ndarray) -> np.ndarray:
        c = self.kin.fk(q, self.cam.hand_frame)[:3, :3] @ self.closing_hand
        c = c[:2]
        return c / max(np.linalg.norm(c), 1e-9)

    # ------------------------------------------------------------------ re-look
    def plan_relook(
        self, target_m: np.ndarray, from_deg: np.ndarray, distance_m: tuple[float, float] = (0.12, 0.22),
        n_samples: int = 1500, seed: int = 0,
    ) -> np.ndarray:
        """Joints that put the camera 12-22 cm from the target, looking at it (near the image centre).

        The camera sits 8-10 cm up the wrist from the fingertips, so the fingertips go 4-14 cm above the
        table, leaning outward (the optical axis is ~30 deg off the approach axis)."""
        rng = np.random.default_rng(seed)
        target_m = np.asarray(target_m, dtype=float)
        best, best_score = None, np.inf
        for _ in range(n_samples):
            tip = target_m + np.array([rng.uniform(-0.10, 0.04), rng.uniform(-0.05, 0.05), rng.uniform(0.04, 0.14)])
            radial = np.append(tip[:2] - [0.0388, 0.0], 0.0)
            radial /= np.linalg.norm(radial)
            pitch = np.radians(rng.uniform(0.0, 50.0))
            axis = np.cos(pitch) * DOWN + np.sin(pitch) * radial
            res = self.kin.ik(tip, from_deg, approach_axis=axis, wrist_roll_deg=0.0, axis_tol_deg=5.0)
            if not res.success:
                continue
            q = res.joints_deg
            T = self.cam.pose_in_base(self.kin, q)
            to_t = target_m - T[:3, 3]
            dist = float(np.linalg.norm(to_t))
            off_axis = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ to_t / dist, -1, 1))))
            if not distance_m[0] <= dist <= distance_m[1] or off_axis > 20.0:
                continue
            score = off_axis + 0.2 * float(np.max(np.abs(q - from_deg))) / 10.0
            if score >= best_score or not self._safe_pose(q):
                continue
            if not path_is_safe(self.kin, self.cc, from_deg, q, self.clearance - 0.01):
                continue
            if not path_is_safe(self.kin, self.cc, q, self.rest, self.clearance - 0.01):
                continue
            best, best_score = q, score
        if best is None:
            raise GraspPlanningError(f"no safe re-look pose for target {np.round(target_m, 3)}")
        return best

    # ------------------------------------------------------------------ grasp
    def plan_grasp(
        self, target_m: np.ndarray, object_axis: np.ndarray | None, from_deg: np.ndarray,
        pregrasp_height_m: float = 0.05, lift_height_m: float = 0.07, hover_axis_tol_deg: float = 20.0,
    ) -> GraspPlan:
        """Straight-down grasp at `target_m` (fingertip point). The wrist roll makes the jaws close across
        `object_axis` (xy) if given. Pregrasp and lift are directly above the target.

        Straight down is only reachable with the fingertips low (<= ~6 cm at x ~ 0.25 m), so pregrasp and
        lift may tilt up to `hover_axis_tol_deg`; only the grasp itself is held to 8 deg."""
        target_m = np.asarray(target_m, dtype=float)
        pre_p = target_m + [0, 0, pregrasp_height_m]
        lift_p = target_m + [0, 0, lift_height_m]
        seed = from_deg
        best = None
        for roll in np.arange(self.roll_limits[0], self.roll_limits[1] + 1e-6, 10.0):
            pre = self.kin.ik(pre_p, seed, approach_axis=DOWN, wrist_roll_deg=roll, axis_tol_deg=hover_axis_tol_deg)
            if not pre.success:
                continue
            grasp = self.kin.ik(target_m, pre.joints_deg, approach_axis=DOWN, wrist_roll_deg=roll, axis_tol_deg=8.0)
            lift = self.kin.ik(lift_p, pre.joints_deg, approach_axis=DOWN, wrist_roll_deg=roll, axis_tol_deg=hover_axis_tol_deg)
            if not (grasp.success and lift.success):
                continue
            c = self._closing_xy(grasp.joints_deg)
            mis = 0.0 if object_axis is None else float(np.degrees(np.arcsin(min(1.0, abs(c @ object_axis)))))
            # Prefer good alignment, then small roll (less wrist cable twist, closer to the IK seed).
            score = mis + abs(roll) / 30.0
            if best is not None and score >= best[0]:
                continue
            best = (score, GraspPlan(target_m, pre.joints_deg, grasp.joints_deg, lift.joints_deg, c, mis))
        if best is None:
            raise GraspPlanningError(f"no top-down IK solution at {np.round(target_m, 3)}")
        plan = best[1]
        # Safety: pregrasp/lift clear of everything; the descent may bring only the fingers near the table.
        for name, q in (("pregrasp", plan.pregrasp_deg), ("lift", plan.lift_deg)):
            if not self._safe_pose(q):
                raise GraspPlanningError(f"{name} pose is too close to the table or in collision")
        for t in np.linspace(0, 1, 15):
            q = plan.pregrasp_deg + t * (plan.grasp_deg - plan.pregrasp_deg)
            if self._self_collision_only(q):
                raise GraspPlanningError("descent collides (other than fingers near the table)")
            if min(self.kin.fk(q, link)[2, 3] for link in ("wrist_link", "lower_arm_link")) < 0.02:
                raise GraspPlanningError("descent brings the wrist too close to the table")
        if not path_is_safe(self.kin, self.cc, from_deg, plan.pregrasp_deg, self.clearance - 0.01):
            raise GraspPlanningError("path to pregrasp is unsafe")
        if not path_is_safe(self.kin, self.cc, plan.lift_deg, self.rest, self.clearance - 0.01):
            raise GraspPlanningError("path from lift back to rest is unsafe")
        return plan


def grasp_height(top_z_m: float, min_tip_z_m: float = 0.010, depth_into_object_m: float = 0.025) -> float:
    """Fingertip height for a top-down grasp: a little below the object's top, never below `min_tip_z_m`."""
    return float(max(min_tip_z_m, min(top_z_m - depth_into_object_m, top_z_m / 2.0)))
