# FK/IK wrapper inspired by LeRobot's lerobot/model/kinematics.py (Apache-2.0), extended with a
# converged, limit-aware top-down solver for the 5-DOF SO-101.
"""Forward and inverse kinematics for the SO-101 using placo and the vendored URDF.

Frames:
- `base_link`: robot base. +x forward, +z up.
- `gripper_frame_link`: point between the fingertips. Its +z axis points out along the fingers.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import placo

from ..hardware.so101 import ARM_JOINTS

TIP_FRAME = "gripper_frame_link"


@dataclass
class IKResult:
    joints_deg: np.ndarray
    position_error_m: float
    axis_error_deg: float
    success: bool


def _urdf_with_tip(urdf_path: Path, tip_xyz_m: tuple[float, float, float]) -> Path:
    """Copy of the URDF with the tip frame origin moved to `tip_xyz_m` (gripper_link frame); the tip
    frame keeps its orientation (+z out of the fingers)."""
    tree = ET.parse(urdf_path)
    joint = tree.getroot().find("joint[@name='gripper_frame_joint']")
    origin = joint.find("origin") if joint is not None else None
    if origin is None:
        raise ValueError(f"{urdf_path} has no gripper_frame_joint origin to move")
    origin.set("xyz", " ".join(f"{float(v):.6g}" for v in tip_xyz_m))
    # Mesh paths are relative to the URDF, so the patched copy must live next to it.
    out = Path(tempfile.mkstemp(prefix=".tip_", suffix=".urdf", dir=urdf_path.parent)[1])
    tree.write(out)
    return out


class SO101Kinematics:
    def __init__(
        self,
        urdf_path: Path,
        joint_limits_deg: dict[str, tuple[float, float]] | None = None,
        tip_xyz_m: tuple[float, float, float] | None = None,
    ):
        """`tip_xyz_m` replaces the URDF's (stock jaw) tool point, in the gripper_link frame."""
        if tip_xyz_m is not None:
            if len(tip_xyz_m) != 3:
                raise ValueError(f"tip_xyz_m must have 3 values, got {tip_xyz_m}")
            patched = _urdf_with_tip(Path(urdf_path), tip_xyz_m)
            try:
                self.robot = placo.RobotWrapper(str(patched), placo.Flags.ignore_collisions)
            finally:
                patched.unlink(missing_ok=True)
        else:
            self.robot = placo.RobotWrapper(str(urdf_path), placo.Flags.ignore_collisions)
        self.joint_names = list(ARM_JOINTS)
        lim = joint_limits_deg or {}
        self._lo = np.array([lim.get(j, (-np.inf, np.inf))[0] for j in self.joint_names], dtype=float)
        self._hi = np.array([lim.get(j, (-np.inf, np.inf))[1] for j in self.joint_names], dtype=float)
        if joint_limits_deg:
            for j, (lo, hi) in joint_limits_deg.items():
                if j in self.joint_names:
                    self.robot.set_joint_limits(j, np.deg2rad(lo), np.deg2rad(hi))
        self.robot.set_joint("gripper", 0.0)

    def _set(self, joints_deg: np.ndarray) -> None:
        for name, q in zip(self.joint_names, np.deg2rad(np.asarray(joints_deg, dtype=float))):
            self.robot.set_joint(name, q)
        self.robot.update_kinematics()

    def _get(self) -> np.ndarray:
        q = np.rad2deg([self.robot.get_joint(n) for n in self.joint_names])
        # The solver keeps joints inside the limits in radians; converting back to degrees can land a
        # rounding error outside (104.52747252747254 vs a 104.52747252747253 limit), which the arm's limit
        # check then rejects. Snap such tiny overshoots onto the limit; real violations are left as they are.
        tol = 1e-6
        q = np.where((q > self._hi) & (q <= self._hi + tol), self._hi, q)
        return np.where((q < self._lo) & (q >= self._lo - tol), self._lo, q)

    def fk(self, joints_deg: np.ndarray, frame: str = TIP_FRAME) -> np.ndarray:
        """4x4 pose of `frame` in the base frame."""
        self._set(joints_deg)
        return np.array(self.robot.get_T_world_frame(frame))

    def ik(
        self,
        target_pos: np.ndarray,
        seed_deg: np.ndarray,
        approach_axis: np.ndarray | None = (0.0, 0.0, -1.0),
        wrist_roll_deg: float | None = None,
        max_iters: int = 200,
        pos_tol_m: float = 0.003,
        axis_tol_deg: float = 15.0,
    ) -> IKResult:
        """Solve for the tip at `target_pos` with the fingers pointing along `approach_axis`.

        The arm has 5 DOF, so a full 6D pose is usually infeasible. We ask for position plus one
        axis (the approach direction). Position has strict priority over the axis: a gripper tilted
        a few degrees still grasps, a tip that lands a centimetre off does not. The returned
        `axis_error_deg` says how far from `approach_axis` the solution ended up. wrist_roll does not change the approach axis, so it is pinned:
        to `wrist_roll_deg` if given (e.g. from the object's PCA angle), else to the seed's value.
        Several seeds are tried to escape local minima at joint limits; the best result is returned.
        Joint limits are enforced. Check `success` before sending the result to the robot.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        seed_deg = np.asarray(seed_deg, dtype=float)
        roll = float(seed_deg[4] if wrist_roll_deg is None else wrist_roll_deg)
        pan = -float(np.degrees(np.arctan2(target_pos[1], target_pos[0])))  # +pan turns toward -y
        seeds = [seed_deg.copy()] + [
            np.array([pan, lift, elbow, flex, roll])
            for lift, elbow, flex in ((5.0, 15.0, 75.0), (-40.0, 55.0, 75.0), (40.0, -40.0, 75.0), (0.0, 0.0, 0.0))
        ]
        seeds[0][4] = roll
        best: IKResult | None = None
        for seed in seeds:
            res = self._solve(target_pos, seed, approach_axis, max_iters, pos_tol_m, axis_tol_deg)
            if res.success:
                return res
            if best is None or res.position_error_m + 0.001 * res.axis_error_deg < (
                best.position_error_m + 0.001 * best.axis_error_deg
            ):
                best = res
        assert best is not None
        return best

    def _solve(
        self,
        target_pos: np.ndarray,
        seed_deg: np.ndarray,
        approach_axis: np.ndarray | None,
        max_iters: int,
        pos_tol_m: float,
        axis_tol_deg: float,
    ) -> IKResult:
        # Strict priority: position is a hard constraint, the approach axis is the best achievable
        # underneath it. If the position is infeasible (out of reach / limits), placo raises; fall
        # back to a soft position task so the caller still gets a best-effort result to inspect.
        try:
            return self._run(target_pos, seed_deg, approach_axis, max_iters, pos_tol_m, axis_tol_deg, "hard")
        except RuntimeError:  # placo QPError: infeasible hard constraint
            return self._run(target_pos, seed_deg, approach_axis, max_iters, pos_tol_m, axis_tol_deg, "soft")

    def _run(
        self,
        target_pos: np.ndarray,
        seed_deg: np.ndarray,
        approach_axis: np.ndarray | None,
        max_iters: int,
        pos_tol_m: float,
        axis_tol_deg: float,
        position_priority: str,
    ) -> IKResult:
        self._set(seed_deg)
        solver = placo.KinematicsSolver(self.robot)
        solver.mask_fbase(True)
        solver.enable_joint_limits(True)
        solver.enable_velocity_limits(False)
        solver.mask_dof("gripper")
        solver.mask_dof("wrist_roll")  # held at the seed's value
        solver.add_position_task(TIP_FRAME, target_pos).configure("pos", position_priority, 1.0)
        axis = None
        if approach_axis is not None:
            axis = np.asarray(approach_axis, dtype=float)
            axis /= np.linalg.norm(axis)
            solver.add_axisalign_task(TIP_FRAME, np.array([0.0, 0.0, 1.0]), axis).configure("axis", "soft", 1.0)
        solver.add_regularization_task(1e-4)

        prev = None
        for _ in range(max_iters):
            solver.solve(True)
            self.robot.update_kinematics()
            T = np.array(self.robot.get_T_world_frame(TIP_FRAME))
            if prev is not None and np.max(np.abs(T - prev)) < 1e-7:
                break
            prev = T
        pos_err = float(np.linalg.norm(T[:3, 3] - target_pos))
        axis_err = 0.0 if axis is None else float(np.degrees(np.arccos(np.clip(T[:3, 2] @ axis, -1, 1))))
        return IKResult(self._get(), pos_err, axis_err, pos_err <= pos_tol_m and axis_err <= axis_tol_deg)
