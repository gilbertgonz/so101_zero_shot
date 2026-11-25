"""Plan arm viewpoints for wrist-camera calibration over a board lying on the table.

The camera looks roughly along the fingers (approach axis), so a viewpoint is "tip somewhere above the
table, approach axis pointing at (or deliberately beside) the board". Off-target aims put the board near
the edge of the fisheye image, which is what constrains the distortion model. The exact camera mount is
unknown before calibration, and the ~180 deg field of view makes rough aiming sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import SO101Kinematics
from ..kinematics.collision import CollisionChecker

# Links checked for table clearance (besides the tip frame).
CLEARANCE_LINKS = ("gripper_link", "wrist_link", "lower_arm_link")
PAN_AXIS_XY = np.array([0.0388, 0.0])  # shoulder_pan joint position in the base frame (URDF)
# Keep the gripper out in front: folding it back over the base risks the (unmodelled) camera mount.
MIN_GRIPPER_RADIUS_M = 0.08
# Aim-offset bins (deg between approach axis and board direction) and their share of the views.
# Centre views give clean board poses for hand-eye; off-centre views constrain the fisheye distortion.
AIM_BINS = ((0.0, 25.0, 0.4), (25.0, 50.0, 0.6))


@dataclass(frozen=True)
class Envelope:
    """Where viewpoints may put the tip. Defaults keep the arm compact and in front of the robot:
    a run that swung the base to 47 deg with the tip 16 cm out to the side hit the desk monitor."""

    x_m: tuple[float, float] = (0.14, 0.26)
    y_m: tuple[float, float] = (-0.08, 0.08)
    z_m: tuple[float, float] = (0.08, 0.16)
    max_pan_deg: float = 25.0
    max_aim_deg: float = 50.0


@dataclass
class Viewpoint:
    joints_deg: np.ndarray
    tip_m: np.ndarray
    aim_offset_deg: float  # angle between approach axis and the direction to the board centre


def gripper_radius(kin: SO101Kinematics, q: np.ndarray) -> float:
    """Horizontal distance of the tip / gripper body from the pan axis (smallest of the two)."""
    return min(float(np.linalg.norm(kin.fk(q, f)[:2, 3] - PAN_AXIS_XY)) for f in ("gripper_frame_link", "gripper_link"))


def min_link_height(kin: SO101Kinematics, q: np.ndarray) -> float:
    return min([kin.fk(q)[2, 3]] + [kin.fk(q, link)[2, 3] for link in CLEARANCE_LINKS])


def path_min_height(kin: SO101Kinematics, a: np.ndarray, b: np.ndarray, steps: int = 30) -> float:
    """Lowest link height along the smoothstep joint-space path move_to will execute from a to b."""
    out = np.inf
    for t in np.linspace(0, 1, steps):
        s = t * t * (3 - 2 * t)
        out = min(out, min_link_height(kin, a + s * (b - a)))
    return out


def path_is_safe(
    kin: SO101Kinematics, cc: CollisionChecker, a: np.ndarray, b: np.ndarray, min_clearance_m: float
) -> bool:
    """Table clearance for the (extended) tip and links, plus MuJoCo self/table collision, along a->b."""
    return path_min_height(kin, a, b) >= min_clearance_m and not cc.path_in_collision(a, b)


def plan_viewpoints(
    kin: SO101Kinematics,
    cc: CollisionChecker,
    board_center_m: np.ndarray,
    return_pose_deg: np.ndarray,
    n_views: int = 10,
    n_candidates: int = 5000,
    min_clearance_m: float = 0.04,
    seed: int = 0,
    envelope: Envelope = Envelope(),
    aim_bins: tuple[tuple[float, float, float], ...] = AIM_BINS,
) -> list[Viewpoint]:
    rng = np.random.default_rng(seed)
    board_center_m = np.asarray(board_center_m, dtype=float)
    ik_seed = np.array([0.0, -40.0, 30.0, 70.0, 0.0])
    cands: list[Viewpoint] = []
    for _ in range(n_candidates):
        tip = np.array([rng.uniform(*envelope.x_m), rng.uniform(*envelope.y_m), rng.uniform(*envelope.z_m)])
        to_board = board_center_m - tip
        dist = np.linalg.norm(to_board)
        if not 0.08 <= dist <= 0.30:
            continue
        # A 5-DOF arm can only point within its vertical plane (through the pan axis and the tip), so
        # sample a pitch in that plane: 0 = straight down, + = leaning outward away from the base.
        radial = tip[:2] - PAN_AXIS_XY
        radial = np.append(radial / np.linalg.norm(radial), 0.0)
        if rng.random() < 0.4:
            # Aim (within the plane) at the board, plus noise: fills the centre of the image.
            pitch_to_board = np.arctan2(to_board @ radial, -to_board[2])
            pitch = pitch_to_board + np.radians(rng.normal(0.0, 8.0))
        else:
            pitch = np.radians(rng.uniform(-20.0, 75.0))
        axis = np.cos(pitch) * np.array([0.0, 0.0, -1.0]) + np.sin(pitch) * radial
        aim = float(np.degrees(np.arccos(np.clip(axis @ (to_board / dist), -1.0, 1.0))))
        if aim > envelope.max_aim_deg:
            continue
        roll = rng.uniform(-90.0, 90.0)
        res = kin.ik(tip, ik_seed, approach_axis=axis, wrist_roll_deg=roll, axis_tol_deg=5.0)
        if not res.success or abs(res.joints_deg[0]) > envelope.max_pan_deg:
            continue
        if min_link_height(kin, res.joints_deg) < min_clearance_m:
            continue
        if gripper_radius(kin, res.joints_deg) < MIN_GRIPPER_RADIUS_M or cc.in_collision(res.joints_deg):
            continue
        # Every view needs a safe direct path back to the return pose, so aborting is always possible.
        if not path_is_safe(kin, cc, res.joints_deg, return_pose_deg, min_clearance_m - 0.01):
            continue
        cands.append(Viewpoint(res.joints_deg, tip, aim))
    # Farthest-point sampling on (tip position, approach direction, roll), stratified by aim offset.
    def feat(v: Viewpoint) -> np.ndarray:
        T = kin.fk(v.joints_deg)
        return np.concatenate([T[:3, 3] / 0.05, T[:3, 2] / 0.3, [v.joints_deg[4] / 45.0]])

    chosen: list[Viewpoint] = []
    for lo, hi, share in aim_bins:
        pool = [c for c in cands if lo <= c.aim_offset_deg < hi]
        k = round(n_views * share) if (lo, hi, share) != aim_bins[-1] else n_views - len(chosen)
        if len(pool) < k:
            raise RuntimeError(f"only {len(pool)} feasible viewpoints with aim {lo}-{hi} deg (wanted {k})")
        F = np.array([feat(c) for c in pool])
        idx = [int(np.argmin(np.linalg.norm(F[:, :3] - F[:, :3].mean(0), axis=1)))]
        d = np.linalg.norm(F - F[idx[0]], axis=1)
        while len(idx) < k:
            i = int(np.argmax(d))
            idx.append(i)
            d = np.minimum(d, np.linalg.norm(F - F[i], axis=1))
        chosen += [pool[i] for i in idx]
    return chosen


def order_and_check(
    kin: SO101Kinematics,
    cc: CollisionChecker,
    start_deg: np.ndarray,
    views: list[Viewpoint],
    min_clearance_m: float = 0.03,
) -> list[Viewpoint]:
    """Greedy nearest-neighbour order in joint space; skip views whose path from the previous one is unsafe."""
    remaining = list(views)
    ordered: list[Viewpoint] = []
    cur = np.asarray(start_deg, dtype=float)
    while remaining:
        remaining.sort(key=lambda v: float(np.max(np.abs(v.joints_deg - cur))))
        for i, v in enumerate(remaining):
            if path_is_safe(kin, cc, cur, v.joints_deg, min_clearance_m):
                ordered.append(remaining.pop(i))
                cur = v.joints_deg
                break
        else:
            break  # nothing reachable safely from here; drop the rest
    return ordered

