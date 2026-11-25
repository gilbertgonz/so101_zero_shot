"""Wrist-camera calibration from capture sessions: fisheye intrinsics, then eye-in-hand extrinsics.

Each capture session (scripts/capture_calibration.py) has the board lying still on the table, so a
session contributes one unknown board pose in the base frame. Several sessions (the board may have moved
between them) are solved together:

1. Intrinsics: cv2.fisheye.calibrate over every frame with enough ChArUco corners.
2. Extrinsics: T_gripper_camera (camera pose in the `gripper_link` frame) and one T_base_board per
   session. These are initialised with cv2.calibrateHandEye and refined by minimising the reprojection
   error of the corners predicted through the arm's forward kinematics.

Optionally (refine_intrinsics=True) step 2 also refines the intrinsics. On clean synthetic data that
removes a ~2% focal-length bias of step 1 (small board near the image centre lets focal length trade
against distortion). On the real arm it overfits: the arm's few-px pose error is absorbed into the lens
model (fx/fy split 3%, large distortion terms, board tilt doubled), so it is off by default. The remaining error therefore
   includes the arm's own accuracy (backlash, joint offsets), which is what grasping will see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..kinematics import SO101Kinematics
from .board import CharucoTarget

HAND_FRAME = "gripper_link"  # fixed to the wrist, unaffected by the configured tip extension
MIN_CORNERS = 6  # fewer coplanar points make the per-frame pose ill-conditioned


@dataclass
class Observation:
    session: str
    file: str
    joints_deg: np.ndarray
    corners: np.ndarray  # (N, 2) px
    object_points: np.ndarray  # (N, 3) board frame, m


@dataclass
class CalibrationResult:
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    D: np.ndarray  # fisheye k1..k4
    intrinsics_rms_px: float
    T_hand_camera: np.ndarray  # camera pose in HAND_FRAME
    T_base_board: dict[str, np.ndarray]  # per session
    robot_rms_px: float  # reprojection error with board poses predicted through the arm FK
    per_frame_rms_px: dict[str, float] = field(default_factory=dict)
    max_corner_radius_px: float = 0.0  # how far from the image centre the data reaches

    def to_yaml_dict(self) -> dict:
        return {
            "model": "opencv_fisheye",
            "image_size": list(self.image_size),
            "K": self.K.tolist(),
            "D": self.D.ravel().tolist(),
            "hand_frame": HAND_FRAME,
            "T_hand_camera": self.T_hand_camera.tolist(),
            "intrinsics_rms_px": round(self.intrinsics_rms_px, 4),
            "robot_rms_px": round(self.robot_rms_px, 4),
            "max_corner_radius_px": round(self.max_corner_radius_px, 1),
            "T_base_board": {k: v.tolist() for k, v in self.T_base_board.items()},
        }


def _to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return T


def _to_rt(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def load_observations(session_dirs: list[Path], target: CharucoTarget, min_corners: int = MIN_CORNERS) -> list[Observation]:
    obs = []
    for d in session_dirs:
        meta = json.loads((d / "session.json").read_text())
        for f in meta.get("frames", []):
            img = cv2.imread(str(d / f["file"]))
            if img is None:
                raise FileNotFoundError(d / f["file"])
            det = target.detect(img)
            if det is None or det.count < min_corners:
                continue
            obs.append(Observation(d.name, f["file"], np.asarray(f["joints_deg"], dtype=float), det.corners, det.object_points))
    return obs


def calibrate(
    obs: list[Observation], kin: SO101Kinematics, image_size: tuple[int, int], refine_intrinsics: bool = False
) -> CalibrationResult:
    sessions = sorted({o.session for o in obs})
    for s in sessions:
        if sum(o.session == s for o in obs) < 3:
            raise ValueError(f"session {s} has fewer than 3 usable frames; drop it")
    if len(obs) < 8:
        raise ValueError(f"only {len(obs)} usable frames; need at least 8 for fisheye intrinsics")

    # 1. Intrinsics.
    w, h = image_size
    K0 = np.array([[w / 3.0, 0, w / 2.0], [0, w / 3.0, h / 2.0], [0, 0, 1]])
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        [o.object_points.reshape(-1, 1, 3) for o in obs],
        [o.corners.reshape(-1, 1, 2) for o in obs],
        image_size, K0, np.zeros((4, 1)), flags=flags,
        criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10),
    )
    T_cam_board = [_to_T(r, t) for r, t in zip(rvecs, tvecs)]
    T_base_hand = [kin.fk(o.joints_deg, HAND_FRAME) for o in obs]

    # 2a. Hand-eye initialisation (Park) on all frames; motion pairs across sessions are invalid, so
    # use the session with the most frames.
    main = max(sessions, key=lambda s: sum(o.session == s for o in obs))
    idx = [i for i, o in enumerate(obs) if o.session == main]
    R_hc, t_hc = cv2.calibrateHandEye(
        [T_base_hand[i][:3, :3] for i in idx], [T_base_hand[i][:3, 3] for i in idx],
        [T_cam_board[i][:3, :3] for i in idx], [T_cam_board[i][:3, 3] for i in idx],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    X0 = np.eye(4)
    X0[:3, :3], X0[:3, 3] = R_hc, t_hc.ravel()
    boards0 = {}
    for s in sessions:
        Ts = [T_base_hand[i] @ X0 @ T_cam_board[i] for i, o in enumerate(obs) if o.session == s]
        boards0[s] = Ts[len(Ts) // 2]

    # 2b. Joint refinement of X, the board poses and (optionally) the intrinsics through the arm FK.
    n_pose = 6 + 6 * len(sessions)

    def unpack(p):
        X = _to_T(p[:3], p[3:6])
        boards = {s: _to_T(p[6 + 6 * k: 9 + 6 * k], p[9 + 6 * k: 12 + 6 * k]) for k, s in enumerate(sessions)}
        if refine_intrinsics:
            fx, fy, cx, cy = p[n_pose: n_pose + 4]
            Kp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            return X, boards, Kp, p[n_pose + 4: n_pose + 8].reshape(4, 1)
        return X, boards, K, D

    def project(X, boards, i, K, D):
        T_cb = np.linalg.inv(T_base_hand[i] @ X) @ boards[obs[i].session]
        rt = _to_rt(T_cb)
        pts, _ = cv2.fisheye.projectPoints(obs[i].object_points.reshape(-1, 1, 3), rt[:3], rt[3:], K, D)
        return pts.reshape(-1, 2)

    def residuals(p):
        X, boards, Kp, Dp = unpack(p)
        return np.concatenate([(project(X, boards, i, Kp, Dp) - o.corners).ravel() for i, o in enumerate(obs)])

    p0 = [_to_rt(X0)] + [_to_rt(boards0[s]) for s in sessions]
    if refine_intrinsics:
        p0 += [np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), D.ravel()]
    sol = least_squares(residuals, np.concatenate(p0), loss="soft_l1", f_scale=3.0, x_scale="jac")
    if not sol.success:
        raise ValueError(f"hand-eye refinement did not converge: {sol.message}")
    X, boards, K, D = unpack(sol.x)

    per_frame = {}
    sq = []
    for i, o in enumerate(obs):
        e = np.linalg.norm(project(X, boards, i, K, D) - o.corners, axis=1)
        per_frame[f"{o.session}/{o.file}"] = float(np.sqrt(np.mean(e**2)))
        sq.append(e**2)
    centre = K[:2, 2]
    max_r = max(float(np.linalg.norm(o.corners - centre, axis=1).max()) for o in obs)
    return CalibrationResult(
        image_size, K, D, float(rms), X, boards, float(np.sqrt(np.mean(np.concatenate(sq)))), per_frame, max_r
    )


def board_sanity(T_base_board: np.ndarray, board_size_m: tuple[float, float]) -> dict:
    """The board lies flat on the table: its normal should be vertical and its centre near z = 0."""
    centre = T_base_board @ np.array([board_size_m[0] / 2, board_size_m[1] / 2, 0.0, 1.0])
    tilt = float(np.degrees(np.arccos(abs(T_base_board[2, 2]))))
    return {"centre_m": centre[:3], "tilt_deg": tilt}
