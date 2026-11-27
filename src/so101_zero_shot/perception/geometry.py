"""Wrist camera geometry: calibrated fisheye model, rectification to a virtual pinhole, table rays.

The rectified camera shares the fisheye camera's frame (same optical axis), so a rectified pixel (u, v)
is the ray (x, y, 1) = ((u - cx) / f, (v - cy) / f, 1) in the camera frame, and a point on that ray at
optical-axis depth Z is Z * (x, y, 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..config import CONFIG_DIR
from ..kinematics import SO101Kinematics

CAMERA_CALIBRATION_PATH = CONFIG_DIR / "calibration" / "wrist_camera.yaml"


@dataclass
class FisheyeCamera:
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    D: np.ndarray
    hand_frame: str
    T_hand_camera: np.ndarray

    @classmethod
    def load(cls, path: Path = CAMERA_CALIBRATION_PATH) -> FisheyeCamera:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/solve_calibration.py")
        d = yaml.safe_load(path.read_text())
        if d.get("model") != "opencv_fisheye":
            raise ValueError(f"{path}: unsupported camera model {d.get('model')!r}")
        return cls(
            tuple(d["image_size"]), np.array(d["K"], dtype=float), np.array(d["D"], dtype=float).reshape(4, 1),
            d["hand_frame"], np.array(d["T_hand_camera"], dtype=float),
        )

    def pose_in_base(self, kin: SO101Kinematics, joints_deg: np.ndarray) -> np.ndarray:
        """T_base_camera at the given arm joints."""
        return kin.fk(joints_deg, self.hand_frame) @ self.T_hand_camera


class Rectifier:
    """Fisheye -> virtual pinhole with the same optical axis.

    The default 100 deg horizontal field of view matches the calibrated region (corners reached ~49 deg
    off-axis); beyond that the lens model is extrapolated.
    """

    def __init__(self, cam: FisheyeCamera, size: tuple[int, int] = (960, 720), hfov_deg: float = 100.0):
        if not 10.0 <= hfov_deg < 170.0:
            raise ValueError(f"hfov_deg must be in [10, 170), got {hfov_deg}")
        self.cam = cam
        self.size = size
        w, h = size
        f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.K = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, np.eye(3), self.K, size, cv2.CV_16SC2)

    def rectify(self, fisheye_bgr: np.ndarray) -> np.ndarray:
        h, w = fisheye_bgr.shape[:2]
        if (w, h) != tuple(self.cam.image_size):
            raise ValueError(f"image is {w}x{h}, calibration is for {self.cam.image_size[0]}x{self.cam.image_size[1]}")
        return cv2.remap(fisheye_bgr, self._map1, self._map2, cv2.INTER_LINEAR)

    def rays(self) -> np.ndarray:
        """(H, W, 3) camera-frame rays (x, y, 1) for every rectified pixel."""
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        return np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], axis=-1)


def to_base(rays_cam: np.ndarray, Z: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Camera rays (N, 3) at depths Z (N,) -> base-frame points (N, 3)."""
    P = rays_cam * Z[:, None]
    return P @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
