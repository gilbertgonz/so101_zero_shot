"""ChArUco board definition (from config/robot.yaml) and detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from ..config import CONFIG_DIR


@dataclass
class BoardDetection:
    corners: np.ndarray  # (N, 2) pixel coords of ChArUco corners
    ids: np.ndarray  # (N,) corner ids
    object_points: np.ndarray  # (N, 3) corner coords in the board frame, metres

    @property
    def count(self) -> int:
        return len(self.ids)


class CharucoTarget:
    def __init__(self, cols: int, rows: int, square_m: float, marker_ratio: float, dictionary: str):
        self.board = cv2.aruco.CharucoBoard(
            (cols, rows), square_m, square_m * marker_ratio, cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        )
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Board is small in a wide fisheye image: accept marker perimeters down to 1% of the image
        # size (side ~0.25%, ~6 px at 2592 wide).
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMax = 53
        self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(), params)
        self.corner_points = self.board.getChessboardCorners()  # (M, 3), board frame
        self.num_corners = len(self.corner_points)
        self.size_m = (cols * square_m, rows * square_m)

    @classmethod
    def from_config(cls) -> CharucoTarget:
        cfg = yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text())["calibration_board"]
        if cfg.get("square_mm") is None:
            raise ValueError("calibration_board.square_mm is not set in config/robot.yaml; measure the board first")
        return cls(int(cfg["cols"]), int(cfg["rows"]), float(cfg["square_mm"]) / 1000.0, float(cfg["marker_ratio"]), cfg["dictionary"])

    def detect(self, image_bgr: np.ndarray) -> BoardDetection | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return None
        ids = ids.ravel()
        return BoardDetection(corners.reshape(-1, 2).astype(np.float64), ids, self.corner_points[ids].astype(np.float64))
