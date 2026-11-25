"""Where the robot's own gripper is in the straightened wrist-camera image.

The camera and the gripper are bolted together, so the gripper's outline in the image depends only on
the jaw opening, not on the arm pose. It is rendered from the MuJoCo parallel-gripper model with a camera
pose fitted to the real fingers' outline (config/calibration/gripper_view.yaml, scripts/fit_gripper_view.py).

That pose is deliberately separate from the hand-eye T_hand_camera in wrist_camera.yaml: the hand-eye
pose is fitted through the arm's forward kinematics and absorbs its errors (good for locating things on
the table, ~2 cm / 10 deg off for the gripper 5-13 cm from the lens). See PROGRESS.md.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ..config import CONFIG_DIR, MJCF_SCENE_PATH  # noqa: E402
from ..gripper_model import DEFAULT_OPEN_GAP_M, set_gripper_qpos  # noqa: E402

log = logging.getLogger(__name__)

GRIPPER_VIEW_PATH = CONFIG_DIR / "calibration" / "gripper_view.yaml"
GRIPPER_BODY = "gripper"
CLAMP_BODIES = ("clamp_right", "clamp_left")


class GripperRenderer:
    """Segmentation renders of the gripper seen from a camera fixed in the gripper frame.

    Uses one EGL context: create, use and close it in a single thread.
    """

    # The camera holder's rim sits around the lens and would cover much of the view if rendered; the
    # fingers and rods are >= 4 cm away, so clip everything nearer than this.
    NEAR_CLIP_M = 0.03

    def __init__(self, K: np.ndarray, size: tuple[int, int], open_gap_m: float = DEFAULT_OPEN_GAP_M,
                 near_clip_m: float = NEAR_CLIP_M):
        self.near_clip_m = near_clip_m
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)  # arm at zero; only the gripper-relative pose matters
        self.K, self.size, self.open_gap_m = np.asarray(K, dtype=float), size, open_gap_m
        w, h = size
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = max(w, 640), max(h, 480)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.renderer.enable_segmentation_rendering()
        body_of = {name: self.model.body(name).id for name in (GRIPPER_BODY, *CLAMP_BODIES)}
        gb = self.model.geom_bodyid
        self._clamp_geoms = np.flatnonzero(np.isin(gb, [body_of[b] for b in CLAMP_BODIES]))
        self._gripper_geoms = np.flatnonzero(np.isin(gb, list(body_of.values())))

    def masks(self, T_gripper_camera: np.ndarray, open_pct: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(clamps, whole gripper) bool masks, (H, W), for an OpenCV-convention camera at T_gripper_camera."""
        set_gripper_qpos(self.model, self.data.qpos, open_pct, self.open_gap_m)
        mujoco.mj_forward(self.model, self.data)
        g = self.model.body(GRIPPER_BODY).id
        T_wg = np.eye(4)
        T_wg[:3, :3], T_wg[:3, 3] = self.data.xmat[g].reshape(3, 3), self.data.xpos[g]
        T = T_wg @ T_gripper_camera
        self.renderer.update_scene(self.data, camera=mujoco.MjvCamera())
        R_gl = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenGL camera: looks down -z, y up
        w, h = self.size
        near = self.near_clip_m
        for cam in self.renderer.scene.camera:
            cam.pos[:] = T[:3, 3]
            cam.forward[:] = -R_gl[:, 2]
            cam.up[:] = R_gl[:, 1]
            cam.frustum_near = near
            # Pinhole with the principal point at K's (cx, cy).
            cam.frustum_top = near * self.K[1, 2] / self.K[1, 1]
            cam.frustum_bottom = -near * (h - 1 - self.K[1, 2]) / self.K[1, 1]
            cam.frustum_width = near * (w / 2.0) / self.K[0, 0]
            cam.frustum_center = near * ((w - 1) / 2.0 - self.K[0, 2]) / self.K[0, 0]
        seg = self.renderer.render()[..., 0]
        return np.isin(seg, self._clamp_geoms), np.isin(seg, self._gripper_geoms)

    def close(self) -> None:
        self.renderer.close()


def load_gripper_view(path: Path = GRIPPER_VIEW_PATH) -> np.ndarray:
    raw = yaml.safe_load(Path(path).read_text())
    T = np.asarray(raw["T_gripper_camera"], dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: T_gripper_camera must be 4x4")
    return T


class GripperMask:
    """Precomputed gripper masks in the straightened view, by jaw opening (thread-safe to use)."""

    STEP_PCT = 5

    def __init__(self, masks_by_pct: dict[int, np.ndarray]):
        if not masks_by_pct:
            raise ValueError("no gripper masks")
        self._masks = masks_by_pct

    @classmethod
    def render(cls, T_gripper_camera: np.ndarray, K: np.ndarray, size: tuple[int, int],
               open_gap_m: float = DEFAULT_OPEN_GAP_M) -> GripperMask:
        r = GripperRenderer(K, size, open_gap_m)
        try:
            return cls({p: r.masks(T_gripper_camera, p)[1] for p in range(0, 101, cls.STEP_PCT)})
        finally:
            r.close()

    @classmethod
    def load(cls, K: np.ndarray, size: tuple[int, int], open_gap_m: float = DEFAULT_OPEN_GAP_M,
             path: Path = GRIPPER_VIEW_PATH) -> GripperMask | None:
        """None (with a warning) if the gripper view has not been fitted or cannot be rendered."""
        try:
            return cls.render(load_gripper_view(path), K, size, open_gap_m)
        except Exception as e:  # noqa: BLE001 - perception still works without it, minus the gripper check
            log.warning("no gripper mask (%s); clicks on the gripper will not be caught", e)
            return None

    def mask(self, open_pct: float = 0.0) -> np.ndarray:
        """(H, W) bool: pixels showing the gripper (frame, rods, clamps) at `open_pct` (nearest 5 %)."""
        p = int(np.clip(round(float(open_pct) / self.STEP_PCT) * self.STEP_PCT, 0, 100))
        return self._masks[p]
