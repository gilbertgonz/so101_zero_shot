"""Offscreen MuJoCo renders of the arm, for showing a human what pose the arm will take."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .config import MJCF_SCENE_PATH  # noqa: E402
from .gripper_model import DEFAULT_OPEN_GAP_M, set_gripper_qpos  # noqa: E402
from .hardware.so101 import ARM_JOINTS  # noqa: E402


def render(joints_deg: np.ndarray, gripper_pct: float = 0.0, size=(640, 480), cameras=None,
           open_gap_m: float = DEFAULT_OPEN_GAP_M) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
    data = mujoco.MjData(model)
    for name, q in zip(ARM_JOINTS, joints_deg):
        data.joint(name).qpos[0] = np.deg2rad(q)
    set_gripper_qpos(model, data.qpos, gripper_pct, open_gap_m)
    mujoco.mj_forward(model, data)
    w, h = size
    model.vis.global_.offwidth, model.vis.global_.offheight = max(w, 640), max(h, 480)
    renderer = mujoco.Renderer(model, height=h, width=w)
    views = []
    cameras = cameras or ((90, -5, "side view (arm faces right)"), (135, -25, "3/4 view"))
    for azimuth, elevation, label in cameras:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.15, 0.0, 0.12]
        cam.distance, cam.azimuth, cam.elevation = 0.75, azimuth, elevation
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        views.append(img)
    renderer.close()
    return np.hstack(views)


class LiveRenderer:
    """Keeps one MuJoCo model + renderer for repeated renders of the current pose (UI live view).

    Not thread-safe: create and use it from a single thread (the EGL context is bound to it).
    """

    def __init__(self, size=(480, 360), azimuth: float = 135.0, elevation: float = -25.0,
                 open_gap_m: float = DEFAULT_OPEN_GAP_M):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        w, h = size
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = max(w, 640), max(h, 480)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.15, 0.0, 0.10]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.65, azimuth, elevation
        self._qadr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.open_gap_m = open_gap_m

    def set_view(self, azimuth: float, elevation: float) -> None:
        self.cam.azimuth, self.cam.elevation = azimuth, elevation

    def _pose(self, joints_deg: np.ndarray, gripper_pct: float) -> None:
        for adr, v in zip(self._qadr, np.asarray(joints_deg, dtype=float)):
            self.data.qpos[adr] = np.deg2rad(v)
        set_gripper_qpos(self.model, self.data.qpos, gripper_pct, self.open_gap_m)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)

    def render(self, joints_deg: np.ndarray, gripper_pct: float = 0.0) -> np.ndarray:
        """BGR image of the arm at `joints_deg` with the gripper `gripper_pct` open."""
        self._pose(joints_deg, gripper_pct)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def arm_mask(self, joints_deg: np.ndarray, gripper_pct: float = 0.0) -> np.ndarray:
        """(H, W) bool: pixels covered by the arm (not the floor) at `joints_deg`."""
        self._pose(joints_deg, gripper_pct)
        self.renderer.enable_segmentation_rendering()
        try:
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        geom = seg[..., 0]
        is_geom = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (geom >= 0)
        return is_geom & (self.model.geom_bodyid[np.clip(geom, 0, None)] != 0)

    def close(self) -> None:
        self.renderer.close()
