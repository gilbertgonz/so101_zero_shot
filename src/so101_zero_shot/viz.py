"""Offscreen MuJoCo renders of the arm, for showing a human what pose the arm will take."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .config import MJCF_SCENE_PATH  # noqa: E402
from .hardware.so101 import ARM_JOINTS  # noqa: E402


def render(joints_deg: np.ndarray, gripper_deg: float = 20.0, size=(640, 480), cameras=None) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
    data = mujoco.MjData(model)
    for name, q in zip((*ARM_JOINTS, "gripper"), (*joints_deg, gripper_deg)):
        data.joint(name).qpos[0] = np.deg2rad(q)
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
