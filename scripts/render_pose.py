"""Render the SO-101 in a given joint pose (deg) to a PNG, from a side and a 3/4 view.

Used to show a human how to place the arm, e.g.:
  render_pose.py --pose 0 0 0 0 0 --out docs/poses/zero.png --title "Pose A"
  render_pose.py --tip 0.20 0 0.0 --out docs/poses/down.png   (top-down IK to a tip position)
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from so101_zero_shot.config import MJCF_SCENE_PATH, RobotConfig  # noqa: E402
from so101_zero_shot.hardware import ARM_JOINTS  # noqa: E402


def render(joints_deg: np.ndarray, gripper_deg: float = 20.0, size=(640, 480)) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
    data = mujoco.MjData(model)
    for name, q in zip((*ARM_JOINTS, "gripper"), (*joints_deg, gripper_deg)):
        data.joint(name).qpos[0] = np.deg2rad(q)
    mujoco.mj_forward(model, data)
    w, h = size
    model.vis.global_.offwidth, model.vis.global_.offheight = max(w, 640), max(h, 480)
    renderer = mujoco.Renderer(model, height=h, width=w)
    views = []
    for azimuth, elevation, label in ((90, -5, "side view (arm faces right)"), (135, -25, "3/4 view")):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.15, 0.0, 0.12]
        cam.distance, cam.azimuth, cam.elevation = 0.75, azimuth, elevation
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        views.append(img)
    renderer.close()
    return np.hstack(views)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pose", type=float, nargs=5, metavar="DEG")
    g.add_argument("--tip", type=float, nargs=3, metavar="M", help="top-down IK target in base frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if args.tip is not None:
        res = RobotConfig.load().make_kinematics().ik(args.tip, np.array([0, -20, 20, 60, 0.0]))
        if not res.success:
            raise SystemExit(f"IK failed for {args.tip}: {res}")
        q = res.joints_deg
    else:
        q = np.array(args.pose)
    img = render(q)
    if args.title:
        cv2.putText(img, args.title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 160), 2)
    cv2.imwrite(args.out, img)
    print("joints (deg):", np.round(q, 1), "->", args.out)


if __name__ == "__main__":
    main()
