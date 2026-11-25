"""Fit where the wrist camera sits relative to the gripper, from the fingers' outline in one image.

Writes config/calibration/gripper_view.yaml (used to mask the robot's own gripper in the camera image)
and IMAGE_gripper_view.jpg next to the image (green = model clamps, red = real fingers, blue = model frame).

Usage:
  fit_gripper_view.py IMAGE                  wrist-camera frame (raw fisheye), gripper closed, fingers
                                             against the table background, nothing touching them
  --points X,Y X,Y                           one point on each finger in the straightened view
                                             (default 330,530 600,530)
  --open-pct P                               gripper opening in the image (default 0)

The hand-eye pose (wrist_camera.yaml) is the starting guess and is not changed.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.grasping.setup import perception_config
from so101_zero_shot.perception.geometry import FisheyeCamera, Rectifier
from so101_zero_shot.perception.gripper_mask import GRIPPER_VIEW_PATH, GripperRenderer, load_gripper_view

MIN_IOU = 0.7


def to_T(p: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """p = rotation vector (rad) and translation (cm) applied in the camera frame of X0."""
    d = np.eye(4)
    d[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    d[:3, 3] = p[3:] / 100.0
    return X0 @ d


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--points", nargs=2, default=["330,530", "600,530"])
    ap.add_argument("--open-pct", type=float, default=0.0)
    args = ap.parse_args()

    from so101_zero_shot.perception.segment import Segmenter

    cfg = RobotConfig.load()
    cam = FisheyeCamera.load()
    pcfg = perception_config()
    rect = Rectifier(cam, tuple(pcfg.get("rectified_size", (960, 720))), float(pcfg.get("rectified_hfov_deg", 100.0)))
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    rect_bgr = rect.rectify(img)
    rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)

    print("segmenting the fingers...")
    seg = Segmenter(perception_config().get("sam_model", "facebook/sam2.1-hiera-small"))
    real = np.zeros(rgb.shape[:2], dtype=bool)
    for s in args.points:
        x, y = (float(v) for v in s.split(","))
        m, _ = seg.mask(rgb, np.array([x, y]), None)
        real |= m
    del seg
    if real.mean() > 0.3 or real.sum() < 2000:
        raise SystemExit(f"finger mask looks wrong ({real.mean():.0%} of the image); pick other --points")

    # Small near clip so the rods hide the clamp bottoms as in the photo (the mask itself uses a larger
    # clip to drop the camera holder; it is a union of all gripper parts, so that does not matter there).
    r = GripperRenderer(rect.K, rect.size, cfg.gripper_open_gap_m, near_clip_m=0.005)
    try:
        # The IoU has local optima: start from the previous fit (if any) and from a small grid around the
        # hand-eye pose (shifted across the jaws and tilted about the jaw-closing axis, where the hand-eye
        # pose was found to be off).
        starts = []
        if GRIPPER_VIEW_PATH.exists():
            starts.append(load_gripper_view())
        for dy_mm in (0.0, -15.0):
            for tilt_deg in (0.0, -8.0, 8.0):
                P = np.eye(4)
                P[:3, :3] = Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()
                P[1, 3] = dy_mm / 1000.0
                starts.append(P @ cam.T_hand_camera)
        best = None
        for k, X0 in enumerate(starts, 1):
            def cost(p, X0=X0):
                return 1.0 - iou(r.masks(to_T(p, X0), args.open_pct)[0], real)
            # Coarse (translation only, where the start is furthest off), then all six.
            p = np.zeros(6)
            for idx in ([3, 4, 5], [0, 1, 2, 3, 4, 5]):
                def sub(v, p=p, idx=idx):
                    q = p.copy()
                    q[idx] = v
                    return cost(q)
                res = minimize(sub, p[idx], method="Powell", options=dict(maxfev=800, xtol=1e-3, ftol=1e-5))
                p[idx] = res.x
            X = to_T(p, X0)
            score = iou(r.masks(X, args.open_pct)[0], real)
            print(f"  start {k}/{len(starts)}: IoU {score:.3f}", flush=True)
            if best is None or score > best[0]:
                best = (score, X)
        score, X = best
        clamps, whole = r.masks(X, args.open_pct)
    finally:
        r.close()

    out = rect_bgr.copy()
    frame = whole & ~clamps
    out[frame] = (0.5 * out[frame] + [127, 100, 0]).astype(np.uint8)
    for m, col in ((real, (0, 0, 255)), (clamps, (0, 255, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 2)
    overlay = args.image.with_name(args.image.stem + "_gripper_view.jpg")
    cv2.imwrite(str(overlay), out)
    dT = np.linalg.inv(cam.T_hand_camera) @ X
    print(f"IoU {score:.3f}; differs from the hand-eye pose by "
          f"{np.degrees(np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec())):.1f} deg, "
          f"{np.linalg.norm(dT[:3, 3]) * 1000:.0f} mm; overlay: {overlay}")
    if score < MIN_IOU:
        raise SystemExit(f"IoU {score:.2f} < {MIN_IOU}: not saved (check the overlay and --points)")
    GRIPPER_VIEW_PATH.write_text(
        "# Generated by scripts/fit_gripper_view.py; do not edit by hand.\n"
        "# Camera pose in the gripper (gripper_link) frame that makes the MuJoCo parallel gripper match the\n"
        "# real fingers in the image. Used only to mask the gripper; 3D geometry uses wrist_camera.yaml.\n"
        + yaml.safe_dump({
            "T_gripper_camera": np.round(X, 6).tolist(),
            "iou": round(score, 3),
            "image": str(args.image),
            "open_pct": args.open_pct,
            "fitted": datetime.now().isoformat(timespec="seconds"),
        }, sort_keys=False)
    )
    print(f"wrote {GRIPPER_VIEW_PATH}")


if __name__ == "__main__":
    main()
