"""Solve wrist-camera intrinsics + hand-eye from capture sessions and write config/calibration/wrist_camera.yaml.

Usage:
  solve_calibration.py SESSION [SESSION ...]   session dirs under runs/ (or absolute)
  --dry-run                                    print results, write nothing to config/
  --refine-intrinsics                          also refine intrinsics through the arm FK (overfits arm error; see solve.py)

Also writes check images into the newest session dir:
  undistort.png   original vs undistorted (lines on the table should become straight)
  robot_fk.png    corners predicted through the arm FK (green) vs detected (red); validates hand-eye
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from so101_zero_shot.calibration.board import CharucoTarget
from so101_zero_shot.calibration.solve import HAND_FRAME, board_sanity, calibrate, load_observations
from so101_zero_shot.config import CONFIG_DIR, RUNS_DIR, RobotConfig

OUT = CONFIG_DIR / "calibration" / "wrist_camera.yaml"


def session_path(arg: str) -> Path:
    p = Path(arg) if Path(arg).is_absolute() else RUNS_DIR / arg
    if not (p / "session.json").exists():
        raise SystemExit(f"{p} has no session.json")
    return p


def undistort_image(res, img: np.ndarray) -> np.ndarray:
    # Same focal length as the fisheye: the centre keeps its scale and the outer part (beyond the
    # calibrated radius, extrapolated) is cropped instead of stretched across the whole output.
    newK = res.K.copy()
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(res.K, res.D, np.eye(3), newK, res.image_size, cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def fk_overlay(res, obs, kin, session_dir: Path) -> np.ndarray:
    tiles = []
    for o in [o for o in obs if o.session == session_dir.name][:6]:
        img = cv2.imread(str(session_dir / o.file))
        T_cb = np.linalg.inv(kin.fk(o.joints_deg, HAND_FRAME) @ res.T_hand_camera) @ res.T_base_board[o.session]
        rvec = cv2.Rodrigues(T_cb[:3, :3])[0]
        pts, _ = cv2.fisheye.projectPoints(o.object_points.reshape(-1, 1, 3), rvec, T_cb[:3, 3], res.K, res.D)
        for p in o.corners:
            cv2.circle(img, tuple(int(v) for v in p), 9, (0, 0, 255), 2)
        for p in pts.reshape(-1, 2):
            cv2.drawMarker(img, tuple(int(v) for v in p), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        c = o.corners.mean(0).astype(int)
        x0, y0 = np.clip(c - [240, 180], 0, [img.shape[1] - 480, img.shape[0] - 360])
        crop = img[y0:y0 + 360, x0:x0 + 480].copy()
        cv2.putText(crop, f"{o.file} {res.per_frame_rms_px[f'{o.session}/{o.file}']:.1f}px", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refine-intrinsics", action="store_true")
    args = ap.parse_args()

    dirs = [session_path(s) for s in args.sessions]
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    target = CharucoTarget.from_config()
    obs = load_observations(dirs, target)
    print(f"{len(obs)} usable frames from {len(dirs)} session(s)")
    try:
        res = calibrate(obs, kin, (cfg.camera_width, cfg.camera_height), refine_intrinsics=args.refine_intrinsics)
    except (ValueError, cv2.error) as e:
        raise SystemExit(f"calibration failed: {e}")

    fx, fy, cx, cy = res.K[0, 0], res.K[1, 1], res.K[0, 2], res.K[1, 2]
    print(f"\nintrinsics: fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}  D {np.round(res.D.ravel(), 4).tolist()}")
    print(f"  rms {res.intrinsics_rms_px:.2f} px; data reaches {res.max_corner_radius_px:.0f} px from centre "
          f"(~{np.degrees(res.max_corner_radius_px / fx):.0f} deg off-axis)")
    X = res.T_hand_camera
    z_cam_in_hand = X[:3, 2]
    print(f"\ncamera in {HAND_FRAME}: position {np.round(X[:3, 3] * 100, 1).tolist()} cm, "
          f"optical axis {np.round(z_cam_in_hand, 2).tolist()}")
    T_tip = np.linalg.inv(kin.fk(np.zeros(5), HAND_FRAME)) @ kin.fk(np.zeros(5))
    ang = np.degrees(np.arccos(np.clip(z_cam_in_hand @ T_tip[:3, 2], -1, 1)))
    print(f"  angle between optical axis and approach axis: {ang:.1f} deg")
    print(f"robot-predicted reprojection rms {res.robot_rms_px:.1f} px")
    for k, v in res.per_frame_rms_px.items():
        print(f"  {k}: {v:.1f} px")
    print("\nboard poses (should lie flat on the table):")
    for s, T in res.T_base_board.items():
        chk = board_sanity(T, target.size_m)
        print(f"  {s}: centre {np.round(chk['centre_m'] * 100, 1).tolist()} cm, tilt {chk['tilt_deg']:.1f} deg")

    newest = dirs[-1]
    # A frame with the board well inside the image, brightened (frames are shot dark for the phone screen).
    best = max((o for o in obs if o.session == newest.name), key=lambda o: len(o.corners))
    img = cv2.imread(str(newest / best.file))
    if img is not None:
        img = cv2.convertScaleAbs(img, alpha=4.0)
        cv2.imwrite(str(newest / "undistort.png"), np.hstack([img, undistort_image(res, img)]))
    cv2.imwrite(str(newest / "robot_fk.png"), fk_overlay(res, obs, kin, newest))
    print(f"\ncheck images: {newest / 'undistort.png'}, {newest / 'robot_fk.png'}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    data = res.to_yaml_dict()
    data["sessions"] = [d.name for d in dirs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# Generated by scripts/solve_calibration.py; do not edit by hand.\n" + yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
