"""Move to the look pose, take one wrist-camera image of the table, return to rest.

Saves runs/scene_<time>/ (so101_zero_shot.scenes): image.jpg (raw fisheye), views/ (frames taken on the
way to the look pose: the other views depth is measured with), scene.json (joints, camera pose, exposure).
Run scripts/locate.py or scripts/grasp.py --dry-run on the result.

Safety: confirmation before torque; any error or Ctrl-C returns to rest before torque-off; a second
Ctrl-C disables torque immediately.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from so101_zero_shot.config import RUNS_DIR, RobotConfig
from so101_zero_shot.grasping.setup import look_settings
from so101_zero_shot.perception.geometry import FisheyeCamera
from so101_zero_shot.scenes import save_scene
from so101_zero_shot.skills.motion_frames import MotionFrames, spread_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="scan", help="pose name from config/robot.yaml to look from")
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure (default: auto)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    cfg = RobotConfig.load()
    if args.pose not in cfg.poses:
        raise SystemExit(f"unknown pose {args.pose!r}; available: {sorted(cfg.poses)}")
    look, rest = cfg.poses[args.pose], cfg.poses["rest"]
    kin = cfg.make_kinematics()
    cam_model = FisheyeCamera.load()

    cam = cfg.make_camera()
    cam.connect()
    arm = cfg.make_arm()
    arm.connect(enable_torque=False)
    out = None
    try:
        arm.check_limits(look)
        if not args.yes and input(f"Move rest -> {args.pose} -> rest and take a picture? Hand near the power switch. [y/N] ").strip().lower() != "y":
            print("aborted, torque stays off")
            return
        arm.bus.sync_write("Goal_Position", arm.bus.sync_read("Present_Position"))
        arm.bus.enable_torque()
        try:
            arm.move_to(rest, speed_deg_s=args.speed)
            cam.set_exposure(args.exposure)
            ls = look_settings()
            center = lambda q: cam_model.pose_in_base(kin, q)[:3, 3]  # noqa: E731
            with MotionFrames(arm, cam, center, min_step_m=ls.min_step_m, camera_latency_s=ls.camera_latency_s) as mf:
                arm.move_to(look, speed_deg_s=args.speed, settle_timeout_s=3.0)
            time.sleep(1.5)  # auto exposure settles
            img = cam.read_fresh(0.3)
            q = arm.read_joints()
            frames = spread_frames(mf.frames(), center, ls.frames, exclude_near=center(q))
            out = RUNS_DIR / time.strftime("scene_%Y%m%d_%H%M%S")
            out.mkdir(parents=True)
            save_scene(out, img, q, [(f.bgr, f.joints_deg) for f in frames], {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pose": args.pose,
                "joints_cmd_deg": look.tolist(),
                "T_base_camera": cam_model.pose_in_base(kin, q).tolist(),
                "exposure": args.exposure,
                "image_size": [img.shape[1], img.shape[0]],
            })
            print(f"saved {out} ({len(frames)} frames from the way for depth)")
        except KeyboardInterrupt:
            print("\ninterrupted: returning to rest (Ctrl-C again = torque off now)")
            arm.hold()
        except Exception as e:  # noqa: BLE001 - always try to bring the arm home before torque-off
            print(f"error: {e!r}; returning to rest before disabling torque")
            arm.move_to(rest, speed_deg_s=min(args.speed, 30.0), settle_timeout_s=3.0)
            raise SystemExit(1) from e
        arm.move_to(rest, speed_deg_s=min(args.speed, 30.0), settle_timeout_s=3.0)
        print("at rest")
    except KeyboardInterrupt:
        print("\nsecond interrupt: disabling torque now")
    finally:
        arm.disconnect(disable_torque=True)
        cam.disconnect()
        print("torque off")
    if out is not None:
        print(f'next: .venv/bin/python scripts/locate.py {out.name} "the object to find"')


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
