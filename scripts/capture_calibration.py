"""Capture wrist-camera calibration data: the arm visits planned viewpoints over the board and saves
(image, joint angles) pairs to runs/calib_<timestamp>/.

Flow: rest -> scan (exposure check: last successful setting first, full sweep only if needed) -> viewpoints -> scan -> rest -> torque off.
Every viewpoint has a verified-clear path to and from scan, so on Ctrl-C or any error the arm returns
scan -> rest before torque is disabled. A second Ctrl-C disables torque immediately.

Camera disconnects (the wrist connector can unseat at steep angles): the arm goes back to scan, you
reseat the connector, and it waits up to 20 s for the camera before retrying the same viewpoint.
A viewpoint that disconnects twice is skipped. Progress is saved after every frame.

  --preview-only        plan + write a montage of the viewpoints, no hardware
  --resume SESSION      continue an earlier session (path or "latest"); the phone must not have moved
"""

import argparse
import json
import logging
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from so101_zero_shot.calibration.board import CharucoTarget
from so101_zero_shot.calibration.viewpoints import Viewpoint, order_and_check, path_is_safe, plan_viewpoints
from so101_zero_shot.config import CONFIG_DIR, RUNS_DIR, RobotConfig
from so101_zero_shot.hardware import CameraError, load_calibration
from so101_zero_shot.kinematics.collision import CollisionChecker
from so101_zero_shot.viz import render

MIN_CLEARANCE_M = 0.03
MIN_CORNERS_AT_SCAN = 6
# Manual exposures tried at the scan pose (V4L2 units). Auto exposure is set for the dim desk and
# blows out the phone screen, washing out the thin black marker borders.
EXPOSURE_SWEEP = (25, 50, 100, 200, 400, 800)
CAMERA_WAIT_S = 20.0
MAX_DISCONNECTS_PER_VIEW = 2


class StopRequested(Exception):
    """User chose to stop (e.g. camera would not come back)."""


def grab(cam, target, exposure):
    """Set exposure, take a fresh frame, detect. Returns (frame, corner count)."""
    cam.set_exposure(exposure)
    frame = cam.read_fresh(settle_s=0.5)
    det = target.detect(frame)
    return frame, 0 if det is None else det.count


def best_exposure_at_scan(cam, target, session, start: float | None = None) -> tuple[float | None, int]:
    """Exposure for the run. Tries `start` (last setting that worked) first; full sweep only if it fails."""
    if start is not None:
        frame, n = grab(cam, target, start)
        print(f"  exposure {start:>4} (last successful): {n:2d}/{target.num_corners} corners")
        if n >= MIN_CORNERS_AT_SCAN:
            cv2.imwrite(str(session / "scan.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return start, n
        print("  no longer works, sweeping")
    results = []
    for e in (None, *EXPOSURE_SWEEP):
        frame, n = grab(cam, target, e)
        print(f"  exposure {'auto' if e is None else e:>4}: {n:2d}/{target.num_corners} corners")
        results.append((n, e, frame))
    n, e, frame = max(results, key=lambda r: r[0])
    cv2.imwrite(str(session / "scan.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return e, n


def shoot(cam, target, exposure):
    """Frame at the current pose, retrying half/double exposure if the board is hard to read."""
    frame, n = grab(cam, target, exposure)
    used = exposure
    if n < MIN_CORNERS_AT_SCAN and exposure is not None:
        # Screen brightness varies with viewing angle: retry darker / brighter.
        for e in (exposure / 2, exposure * 2):
            f2, n2 = grab(cam, target, e)
            if n2 > n:
                frame, n, used = f2, n2, e
    return frame, n, used


def recover_camera(arm, cam, scan, speed) -> None:
    """Park at scan, ask for the connector to be reseated, wait for frames. Raises StopRequested."""
    print("\n!! camera disconnected. Moving to the scan pose; please reseat the camera connector.")
    arm.move_to(scan, speed_deg_s=speed)
    while True:
        print(f"   waiting up to {CAMERA_WAIT_S:.0f} s for the camera...")
        if cam.wait_until_streaming(CAMERA_WAIT_S):
            print("   camera is back, continuing")
            return
        if input("   camera still not back. ENTER = wait again, q = stop and return to rest: ").strip().lower() == "q":
            raise StopRequested


def views_to_json(views: list[Viewpoint]) -> list[dict]:
    return [{"joints_deg": v.joints_deg.tolist(), "tip_m": v.tip_m.tolist(), "aim_offset_deg": v.aim_offset_deg} for v in views]


def views_from_json(items: list[dict]) -> list[Viewpoint]:
    return [Viewpoint(np.array(d["joints_deg"]), np.array(d["tip_m"]), float(d["aim_offset_deg"])) for d in items]


def last_good_exposure(meta: dict) -> float | None:
    """Exposure of the most recent usable frame (or the session's chosen exposure), else None."""
    for f in reversed(meta.get("frames", [])):
        if f.get("corners", 0) >= MIN_CORNERS_AT_SCAN and f.get("exposure") is not None:
            return f["exposure"]
    return meta.get("exposure")


def frame_view_index(f: dict) -> int:
    return int(f["view_index"]) if "view_index" in f else int(f["file"][len("view_") : -len(".jpg")])


def plan(cfg: RobotConfig, kin, board_center: np.ndarray, n_views: int, seed: int = 0):
    """Viewpoints whose own pose, path from the previous view, and abort path back to scan are all safe."""
    scan, rest = cfg.poses["scan"], cfg.poses["rest"]
    cc = CollisionChecker()
    for a, b, name in ((rest, scan, "rest->scan"), (scan, rest, "scan->rest")):
        if not path_is_safe(kin, cc, a, b, MIN_CLEARANCE_M):
            raise SystemExit(f"{name} path is not collision-free; fix the poses in config/robot.yaml")
    views = plan_viewpoints(kin, cc, board_center, scan, n_views=n_views, seed=seed)
    return order_and_check(kin, cc, scan, views, MIN_CLEARANCE_M)


def montage(views, n_tiles: int = 8) -> np.ndarray:
    idx = np.linspace(0, len(views) - 1, n_tiles).round().astype(int)
    tiles = []
    for i in idx:
        img = render(views[i].joints_deg, size=(320, 240), cameras=((120, -20, f"view {i + 1}/{len(views)}"),))
        tiles.append(img)
    rows = [np.hstack(tiles[r : r + 4]) for r in range(0, len(tiles), 4)]
    return np.vstack(rows)


def resolve_session(arg: str):
    if arg == "latest":
        sessions = sorted(p for p in RUNS_DIR.glob("calib_*") if (p / "session.json").exists())
        if not sessions:
            raise SystemExit("no previous calibration session with session.json in runs/")
        return sessions[-1]
    path = Path(arg) if Path(arg).is_absolute() else RUNS_DIR / arg
    if not (path / "session.json").exists():
        raise SystemExit(f"{path} has no session.json")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--board-center", type=float, nargs=2, default=(0.22, 0.0), metavar=("X", "Y"))
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--resume", metavar="SESSION", help='session dir under runs/ (or absolute), or "latest"')
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    cfg = RobotConfig.load()
    target = CharucoTarget.from_config()
    kin = cfg.make_kinematics()

    if args.resume:
        session = resolve_session(args.resume)
        meta = json.loads((session / "session.json").read_text())
        if meta.get("robot_yaml") != (CONFIG_DIR / "robot.yaml").read_text():
            print("WARNING: config/robot.yaml changed since this session started (offsets/limits may differ).")
        if "plan" in meta:
            views = views_from_json(meta["plan"])
        else:  # sessions from before the plan was stored: the planner is deterministic, so replan and verify
            views = plan(cfg, kin, np.array(meta["board_center_guess_m"]), args.views)
            for f in meta["frames"]:
                i = frame_view_index(f)
                if i >= len(views) or not np.allclose(views[i].joints_deg, f["joints_cmd_deg"], atol=1e-3):
                    raise SystemExit(f"replanned viewpoint {i} does not match the session; cannot resume safely")
            meta["plan"] = views_to_json(views)
        meta.setdefault("disconnects", [])
        meta.setdefault("skipped_views", [])
        meta.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))
        done = {frame_view_index(f) for f in meta["frames"]} | set(meta["skipped_views"])
        print(f"resuming {session.name}: {len(done)}/{len(views)} viewpoints done or skipped")
    else:
        board_center = np.array([*args.board_center, 0.008])  # phone lying on the table
        views = plan(cfg, kin, board_center, args.views)
        session = RUNS_DIR / datetime.now().strftime("calib_%Y%m%d_%H%M%S")
        session.mkdir(parents=True)
        cv2.imwrite(str(session / "preview.png"), montage(views))
        print(f"planned {len(views)} viewpoints; preview: {session / 'preview.png'}")
        if args.preview_only:
            return
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "board": {"cols": target.board.getChessboardSize()[0], "rows": target.board.getChessboardSize()[1],
                      "square_m": target.board.getSquareLength(), "marker_m": target.board.getMarkerLength()},
            "board_center_guess_m": board_center.tolist(),
            "camera": {"device": cfg.camera_device, "width": cfg.camera_width, "height": cfg.camera_height},
            "arm_calibration": {k: asdict(v) for k, v in load_calibration(cfg.calibration_file).items()},
            "joint_offsets_deg": cfg.joint_offsets_deg,
            "tip_xyz_m": cfg.tip_xyz_m,
            "robot_yaml": (CONFIG_DIR / "robot.yaml").read_text(),
            "plan": views_to_json(views),
            "frames": [],
            "disconnects": [],
            "skipped_views": [],
        }
        done = set()

    def save_meta() -> None:
        (session / "session.json").write_text(json.dumps(meta, indent=2))

    todo = [i for i in range(len(views)) if i not in done]
    if not todo:
        print("nothing left to capture in this session")
        return
    if not args.yes and input(f"Run {len(todo)} viewpoints (~{len(todo) * 4 / 60:.1f} min) at {args.speed:.0f} deg/s? "
                              "Phone flat on the table (not moved), hand near the power switch. [y/N] ").strip().lower() != "y":
        print("aborted, nothing moved")
        return
    save_meta()

    arm, cam = cfg.make_arm(), cfg.make_camera()
    cam.connect()
    arm.connect(enable_torque=False)
    scan, rest = cfg.poses["scan"], cfg.poses["rest"]
    failure: BaseException | None = None
    try:
        arm.bus.sync_write("Goal_Position", arm.bus.sync_read("Present_Position"))
        arm.bus.enable_torque()
        try:
            arm.move_to(scan, speed_deg_s=args.speed)
            while True:
                try:
                    print("scan pose: sweeping exposure")
                    exposure, n = best_exposure_at_scan(cam, target, session, start=last_good_exposure(meta))
                    break
                except CameraError:
                    meta["disconnects"].append({"view": None, "time": time.time()})
                    save_meta()
                    recover_camera(arm, cam, scan, args.speed)
            meta["exposure"] = exposure
            print(f"using exposure {'auto' if exposure is None else exposure} ({n}/{target.num_corners} corners at scan)")
            if n < MIN_CORNERS_AT_SCAN:
                raise StopRequested(f"board not visible from scan (see {session / 'scan.jpg'}); check the phone screen")
            for i in todo:
                v = views[i]
                frame = None
                for attempt in range(1, MAX_DISCONNECTS_PER_VIEW + 1):
                    try:
                        t0 = time.monotonic()
                        err = arm.move_to(v.joints_deg, speed_deg_s=args.speed)
                        t1 = time.monotonic()
                        t_cap = time.time()
                        frame, n, used = shoot(cam, target, exposure)
                        t2 = time.monotonic()
                        break
                    except CameraError:
                        meta["disconnects"].append({"view": i, "time": time.time()})
                        save_meta()
                        recover_camera(arm, cam, scan, args.speed)  # back at scan afterwards
                if frame is not None and n >= MIN_CORNERS_AT_SCAN:
                    exposure = used  # carry the last successful setting to the next viewpoint
                if frame is None:
                    print(f"  {i + 1:2d}/{len(views)}  SKIPPED (camera disconnected {MAX_DISCONNECTS_PER_VIEW}x at this angle)")
                    meta["skipped_views"].append(i)
                    save_meta()
                    continue
                q = arm.read_joints()
                name = f"view_{i:03d}.jpg"
                cv2.imwrite(str(session / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                meta["frames"].append({
                    "file": name, "view_index": i, "time": t_cap, "joints_cmd_deg": v.joints_deg.tolist(),
                    "joints_deg": q.tolist(), "settle_error_deg": err.tolist(), "aim_offset_deg": v.aim_offset_deg,
                    "corners": n, "exposure": used, "move_s": round(t1 - t0, 2), "grab_s": round(t2 - t1, 2),
                })
                save_meta()
                print(f"  {i + 1:2d}/{len(views)}  corners {n:2d}/{target.num_corners}  aim {v.aim_offset_deg:4.0f} deg  "
                      f"exposure {used}  (move {t1 - t0:.1f} s, grab {t2 - t1:.1f} s)")
        except KeyboardInterrupt:
            print("\ninterrupted: returning to scan then rest (Ctrl-C again = torque off now)")
        except StopRequested as e:
            print(f"stopping{': ' + str(e) if str(e) else ''}; returning to scan then rest")
        except Exception as e:  # noqa: BLE001 - always try to bring the arm home before torque-off
            failure = e
            traceback.print_exc()
            print("error above: returning to scan then rest before disabling torque")
        arm.hold()
        time.sleep(0.3)
        arm.move_to(scan, speed_deg_s=args.speed)
        arm.move_to(rest, speed_deg_s=args.speed)
        print("at rest")
    except KeyboardInterrupt:
        print("\nsecond interrupt: disabling torque now")
    finally:
        arm.disconnect(disable_torque=True)
        meta["camera_reconnects"] = meta.get("camera_reconnects", 0) + cam.reconnects
        if cam.reconnects:
            print(f"note: the camera dropped off USB and reconnected {cam.reconnects} time(s) during this run")
        try:
            cam.set_exposure(None)  # leave the camera in auto exposure for other tools
        except CameraError as e:
            print(f"warning: could not restore auto exposure: {e}")
        cam.disconnect()
        save_meta()
        done_now = {frame_view_index(f) for f in meta["frames"]}
        good = sum(f["corners"] >= MIN_CORNERS_AT_SCAN for f in meta["frames"])
        remaining = len(views) - len(done_now) - len(set(meta["skipped_views"]) - done_now)
        print(f"torque off. {len(meta['frames'])} frames ({good} usable), {len(meta['skipped_views'])} skipped, "
              f"{remaining} remaining -> {session}")
        if remaining:
            print(f"resume with: .venv/bin/python scripts/capture_calibration.py --resume {session.name}")
    if failure is not None:
        raise SystemExit(f"capture stopped by an error: {failure!r}")


if __name__ == "__main__":
    main()
