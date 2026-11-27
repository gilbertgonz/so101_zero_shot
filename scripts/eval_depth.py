"""Measure multi-view depth against exact simulated depth (MuJoCo), on the views a grasp actually gets.

  eval_depth.py capture [--layouts 4]            sim: photos, frames on the way, ground truth
  eval_depth.py run METHOD [--pose-noise-mm 1]   depth for every sample
  eval_depth.py report

Samples (runs/sim/depth_eval/sNNN/), each a reference photo (where depth is wanted) and other views:
  scan:   reference = photo at the scan pose; frames taken while moving rest -> scan
  relook: reference = photo at a closer-look pose over an object (planned as the grasp does it); the
          scan photo and the frames taken scan -> closer look
Ground truth: rendered depth at the reference, the objects' true centres and tops.

METHODs (MVSAnywhere, perception.multiview):
  photo        reference + the previous photo (closer looks only)
  moveK        reference + K frames from the move, spread along the path
  allK         reference + previous photo + K frames (what the grasp uses; K = perception.look.frames)
  suffix _sHxW model input size (default 480x640)
--pose-noise-mm perturbs every non-reference view's pose (and a matching rotation): FK error.

Scores (object pixels = more than 3 mm above the table in the true depth): coverage, median |depth
error|, share within 5 mm, per object top-height and top-centre xy error as perception.locate measures
them (xy-gt: the same measurement on true depth, i.e. the part of the xy error depth cannot fix).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from so101_zero_shot.config import RUNS_DIR, RobotConfig
from so101_zero_shot.grasping.setup import make_rectifier as make_rect
from so101_zero_shot.perception.geometry import FisheyeCamera, to_base

OUT = RUNS_DIR / "sim" / "depth_eval"


# ---------------------------------------------------------------------------------------------- capture
def capture(args) -> None:
    from so101_zero_shot.grasping.topdown import TopDownPlanner
    from so101_zero_shot.kinematics.collision import CollisionChecker
    from so101_zero_shot.mujoco_sim import SimWorld, SimWristCamera, make_mujoco_arm
    from so101_zero_shot.skills.motion_frames import MotionFrames

    cfg, cam = RobotConfig.load(), FisheyeCamera.load()
    kin, rect = cfg.make_kinematics(), make_rect(cam)
    planner = TopDownPlanner(kin, CollisionChecker(), cam, cfg.poses["rest"])
    world = SimWorld(cfg, cam)
    world.start()
    arm, camera = make_mujoco_arm(cfg, world), SimWristCamera(world)
    center = lambda q: cam.pose_in_base(kin, q)[:3, 3]  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(list(OUT.glob("s[0-9]*")))

    def move(q, collect=False):
        if not collect:
            arm.move_to(q, speed_deg_s=args.speed, settle_timeout_s=3.0)
            return []
        with MotionFrames(arm, camera, center) as mf:
            arm.move_to(q, speed_deg_s=args.speed, settle_timeout_s=3.0)
        return mf.frames()

    def photo():
        time.sleep(0.4)
        img = camera.read_fresh(0.2)
        return rect.rectify(img), arm.read_joints()

    def save(tag, ref, views, truth):
        nonlocal n
        d = OUT / f"s{n:03d}"
        d.mkdir(exist_ok=True)
        Tr = cam.pose_in_base(kin, ref[1])
        _, depth = world.render_truth(Tr, rect.K, rect.size)
        cv2.imwrite(str(d / "ref.png"), ref[0])
        np.save(d / "depth_true.npy", depth.astype(np.float32))
        np.save(d / "robot_mask.npy", world.last_robot_mask)
        vmeta = []
        for i, (kind, img, q) in enumerate(views):
            f = f"{kind}_{i:02d}.png"
            cv2.imwrite(str(d / f), img)
            vmeta.append({"file": f, "kind": kind, "joints": np.asarray(q).tolist(),
                          "T": cam.pose_in_base(kin, q).tolist()})
        (d / "meta.json").write_text(json.dumps({
            "tag": tag, "q_ref": np.asarray(ref[1]).tolist(), "T_ref": Tr.tolist(), "K": rect.K.tolist(),
            "views": vmeta, "objects": truth}, indent=1))
        kinds = {k: sum(v["kind"] == k for v in vmeta) for k in ("photo", "move")}
        print(f"s{n:03d} {tag:<22} {kinds}")
        n += 1

    try:
        for layout in range(args.layouts):
            world.reset(jitter_m=0.0 if layout == 0 else 0.03, seed=layout)
            arm.hold()
            arm.bus.enable_torque()
            time.sleep(0.5)
            truth = {k: {"center": v["center"].tolist(), "top_z": v["top_z"]} for k, v in world.object_geometry().items()}
            scan = cfg.poses["scan"]
            move(cfg.poses["rest"])
            frames = move(scan, collect=True)
            scan_photo = photo()
            save("scan", scan_photo, [("move", rect.rectify(f.bgr), f.joints_deg) for f in frames], truth)
            for name, g in truth.items():
                try:
                    relook = planner.plan_relook(np.array([*g["center"][:2], 0.0]), scan_photo[1])
                except Exception as e:  # noqa: BLE001
                    print(f"  no re-look pose for {name}: {e}")
                    continue
                frames = move(relook, collect=True)
                ref = photo()
                views = [("photo", *scan_photo)] + [("move", rect.rectify(f.bgr), f.joints_deg) for f in frames]
                save(f"relook_{name}", ref, views, truth)
                move(scan)
            move(cfg.poses["rest"])
    finally:
        world.close()


# ---------------------------------------------------------------------------------------------- methods
def pick(views, method: str, center_ref: np.ndarray):
    """The other views a method uses: previous photo and/or K frames spread along the move."""
    from so101_zero_shot.skills.motion_frames import Frame, spread_frames

    by = lambda k: [v for kind, v in views if kind == k]  # noqa: E731
    if method.startswith("photo"):
        return by("photo")
    m = re.match(r"(move|all)(\d+)", method)
    if not m:
        raise SystemExit(f"unknown method {method}")
    moves = by("move")
    # spread_frames takes Frames and a joints -> camera centre function; here "joints" = the centre itself
    fr = [Frame(None, v.T_base_cam[:3, 3].copy(), 0.0) for v in moves]
    chosen = spread_frames(fr, lambda c: c, int(m.group(2)), exclude_near=center_ref)
    idx = [next(i for i, f in enumerate(fr) if f is c) for c in chosen]
    return (by("photo") if m.group(1) == "all" else []) + [moves[i] for i in idx]


def build(method: str):
    from so101_zero_shot.perception.multiview import MvsAnywhereDepth

    sz = re.search(r"_s(\d+)x(\d+)$", method)
    model = MvsAnywhereDepth(size=(int(sz.group(1)), int(sz.group(2))) if sz else (480, 640))

    def fn(ref, views, meta):
        others = pick(views, method, ref.T_base_cam[:3, 3])
        return model.depth(ref, others)[0] if others else None
    return fn


def perturb(T: np.ndarray, mm: float, rng) -> np.ndarray:
    if mm <= 0:
        return T
    d = rng.normal(size=3)
    d *= mm / 1000.0 / np.linalg.norm(d)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * np.radians(0.1 * mm))  # ~0.3 deg for 3 mm
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] += d
    return out


def run(args) -> None:
    from so101_zero_shot.perception.multiview import View

    fn = build(args.method)
    rng = np.random.default_rng(0)
    name = args.method + (f"_noise{args.pose_noise_mm:g}mm" if args.pose_noise_mm else "")
    rgb = lambda f: cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)  # noqa: E731
    for d in sorted(OUT.glob("s[0-9]*")):
        meta = json.loads((d / "meta.json").read_text())
        K = np.array(meta["K"], dtype=float)
        ref = View(rgb(d / "ref.png"), K, np.array(meta["T_ref"]))
        views = [(v["kind"], View(rgb(d / v["file"]), K, perturb(np.array(v["T"]), args.pose_noise_mm, rng)))
                 for v in meta["views"] if v["kind"] in ("photo", "move")]
        t = time.time()
        z = fn(ref, views, meta)
        dt = time.time() - t
        if z is None:
            continue
        np.save(d / f"depth_{name}.npy", z.astype(np.float32))
        (d / f"time_{name}.txt").write_text(f"{dt:.3f}")
        print(f"{d.name} {meta['tag']:<22} {dt:5.2f}s")


# ---------------------------------------------------------------------------------------------- report
def top_centre(pts: np.ndarray) -> tuple[float, np.ndarray]:
    """Top height (90th percentile z) and xy centre of the points near the top, as perception.locate does."""
    top = float(np.percentile(pts[:, 2], 90))
    return top, np.median(pts[pts[:, 2] > top - 0.01][:, :2], axis=0)


def score(d: Path, name: str, rays: np.ndarray) -> dict | None:
    f = d / f"depth_{name}.npy"
    if not f.exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    T = np.array(meta["T_ref"])
    zt, ze = np.load(d / "depth_true.npy").astype(float), np.load(f).astype(float)
    Pt = to_base(rays.reshape(-1, 3), np.nan_to_num(zt).reshape(-1), T).reshape(zt.shape + (3,))
    obj = np.isfinite(zt) & (Pt[..., 2] > 0.003)
    obj &= ~np.load(d / "robot_mask.npy")  # the gripper is always in view; perception masks it out
    err = np.abs(ze - zt)[obj]
    valid = np.isfinite(err)
    out = {"tag": meta["tag"], "coverage": float(valid.mean()) if obj.any() else 0.0,
           "med_err_mm": float(np.median(err[valid]) * 1000) if valid.any() else np.nan,
           "within5mm": float((err[valid] < 0.005).mean()) if valid.any() else 0.0, "objects": {}}
    Pe = to_base(rays.reshape(-1, 3), np.nan_to_num(ze).reshape(-1), T).reshape(ze.shape + (3,))
    for oname, g in meta["objects"].items():
        c = np.array(g["center"])
        region = obj & (np.linalg.norm(Pt[..., :2] - c[:2], axis=-1) < 0.045) & np.isfinite(ze)
        if region.sum() < 50:
            continue
        top, cxy = top_centre(Pe[region])
        _, cxy_true = top_centre(Pt[region])
        out["objects"][oname] = {"top_err_mm": (top - g["top_z"]) * 1000,
                                 "xy_err_mm": float(np.linalg.norm(cxy - c[:2]) * 1000),
                                 "xy_vs_true_mm": float(np.linalg.norm(cxy - cxy_true) * 1000)}
    return out


def report(args) -> None:
    rays = make_rect(FisheyeCamera.load()).rays()
    samples = sorted(OUT.glob("s[0-9]*"))
    names = sorted({p.stem[len("depth_"):] for d in samples for p in d.glob("depth_*.npy")} - {"true"})
    print(f"{len(samples)} samples\n")
    print(f"{'method':<28}{'view':<8}{'n':>4}{'cover':>7}{'med mm':>8}{'<5mm':>6}{'|top| mm':>10}{'xy mm':>8}"
          f"{'xy-gt':>7}{'sec':>6}")
    for kind in ("scan", "relook"):
        for name in names:
            rows = [r for d in samples if (r := score(d, name, rays)) and r["tag"].startswith(kind)]
            if not rows:
                continue
            objs = [o for r in rows for o in r["objects"].values()]
            secs = [float((d / f"time_{name}.txt").read_text()) for d in samples
                    if (d / f"time_{name}.txt").exists() and json.loads((d / "meta.json").read_text())["tag"].startswith(kind)]
            print(f"{name:<28}{kind:<8}{len(rows):>4}{np.mean([r['coverage'] for r in rows]):>7.2f}"
                  f"{np.nanmedian([r['med_err_mm'] for r in rows]):>8.1f}{np.mean([r['within5mm'] for r in rows]):>6.2f}"
                  f"{np.median([abs(o['top_err_mm']) for o in objs]):>10.1f}{np.median([o['xy_err_mm'] for o in objs]):>8.1f}"
                  f"{np.median([o['xy_vs_true_mm'] for o in objs]):>7.1f}{np.median(secs):>6.2f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--layouts", type=int, default=4)
    c.add_argument("--baseline", type=float, default=0.05)
    c.add_argument("--speed", type=float, default=30.0, help="deg/s, as the grasp runs")
    r = sub.add_parser("run")
    r.add_argument("method")
    r.add_argument("--pose-noise-mm", type=float, default=0.0)
    sub.add_parser("report")
    args = ap.parse_args()
    {"capture": capture, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
