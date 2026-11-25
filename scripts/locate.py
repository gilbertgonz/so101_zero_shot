"""Find objects in a captured scene: VLM point -> SAM mask -> table-scaled depth -> 3D in the base frame.

Usage: locate.py SCENE QUERY [QUERY ...]
  SCENE: runs/scene_* dir name (or absolute path), or "latest"

Writes locate_<query>.png (mask, point, box, depth) and locate_<query>.json into the scene dir.
Nothing moves; this only reads the saved image.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from so101_zero_shot.config import CONFIG_DIR, RUNS_DIR, RobotConfig
from so101_zero_shot.perception.depth import DepthAnything
from so101_zero_shot.perception.geometry import FisheyeCamera, Rectifier
from so101_zero_shot.perception.locate import Locator, PerceptionError, draw_debug
from so101_zero_shot.perception.pointer import VlmPointer
from so101_zero_shot.perception.segment import Segmenter


def scene_dir(arg: str) -> Path:
    if arg == "latest":
        scenes = sorted(p for p in RUNS_DIR.glob("scene_*") if (p / "scene.json").exists())
        if not scenes:
            raise SystemExit("no runs/scene_* with scene.json; run scripts/capture_scene.py first")
        return scenes[-1]
    p = Path(arg) if Path(arg).is_absolute() else RUNS_DIR / arg
    if not (p / "scene.json").exists():
        raise SystemExit(f"{p} has no scene.json")
    return p


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "query"


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("queries", nargs="+")
    args = ap.parse_args()

    d = scene_dir(args.scene)
    meta = json.loads((d / "scene.json").read_text())
    img = cv2.imread(str(d / "image.jpg"))
    if img is None:
        raise SystemExit(f"cannot read {d / 'image.jpg'}")
    cfg = RobotConfig.load()
    pcfg = yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()).get("perception") or {}
    cam = FisheyeCamera.load()
    rect = Rectifier(cam, tuple(pcfg.get("rectified_size", (960, 720))), float(pcfg.get("rectified_hfov_deg", 100.0)))

    t0 = time.monotonic()
    loc = Locator(
        cam, cfg.make_kinematics(),
        VlmPointer(pcfg.get("vlm_model", "Qwen/Qwen3-VL-2B-Instruct")),
        Segmenter(pcfg.get("sam_model", "facebook/sam2.1-hiera-small")),
        DepthAnything(pcfg.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf")),
        rectifier=rect, table_z_m=float(pcfg.get("table_z_m", 0.0)),
    )
    print(f"models loaded in {time.monotonic() - t0:.0f} s")
    cv2.imwrite(str(d / "rectified.jpg"), rect.rectify(img))

    failures = 0
    for q in args.queries:
        t0 = time.monotonic()
        try:
            est, debug = loc.locate(img, np.asarray(meta["joints_deg"]), q)
        except PerceptionError as e:
            print(f'"{q}": NOT LOCATED: {e}')
            failures += 1
            continue
        dt = time.monotonic() - t0
        s = est.summary()
        s["seconds"] = round(dt, 2)
        name = slug(q)
        (d / f"locate_{name}.json").write_text(json.dumps(s, indent=1))
        cv2.imwrite(str(d / f"locate_{name}.png"), draw_debug(est, debug))
        c = est.center_base * 100
        try:
            cfg.check_in_workspace(np.array([est.center_base[0], est.center_base[1], est.top_z_m]))
            ws = "inside workspace"
        except ValueError as e:
            ws = str(e)
        print(f'"{q}": centre ({c[0]:.1f}, {c[1]:.1f}) cm, top {est.top_z_m * 100:.1f} cm, '
              f"table fit rmse {est.fit.rmse_m * 1000:.1f} mm, {dt:.1f} s; {ws}"
              + "".join(f"\n   warning: {w}" for w in est.warnings))
        print(f"   {d / f'locate_{name}.png'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
