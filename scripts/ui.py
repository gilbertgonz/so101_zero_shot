"""Web UI for zero-shot grasping: live camera, live arm view, prompt, plan panel, approval, stop / E-stop.

Usage:
  ui.py                       serve on http://127.0.0.1:8000 (this machine only)
  ui.py --port 8080
  ui.py --fake [SCENE]        simulated arm + a saved scene image as the camera (no hardware moves)

The arm connects with torque off; torque is enabled only while a run is in progress (Run = consent,
the grasp itself still needs Approve). Stop = hold, then return to rest (press again while returning to
cut torque where it is). E-STOP = torque off immediately (the arm drops onto its stops).
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import uvicorn  # noqa: E402

from so101_zero_shot.config import RUNS_DIR  # noqa: E402
from so101_zero_shot.ui.app import create_app  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", nargs="?", const="latest", metavar="SCENE",
                    help="simulate the arm; camera = runs/SCENE/image.jpg (default: latest scene)")
    args = ap.parse_args()

    fake_image = None
    if args.fake:
        scenes = sorted(p for p in RUNS_DIR.glob("scene_*") if (p / "image.jpg").exists())
        sd = (scenes[-1] if scenes else None) if args.fake == "latest" else RUNS_DIR / args.fake
        if sd is None or not (sd / "image.jpg").exists():
            raise SystemExit(f"no scene image for --fake {args.fake!r}")
        fake_image = sd / "image.jpg"
    app = create_app(fake=bool(args.fake), fake_image=fake_image)
    print(f"open http://127.0.0.1:{args.port}" + ("  (SIMULATED arm and camera)" if args.fake else ""))
    # Local only: anyone who can reach this port can move the arm.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
