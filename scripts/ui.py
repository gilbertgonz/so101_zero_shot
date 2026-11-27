"""Web UI for zero-shot grasping: live camera, live arm view, prompt, plan panel, approval, stop / E-stop.

Usage:
  ui.py                       serve on http://127.0.0.1:8000 (this machine only)
  ui.py --port 8080
  ui.py --sim                 MuJoCo simulation: arm on a table with a few objects, rendered wrist camera,
                              physics grasping (no hardware); runs and episodes go to runs/sim/

The arm connects with torque off; torque is enabled only while a run is in progress (Run = consent,
the grasp itself still needs Approve). Stop = hold, then return to rest (press again while returning to
cut torque where it is). E-STOP = torque off immediately (the arm drops onto its stops).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import uvicorn  # noqa: E402

from so101_zero_shot.ui.app import create_app  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sim", action="store_true", help="MuJoCo simulation of the arm, camera and objects")
    args = ap.parse_args()
    app = create_app(sim=args.sim)
    print(f"open http://127.0.0.1:{args.port}" + ("  (SIMULATED arm and camera)" if args.sim else ""))
    backend = app.state.backend

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # End the open browser streams first; otherwise uvicorn waits for them forever and the app
            # shutdown (stop the arm, torque off) never runs.
            backend.close_streams()
            super().handle_exit(sig, frame)

    # Local only: anyone who can reach this port can move the arm.
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=5)
    Server(config).run()  # returns after the app shutdown: arm stopped, torque off, camera closed
    # Exit without running native destructors: torch/CUDA and the MuJoCo EGL context abort during
    # interpreter teardown ("terminate called without an active exception"), and the core dump of a
    # multi-GB GPU process made the UI look like it ignored Ctrl-C.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
