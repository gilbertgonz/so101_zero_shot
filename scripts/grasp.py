"""Zero-shot top-down grasp of a named object with the wrist camera (terminal version of scripts/ui.py).

  1. rest -> scan: photo, coarse locate (VLM point + SAM mask + multi-view depth from the frames on the way)
  2. re-look: camera ~20 cm from the target, photo, locate again (more accurate; nothing hidden under
     the fingers)
  3. plan: fingertip point, wrist roll so the jaws close across the object's long axis; render the plan
     and ask before moving
  4. open, pregrasp, descend slowly, close, lift; held = gripper squeezing (load) or stopped open
  5. put it back where it was, open, back off, rest

Usage:
  grasp.py QUERY                  run on the arm
  grasp.py QUERY --dry-run SCENE  plan from a saved scene (runs/scene_* or 'latest'), render only
  --keep                          keep the object in the gripper (it is still returned to rest)
  --sim                           MuJoCo simulation instead of the arm (runs go to runs/sim/)
  --no-record                     do not save an episode for this run (default: config recording.enabled)

Everything is saved to runs/grasp_<time>/. Safety: confirmation before torque and before the grasp;
Ctrl-C stops the motion and returns to rest; a second Ctrl-C disables torque where the arm is.
"""

from __future__ import annotations

import argparse
import logging

from so101_zero_shot.config import RUNS_DIR
from so101_zero_shot.grasping.runner import STEPS, GraspRunner
from so101_zero_shot.grasping.setup import build_runner_parts

LABELS = dict(STEPS)


class ConsoleReporter:
    def step(self, step_id, status, message="", detail=None, images=None):
        if status == "active":
            print(f"- {LABELS.get(step_id, step_id)}...")
            return
        print(f"  {status.upper()}" + (f": {message}" if message else ""))
        for k, v in (detail or {}).items():
            if k != "target_deg":
                print(f"    {k}: {v}")
        for im in images or []:
            print(f"    {im}")

    def log(self, message):
        print(message)

    def confirm(self, kind, prompt, images=None):
        for im in images or []:
            print(f"    see {im}")
        return input(f"{prompt} [y/N] ").strip().lower() == "y"

    def choose(self, kind, title, prompt, options, images=None):
        for im in images or []:
            print(f"    see {im}")
        keys = "/".join(v for v, _ in options)
        ans = input(f"{title} {prompt} [{keys}] ").strip().lower()
        return ans if ans in {v for v, _ in options} else "cancel"

    def pick(self, prompt):
        return None  # clicking needs the UI (scripts/ui.py)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--dry-run", metavar="SCENE")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    print("loading models...")
    parts = build_runner_parts(sim=args.sim)
    runner = GraspRunner(**parts, reporter=ConsoleReporter(), speed_deg_s=args.speed,
                         environment="sim" if args.sim else "real", runs_dir=RUNS_DIR / "sim" if args.sim else None)

    if args.dry_run:
        if args.dry_run == "latest":
            scenes = sorted(p for p in RUNS_DIR.glob("scene_*") if (p / "scene.json").exists())
            sd = scenes[-1] if scenes else None
        else:
            sd = RUNS_DIR / args.dry_run
        if sd is None or not (sd / "scene.json").exists():
            raise SystemExit(f"no scene {args.dry_run!r}")
        res = runner.dry_run(sd, args.query)
        print(f"dry run: {res.result}; {res.out_dir}")
        return

    cfg = parts["cfg"]
    world = None
    if args.sim:
        from so101_zero_shot.mujoco_sim import SimWorld, SimWristCamera, make_mujoco_arm

        world = SimWorld(cfg, parts["cam_model"])
        world.start()
        camera, arm = SimWristCamera(world), make_mujoco_arm(cfg, world)
    else:
        camera = cfg.make_camera()
        camera.connect()
        arm = cfg.make_arm()
        arm.connect(enable_torque=False)
    try:
        runner.run(arm, camera, args.query, keep=args.keep, record=False if args.no_record else None)
    except KeyboardInterrupt:
        print("\nsecond interrupt: disabling torque now")
    finally:
        arm.disconnect(disable_torque=True)
        camera.disconnect()
        if world is not None:
            world.close()
        print("torque off")


if __name__ == "__main__":
    main()
