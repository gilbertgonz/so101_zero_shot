"""Run a free-form request with the agent (terminal version of the UI's Agent mode).

The robot looks around, the local VLM proposes a plan, you approve it, then it calls one tool at a time
(look, pick up, place, move the gripper along any path) until it finishes. Grasps and paths near a hand
ask again unless --yes.

Usage:
  agent.py "knock over the blue cylinder" --sim
  agent.py REQUEST                                      on the real arm
  --reasoner SPEC  reasoning model, e.g. hf:Qwen/Qwen3-VL-8B-Instruct@4bit (default: config agent.reasoner)
  --yes          approve everything automatically (sim testing)
  --no-record    do not save an episode
  --video PATH   sim: save a video of the run (3/4 view + wrist camera, 4x speed)

Runs go to runs/agent_<time>/ (runs/sim/ in the simulation).
"""

from __future__ import annotations

import argparse
import logging
import time

from so101_zero_shot.agent import Agent, AgentConfig, build_reasoner
from so101_zero_shot.config import RUNS_DIR
from so101_zero_shot.grasping.runner import GraspRunner
from so101_zero_shot.grasping.setup import build_runner_parts


class ConsoleReporter:
    def __init__(self, yes: bool):
        self.yes, self.labels = yes, {}

    def add_step(self, step_id, label):
        self.labels[step_id] = label

    def step(self, step_id, status, message="", detail=None, images=None):
        label = self.labels.get(step_id, step_id)
        if status == "active":
            print(f"- {label.strip()}" + (f": {message}" if message else ""))
            return
        print(f"  {status.upper()}" + (f": {message}" if message else ""))
        for im in images or []:
            print(f"    {im}")

    def log(self, message):
        print(message)

    def confirm(self, kind, prompt, images=None):
        print(f"[{kind}] {prompt}")
        if self.yes or kind == "start":
            return True
        return input("approve? [y/N] ").strip().lower() == "y"

    def choose(self, kind, title, prompt, options, images=None):
        return options[0][0] if self.yes else input(f"{title} {prompt} {[v for v, _ in options]} ").strip()

    def pick(self, prompt):
        return None


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--reasoner", default=None, help="backend:model[@options] (default: config agent.reasoner)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--max-steps", type=int, default=AgentConfig.max_steps)
    ap.add_argument("--video", default=None, help="sim: save a video of the run (mp4, 4x speed)")
    ap.add_argument("--hand-down", action="store_true", help="sim: a hand lying palm down on the table")
    args = ap.parse_args()

    print("loading models...")
    parts = build_runner_parts(sim=args.sim)
    rep = ConsoleReporter(args.yes)
    runner = GraspRunner(**parts, reporter=rep, speed_deg_s=args.speed, environment="sim" if args.sim else "real",
                         runs_dir=RUNS_DIR / "sim" if args.sim else None)
    reasoner = build_reasoner(args.reasoner, shared_vlm=parts["locator"].pointer)
    print(f"reasoner: {reasoner.name}")
    agent = Agent(runner, reasoner, AgentConfig(max_steps=args.max_steps))
    cfg, world = parts["cfg"], None
    if args.sim:
        from so101_zero_shot.mujoco_sim import SimWorld, SimWristCamera, make_mujoco_arm

        world = SimWorld(cfg, parts["cam_model"])
        world.start()
        if args.hand_down:
            world.set_hand(*world.HAND_PLACEMENTS["down"])
        camera, arm = SimWristCamera(world), make_mujoco_arm(cfg, world)
    else:
        camera = cfg.make_camera()
        camera.connect()
        arm = cfg.make_arm()
        arm.connect(enable_torque=False)
    try:
        from contextlib import nullcontext

        if args.video and world is not None:
            from so101_zero_shot.mujoco_sim import SimVideo
            recorder = SimVideo(world, args.video, f'"{args.request}"')
        else:
            recorder = nullcontext()
        with recorder:
            res = agent.run(arm, camera, args.request, record=False if args.no_record else None)
            if args.video and world is not None:
                recorder.caption = f'"{args.request}" -> {res.result}'
                time.sleep(0.3)
        print(f"\n{res.result}\n{res.out_dir}")
        if world is not None:
            print("objects now:", world.object_positions())
    except KeyboardInterrupt:
        print("\nsecond interrupt: disabling torque now")
    finally:
        arm.disconnect(disable_torque=True)
        camera.disconnect()
        if world is not None:
            world.close()


if __name__ == "__main__":
    main()
