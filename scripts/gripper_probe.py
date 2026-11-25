"""Measure gripper feedback when closing on nothing vs on an object (to tell "held" from "missed").

Only the gripper moves; the arm holds its current pose (start it at rest). Opens to the safe maximum,
closes, and prints position / load / current over 1.5 s, first empty, then with an object you place
between the jaws. Gripper torque is limited (Max_Torque_Limit 500), but keep fingers clear when closing.
"""

from __future__ import annotations

import time

import numpy as np

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.hardware import GRIPPER


def sample(arm, seconds: float = 1.5) -> dict:
    pos, load, cur = [], [], []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        pos.append(arm.read_gripper())
        load.append(arm.bus.read("Present_Load", GRIPPER))
        cur.append(arm.bus.read("Present_Current", GRIPPER))
        time.sleep(0.05)
    return {"pos_pct": float(np.median(pos)), "load": float(np.median(load)), "current": float(np.median(cur))}


def main() -> None:
    cfg = RobotConfig.load()
    arm = cfg.make_arm()
    arm.connect(enable_torque=False)
    try:
        if input("Enable torque (arm holds still) and test the gripper? [y/N] ").strip().lower() != "y":
            return
        arm.bus.sync_write("Goal_Position", arm.bus.sync_read("Present_Position"))
        arm.bus.enable_torque()
        results = {}
        for name, prompt in (("empty", "Nothing between the jaws. ENTER to close..."),
                             ("object", "Opening. Hold the object between the jaws, ENTER to close (fingers clear!)...")):
            arm.set_gripper(100.0, duration_s=0.8)
            input(prompt)
            arm.set_gripper(0.0, duration_s=1.0)
            time.sleep(0.5)
            results[name] = sample(arm)
            print(f"  {name:6s} {results[name]}")
        arm.set_gripper(100.0, duration_s=0.8)
        input("Remove the object, ENTER to close the gripper and finish...")
        arm.set_gripper(0.0, duration_s=0.8)
    finally:
        arm.disconnect(disable_torque=True)
        print("torque off")


if __name__ == "__main__":
    main()
