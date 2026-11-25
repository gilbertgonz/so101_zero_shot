"""Stream joint angles and FK tip position with torque OFF (move the arm by hand).

Use this to sanity-check calibration against the URDF: put the arm in a known pose and compare.
"""

import argparse
import time

import numpy as np

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.hardware import ARM_JOINTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="print one reading and exit")
    args = ap.parse_args()
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    arm = cfg.make_arm()
    arm.connect(enable_torque=False)
    try:
        print("  ".join(f"{j[:11]:>11s}" for j in ARM_JOINTS) + "    grip%    tip xyz (m)        approach axis")
        while True:
            q = arm.read_joints()
            T = kin.fk(q)
            line = (
                "  ".join(f"{v:11.1f}" for v in q)
                + f"  {arm.read_gripper():6.1f}   {np.array2string(T[:3, 3], precision=3, suppress_small=True)}"
                + f"  {np.array2string(T[:3, 2], precision=2, suppress_small=True)}"
            )
            if args.once:
                print(line)
                return
            print(line, end="\r", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
