"""Move the arm to a named pose from config/robot.yaml, hold it, then return to rest and relax.

Safety:
- Shows the plan and asks for confirmation before enabling torque.
- Speed is capped (default 30 deg/s for early testing).
- Ctrl-C while moving or holding: stop, then go back to `rest` slowly and disable torque.
  A second Ctrl-C disables torque immediately (the arm will drop onto its stops).
- Targets outside the soft limits are rejected before anything moves.
"""

import argparse
import logging
import time

import numpy as np

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.hardware import ARM_JOINTS


def fmt(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:6.1f}" for v in q) + "]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", help="pose name from config/robot.yaml")
    ap.add_argument("--speed", type=float, default=30.0, help="max joint speed, deg/s")
    ap.add_argument("--gripper", type=float, default=None, help="also move gripper to this %% (0 closed, 100 max safe open)")
    ap.add_argument("--stay", action="store_true", help="do not return to rest; leave torque ON holding the pose")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    cfg = RobotConfig.load()
    if args.pose not in cfg.poses or "rest" not in cfg.poses:
        raise SystemExit(f"unknown pose {args.pose!r}; available: {sorted(cfg.poses)}")
    target, rest = cfg.poses[args.pose], cfg.poses["rest"]
    kin = cfg.make_kinematics()

    arm = cfg.make_arm()
    arm.connect(enable_torque=False)
    try:
        arm.check_limits(target)
        arm.check_limits(rest)
        start = arm.read_joints()
        tip = kin.fk(target)[:3, 3]
        print(f"joints        {'  '.join(f'{j[:6]:>6s}' for j in ARM_JOINTS)}")
        print(f"now           {fmt(start)}")
        print(f"{args.pose:<13} {fmt(target)}   tip -> {np.round(tip, 3)} m")
        print(f"largest joint move {np.max(np.abs(target - start)):.0f} deg at <= {args.speed:.0f} deg/s "
              f"(~{np.max(np.abs(target - start)) / args.speed:.1f} s)")
        if not args.yes and input("Enable torque and move? Hand near the power switch. [y/N] ").strip().lower() != "y":
            print("aborted, torque stays off")
            return

        arm.bus.sync_write("Goal_Position", arm.bus.sync_read("Present_Position"))
        arm.bus.enable_torque()
        try:
            err = arm.move_to(target, gripper=args.gripper, speed_deg_s=args.speed, settle_timeout_s=3.0)
            print(f"reached {args.pose}; error (target - measured) {fmt(err)} deg; gripper {arm.read_gripper():.0f}%")
            print(f"tip (FK of measured joints) {np.round(kin.fk(arm.read_joints())[:3, 3], 3)} m")
            if args.stay:
                input("Holding. Press ENTER to disable torque (arm will drop onto its stops)...")
                return
            input("Holding. Press ENTER to return to rest...")
        except KeyboardInterrupt:
            print("\ninterrupted: stopping and returning to rest (Ctrl-C again = torque off now)")
            arm.hold()
            time.sleep(0.3)
        arm.move_to(rest, speed_deg_s=min(args.speed, 30.0), settle_timeout_s=3.0)
        print("at rest")
    except KeyboardInterrupt:
        print("\nsecond interrupt: disabling torque now")
    finally:
        arm.disconnect(disable_torque=True)
        print("torque off")


if __name__ == "__main__":
    main()
