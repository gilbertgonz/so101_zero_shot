"""Create config/calibration/so101_follower.json.

Modes:
  --from-eeprom  Read the calibration already stored in the servos (e.g. from a previous LeRobot
                 calibration) and save it. No motion, torque stays off.
  (default)      Full interactive calibration, same procedure as LeRobot:
                 1) move every arm joint to the middle of its range, press ENTER
                 2) sweep every arm joint through its full range, press ENTER
                 The GRIPPER is never re-homed or swept: the parallel gripper's gears jump their rail
                 if opened past ~49% of the old range, so its safe range (set by hand, also stored in
                 the servo's position limits) is carried over from the existing calibration file.
"""

import argparse
import select
import sys

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.hardware import FeetechBus, MotorCalibration, load_calibration, save_calibration
from so101_zero_shot.hardware.so101 import ARM_JOINTS, GRIPPER, MOTOR_IDS

# Wrist roll can spin freely, so record the full turn instead of a swept range (as LeRobot does).
FULL_TURN = ("wrist_roll",)


def enter_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.readline() is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-eeprom", action="store_true")
    args = ap.parse_args()

    cfg = RobotConfig.load()
    bus = FeetechBus(cfg.port, MOTOR_IDS)
    bus.connect()
    try:
        bus.disable_torque()
        if args.from_eeprom:
            cal = bus.read_calibration()
        else:
            # Load first so a missing/broken gripper calibration aborts before touching anything.
            gripper_cal = load_calibration(cfg.calibration_file)[GRIPPER]
            input("Move the arm to the MIDDLE of every joint's range (do NOT touch the gripper), then press ENTER...")
            offsets = bus.set_half_turn_homings(list(ARM_JOINTS))
            swept = [m for m in ARM_JOINTS if m not in FULL_TURN]
            print("Sweep every joint (except wrist_roll) through its full range. Press ENTER when done.")
            pos = bus.sync_read("Present_Position", swept)
            mins, maxes = dict(pos), dict(pos)
            while not enter_pressed():
                pos = bus.sync_read("Present_Position", swept)
                for m, p in pos.items():
                    mins[m], maxes[m] = min(mins[m], p), max(maxes[m], p)
                print("  " + "  ".join(f"{m[:8]}:{mins[m]}-{maxes[m]}" for m in swept), end="\r")
            print()
            for m in FULL_TURN:
                mins[m], maxes[m] = 0, 4095
            narrow = [m for m in ARM_JOINTS if maxes[m] - mins[m] < 200]
            if narrow:
                raise SystemExit(f"range too small for {narrow}; did you move them? Nothing saved.")
            cal = {
                m: MotorCalibration(
                    id=MOTOR_IDS[m], drive_mode=0, homing_offset=offsets[m], range_min=mins[m], range_max=maxes[m]
                )
                for m in ARM_JOINTS
            }
            cal[GRIPPER] = gripper_cal
            bus.write_calibration(cal)
        for m, c in cal.items():
            print(f"{m:14s} {c}")
        save_calibration(cfg.calibration_file, cal)
        print("saved", cfg.calibration_file)
    finally:
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
