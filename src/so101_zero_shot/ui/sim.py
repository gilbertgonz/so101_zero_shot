"""Simulated hardware for `scripts/ui.py --fake`: exercise the whole UI and grasp flow without the arm.

SimBus stands in for FeetechBus under the real SO101Arm, so interpolation, stopping and limits are the
real code. Servos reach their goal immediately (one control tick); the gripper reports a squeeze load
when commanded closed, as if it were holding something. SimCamera serves a saved image.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import RobotConfig
from ..hardware.so101 import GRIPPER, MOTOR_IDS, SO101Arm, joint_limits_from_calibration, load_calibration


class SimBus:
    def __init__(self):
        self.pos: dict[str, int] = {m: 2048 for m in MOTOR_IDS}
        self.torque = False
        self._lock = threading.RLock()

    def sync_write(self, register, values, num_retry=1):
        if register == "Goal_Position" and self.torque:
            with self._lock:
                self.pos.update({m: int(v) for m, v in values.items()})

    def sync_read(self, register, motors=None, num_retry=2):
        with self._lock:
            return {m: self.pos[m] for m in (motors or MOTOR_IDS)}

    def read(self, register, motor, num_retry=2):
        if register == "Present_Load":
            return -72 if motor == GRIPPER else 0
        with self._lock:
            return self.pos[motor]

    def write(self, register, motor, value, num_retry=2):
        if register == "Torque_Enable":
            self.torque = bool(value)

    def enable_torque(self, motors=None, num_retry=2):
        self.torque = True

    def disable_torque(self, motors=None, num_retry=2):
        self.torque = False

    def disconnect(self, disable_torque=True):
        self.torque = False


def make_sim_arm(cfg: RobotConfig) -> SO101Arm:
    arm = SO101Arm("/dev/null", cfg.calibration_file, cfg.joint_limits_deg or None, cfg.max_speed_deg_s,
                   joint_offsets_deg=cfg.joint_offsets_deg, gripper_inverted=cfg.gripper_inverted)
    arm.calibration = load_calibration(cfg.calibration_file)
    arm.joint_limits = {**joint_limits_from_calibration(arm.calibration, arm.joint_offsets), **cfg.joint_limits_deg}
    arm.bus = SimBus()
    for m, q in zip(arm.joint_offsets, cfg.poses["rest"]):
        arm.bus.pos[m] = arm.deg_to_ticks(m, q)
    arm.bus.pos[GRIPPER] = arm.gripper_to_ticks(0.0)
    arm.connect = lambda enable_torque=True: None  # type: ignore[method-assign]
    return arm


class SimCamera:
    def __init__(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        self.img = img
        self.reconnects = 0

    def connect(self, warmup_s: float = 0.0):
        pass

    def set_exposure(self, value):
        pass

    def read(self, newer_than=None, timeout_s=8.0):
        time.sleep(1 / 20)
        noisy = cv2.add(self.img, np.random.default_rng().integers(0, 3, self.img.shape, dtype=np.uint8))
        return noisy, time.monotonic()

    def read_fresh(self, settle_s: float = 0.15):
        return self.read()[0]

    def disconnect(self):
        pass
