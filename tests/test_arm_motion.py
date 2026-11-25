"""SO101Arm motion logic against a fake bus (no hardware)."""

import time

import numpy as np
import pytest

from so101_zero_shot.hardware import so101
from so101_zero_shot.hardware.feetech import MotorCalibration
from so101_zero_shot.hardware.so101 import ARM_JOINTS, GRIPPER, MOTOR_IDS, SO101Arm

CAL = {m: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=900, range_max=3200) for m, i in MOTOR_IDS.items()}


class SaggingBus:
    """Servo model: position = goal - sag (constant steady-state error, like gravity with low P gain)."""

    def __init__(self, sag_ticks: dict[str, int]):
        self.pos = {m: 2050 for m in MOTOR_IDS}
        self.sag = sag_ticks
        self.goals: list[dict[str, int]] = []

    def sync_write(self, register, values):
        assert register == "Goal_Position"
        self.goals.append(dict(values))
        for m, v in values.items():
            self.pos[m] = v - self.sag.get(m, 0)

    def sync_read(self, register, motors=None):
        return {m: self.pos[m] for m in (motors or MOTOR_IDS)}

    def read(self, register, motor):
        return self.pos[motor]


def make_arm(sag_ticks, offsets=None):
    arm = SO101Arm("/dev/null", "/nonexistent.json", control_hz=1000.0, joint_offsets_deg=offsets)
    arm.calibration = CAL
    arm.joint_limits = so101.joint_limits_from_calibration(CAL, arm.joint_offsets)
    arm.bus = SaggingBus(sag_ticks)
    return arm


def test_sag_is_compensated():
    arm = make_arm({"elbow_flex": -57})  # ~5 deg droop
    target = np.array([0.0, -20.0, 10.0, 40.0, 0.0])
    err = arm.move_to(target, speed_deg_s=1000.0)
    assert np.max(np.abs(err)) <= 0.7
    elbow_cmd = arm.ticks_to_deg("elbow_flex", arm.bus.goals[-1]["elbow_flex"])
    assert elbow_cmd == pytest.approx(10.0 - 5.0, abs=0.3)  # commanded past the target to cancel the droop


def test_correction_is_capped():
    arm = make_arm({"elbow_flex": -400})  # 35 deg "sag" = blocked by an obstacle
    target = np.array([0.0, -20.0, 10.0, 40.0, 0.0])
    arm.move_to(target, speed_deg_s=1000.0, max_correction_deg=10.0)
    elbow_cmd = arm.ticks_to_deg("elbow_flex", arm.bus.goals[-1]["elbow_flex"])
    assert abs(elbow_cmd - 10.0) <= 10.0 + 0.1


def test_no_compensation_when_disabled():
    arm = make_arm({"elbow_flex": -57})
    target = np.array([0.0, -20.0, 10.0, 40.0, 0.0])
    err = arm.move_to(target, speed_deg_s=1000.0, compensate=False)
    assert err[ARM_JOINTS.index("elbow_flex")] == pytest.approx(-5.0, abs=0.2)


def test_out_of_limits_rejected_before_motion():
    arm = make_arm({})
    with pytest.raises(ValueError):
        arm.move_to(np.array([0.0, 0.0, 0.0, 179.0, 0.0]))
    assert arm.bus.goals == []


def test_offsets_and_gripper_inversion_roundtrip():
    arm = make_arm({}, offsets={"elbow_flex": 11.5})
    t = arm.deg_to_ticks("elbow_flex", 0.0)
    assert arm.ticks_to_deg("elbow_flex", t) == pytest.approx(0.0, abs=0.1)
    assert t == pytest.approx((900 + 3200) / 2 + 11.5 * 4095 / 360, abs=1)
    arm.gripper_inverted = True
    assert arm.gripper_to_ticks(0.0) == CAL[GRIPPER].range_max  # closed = max ticks on the parallel gripper
    assert arm.gripper_to_ticks(150.0) == CAL[GRIPPER].range_min  # clamped: never past max safe open


def test_request_stop_interrupts_a_move_and_holds():
    import threading

    from so101_zero_shot.hardware import MotionStopped

    arm = make_arm(sag_ticks={})
    target = np.array([0.0, -30.0, 10.0, 60.0, 0.0])
    errors = []

    def run():
        try:
            arm.move_to(target, speed_deg_s=30.0)  # ~1.4 s move
        except MotionStopped as e:
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)
    arm.request_stop()
    t.join(2.0)
    assert not t.is_alive() and errors, "move did not stop"
    assert np.max(np.abs(arm.read_joints() - target)) > 5.0  # stopped partway
    with pytest.raises(MotionStopped):  # stays stopped until cleared
        arm.move_to(target)
    arm.clear_stop()
    arm.move_to(target, speed_deg_s=200.0, compensate=False)
    assert np.max(np.abs(arm.read_joints() - target)) < 1.0
