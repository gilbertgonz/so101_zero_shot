# Adapted from HuggingFace LeRobot (lerobot/robots/so101_follower).
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
"""SO-101 follower arm: calibration, unit conversion, and safe blocking moves.

Units:
- Arm joints (first five) are in degrees, using LeRobot's `use_degrees=True` convention:
  deg = (ticks - mid) * 360 / 4095, mid = (range_min + range_max) / 2.
  This matches the zero pose of `so101_new_calib.urdf`.
  `joint_offsets_deg` corrects per-joint calibration error vs the URDF:
  urdf_deg = lerobot_deg - offset. Measured by placing the arm in known poses (see PROGRESS.md).
- Gripper is 0..100, 0 = closed, 100 = open (`gripper_inverted` flips the servo direction for
  grippers that close toward the max tick, e.g. the parallel-jaw gripper).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .feetech import RESOLUTION, FeetechBus, FeetechError, MotorCalibration

logger = logging.getLogger(__name__)

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
MOTOR_IDS: dict[str, int] = {name: i + 1 for i, name in enumerate((*ARM_JOINTS, GRIPPER))}

# wrist_roll is calibrated over a full turn, so its limits come from here rather than calibration.
WRIST_ROLL_LIMITS_DEG = (-157.0, 162.0)
LIMIT_MARGIN_DEG = 4.0

_TICKS_PER_DEG = (RESOLUTION - 1) / 360.0


class CalibrationError(RuntimeError):
    pass


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError as e:
        raise CalibrationError(f"no calibration file at {path}; run scripts/calibrate_arm.py") from e
    except json.JSONDecodeError as e:
        raise CalibrationError(f"calibration file {path} is not valid JSON: {e}") from e
    missing = set(MOTOR_IDS) - set(raw)
    if missing:
        raise CalibrationError(f"calibration file {path} is missing motors {sorted(missing)}")
    try:
        cal = {name: MotorCalibration(**raw[name]) for name in MOTOR_IDS}
    except TypeError as e:
        raise CalibrationError(f"calibration file {path} has unexpected fields: {e}") from e
    for name, c in cal.items():
        if c.range_max <= c.range_min:
            raise CalibrationError(f"{name}: range_max ({c.range_max}) <= range_min ({c.range_min})")
    return cal


def joint_limits_from_calibration(
    calibration: dict[str, MotorCalibration],
    joint_offsets_deg: dict[str, float] | None = None,
    margin_deg: float = LIMIT_MARGIN_DEG,
) -> dict[str, tuple[float, float]]:
    """Soft limits (deg) from the recorded range of motion, shrunk by `margin_deg` on each side.

    These are the arm's real mechanical limits, which are wider than the generic URDF limits
    (e.g. wrist_flex is about +-108 deg vs +-95 in the URDF) and matter for top-down reach.
    """
    limits = {}
    for m in ARM_JOINTS:
        if m == "wrist_roll":
            limits[m] = WRIST_ROLL_LIMITS_DEG
            continue
        c = calibration[m]
        half = (c.range_max - c.range_min) / 2.0 / _TICKS_PER_DEG
        off = (joint_offsets_deg or {}).get(m, 0.0)
        limits[m] = (-half - off + margin_deg, half - off - margin_deg)
    return limits


def save_calibration(path: Path, calibration: dict[str, MotorCalibration]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in calibration.items()}, indent=4) + "\n")


class MotionStopped(RuntimeError):
    """A move was interrupted by SO101Arm.request_stop(); the arm is holding where it was."""


class SO101Arm:
    """Blocking, safety-first interface to the SO-101 follower arm.

    Motion commands run in the caller's thread. Another thread may call request_stop() (the move holds
    and raises MotionStopped at the next control tick) or emergency_torque_off(); bus access is locked.
    """

    def __init__(
        self,
        port: str,
        calibration_path: Path,
        joint_limits_deg: dict[str, tuple[float, float]] | None = None,
        max_speed_deg_s: float = 60.0,
        control_hz: float = 50.0,
        joint_offsets_deg: dict[str, float] | None = None,
        gripper_inverted: bool = False,
    ):
        unknown = set(joint_offsets_deg or {}) - set(ARM_JOINTS)
        if unknown:
            raise ValueError(f"joint_offsets_deg has unknown joints {sorted(unknown)}")
        self.bus = FeetechBus(port, MOTOR_IDS)
        self.calibration_path = Path(calibration_path)
        self._limit_overrides = dict(joint_limits_deg or {})
        self.joint_limits: dict[str, tuple[float, float]] = {}
        self.max_speed_deg_s = max_speed_deg_s
        self.control_hz = control_hz
        self.calibration: dict[str, MotorCalibration] = {}
        self.joint_offsets = {m: float((joint_offsets_deg or {}).get(m, 0.0)) for m in ARM_JOINTS}
        self.gripper_inverted = gripper_inverted
        self.stop_event = threading.Event()
        self.estopped = False  # set by emergency_torque_off(); cleared by clear_stop()
        # Last goal sent per motor (5 joints in deg, gripper in %), NaN until commanded; for recording.
        self._command = np.full(len(MOTOR_IDS), np.nan)
        self._command_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def connect(self, enable_torque: bool = True) -> None:
        self.calibration = load_calibration(self.calibration_path)
        self.joint_limits = {
            **joint_limits_from_calibration(self.calibration, self.joint_offsets),
            **self._limit_overrides,
        }
        self.bus.connect()
        try:
            # Everything below runs with torque off. Torque is only enabled after Goal_Position has
            # been set to the present position, otherwise the servos jump to a stale goal.
            self.bus.disable_torque()
            on_motor = self.bus.read_calibration()
            if on_motor != self.calibration:
                logger.warning("servo EEPROM calibration differs from %s; writing file values", self.calibration_path)
                self.bus.write_calibration(self.calibration)
            self._configure_torque_off()
            if enable_torque:
                self.bus.sync_write("Goal_Position", self.bus.sync_read("Present_Position"))
                self.bus.enable_torque()
        except Exception:
            self.bus.disconnect(disable_torque=True)
            raise

    def _configure_torque_off(self) -> None:
        self.bus.configure_motors()
        for m in MOTOR_IDS:
            self.bus.write("Operating_Mode", m, 0)  # position mode
            self.bus.write("P_Coefficient", m, 16)  # default 32 is shaky
            self.bus.write("I_Coefficient", m, 0)
            self.bus.write("D_Coefficient", m, 32)
        # Keep the gripper from cooking itself when squeezing an object.
        self.bus.write("Max_Torque_Limit", GRIPPER, 500)
        self.bus.write("Protection_Current", GRIPPER, 250)
        self.bus.write("Overload_Torque", GRIPPER, 25)

    def disconnect(self, disable_torque: bool = True) -> None:
        self.bus.disconnect(disable_torque=disable_torque)

    def __enter__(self) -> SO101Arm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ conversions
    def _mid(self, m: str) -> float:
        c = self.calibration[m]
        return (c.range_min + c.range_max) / 2.0

    def ticks_to_deg(self, m: str, ticks: int) -> float:
        return (ticks - self._mid(m)) / _TICKS_PER_DEG - self.joint_offsets[m]

    def deg_to_ticks(self, m: str, deg: float) -> int:
        c = self.calibration[m]
        t = int(round((deg + self.joint_offsets[m]) * _TICKS_PER_DEG + self._mid(m)))
        return min(max(t, c.range_min), c.range_max)

    def ticks_to_gripper(self, ticks: int) -> float:
        c = self.calibration[GRIPPER]
        pct = 100.0 * (min(max(ticks, c.range_min), c.range_max) - c.range_min) / (c.range_max - c.range_min)
        return 100.0 - pct if self.gripper_inverted else pct

    def gripper_to_ticks(self, pct: float) -> int:
        c = self.calibration[GRIPPER]
        pct = min(max(pct, 0.0), 100.0)
        if self.gripper_inverted:
            pct = 100.0 - pct
        return int(round(c.range_min + pct / 100.0 * (c.range_max - c.range_min)))

    # ------------------------------------------------------------------ state
    def read_joints(self) -> np.ndarray:
        """Arm joint angles in degrees, shape (5,)."""
        ticks = self.bus.sync_read("Present_Position", ARM_JOINTS)
        return np.array([self.ticks_to_deg(m, ticks[m]) for m in ARM_JOINTS])

    def read_gripper(self) -> float:
        return self.ticks_to_gripper(self.bus.read("Present_Position", GRIPPER))

    def check_limits(self, joints_deg: np.ndarray) -> None:
        if not self.joint_limits:
            raise RuntimeError("arm not connected (limits come from the calibration loaded in connect())")
        joints_deg = np.asarray(joints_deg, dtype=float)
        if joints_deg.shape != (len(ARM_JOINTS),) or not np.all(np.isfinite(joints_deg)):
            raise ValueError(f"expected 5 finite joint angles, got {joints_deg}")
        bad = [
            f"{m}={q:.1f} not in [{lo}, {hi}]"
            for m, q in zip(ARM_JOINTS, joints_deg)
            for lo, hi in [self.joint_limits[m]]
            if not lo <= q <= hi
        ]
        if bad:
            raise ValueError("joint target outside soft limits: " + "; ".join(bad))

    # ------------------------------------------------------------------ stopping
    def request_stop(self) -> None:
        """Ask the running (or next) motion command to hold and raise MotionStopped."""
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()
        self.estopped = False

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            self.hold()
            raise MotionStopped("motion stopped on request")

    def emergency_torque_off(self) -> None:
        """Disable torque on every servo now (the arm drops onto its stops). Safe from any thread."""
        self.estopped = True
        self.stop_event.set()
        self.bus.disable_torque(num_retry=5)

    def last_command(self) -> np.ndarray:
        """Last goal sent, shape (6,): 5 joints (deg) then gripper (%); NaN for motors not yet commanded."""
        with self._command_lock:
            return self._command.copy()

    def read_state(self, num_retry: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(position, load), shape (6,) each, in two bus transactions: 5 joints (deg) + gripper (%), and
        Present_Load (0.1 % of max torque, signed) per motor. No retries by default so a recorder never
        holds the bus long. Raises FeetechError."""
        names = list(MOTOR_IDS)
        ticks = self.bus.sync_read("Present_Position", names, num_retry=num_retry)
        load = self.bus.sync_read("Present_Load", names, num_retry=num_retry)
        pos = [self.ticks_to_deg(m, ticks[m]) for m in ARM_JOINTS] + [self.ticks_to_gripper(ticks[GRIPPER])]
        return np.array(pos), np.array([float(load[m]) for m in names])

    # ------------------------------------------------------------------ commands
    def _note_command(self, goal_ticks: dict[str, int]) -> None:
        with self._command_lock:
            for m, t in goal_ticks.items():
                i = list(MOTOR_IDS).index(m)
                self._command[i] = self.ticks_to_gripper(t) if m == GRIPPER else self.ticks_to_deg(m, t)

    def _send(self, joints_deg: np.ndarray, gripper: float | None) -> None:
        goal = {m: self.deg_to_ticks(m, q) for m, q in zip(ARM_JOINTS, joints_deg)}
        if gripper is not None:
            goal[GRIPPER] = self.gripper_to_ticks(gripper)
        self.bus.sync_write("Goal_Position", goal)
        self._note_command(goal)

    def move_to(
        self,
        joints_deg: np.ndarray,
        gripper: float | None = None,
        speed_deg_s: float | None = None,
        settle_tol_deg: float = 0.7,
        settle_timeout_s: float = 2.0,
        compensate: bool = True,
        max_correction_deg: float = 10.0,
        correction_iters: int = 6,
    ) -> np.ndarray:
        """Interpolate in joint space to the target, wait to settle, then correct steady-state sag.

        With the low P gain used to avoid shaking, gravity leaves a steady error (5 deg on the elbow
        in the scan pose, ~2 cm at the tip). Since the encoders are on the output shafts, the error is
        measured directly; `compensate` re-commands target + accumulated error a few times, with the
        offset capped at `max_correction_deg` per joint so it cannot push hard against an obstacle.
        Pass `compensate=False` for moves that are expected to end in contact.

        Returns the final joint error (target - measured) in degrees. Raises ValueError for targets
        outside the soft limits before moving at all.
        """
        target = np.asarray(joints_deg, dtype=float)
        self.check_limits(target)
        self._check_stop()
        speed = min(speed_deg_s or self.max_speed_deg_s, self.max_speed_deg_s)
        start = self.read_joints()
        g_start = self.read_gripper() if gripper is not None else None

        duration = max(float(np.max(np.abs(target - start))) / speed, 0.1)
        n = max(int(duration * self.control_hz), 1)
        period = 1.0 / self.control_hz
        t_next = time.perf_counter()
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)  # smoothstep: zero velocity at both ends
            g = None if gripper is None else g_start + a * (gripper - g_start)
            self._check_stop()
            self._send(start + a * (target - start), g)
            t_next += period
            time.sleep(max(0.0, t_next - time.perf_counter()))

        err = self._wait_settled(target, settle_tol_deg, settle_timeout_s)
        if compensate:
            command = target.copy()
            for _ in range(correction_iters):
                if np.max(np.abs(err)) <= settle_tol_deg:
                    break
                command = target + np.clip(command - target + err, -max_correction_deg, max_correction_deg)
                self._send(command, None)
                err = self._wait_settled(target, settle_tol_deg, 0.4)
        if np.max(np.abs(err)) > settle_tol_deg:
            logger.warning("move_to did not settle: error per joint (deg) = %s", np.round(err, 1))
        return err

    def _wait_settled(self, target: np.ndarray, tol_deg: float, timeout_s: float) -> np.ndarray:
        """Poll until within tolerance or the joints stop moving (or timeout). Returns target - measured."""
        deadline = time.perf_counter() + timeout_s
        prev = None
        while True:
            self._check_stop()
            q = self.read_joints()
            err = target - q
            stopped = prev is not None and np.max(np.abs(q - prev)) < 0.1
            if np.max(np.abs(err)) <= tol_deg or stopped or time.perf_counter() > deadline:
                return err
            prev = q
            time.sleep(0.05)

    def set_gripper(self, pct: float, duration_s: float = 0.6) -> float:
        """Move the gripper (0 closed, 100 open). Returns the measured opening after the move.

        When grasping, the measured value staying well above the command means an object is held.
        """
        start = self.read_gripper()
        n = max(int(duration_s * self.control_hz), 1)
        for i in range(1, n + 1):
            self._check_stop()
            goal = {GRIPPER: self.gripper_to_ticks(start + (pct - start) * i / n)}
            self.bus.sync_write("Goal_Position", goal)
            self._note_command(goal)
            time.sleep(1.0 / self.control_hz)
        time.sleep(0.3)
        return self.read_gripper()

    def hold(self) -> None:
        """Re-command the present position (stops any motion in progress)."""
        present = self.bus.sync_read("Present_Position")
        self.bus.sync_write("Goal_Position", present)
        self._note_command(present)


__all__ = [
    "ARM_JOINTS",
    "GRIPPER",
    "MOTOR_IDS",
    "CalibrationError",
    "FeetechError",
    "MotionStopped",
    "SO101Arm",
    "load_calibration",
    "joint_limits_from_calibration",
    "save_calibration",
]
