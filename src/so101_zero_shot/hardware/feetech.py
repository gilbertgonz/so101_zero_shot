# Adapted from HuggingFace LeRobot (lerobot/motors/feetech, lerobot/motors/motors_bus.py).
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
# Trimmed to what the SO-101 needs: STS3215 servos, protocol 0, raw (unnormalized) register access.
# Unit conversion (ticks <-> degrees) lives in so101.py, not here.
"""Minimal Feetech STS3215 bus driver on top of scservo_sdk."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import scservo_sdk as scs

logger = logging.getLogger(__name__)

STS3215_MODEL_NUMBER = 777
RESOLUTION = 4096  # ticks per revolution
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

# name: (address, size_bytes). Source: Feetech STS/SMS series manual.
CONTROL_TABLE: dict[str, tuple[int, int]] = {
    # EEPROM
    "Firmware_Major_Version": (0, 1),
    "Firmware_Minor_Version": (1, 1),
    "Model_Number": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Protection_Current": (28, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Overload_Torque": (36, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    "Maximum_Acceleration": (85, 1),
}

# Registers stored in sign-magnitude form, with the index of the sign bit.
SIGN_BIT: dict[str, int] = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Goal_Velocity": 15,
    "Present_Position": 15,
    "Present_Velocity": 15,
}


class FeetechError(RuntimeError):
    """Communication or servo-reported error on the Feetech bus."""


@dataclass(frozen=True)
class MotorCalibration:
    """Per-motor calibration. Field names match LeRobot's JSON so old files load as-is."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    max_magnitude = (1 << sign_bit) - 1
    if abs(value) > max_magnitude:
        raise ValueError(f"|{value}| exceeds {max_magnitude} (sign bit {sign_bit})")
    return ((1 if value < 0 else 0) << sign_bit) | abs(value)


def decode_sign_magnitude(encoded: int, sign_bit: int) -> int:
    magnitude = encoded & ((1 << sign_bit) - 1)
    return -magnitude if (encoded >> sign_bit) & 1 else magnitude


def _to_bytes(value: int, length: int) -> list[int]:
    if value < 0 or value >= 1 << (8 * length):
        raise ValueError(f"value {value} does not fit in {length} byte(s)")
    return [(value >> (8 * i)) & 0xFF for i in range(length)]


def _patched_set_packet_timeout(self, packet_length):  # noqa: N802
    # Fixes too-short packet timeouts in the PyPI scservo_sdk (see LeRobot / gitee issue IBY2S6).
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class FeetechBus:
    """Raw register access to a set of named STS3215 servos on one serial port."""

    def __init__(self, port: str, motors: dict[str, int], baudrate: int = DEFAULT_BAUDRATE):
        if len(set(motors.values())) != len(motors):
            raise ValueError(f"duplicate motor ids in {motors}")
        self.port = port
        self.motors = dict(motors)  # name -> id
        self._names_by_id = {id_: name for name, id_ in motors.items()}
        self.baudrate = baudrate
        self._port_handler = scs.PortHandler(port)
        self._port_handler.setPacketTimeout = _patched_set_packet_timeout.__get__(
            self._port_handler, scs.PortHandler
        )
        self._packet_handler = scs.PacketHandler(0)
        self._sync_reader = scs.GroupSyncRead(self._port_handler, self._packet_handler, 0, 0)
        self._sync_writer = scs.GroupSyncWrite(self._port_handler, self._packet_handler, 0, 0)

    # ------------------------------------------------------------------ connection
    @property
    def is_connected(self) -> bool:
        return bool(self._port_handler.is_open)

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            return
        try:
            opened = self._port_handler.openPort()
        except OSError as e:
            raise FeetechError(f"could not open serial port {self.port!r}: {e}") from e
        if not opened:
            raise FeetechError(f"could not open serial port {self.port!r}")
        if not self._port_handler.setBaudRate(self.baudrate):
            self._port_handler.closePort()
            raise FeetechError(f"could not set baudrate {self.baudrate} on {self.port!r}")
        self._port_handler.setPacketTimeoutMillis(DEFAULT_TIMEOUT_MS)
        if handshake:
            try:
                self._handshake()
            except Exception:
                self._port_handler.closePort()
                raise

    def _handshake(self) -> None:
        missing, wrong_model = [], []
        for name, id_ in self.motors.items():
            model, comm, _ = self._packet_handler.ping(self._port_handler, id_)
            if comm != scs.COMM_SUCCESS:
                missing.append(f"{name}(id={id_})")
            elif model != STS3215_MODEL_NUMBER:
                wrong_model.append(f"{name}(id={id_}, model={model})")
        if missing or wrong_model:
            raise FeetechError(
                f"bus handshake failed on {self.port!r}: missing={missing} wrong_model={wrong_model}. "
                "Check servo power and cabling."
            )

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            return
        if disable_torque:
            self._port_handler.clearPort()
            self._port_handler.is_using = False
            try:
                self.disable_torque(num_retry=5)
            except FeetechError as e:
                logger.error("failed to disable torque on disconnect: %s", e)
        self._port_handler.closePort()

    # ------------------------------------------------------------------ single motor
    def _check(self, comm: int, error: int, what: str) -> None:
        if comm != scs.COMM_SUCCESS:
            raise FeetechError(f"{what}: {self._packet_handler.getTxRxResult(comm)}")
        if error != 0:
            raise FeetechError(f"{what}: {self._packet_handler.getRxPacketError(error)}")

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise FeetechError(f"bus on {self.port!r} is not connected")

    def read(self, register: str, motor: str, num_retry: int = 2) -> int:
        self._require_connected()
        addr, length = CONTROL_TABLE[register]
        fn = {1: self._packet_handler.read1ByteTxRx, 2: self._packet_handler.read2ByteTxRx}[length]
        id_ = self.motors[motor]
        for _ in range(1 + num_retry):
            value, comm, error = fn(self._port_handler, id_, addr)
            if comm == scs.COMM_SUCCESS:
                break
        self._check(comm, error, f"read {register} from {motor}")
        return decode_sign_magnitude(value, SIGN_BIT[register]) if register in SIGN_BIT else value

    def write(self, register: str, motor: str, value: int, num_retry: int = 2) -> None:
        self._require_connected()
        addr, length = CONTROL_TABLE[register]
        if register in SIGN_BIT:
            value = encode_sign_magnitude(int(value), SIGN_BIT[register])
        data = _to_bytes(int(value), length)
        id_ = self.motors[motor]
        for _ in range(1 + num_retry):
            comm, error = self._packet_handler.writeTxRx(self._port_handler, id_, addr, length, data)
            if comm == scs.COMM_SUCCESS:
                break
        self._check(comm, error, f"write {register}={value} to {motor}")

    # ------------------------------------------------------------------ group
    def sync_read(self, register: str, motors: Sequence[str] | None = None, num_retry: int = 2) -> dict[str, int]:
        self._require_connected()
        names = list(self.motors) if motors is None else list(motors)
        addr, length = CONTROL_TABLE[register]
        self._sync_reader.clearParam()
        self._sync_reader.start_address = addr
        self._sync_reader.data_length = length
        for name in names:
            self._sync_reader.addParam(self.motors[name])
        for _ in range(1 + num_retry):
            comm = self._sync_reader.txRxPacket()
            if comm == scs.COMM_SUCCESS:
                break
        if comm != scs.COMM_SUCCESS:
            raise FeetechError(f"sync_read {register}: {self._packet_handler.getTxRxResult(comm)}")
        out = {}
        for name in names:
            raw = self._sync_reader.getData(self.motors[name], addr, length)
            out[name] = decode_sign_magnitude(raw, SIGN_BIT[register]) if register in SIGN_BIT else raw
        return out

    def sync_write(self, register: str, values: dict[str, int], num_retry: int = 1) -> None:
        """Fire-and-forget write (no status packet). Use for streaming goals, not configuration."""
        self._require_connected()
        addr, length = CONTROL_TABLE[register]
        self._sync_writer.clearParam()
        self._sync_writer.start_address = addr
        self._sync_writer.data_length = length
        for name, value in values.items():
            v = encode_sign_magnitude(int(value), SIGN_BIT[register]) if register in SIGN_BIT else int(value)
            self._sync_writer.addParam(self.motors[name], _to_bytes(v, length))
        for _ in range(1 + num_retry):
            comm = self._sync_writer.txPacket()
            if comm == scs.COMM_SUCCESS:
                return
        raise FeetechError(f"sync_write {register}: {self._packet_handler.getTxRxResult(comm)}")

    # ------------------------------------------------------------------ torque
    def enable_torque(self, motors: Sequence[str] | None = None, num_retry: int = 2) -> None:
        for m in self.motors if motors is None else motors:
            self.write("Torque_Enable", m, 1, num_retry=num_retry)
            self.write("Lock", m, 1, num_retry=num_retry)

    def disable_torque(self, motors: Sequence[str] | None = None, num_retry: int = 2) -> None:
        for m in self.motors if motors is None else motors:
            self.write("Torque_Enable", m, 0, num_retry=num_retry)
            self.write("Lock", m, 0, num_retry=num_retry)

    # ------------------------------------------------------------------ calibration
    def read_calibration(self) -> dict[str, MotorCalibration]:
        return {
            name: MotorCalibration(
                id=id_,
                drive_mode=0,
                homing_offset=self.read("Homing_Offset", name),
                range_min=self.read("Min_Position_Limit", name),
                range_max=self.read("Max_Position_Limit", name),
            )
            for name, id_ in self.motors.items()
        }

    def write_calibration(self, calibration: dict[str, MotorCalibration]) -> None:
        """Write to EEPROM. Torque must be disabled (Lock=0) for EEPROM writes to stick."""
        for name, cal in calibration.items():
            self.write("Homing_Offset", name, cal.homing_offset)
            self.write("Min_Position_Limit", name, cal.range_min)
            self.write("Max_Position_Limit", name, cal.range_max)

    def reset_calibration(self, motors: Sequence[str]) -> None:
        for m in motors:
            self.write("Homing_Offset", m, 0)
            self.write("Min_Position_Limit", m, 0)
            self.write("Max_Position_Limit", m, RESOLUTION - 1)

    def set_half_turn_homings(self, motors: Sequence[str]) -> dict[str, int]:
        """Set homing offsets so the current position reads as mid-range (2047).

        On Feetech servos: Present_Position = Actual_Position - Homing_Offset.
        """
        self.reset_calibration(motors)
        actual = self.sync_read("Present_Position", motors)
        offsets = {m: pos - (RESOLUTION - 1) // 2 for m, pos in actual.items()}
        for m, off in offsets.items():
            self.write("Homing_Offset", m, off)
        return offsets

    def configure_motors(self) -> None:
        for m in self.motors:
            self.write("Return_Delay_Time", m, 0)  # 2us instead of default 500us
            self.write("Maximum_Acceleration", m, 254)
            self.write("Acceleration", m, 254)

    def name_of(self, id_: int) -> str:
        return self._names_by_id[id_]
