"""Did the gripper close on something? Decided from servo feedback after a close command.

Measured on the parallel gripper (scripts/gripper_probe.py, 2026-11-25):
  closed on nothing:     position 0.2 %, |load| 28, current 1
  closed on a marker:    position 0.9 %, |load| 72, current 4
Position alone cannot tell a thin object from nothing (the jaws close almost fully around it), so the
squeeze load decides; position covers thicker objects.
"""

from __future__ import annotations

import time

import numpy as np

from ..hardware import GRIPPER

HELD_MIN_LOAD = 50.0  # |Present_Load| (0.1 % of max torque); between 28 (empty) and 72 (marker)
HELD_MIN_OPEN_PCT = 4.0  # stopped this far open = something is between the jaws


def is_holding(open_pct: float, load: float) -> bool:
    return abs(load) >= HELD_MIN_LOAD or open_pct >= HELD_MIN_OPEN_PCT


def read_grip(arm, seconds: float = 0.5) -> tuple[float, float]:
    """Median (opening %, load) over `seconds`, robust to single noisy reads."""
    pos, load = [], []
    t_end = time.monotonic() + seconds
    while True:
        pos.append(arm.read_gripper())
        load.append(arm.bus.read("Present_Load", GRIPPER))
        if time.monotonic() >= t_end:
            break
        time.sleep(0.05)
    return float(np.median(pos)), float(np.median(load))
