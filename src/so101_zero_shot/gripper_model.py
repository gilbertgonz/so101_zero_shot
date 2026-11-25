"""Parallel gripper in the MuJoCo model: two slide joints (clamps), driven from the arm's gripper %."""

from __future__ import annotations

import mujoco
import numpy as np

GRIPPER_JOINTS = ("gripper", "gripper_left")  # right / left clamp, metres open from closed (0)
DEFAULT_OPEN_GAP_M = 0.046  # jaw gap at 100 % (config tool.gripper_open_gap_m)


def gripper_qadr(model: mujoco.MjModel) -> list[int]:
    return [int(model.joint(n).qposadr[0]) for n in GRIPPER_JOINTS]


def set_gripper_qpos(model: mujoco.MjModel, qpos: np.ndarray, open_pct: float,
                     open_gap_m: float = DEFAULT_OPEN_GAP_M, qadr: list[int] | None = None) -> None:
    """Open both clamps symmetrically: each moves half the gap; clamped to the joint ranges."""
    each = float(np.clip(open_pct, 0.0, 100.0)) / 100.0 * open_gap_m / 2.0
    for adr, name in zip(qadr or gripper_qadr(model), GRIPPER_JOINTS):
        lo, hi = model.joint(name).range
        qpos[adr] = min(max(each, lo), hi)
