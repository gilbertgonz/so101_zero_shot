"""Self- and table-collision checking with the MuJoCo model (assets/so101/scene.xml).

The model has the stock gripper and no camera mount, so a clearance margin is applied: anything closer
than `margin_m` counts as a collision. Pair with an explicit tip-height check for the longer
parallel-gripper fingers (see viewpoints.min_link_height).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..config import MJCF_SCENE_PATH  # noqa: E402
from ..gripper_model import gripper_qadr, set_gripper_qpos  # noqa: E402
from ..hardware.so101 import ARM_JOINTS  # noqa: E402

# Pairs that are always close by construction (their distance does not depend on the joints that
# matter here): base on the table, shoulder just above it.
_IGNORED_PAIRS = {
    frozenset({"base", "world"}),
    frozenset({"shoulder", "world"}),
}


class CollisionChecker:
    def __init__(self, margin_m: float = 0.015, gripper_pct: float = 100.0):
        """Checks with the gripper `gripper_pct` open (default fully open: the widest it gets)."""
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.margin_m = margin_m
        # MuJoCo reports a contact when dist < margin_a + margin_b; filter to dist < margin_m below.
        self.model.geom_margin[:] = margin_m
        self.gripper_pct = gripper_pct
        self._joint_qpos = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self._gripper_qpos = gripper_qadr(self.model)

    def contacts(self, joints_deg: np.ndarray) -> list[tuple[str, str, float]]:
        """(body_a, body_b, signed distance m) for every pair closer than the margin."""
        self.data.qpos[:] = 0.0
        for adr, v in zip(self._joint_qpos, np.asarray(joints_deg, dtype=float)):
            self.data.qpos[adr] = np.deg2rad(v)
        set_gripper_qpos(self.model, self.data.qpos, self.gripper_pct, qadr=self._gripper_qpos)
        mujoco.mj_forward(self.model, self.data)
        out = []
        for c in self.data.contact[: self.data.ncon]:
            a = self.model.body(self.model.geom_bodyid[c.geom1]).name
            b = self.model.body(self.model.geom_bodyid[c.geom2]).name
            if c.dist < self.margin_m and frozenset({a, b}) not in _IGNORED_PAIRS:
                out.append((a, b, float(c.dist)))
        return out

    def in_collision(self, joints_deg: np.ndarray) -> bool:
        return bool(self.contacts(joints_deg))

    def path_in_collision(self, a: np.ndarray, b: np.ndarray, steps: int = 30) -> bool:
        """Check the smoothstep joint-space path SO101Arm.move_to executes from a to b."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        for t in np.linspace(0, 1, steps):
            s = t * t * (3 - 2 * t)
            if self.in_collision(a + s * (b - a)):
                return True
        return False
