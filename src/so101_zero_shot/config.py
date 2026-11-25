"""Repo paths and the robot config file (config/robot.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "so101"
URDF_PATH = ASSETS_DIR / "so101_new_calib.urdf"
MJCF_SCENE_PATH = ASSETS_DIR / "scene.xml"
CONFIG_DIR = REPO_ROOT / "config"
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class RobotConfig:
    port: str
    calibration_file: Path
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fps: int
    max_speed_deg_s: float
    joint_offsets_deg: dict[str, float] = field(default_factory=dict)
    gripper_inverted: bool = False
    tip_extension_m: float = 0.0
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    poses: dict[str, np.ndarray] = field(default_factory=dict)
    workspace: dict[str, tuple[float, float]] = field(default_factory=dict)

    def check_in_workspace(self, point_m: np.ndarray) -> None:
        """Raise ValueError if a base-frame point lies outside the configured workspace box."""
        p = np.asarray(point_m, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"expected a finite 3D point, got {point_m}")
        bad = [
            f"{ax}={v:.3f} not in [{lo}, {hi}]"
            for ax, v in zip("xyz", p)
            for lo, hi in [self.workspace[ax]]
            if not lo <= v <= hi
        ]
        if bad:
            raise ValueError("target outside workspace: " + "; ".join(bad))

    @classmethod
    def load(cls, path: Path = CONFIG_DIR / "robot.yaml") -> RobotConfig:
        raw = yaml.safe_load(Path(path).read_text())
        cam = raw["camera"]
        return cls(
            port=raw["arm"]["port"],
            calibration_file=(CONFIG_DIR / raw["arm"]["calibration_file"]).resolve(),
            camera_device=cam["device"],
            camera_width=int(cam["width"]),
            camera_height=int(cam["height"]),
            camera_fps=int(cam["fps"]),
            max_speed_deg_s=float(raw["arm"].get("max_speed_deg_s", 60.0)),
            joint_offsets_deg={k: float(v) for k, v in (raw["arm"].get("joint_offsets_deg") or {}).items()},
            gripper_inverted=bool(raw["arm"].get("gripper_inverted", False)),
            tip_extension_m=float((raw.get("tool") or {}).get("tip_extension_m", 0.0)),
            joint_limits_deg={k: tuple(v) for k, v in (raw["arm"].get("joint_limits_deg") or {}).items()},
            poses={k: np.asarray(v, dtype=float) for k, v in (raw.get("poses") or {}).items()},
            workspace={ax: (float(raw["workspace"][ax][0]), float(raw["workspace"][ax][1])) for ax in "xyz"},
        )

    def make_arm(self):
        from .hardware import SO101Arm

        return SO101Arm(
            self.port,
            self.calibration_file,
            self.joint_limits_deg or None,
            self.max_speed_deg_s,
            joint_offsets_deg=self.joint_offsets_deg,
            gripper_inverted=self.gripper_inverted,
        )

    def make_camera(self):
        from .hardware import WristCamera

        return WristCamera(self.camera_device, self.camera_width, self.camera_height, self.camera_fps)

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Soft limits shared by the arm and IK: calibrated range minus margin, then config overrides."""
        from .hardware import joint_limits_from_calibration, load_calibration

        cal = load_calibration(self.calibration_file)
        return {**joint_limits_from_calibration(cal, self.joint_offsets_deg), **self.joint_limits_deg}

    def make_kinematics(self):
        from .kinematics import SO101Kinematics

        return SO101Kinematics(URDF_PATH, self.joint_limits(), tip_extension_m=self.tip_extension_m)
