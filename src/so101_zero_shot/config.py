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


def _tip_xyz(v) -> tuple[float, float, float] | None:
    if v is None:
        return None
    if len(v) != 3:
        raise ValueError(f"tool.tip_xyz_m must have 3 values, got {v}")
    return tuple(float(x) for x in v)


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = True
    fps: float = 10.0
    image_width: int = 800  # frames are stored resized to this width (aspect kept); raw is 1600x1200
    jpeg_quality: int = 85

    def __post_init__(self):
        if not 0.5 <= self.fps <= 30:
            raise ValueError(f"recording.fps must be in [0.5, 30], got {self.fps}")
        if not 64 <= self.image_width <= 4096:
            raise ValueError(f"recording.image_width must be in [64, 4096], got {self.image_width}")
        if not 10 <= self.jpeg_quality <= 100:
            raise ValueError(f"recording.jpeg_quality must be in [10, 100], got {self.jpeg_quality}")

    @classmethod
    def from_dict(cls, d: dict | None) -> RecordingConfig:
        d = dict(d or {})
        unknown = set(d) - {"enabled", "fps", "image_width", "jpeg_quality"}
        if unknown:
            raise ValueError(f"unknown recording settings {sorted(unknown)}")
        casts = {"enabled": bool, "fps": float, "image_width": int, "jpeg_quality": int}
        return cls(**{k: casts[k](v) for k, v in d.items()})


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
    tip_xyz_m: tuple[float, float, float] | None = None
    gripper_open_gap_m: float = 0.046
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    poses: dict[str, np.ndarray] = field(default_factory=dict)
    workspace: dict[str, tuple[float, float]] = field(default_factory=dict)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

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
            tip_xyz_m=_tip_xyz((raw.get("tool") or {}).get("tip_xyz_m")),
            gripper_open_gap_m=float((raw.get("tool") or {}).get("gripper_open_gap_m", 0.046)),
            joint_limits_deg={k: tuple(v) for k, v in (raw["arm"].get("joint_limits_deg") or {}).items()},
            poses={k: np.asarray(v, dtype=float) for k, v in (raw.get("poses") or {}).items()},
            workspace={ax: (float(raw["workspace"][ax][0]), float(raw["workspace"][ax][1])) for ax in "xyz"},
            recording=RecordingConfig.from_dict(raw.get("recording")),
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

        return SO101Kinematics(URDF_PATH, self.joint_limits(), tip_xyz_m=self.tip_xyz_m)
