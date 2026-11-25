"""Build everything GraspRunner needs from config/robot.yaml (models are loaded here: ~10-15 s)."""

from __future__ import annotations

import yaml

from ..config import CONFIG_DIR, RobotConfig
from ..kinematics.collision import CollisionChecker
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import Locator
from .topdown import TopDownPlanner


def perception_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()).get("perception") or {}


def build_runner_parts() -> dict:
    from ..perception.depth import DepthAnything
    from ..perception.gripper_mask import GripperMask
    from ..perception.pointer import VlmPointer
    from ..perception.segment import Segmenter

    cfg = RobotConfig.load()
    pcfg = perception_config()
    kin = cfg.make_kinematics()
    cam_model = FisheyeCamera.load()
    rect = Rectifier(cam_model, tuple(pcfg.get("rectified_size", (960, 720))), float(pcfg.get("rectified_hfov_deg", 100.0)))
    locator = Locator(
        cam_model, kin,
        VlmPointer(pcfg.get("vlm_model", "Qwen/Qwen3-VL-2B-Instruct")),
        Segmenter(pcfg.get("sam_model", "facebook/sam2.1-hiera-small")),
        DepthAnything(pcfg.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf")),
        rectifier=rect, table_z_m=float(pcfg.get("table_z_m", 0.0)),
        gripper_mask=GripperMask.load(rect.K, rect.size, cfg.gripper_open_gap_m),
    )
    planner = TopDownPlanner(kin, CollisionChecker(), cam_model, cfg.poses["rest"])
    return {"cfg": cfg, "kin": kin, "cam_model": cam_model, "rect": rect, "planner": planner, "locator": locator}
