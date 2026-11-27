"""Build everything GraspRunner needs from config/robot.yaml (models are loaded here: ~10-15 s)."""

from __future__ import annotations

import yaml

from ..config import CONFIG_DIR, RobotConfig
from ..kinematics.collision import CollisionChecker
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import Locator
from ..skills.motion_frames import LookSettings
from .topdown import TopDownPlanner


def perception_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()).get("perception") or {}


def make_rectifier(cam_model: FisheyeCamera, pcfg: dict | None = None) -> Rectifier:
    pcfg = perception_config() if pcfg is None else pcfg
    return Rectifier(cam_model, tuple(pcfg.get("rectified_size", (960, 720))), float(pcfg.get("rectified_hfov_deg", 100.0)))


def look_settings(pcfg: dict | None = None) -> LookSettings:
    look = ((perception_config() if pcfg is None else pcfg).get("look")) or {}
    d = LookSettings()
    return LookSettings(frames=int(look.get("frames", d.frames)), min_step_m=float(look.get("min_step_m", d.min_step_m)),
                        camera_latency_s=float(look.get("camera_latency_s", d.camera_latency_s)))


def build_locator(cfg: RobotConfig, kin, cam_model: FisheyeCamera, rect: Rectifier, sim: bool = False,
                  pcfg: dict | None = None) -> Locator:
    """VLM pointer, SAM segmenter and MVSAnywhere depth, as config/robot.yaml `perception:` says.

    sim: the MuJoCo camera renders from the hand-eye pose, so mask the gripper as seen from there
    (instead of the pose fitted to the real fingers)."""
    from ..perception.gripper_mask import GripperMask
    from ..perception.multiview import MvsAnywhereDepth
    from ..perception.pointer import VlmPointer
    from ..perception.segment import Segmenter

    pcfg = perception_config() if pcfg is None else pcfg
    return Locator(
        cam_model, kin,
        VlmPointer(pcfg.get("vlm_model", "Qwen/Qwen3-VL-4B-Instruct"), quantization=pcfg.get("vlm_quantization")),
        Segmenter(pcfg.get("sam_model", "facebook/sam2.1-hiera-small")),
        MvsAnywhereDepth(),
        rectifier=rect, table_z_m=float(pcfg.get("table_z_m", 0.0)),
        gripper_mask=(GripperMask.render(cam_model.T_hand_camera, rect.K, rect.size, cfg.gripper_open_gap_m) if sim
                      else GripperMask.load(rect.K, rect.size, cfg.gripper_open_gap_m)),
    )


def build_runner_parts(sim: bool = False) -> dict:
    cfg = RobotConfig.load()
    pcfg = perception_config()
    kin = cfg.make_kinematics()
    cam_model = FisheyeCamera.load()
    rect = make_rectifier(cam_model, pcfg)
    planner = TopDownPlanner(kin, CollisionChecker(), cam_model, cfg.poses["rest"])
    return {"cfg": cfg, "kin": kin, "cam_model": cam_model, "rect": rect, "planner": planner,
            "locator": build_locator(cfg, kin, cam_model, rect, sim, pcfg), "look_settings": look_settings(pcfg)}
