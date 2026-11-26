"""MuJoCo simulation backend: the real SO101Arm on the simulated bus, the rendered camera, physics grasps."""

import time

import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.grasping.feedback import is_holding, read_grip
from so101_zero_shot.grasping.topdown import TopDownPlanner
from so101_zero_shot.kinematics.collision import CollisionChecker
from so101_zero_shot.mujoco_sim import SimWorld, SimWristCamera, make_mujoco_arm
from so101_zero_shot.perception.geometry import FisheyeCamera


@pytest.fixture(scope="module")
def sim():
    cfg = RobotConfig.load()
    cam = FisheyeCamera.load()
    world = SimWorld(cfg, cam)
    world.start()
    arm = make_mujoco_arm(cfg, world)
    yield cfg, cam, world, arm, SimWristCamera(world)
    world.close()


def test_sim_matches_planner_kinematics(sim):
    cfg, cam, world, arm, _ = sim
    kin = cfg.make_kinematics()
    world.reset()
    arm.hold()
    arm.bus.enable_torque()
    try:
        arm.move_to(cfg.poses["scan"], speed_deg_s=90, settle_timeout_s=3.0)
        q = arm.read_joints()
        assert np.max(np.abs(q - cfg.poses["scan"])) < 1.0
        with world._lock:
            g = world.model.body("gripper").id
            pos = world.data.xpos[g].copy()
        assert np.linalg.norm(pos - kin.fk(q, "gripper_link")[:3, 3]) < 1e-3
    finally:
        arm.bus.disable_torque()


def test_camera_renders_calibrated_fisheye(sim):
    cfg, cam, world, arm, camera = sim
    img, _ = camera.read(timeout_s=5.0)
    w, h = cam.image_size
    assert img.shape == (h, w, 3)
    cx, cy = int(cam.K[0, 2]), int(cam.K[1, 2])
    assert img[cy - 50:cy + 50, cx - 50:cx + 50].mean() > 20  # scene in the middle
    assert img[:20, :20].max() == 0  # black outside the rendered field of view
    t = time.monotonic()
    assert camera.read_fresh(0.05) is not None and camera.read(newer_than=t)[1] > t


def test_torque_off_disables_actuators(sim):
    _, _, world, arm, _ = sim
    world.reset()
    assert not arm.bus.torque and arm.bus.read("Present_Load", "elbow_flex") == 0
    arm.bus.enable_torque()
    assert arm.bus.read("Torque_Enable", "shoulder_pan") == 1
    arm.bus.disable_torque()


@pytest.mark.parametrize("name,top_z", [("red_cube", 0.03), ("blue_can", 0.07)])
def test_top_down_grasp_lifts_object(sim, name, top_z):
    """With the true object position the planned top-down grasp holds and lifts it (physics sanity)."""
    cfg, cam, world, arm, _ = sim
    kin = cfg.make_kinematics()
    world.reset()
    xy = np.array(world.object_positions()[name][:2])
    target = np.array([*xy, max(0.01, min(top_z - 0.025, top_z / 2))])
    plan = TopDownPlanner(kin, CollisionChecker(), cam, cfg.poses["rest"]).plan_grasp(target, None, cfg.poses["scan"])
    arm.hold()
    arm.bus.enable_torque()
    try:
        arm.move_to(plan.pregrasp_deg, gripper=100.0, speed_deg_s=90, settle_timeout_s=3.0)
        arm.move_to(plan.grasp_deg, speed_deg_s=40, settle_timeout_s=3.0)
        arm.set_gripper(0.0, duration_s=0.8)
        arm.move_to(plan.lift_deg, speed_deg_s=30, settle_timeout_s=3.0)
        assert is_holding(*read_grip(arm, 0.2))
        assert world.object_positions()[name][2] > top_z / 2 + 0.03
    finally:
        arm.bus.disable_torque()
        world.reset()
