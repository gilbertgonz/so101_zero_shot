"""Top-down grasp planning: jaw alignment, reachability, safety checks, re-look geometry."""

import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.grasping.topdown import (
    GraspPlanningError, TopDownPlanner, closing_axis_hand, grasp_height, object_axis_xy,
)
from so101_zero_shot.kinematics.collision import CollisionChecker
from so101_zero_shot.perception.geometry import FisheyeCamera


@pytest.fixture(scope="module")
def planner():
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    cam = FisheyeCamera.load()
    return cfg, kin, cam, TopDownPlanner(kin, CollisionChecker(), cam, cfg.poses["rest"])


def test_closing_axis_is_perpendicular_to_approach(planner):
    _, kin, cam, _ = planner
    c = closing_axis_hand(cam, kin)
    approach = (np.linalg.inv(kin.fk(np.zeros(5), cam.hand_frame)) @ kin.fk(np.zeros(5)))[:3, 2]
    assert abs(c @ approach) < 1e-9 and np.linalg.norm(c) == pytest.approx(1.0)


def test_object_axis():
    rng = np.random.default_rng(0)
    t = rng.uniform(-0.06, 0.06, 500)
    d = np.array([np.cos(0.5), np.sin(0.5)])
    pts = np.c_[0.25 + t * d[0], t * d[1], np.zeros(500)] + np.r_[rng.normal(0, 0.004, (500, 2)).T, [np.zeros(500)]].T
    a = object_axis_xy(pts)
    assert abs(a @ d) > 0.99
    assert object_axis_xy(np.c_[rng.normal(0, 0.02, (500, 2)), np.zeros(500)]) is None  # round blob


@pytest.mark.parametrize("yaw_deg", [0, 45, 90, 135])
def test_jaws_close_across_object(planner, yaw_deg):
    cfg, kin, _, pl = planner
    axis = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
    target = np.array([0.24, 0.02, 0.01])
    plan = pl.plan_grasp(target, axis, cfg.poses["scan"])
    assert plan.misalignment_deg < 6.0
    T = kin.fk(plan.grasp_deg)
    assert np.linalg.norm(T[:3, 3] - target) < 0.004
    assert np.degrees(np.arccos(-T[2, 2])) < 8.0  # pointing down
    for q in (plan.pregrasp_deg, plan.lift_deg):
        assert kin.fk(q)[2, 3] > target[2] + 0.03


def test_unreachable_target_raises(planner):
    cfg, _, _, pl = planner
    with pytest.raises(GraspPlanningError):
        pl.plan_grasp(np.array([0.45, 0.0, 0.01]), None, cfg.poses["scan"])


def test_relook_centres_target(planner):
    cfg, kin, cam, pl = planner
    t = np.array([0.25, 0.0, 0.0])
    q = pl.plan_relook(t, cfg.poses["scan"])
    T = cam.pose_in_base(kin, q)
    d = t - T[:3, 3]
    assert 0.12 <= np.linalg.norm(d) <= 0.22
    assert np.degrees(np.arccos(T[:3, 2] @ d / np.linalg.norm(d))) < 20.0


def test_grasp_height():
    assert grasp_height(0.015) == pytest.approx(0.010)  # marker: floor
    assert grasp_height(0.08) == pytest.approx(0.040)  # can: half height
    assert grasp_height(-0.02) == pytest.approx(0.010)  # bad depth never sends the tips into the table


def test_holding_decision_matches_probe_measurements():
    from so101_zero_shot.grasping.feedback import is_holding

    assert not is_holding(0.23, -28.0)  # closed on nothing (gripper_probe, 2026-11-25)
    assert is_holding(0.85, -72.0)  # closed on the marker
    assert is_holding(20.0, -10.0)  # thick object: stopped well open
