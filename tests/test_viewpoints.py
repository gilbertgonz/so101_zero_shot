"""Every planned calibration viewpoint must satisfy the safety properties the capture relies on."""

import numpy as np
import pytest

from so101_zero_shot.calibration.viewpoints import (
    MIN_GRIPPER_RADIUS_M,
    gripper_radius,
    min_link_height,
    order_and_check,
    path_is_safe,
    plan_viewpoints,
)
from so101_zero_shot.config import RobotConfig
from so101_zero_shot.kinematics.collision import CollisionChecker


@pytest.fixture(scope="module")
def setup():
    cfg = RobotConfig.load()
    kin, cc = cfg.make_kinematics(), CollisionChecker()
    scan = cfg.poses["scan"]
    views = plan_viewpoints(kin, cc, np.array([0.22, 0.0, 0.008]), scan, n_views=10, n_candidates=2500)
    return cfg, kin, cc, scan, order_and_check(kin, cc, scan, views)


def test_views_are_safe(setup):
    cfg, kin, cc, scan, views = setup
    assert len(views) >= 9
    prev = scan
    for v in views:
        assert not cc.in_collision(v.joints_deg)
        assert min_link_height(kin, v.joints_deg) >= 0.04
        assert gripper_radius(kin, v.joints_deg) >= MIN_GRIPPER_RADIUS_M - 1e-9
        assert path_is_safe(kin, cc, prev, v.joints_deg, 0.03)
        assert path_is_safe(kin, cc, v.joints_deg, scan, 0.02)  # abort path
        prev = v.joints_deg


def test_views_within_soft_limits(setup):
    cfg, _, _, _, views = setup
    limits = cfg.joint_limits()
    for v in views:
        for (lo, hi), q in zip(limits.values(), v.joints_deg):
            assert lo <= q <= hi


def test_collision_checker_known_poses():
    cfg = RobotConfig.load()
    cc = CollisionChecker()
    assert not cc.in_collision(cfg.poses["scan"]) and not cc.in_collision(cfg.poses["rest"])
    assert cc.in_collision(np.array([0.0, 30.0, 40.0, 60.0, 0.0]))  # gripper through the table
    assert cc.in_collision(np.array([0.0, -20.0, 96.0, 100.0, 0.0]))  # folded into the shoulder


def test_views_stay_in_compact_envelope(setup):
    from so101_zero_shot.calibration.viewpoints import Envelope

    _, kin, _, _, views = setup
    env = Envelope()
    for v in views:
        assert abs(v.joints_deg[0]) <= env.max_pan_deg  # no swinging out toward desk objects
        assert v.aim_offset_deg <= env.max_aim_deg
        assert env.y_m[0] - 1e-6 <= v.tip_m[1] <= env.y_m[1] + 1e-6 and v.tip_m[2] <= env.z_m[1] + 1e-6
