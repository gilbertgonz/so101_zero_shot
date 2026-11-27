import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig

SEED = np.array([0.0, -20.0, 20.0, 60.0, 0.0])


@pytest.fixture(scope="module")
def kin():
    # Uses the real calibrated limits (config/calibration/so101_follower.json), same as on the robot.
    return RobotConfig.load().make_kinematics()


def test_zero_pose_points_forward(kin):
    T = kin.fk(np.zeros(5))
    assert T[0, 3] > 0.3  # stretched out in front of the base
    np.testing.assert_allclose(T[:3, 2], [1, 0, 0], atol=0.01)  # fingers point along +x


def test_ik_roundtrip_free_orientation(kin):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform([-60, -60, -60, -60, -90], [60, 40, 60, 60, 90])
        target = kin.fk(q)[:3, 3]
        res = kin.ik(target, SEED, approach_axis=None)
        assert res.success, res
        np.testing.assert_allclose(kin.fk(res.joints_deg)[:3, 3], target, atol=0.003)


@pytest.mark.parametrize("x,y", [(0.20, 0.0), (0.24, 0.08), (0.16, -0.10), (0.26, 0.0)])
def test_top_down_reachable_on_table(kin, x, y):
    for z in (0.0, 0.04):
        res = kin.ik([x, y, z], SEED)
        assert res.success, (x, y, z, res)
        T = kin.fk(res.joints_deg)
        assert T[2, 2] < -0.99  # fingers point down


def test_high_hover_needs_tilt_but_hits_position(kin):
    # Above ~6 cm tip height (parallel gripper) exact vertical is infeasible; position must still be exact.
    res = kin.ik([0.20, 0.0, 0.12], SEED, axis_tol_deg=30.0)
    assert res.success and res.position_error_m < 1e-3
    assert 5.0 < res.axis_error_deg < 30.0


def test_unreachable_reports_failure(kin):
    res = kin.ik([0.60, 0.0, 0.02], SEED)  # beyond arm length
    assert not res.success and res.position_error_m > 0.05


def test_wrist_roll_pinned(kin):
    res = kin.ik([0.22, 0.0, 0.05], SEED, wrist_roll_deg=30.0)
    assert res.success
    assert abs(res.joints_deg[4] - 30.0) < 1e-6


def test_workspace_check():
    cfg = RobotConfig.load()
    cfg.check_in_workspace(np.array([0.2, 0.0, 0.02]))
    for bad in ([0.5, 0.0, 0.02], [0.2, 0.0, -0.01], [0.2, 0.3, 0.02], [np.nan, 0, 0]):
        with pytest.raises(ValueError):
            cfg.check_in_workspace(np.array(bad))


def test_ik_results_pass_the_arm_limit_check():
    """Regression: plans that reach a joint limit came back 1e-14 deg outside it after the rad -> deg
    conversion (wrist_flex 104.52747252747254 vs 104.52747252747253) and the arm refused to move."""
    from so101_zero_shot.grasping.topdown import GraspPlanningError, TopDownPlanner
    from so101_zero_shot.kinematics.collision import CollisionChecker
    from so101_zero_shot.perception.geometry import FisheyeCamera
    from fake_hardware import make_sim_arm

    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    arm = make_sim_arm(cfg)
    planner = TopDownPlanner(kin, CollisionChecker(), FisheyeCamera.load(), cfg.poses["rest"])
    at_limit = 0
    for x in np.linspace(0.14, 0.30, 5):
        for y in np.linspace(-0.12, 0.12, 4):
            try:
                plan = planner.plan_grasp(np.array([x, y, 0.02]), None, cfg.poses["scan"])
            except GraspPlanningError:
                continue
            for q in (plan.pregrasp_deg, plan.grasp_deg, plan.lift_deg):
                arm.check_limits(q)  # raises if any joint is outside, by any amount
                at_limit += any(abs(v - b) < 1e-9 for v, m in zip(q, arm.joint_limits) for b in arm.joint_limits[m])
    assert at_limit > 0  # the grid does include poses at a limit
