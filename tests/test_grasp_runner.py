"""GraspRunner flow and stop/E-stop safety on the simulated arm (no models, no hardware)."""

import threading

import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.grasping import runner as runner_mod
from so101_zero_shot.grasping.runner import GraspRunner
from so101_zero_shot.grasping.topdown import TopDownPlanner
from so101_zero_shot.kinematics.collision import CollisionChecker
from so101_zero_shot.perception.depth import TableFit
from so101_zero_shot.perception.geometry import FisheyeCamera, Rectifier
from so101_zero_shot.perception.locate import ObjectEstimate, PerceptionError
from so101_zero_shot.perception.pointer import Pointing
from so101_zero_shot.ui.sim import SimCamera, make_sim_arm


class FakeLocator:
    """A marker lying along y at (25, 0) cm, wherever the camera is."""

    def __init__(self, cam, kin, rect, bad_clicks=0):
        self.cam, self.kin, self.rect = cam, kin, rect
        self.bad_clicks = bad_clicks  # first N clicks hit "the gripper"
        self.pointer = self
        self.clicks = 0

    def describe(self, rgb, mask):
        return "black marker"

    def locate_at(self, img, q, point_px, box_px=None, label="", gripper_pct=0.0):
        self.clicks += 1
        if self.clicks <= self.bad_clicks:
            raise PerceptionError("that is the robot's own gripper")
        return self.locate(img, q, label)

    def locate(self, img, q, query, gripper_pct=0.0):
        y = np.linspace(-0.06, 0.06, 200)
        pts = np.c_[np.full(200, 0.25), y, np.full(200, 0.015)]
        h, w = self.rect.size[1], self.rect.size[0]
        est = ObjectEstimate(query, Pointing(np.array([w / 2, h / 2]), None, "{}"), np.zeros((h, w), bool), 0.9, pts,
                             np.array([0.25, 0.0, 0.015]), 0.015, TableFit(1, 0, 1000, 0.9, 0.002),
                             self.cam.pose_in_base(self.kin, q))
        est.mask[h // 2 - 5:h // 2 + 5, w // 2 - 40:w // 2 + 40] = True
        img = np.zeros((h, w, 3), np.uint8)
        return est, {"rect_bgr": img, "depth_m": np.full((h, w), 0.3)}


class Rep:
    def __init__(self, answers=None, on_confirm=None, on_step=None, choices=("grasp",)):
        self.events, self.answers, self.on_confirm, self.on_step = [], dict(answers or {}), on_confirm, on_step
        self.choices, self.picks = list(choices), 0

    def choose(self, kind, title, prompt, options, images=None):
        return self.choices.pop(0) if self.choices else "cancel"

    def pick(self, prompt):
        self.picks += 1
        return np.array([480.0, 400.0])

    def step(self, step_id, status, message="", detail=None, images=None):
        self.events.append((step_id, status))
        if self.on_step:
            self.on_step(step_id, status)

    def log(self, message):
        pass

    def confirm(self, kind, prompt, images=None):
        if self.on_confirm:
            self.on_confirm(kind)
        return self.answers.get(kind, True)


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    cam = FisheyeCamera.load()
    rect = Rectifier(cam)
    planner = TopDownPlanner(kin, CollisionChecker(), cam, cfg.poses["rest"])
    return dict(cfg=cfg, kin=kin, cam_model=cam, rect=rect, planner=planner, locator=FakeLocator(cam, kin, rect))


@pytest.fixture
def setup(parts, tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "render_plan", lambda poses: np.zeros((10, 10, 3), np.uint8))
    import cv2

    img = tmp_path / "scene.jpg"
    cv2.imwrite(str(img), np.zeros((1200, 1600, 3), np.uint8))
    arm = make_sim_arm(parts["cfg"])
    arm.control_hz = 500.0  # fast simulated moves
    return parts, arm, SimCamera(img)


def at_rest(parts, arm):
    return np.max(np.abs(arm.read_joints() - parts["cfg"].poses["rest"])) < 1.0


def test_full_grasp_holds_and_returns(setup):
    parts, arm, cam = setup
    rep = Rep()
    res = GraspRunner(**parts, reporter=rep, speed_deg_s=200).run(arm, cam, "marker")
    assert res.result == "held"
    assert ("lift", "done") in rep.events and ("place", "done") in rep.events and ("return", "done") in rep.events
    assert at_rest(parts, arm) and not arm.bus.torque
    assert arm.read_gripper() < 2.0  # closed again after putting the object back
    assert (res.out_dir / "grasp.json").exists()


def test_declined_grasp_returns_to_rest(setup):
    parts, arm, cam = setup
    res = GraspRunner(**parts, reporter=Rep({"grasp": False}), speed_deg_s=200).run(arm, cam, "marker")
    assert res.result.startswith("aborted") and at_rest(parts, arm) and not arm.bus.torque


def test_stop_during_approval_still_returns_to_rest(setup):
    """Regression: a Stop that ends the run at the approval card must not cut torque mid-air."""
    parts, arm, cam = setup

    def on_confirm(kind):
        if kind == "grasp":
            arm.request_stop()

    res = GraspRunner(**parts, reporter=Rep({"grasp": False}, on_confirm=on_confirm), speed_deg_s=200).run(arm, cam, "marker")
    assert res.result.startswith("aborted") and at_rest(parts, arm)


def test_stop_mid_move_returns_to_rest(setup):
    parts, arm, cam = setup

    def on_step(step_id, status):
        if step_id == "descend" and status == "active":
            threading.Timer(0.02, arm.request_stop).start()

    rep = Rep(on_step=on_step)
    res = GraspRunner(**parts, reporter=rep, speed_deg_s=20).run(arm, cam, "marker")
    assert res.result == "stopped" and ("descend", "stopped") in rep.events
    assert ("return", "done") in rep.events and at_rest(parts, arm)


def test_second_stop_while_returning_cuts_torque_in_place(setup):
    parts, arm, cam = setup
    state = {"n": 0}

    def on_step(step_id, status):
        if status == "active" and step_id in ("descend", "return"):
            threading.Timer(0.02, arm.request_stop).start()

    res = GraspRunner(**parts, reporter=Rep(on_step=on_step), speed_deg_s=20).run(arm, cam, "marker")
    assert res.result == "stopped" and not at_rest(parts, arm) and not arm.bus.torque


def test_estop_never_moves_again(setup):
    parts, arm, cam = setup

    def on_step(step_id, status):
        if step_id == "pregrasp" and status == "active":
            threading.Timer(0.02, arm.emergency_torque_off).start()

    rep = Rep(on_step=on_step)
    res = GraspRunner(**parts, reporter=rep, speed_deg_s=20).run(arm, cam, "marker")
    assert res.result == "estopped" and not arm.bus.torque
    assert ("return", "active") not in rep.events


def test_click_mode_identifies_and_grasps(setup):
    parts, arm, cam = setup
    rep = Rep()
    res = GraspRunner(**parts, reporter=rep, speed_deg_s=200).run(arm, cam, None)
    assert res.result == "held" and res.log["identified_as"] == "black marker"
    assert ("scan", "done") in rep.events and ("identify", "done") in rep.events and at_rest(parts, arm)


def test_click_mode_pick_again_and_bad_click(setup):
    parts, arm, cam = setup
    locator = FakeLocator(parts["cam_model"], parts["kin"], parts["rect"], bad_clicks=1)
    rep = Rep(choices=("retry", "grasp"))
    res = GraspRunner(**{**parts, "locator": locator}, reporter=rep, speed_deg_s=200).run(arm, cam, None)
    # click 1: gripper -> asked again; click 2: "pick again"; click 3: confirmed
    assert res.result == "held" and rep.picks == 3 and ("identify", "failed") in rep.events


def test_click_mode_cancel_returns_to_rest(setup):
    parts, arm, cam = setup
    res = GraspRunner(**parts, reporter=Rep(choices=("cancel",)), speed_deg_s=200).run(arm, cam, None)
    assert res.result.startswith("aborted") and at_rest(parts, arm) and not arm.bus.torque
