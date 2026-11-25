"""Control flow of scripts/capture_calibration.py with a fake arm and camera (no hardware)."""

import importlib.util
import json
import sys

import numpy as np
import pytest

from so101_zero_shot.calibration.viewpoints import Viewpoint
from so101_zero_shot.config import REPO_ROOT, RobotConfig
from so101_zero_shot.hardware import CameraError

spec = importlib.util.spec_from_file_location("capture_calibration", REPO_ROOT / "scripts" / "capture_calibration.py")
cap_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap_mod)

VIEWS = [Viewpoint(np.array([float(i), -40.0, 30.0, 70.0, 0.0]), np.zeros(3), 10.0) for i in range(4)]


class FakeArm:
    def __init__(self):
        self.log, self.torque = [], False
        self.bus = self

    def connect(self, enable_torque):
        pass

    def sync_read(self, *_):
        return {}

    def sync_write(self, *_):
        pass

    def enable_torque(self):
        self.torque = True

    def move_to(self, q, **_):
        self.log.append(("move", tuple(np.round(q, 1))))
        return np.zeros(5)

    def read_joints(self):
        return np.zeros(5)

    def hold(self):
        pass

    def disconnect(self, disable_torque):
        self.log.append(("torque_off",))


class FakeCam:
    def __init__(self, fail_on_views=(), fail_times=1, crash_on_view=-1):
        self.fail = {v: fail_times for v in fail_on_views}
        self.crash_on_view, self.reconnects, self.current = crash_on_view, 0, None
        self.exposures: list = []

    def connect(self):
        pass

    def set_exposure(self, e):
        self.exposures.append(e)
        if self.current in self.fail and self.fail[self.current] > 0:
            self.fail[self.current] -= 1
            raise CameraError("device gone")
        if self.current == self.crash_on_view:
            raise RuntimeError("unexpected bug")

    def read_fresh(self, settle_s=0.0):
        return np.zeros((8, 8, 3), np.uint8)

    def wait_until_streaming(self, timeout_s):
        self.reconnects += 1
        return True

    def disconnect(self):
        pass


@pytest.fixture
def harness(tmp_path, monkeypatch):
    def run(cam, argv=()):
        arm = FakeArm()
        orig_move = arm.move_to

        def move(q, **kw):  # tell the fake camera which view we are at
            q = np.asarray(q)
            cam.current = next((i for i, v in enumerate(VIEWS) if np.allclose(v.joints_deg, q)), None)
            return orig_move(q, **kw)

        arm.move_to = move
        monkeypatch.setattr(cap_mod, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(cap_mod, "plan", lambda *a, **k: list(VIEWS))
        monkeypatch.setattr(cap_mod, "montage", lambda views: np.zeros((4, 4, 3), np.uint8))
        monkeypatch.setattr(RobotConfig, "make_arm", lambda self: arm)
        monkeypatch.setattr(RobotConfig, "make_camera", lambda self: cam)
        # Like the real phone screen: only exposure 25 reads well (auto is blown out).
        good, poor = type("D", (), {"count": 15})(), type("D", (), {"count": 3})()
        monkeypatch.setattr(
            cap_mod.CharucoTarget, "detect", lambda self, img: good if cam.exposures and cam.exposures[-1] == 25 else poor
        )
        monkeypatch.setattr("builtins.input", lambda *_: "y")
        monkeypatch.setattr(sys, "argv", ["capture", "--yes", *argv])
        try:
            cap_mod.main()
            exc = None
        except SystemExit as e:
            exc = e
        session = sorted(tmp_path.glob("calib_*"))[-1]
        return arm, json.loads((session / "session.json").read_text()), session, exc

    return run


def test_disconnect_recovers_and_retries_same_view(harness):
    arm, meta, _, exc = harness(FakeCam(fail_on_views=[2], fail_times=1))
    assert exc is None
    assert sorted(f["view_index"] for f in meta["frames"]) == [0, 1, 2, 3]
    assert [d["view"] for d in meta["disconnects"]] == [2]
    assert meta["skipped_views"] == []


def test_view_skipped_after_two_disconnects(harness):
    arm, meta, _, exc = harness(FakeCam(fail_on_views=[1], fail_times=5))
    assert sorted(f["view_index"] for f in meta["frames"]) == [0, 2, 3]
    assert meta["skipped_views"] == [1]


def test_unexpected_error_returns_to_rest_before_torque_off(harness):
    cfg = RobotConfig.load()
    arm, meta, _, exc = harness(FakeCam(crash_on_view=1))
    assert exc is not None  # error is still reported
    moves = [e for e in arm.log if e[0] == "move"]
    assert moves[-1][1] == tuple(np.round(cfg.poses["rest"], 1))  # went home...
    assert arm.log[-1] == ("torque_off",)  # ...before torque off
    assert [f["view_index"] for f in meta["frames"]] == [0]


def test_resume_continues_where_it_stopped(harness):
    _, meta, session, _ = harness(FakeCam(crash_on_view=2))
    assert [f["view_index"] for f in meta["frames"]] == [0, 1]
    _, meta2, session2, exc = harness(FakeCam(), argv=["--resume", session.name])
    assert exc is None and session2 == session
    assert [f["view_index"] for f in meta2["frames"]] == [0, 1, 2, 3]


def test_resume_reuses_last_good_exposure_without_sweep(harness):
    _, meta, session, _ = harness(FakeCam(crash_on_view=2))
    assert meta["frames"][-1]["exposure"] == 25  # first sweep value that gives 15 corners with the fake detector
    cam = FakeCam()
    _, meta2, _, exc = harness(cam, argv=["--resume", session.name])
    assert exc is None
    assert cam.exposures[0] == 25  # scan check used the last successful setting...
    assert None not in cam.exposures[:-1]  # ...and never ran the full sweep (None = auto is only restored at exit)
    assert all(e == 25 for e in cam.exposures[:-1])
