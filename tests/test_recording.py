"""Episode recording on the simulated arm and camera (no hardware)."""

import json
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from so101_zero_shot.config import RecordingConfig, RobotConfig
from so101_zero_shot.perception.geometry import FisheyeCamera
from so101_zero_shot.recording import (EPISODE_COLUMNS, IMAGE_KEY, EpisodeRecorder, decode_image, effective_success,
                                       load_episode, outcome_of)
from so101_zero_shot.ui.sim import SimCamera, make_sim_arm


@pytest.fixture
def rig(tmp_path):
    img = tmp_path / "scene.jpg"
    cv2.imwrite(str(img), np.full((1200, 1600, 3), 90, np.uint8))
    arm = make_sim_arm(RobotConfig.load())
    arm.control_hz = 500.0
    arm.bus.torque = True
    return arm, SimCamera(img), tmp_path


def record(arm, cam, out, seconds=0.6, during=None, **cfg):
    rec = EpisodeRecorder(out, arm, cam, RecordingConfig(**{"fps": 20.0, **cfg}))
    rec.start()
    rec.set_phase("home")
    if during:
        during(rec)
    time.sleep(seconds)
    rec.stop()
    return rec


def test_episode_rows_images_and_commands(rig):
    arm, cam, out = rig

    def move(rec):
        time.sleep(0.2)
        rec.set_phase("scan")
        rec.event("step", step="scan", status="active")
        arm.move_to(np.array([0.0, -50.0, 10.0, 85.0, 0.0]), gripper=50.0, speed_deg_s=500)

    rec = record(arm, cam, out, during=move)
    meta_path = rec.finish({"task": {"mode": "click"}, "outcome": outcome_of("held", rec.events)},
                           cam_model=FisheyeCamera.load())
    meta, table = load_episode(out)
    assert table.column_names == EPISODE_COLUMNS and table.num_rows == rec.num_frames >= 8
    ts = table["timestamp"].to_pylist()
    assert all(b > a for a, b in zip(ts, ts[1:]))
    img = decode_image(table[IMAGE_KEY][0].as_py())
    assert img.shape == (600, 800, 3)
    state = np.array(table["observation.state"].to_pylist())
    action = np.array(table["action"].to_pylist())
    assert state.shape == (table.num_rows, 6) and table["state_ok"].to_pylist() == [True] * table.num_rows
    assert np.isnan(action[0]).all() and np.allclose(action[-1], [0, -50, 10, 85, 0, 50], atol=0.2)
    assert np.allclose(state[-1, :5], [0, -50, 10, 85, 0], atol=0.2)
    assert {"home", "scan"} <= set(table["phase"].to_pylist())
    assert meta["episode_id"] == out.name and meta["num_frames"] == table.num_rows
    assert meta["camera"]["stored_size"] == [800, 600] and abs(meta["camera"]["K_stored"][0][0] - 242.2) < 0.5
    assert json.loads(pq.read_schema(out / "episode.parquet").metadata[b"huggingface"])["info"]["features"][IMAGE_KEY]
    assert meta_path.name == "episode.json"


def test_bus_failure_keeps_recording(rig):
    arm, cam, out = rig
    real = arm.bus.sync_read
    calls = {"n": 0}

    def flaky(register, motors=None, num_retry=2):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("no status packet")
        return real(register, motors, num_retry)

    arm.bus.sync_read = flaky
    rec = record(arm, cam, out)
    table = pq.read_table(out / "episode.parquet")
    ok = table["state_ok"].to_pylist()
    assert rec.state_errors > 0 and not all(ok) and any(ok) and table.num_rows == rec.num_frames
    bad = np.array(table["observation.state"].to_pylist())[[not v for v in ok]]
    assert np.isnan(bad).all()


def test_outcome_parsing():
    close = [{"kind": "step", "step": "close", "status": "active"}]
    assert outcome_of("held", close) == {"result": "held", "status": "held", "success": True,
                                         "grasp_attempted": True, "failed_step": None, "reason": None}
    o = outcome_of("aborted: cannot grasp: target outside workspace", [
        {"kind": "step", "step": "plan", "status": "active"}, {"kind": "step", "step": "plan", "status": "failed"}])
    assert o["status"] == "aborted" and o["success"] is None and o["failed_step"] == "plan"
    assert o["reason"] == "cannot grasp: target outside workspace" and not o["grasp_attempted"]
    assert outcome_of("missed", close)["success"] is False
    meta = {"outcome": {"success": False}, "label": {"success": True}}
    assert effective_success(meta) is True and effective_success({"outcome": {"success": False}}) is False


def test_config_validation():
    with pytest.raises(ValueError):
        RecordingConfig(fps=100)
    with pytest.raises(ValueError):
        RecordingConfig.from_dict({"fps": 10, "colour": True})
    assert RecordingConfig.from_dict({"fps": 5}).fps == 5.0
