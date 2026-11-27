"""Frames during a move: joints interpolated at each frame's exposure time, spacing, spreading."""

import threading
import time

import numpy as np

from so101_zero_shot.skills.motion_frames import Frame, MotionFrames, spread_frames

SPEED = 0.2  # joint 0 moves 0.2 units/s; the "camera centre" is (joint 0, 0, 0)


class LinearArm:
    """Joint 0 = SPEED * (t - t0), read at any time."""

    def __init__(self):
        self.t0 = time.monotonic()

    def read_joints(self):
        return np.array([SPEED * (time.monotonic() - self.t0), 0.0, 0.0, 0.0, 0.0])


class Camera:
    """20 fps; each frame's pixel value encodes its exposure time, stamped `latency` later."""

    def __init__(self, latency):
        self.latency, self.lock = latency, threading.Lock()

    def read(self, newer_than=None, timeout_s=1.0):
        time.sleep(0.05)
        exposed = time.monotonic() - self.latency
        return np.full((2, 2, 3), 0, np.uint8), exposed + self.latency


def center(q):
    return np.array([q[0], 0.0, 0.0])


def test_frames_get_joints_at_exposure_time_and_are_spaced():
    arm, latency = LinearArm(), 0.03
    with MotionFrames(arm, Camera(latency), center, min_step_m=0.02, camera_latency_s=latency) as mf:
        time.sleep(1.0)
    frames = mf.frames()
    assert 5 <= len(frames) <= 12  # ~0.2 m of travel at >= 2 cm spacing
    xs = np.array([f.joints_deg[0] for f in frames])
    expected = SPEED * (np.array([f.stamp for f in frames]) - latency - arm.t0)
    assert np.abs(xs - expected).max() < 0.004  # within ~one joint-sample interval of motion
    assert np.diff(xs).min() > 0.015


def test_no_joint_samples_means_no_frames():
    class DeadArm:
        def read_joints(self):
            raise OSError("bus gone")

    with MotionFrames(DeadArm(), Camera(0.0), center) as mf:
        time.sleep(0.3)
    assert mf.frames() == []


def test_spread_frames_even_along_path_and_away_from_reference():
    frames = [Frame(None, np.array([x, 0, 0, 0, 0.0]), 0.0) for x in np.linspace(0, 0.10, 21)]
    picked = spread_frames(frames, center, 3, exclude_near=np.array([0.10, 0, 0]), min_dist_m=0.02)
    xs = [f.joints_deg[0] for f in picked]
    assert len(xs) == 3 and xs[0] == 0.0 and max(xs) <= 0.08 + 1e-9
    assert 0.03 < xs[1] < 0.05
    assert spread_frames(frames[:2], center, 4) == frames[:2]
