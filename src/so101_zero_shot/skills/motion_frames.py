"""Camera frames taken while the arm moves, each with the joint angles at the moment it was captured.

The wrist camera streams the whole time, so every move (rest -> scan, scan -> closer look) passes the
object from many viewpoints for free. MotionFrames runs two threads during a move: one samples the joints
(~30 Hz), one keeps frames whenever the camera has moved far enough since the last kept one. Afterwards
each kept frame gets the joints interpolated at its timestamp (minus the camera's latency), which is
what multi-view depth needs as the camera pose.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

MIN_BASELINE_M = 0.015  # another view this close to the photo's camera adds no parallax


@dataclass
class LookSettings:
    """How photos get their other views for multi-view depth (perception: in config/robot.yaml)."""
    frames: int = 6  # frames from the move to a photo, spread along the camera path
    min_step_m: float = 0.01  # keep a frame only once the camera moved this far since the last kept one
    camera_latency_s: float = 0.0  # frame timestamp minus exposure time (0 in sim; measure on the real camera)


@dataclass
class Frame:
    bgr: np.ndarray  # fisheye frame as the camera delivered it
    joints_deg: np.ndarray  # interpolated at the capture time
    stamp: float  # camera timestamp (time.monotonic)


class MotionFrames:
    """Context manager: frames + joints while the body of the `with` moves the arm.

    `camera_center(joints) -> (3,)` lets it keep only frames at least `min_step_m` apart (camera path),
    capped at `max_frames`. `camera_latency_s`: how long before its timestamp a frame was exposed (0 in
    the simulation; on the real camera, measure it). Errors in the threads are logged, never raised: a
    failed collection just yields fewer frames."""

    def __init__(self, arm, camera, camera_center: Callable[[np.ndarray], np.ndarray], min_step_m: float = 0.01,
                 max_frames: int = 30, joint_hz: float = 30.0, camera_latency_s: float = 0.0):
        self.arm, self.camera, self.camera_center = arm, camera, camera_center
        self.min_step_m, self.max_frames = min_step_m, max_frames
        self.joint_period, self.latency = 1.0 / joint_hz, camera_latency_s
        self._stop = threading.Event()
        self._joints: list[tuple[float, np.ndarray]] = []
        self._raw: list[tuple[np.ndarray, float]] = []
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> MotionFrames:
        self._stop.clear()
        self._threads = [threading.Thread(target=self._sample_joints, daemon=True, name="motion-joints"),
                         threading.Thread(target=self._grab_frames, daemon=True, name="motion-frames")]
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    def _sample_joints(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                q = np.asarray(self.arm.read_joints(), dtype=float)
            except Exception as e:  # noqa: BLE001 - a bus hiccup or a stop: keep what we have
                logger.debug("joint sample failed: %s", e)
                time.sleep(self.joint_period)
                continue
            t1 = time.monotonic()
            with self._lock:
                self._joints.append(((t0 + t1) / 2, q))
            time.sleep(max(0.0, self.joint_period - (time.monotonic() - t0)))

    def _latest_joints(self) -> np.ndarray | None:
        with self._lock:
            return self._joints[-1][1] if self._joints else None

    def _grab_frames(self) -> None:
        last_stamp, last_center = time.monotonic(), None
        while not self._stop.is_set() and len(self._raw) < self.max_frames:
            try:
                frame, stamp = self.camera.read(newer_than=last_stamp, timeout_s=0.5)
            except Exception as e:  # noqa: BLE001 - no frame in time: try again until the move ends
                logger.debug("frame grab failed: %s", e)
                continue
            last_stamp = stamp
            q = self._latest_joints()
            if q is None:
                continue
            c = np.asarray(self.camera_center(q), dtype=float)
            if last_center is not None and np.linalg.norm(c - last_center) < self.min_step_m:
                continue
            last_center = c
            self._raw.append((frame, stamp))

    def frames(self) -> list[Frame]:
        """Kept frames with joints interpolated at their exposure time; frames outside the sampled
        joint history (before the first / after the last joint sample) are dropped."""
        with self._lock:
            hist = list(self._joints)
        if len(hist) < 2:
            return []
        ts = np.array([t for t, _ in hist])
        qs = np.stack([q for _, q in hist])
        out = []
        for frame, stamp in self._raw:
            t = stamp - self.latency
            if not ts[0] <= t <= ts[-1]:
                continue
            q = np.array([np.interp(t, ts, qs[:, j]) for j in range(qs.shape[1])])
            out.append(Frame(frame, q, stamp))
        return out


def spread_frames(frames: list[Frame], camera_center: Callable[[np.ndarray], np.ndarray], k: int,
                  exclude_near: np.ndarray | None = None, min_dist_m: float = MIN_BASELINE_M) -> list[Frame]:
    """Up to `k` frames spread evenly along the camera's path, skipping any within `min_dist_m` of
    `exclude_near` (the camera centre of the photo they will be paired with: no baseline, no depth)."""
    cands = [(f, np.asarray(camera_center(f.joints_deg))) for f in frames]
    if exclude_near is not None:
        cands = [(f, c) for f, c in cands if np.linalg.norm(c - exclude_near) >= min_dist_m]
    if len(cands) <= k:
        return [f for f, _ in cands]
    path = np.r_[0.0, np.cumsum([np.linalg.norm(b[1] - a[1]) for a, b in zip(cands, cands[1:])])]
    picks = {int(np.argmin(np.abs(path - s))) for s in np.linspace(0, path[-1], k)}
    return [cands[i][0] for i in sorted(picks)]
