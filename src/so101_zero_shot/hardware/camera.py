"""Wrist camera (V4L2 / OpenCV) that always returns a fresh frame."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class WristCamera:
    """Background-grabbing camera so `read()` returns the newest frame, not a stale buffered one.

    The robot is usually still for a moment before a frame is used, but V4L2 keeps a few frames
    buffered; grabbing continuously in a thread avoids acting on an image from before the last move.
    """

    def __init__(self, device: str, width: int = 1600, height: int = 1200, fps: int = 20, fourcc: str = "MJPG"):
        self.device = device
        self.width, self.height, self.fps, self.fourcc = width, height, fps, fourcc
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def connect(self, warmup_s: float = 1.0) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"could not open camera {self.device!r}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError(f"camera {self.device!r} opened but returned no frame")
        if frame.shape[:2] != (self.height, self.width):
            cap.release()
            raise CameraError(
                f"camera {self.device!r} gave {frame.shape[1]}x{frame.shape[0]}, requested {self.width}x{self.height}"
            )
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="wrist-camera", daemon=True)
        self._thread.start()
        time.sleep(warmup_s)  # let auto-exposure settle

    def _loop(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures in (10, 100):
                    logger.error("camera %s: %d consecutive read failures", self.device, failures)
                time.sleep(0.01)
                continue
            failures = 0
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()

    def read(self, newer_than: float | None = None, timeout_s: float = 2.0) -> tuple[np.ndarray, float]:
        """Return (BGR frame, monotonic timestamp). If `newer_than` is given, wait for a frame after it."""
        if not self._running:
            raise CameraError("camera not connected")
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._frame is not None and (newer_than is None or self._stamp > newer_than):
                    return self._frame.copy(), self._stamp
            if time.monotonic() > deadline:
                raise CameraError(f"no new frame from {self.device!r} within {timeout_s}s")
            time.sleep(0.005)

    def read_fresh(self, settle_s: float = 0.15) -> np.ndarray:
        """Frame captured strictly after this call (plus a short settle for motion blur)."""
        t = time.monotonic() + settle_s
        time.sleep(settle_s)
        return self.read(newer_than=t)[0]

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def __enter__(self) -> WristCamera:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
