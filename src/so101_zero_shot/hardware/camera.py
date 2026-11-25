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
    """Background-grabbing, self-reconnecting camera so `read()` returns the newest frame, not a stale buffered one.

    The robot is usually still for a moment before a frame is used, but V4L2 keeps a few frames
    buffered; grabbing continuously in a thread avoids acting on an image from before the last move.
    The camera can drop off USB briefly (seen on this setup); the grab thread reopens it automatically,
    so callers only see a short wait in read().
    """

    REOPEN_AFTER_FAILURES = 30

    def __init__(self, device: str, width: int = 1600, height: int = 1200, fps: int = 20, fourcc: str = "MJPG"):
        self.device = device
        self.width, self.height, self.fps, self.fourcc = width, height, fps, fourcc
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()  # guards _frame/_stamp
        self._cap_lock = threading.Lock()  # serialises V4L2 reads and control changes
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._exposure: float | None = None  # None = auto
        self.reconnects = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _open(self) -> cv2.VideoCapture:
        """Open and configure the device, re-applying the requested exposure. Raises CameraError."""
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
        if not self._apply_exposure(cap, self._exposure):
            cap.release()
            raise CameraError(f"camera {self.device!r} rejected exposure={self._exposure}")
        return cap

    @staticmethod
    def _apply_exposure(cap: cv2.VideoCapture, value: float | None) -> bool:
        if value is None:
            return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3))  # V4L2 aperture-priority = auto
        return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) and cap.set(cv2.CAP_PROP_EXPOSURE, float(value)))

    def connect(self, warmup_s: float = 1.0) -> None:
        self._cap = self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="wrist-camera", daemon=True)
        self._thread.start()
        time.sleep(warmup_s)  # let auto-exposure settle

    def _loop(self) -> None:
        """Grab frames forever; if the device drops off USB, keep trying to reopen it."""
        failures = 0
        while self._running:
            with self._cap_lock:
                cap = self._cap
                ok, frame = cap.read() if cap is not None else (False, None)
            if ok:
                failures = 0
                with self._lock:
                    self._frame, self._stamp = frame, time.monotonic()
                continue
            failures += 1
            if failures < self.REOPEN_AFTER_FAILURES:
                time.sleep(0.01)
                continue
            # Persistent failure: the device was most likely disconnected (USB glitch). Reopen.
            logger.warning("camera %s stopped delivering frames; reconnecting", self.device)
            with self._cap_lock:
                if self._cap is not None:
                    self._cap.release()
                self._cap = None
            while self._running:
                try:
                    cap = self._open()
                except CameraError as e:
                    logger.debug("reconnect failed: %s", e)
                    time.sleep(0.5)
                    continue
                with self._cap_lock:
                    self._cap = cap
                self.reconnects += 1
                logger.warning("camera %s reconnected (%d so far)", self.device, self.reconnects)
                break
            failures = 0

    def set_exposure(self, value: float | None) -> None:
        """Manual exposure in V4L2 units (camera-specific scale), or None for auto exposure.

        The setting is remembered and re-applied after a reconnect. Frames already buffered were
        taken with the old setting; use read_fresh() afterwards.
        """
        if not self._running:
            raise CameraError("camera not connected")
        self._exposure = value
        with self._cap_lock:
            ok = self._cap is None or self._apply_exposure(self._cap, value)  # None: applied on reconnect
        if not ok:
            raise CameraError(f"camera {self.device!r} rejected exposure={value}")

    def read(self, newer_than: float | None = None, timeout_s: float = 8.0) -> tuple[np.ndarray, float]:
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

    def wait_until_streaming(self, timeout_s: float) -> bool:
        """True once a new frame arrives within `timeout_s` (the grab thread reconnects by itself)."""
        try:
            self.read(newer_than=time.monotonic(), timeout_s=timeout_s)
            return True
        except CameraError:
            return False

    def read_fresh(self, settle_s: float = 0.15) -> np.ndarray:
        """Frame captured strictly after this call (plus a short settle for motion blur)."""
        t = time.monotonic() + settle_s
        time.sleep(settle_s)
        return self.read(newer_than=t)[0]

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = None

    def __enter__(self) -> WristCamera:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
