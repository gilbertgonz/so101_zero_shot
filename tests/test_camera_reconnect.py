"""WristCamera keeps delivering frames across a simulated USB disconnect (no hardware)."""

import time

import cv2
import numpy as np

from so101_zero_shot.hardware import camera as camera_mod
from so101_zero_shot.hardware.camera import WristCamera


class FakeCap:
    """Delivers frames, then fails forever after `fail_after` reads (device unplugged)."""

    instances: list["FakeCap"] = []

    def __init__(self, fail_after: int | None):
        self.fail_after, self.reads, self.props = fail_after, 0, {}
        FakeCap.instances.append(self)

    def read(self):
        self.reads += 1
        time.sleep(0.002)
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def release(self):
        pass


def test_reconnects_and_reapplies_exposure(monkeypatch):
    FakeCap.instances.clear()
    plan = iter([20, None])  # first device dies after 20 frames, the reopened one is healthy

    def fake_open(self):
        cap = FakeCap(next(plan))
        self._apply_exposure(cap, self._exposure)
        return cap

    monkeypatch.setattr(WristCamera, "_open", fake_open)
    cam = WristCamera("/dev/fake", 4, 4)
    cam.connect(warmup_s=0.0)
    try:
        cam.set_exposure(200)
        deadline = time.monotonic() + 3.0
        while cam.reconnects == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cam.reconnects == 1
        _, t = cam.read()
        _, t2 = cam.read(newer_than=t, timeout_s=1.0)  # frames keep coming after the reconnect
        assert t2 > t
        new = FakeCap.instances[-1]
        assert new.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1 and new.props[cv2.CAP_PROP_EXPOSURE] == 200.0
    finally:
        cam.disconnect()


def test_reopen_threshold_is_short():
    # ~30 failed reads at ~10 ms each: reconnect starts well within the 8 s read() timeout.
    assert WristCamera.REOPEN_AFTER_FAILURES * 0.01 < 1.0
    assert camera_mod.WristCamera.read.__defaults__[-1] >= 5.0
