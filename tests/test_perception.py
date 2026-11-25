"""Perception geometry with fake models: a box on the table must come back at the right place and height."""

import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.perception.depth import DepthFitError, fit_table_scale
from so101_zero_shot.perception.geometry import FisheyeCamera, Rectifier, table_depth, to_base
from so101_zero_shot.perception.locate import Locator, PerceptionError
from so101_zero_shot.perception.pointer import Pointing, parse_pointing

BOX_XY = ((0.24, 0.28), (-0.03, 0.01))
BOX_H = 0.05


@pytest.fixture(scope="module")
def scene():
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    cam = FisheyeCamera.load()
    rect = Rectifier(cam)
    q = cfg.poses["scan"]
    T = cam.pose_in_base(kin, q)
    rays = rect.rays()
    Z = table_depth(rays, T)
    Ztop = table_depth(rays, T, BOX_H)
    top = to_base(rays.reshape(-1, 3), np.nan_to_num(Ztop.reshape(-1)), T).reshape(Z.shape + (3,))
    (x0, x1), (y0, y1) = BOX_XY
    on_box = np.isfinite(Ztop) & (top[..., 0] > x0) & (top[..., 0] < x1) & (top[..., 1] > y0) & (top[..., 1] < y1)
    Z = np.where(on_box, Ztop, Z)
    return cfg, kin, cam, rect, q, Z, on_box


class FakeDepth:
    def __init__(self, Z, a=3.0, b=0.7, noise=0.0):
        rng = np.random.default_rng(0)
        self.disp = (1.0 / Z - b) / a + rng.normal(0, noise, Z.shape)  # affine-invariant inverse depth

    def disparity(self, rgb):
        return self.disp


class FakePointer:
    def __init__(self, mask):
        v, u = np.nonzero(mask)
        self.p = Pointing(np.array([u.mean(), v.mean()]), np.array([u.min(), v.min(), u.max(), v.max()], float), "{}")

    def point(self, rgb, query):
        return self.p


class FakeSegmenter:
    def __init__(self, mask):
        self.m = mask

    def mask(self, rgb, point, box=None):
        return self.m, 0.95


def test_scan_pose_sees_whole_workspace(scene):
    cfg, _, cam, rect, q, _, _ = scene
    T = cam.pose_in_base(cfg.make_kinematics(), q)
    for x in cfg.workspace["x"]:
        for y in cfg.workspace["y"]:
            p = np.linalg.inv(T) @ np.array([x, y, 0.0, 1.0])
            u, v = rect.K[:2, :2] @ (p[:2] / p[2]) + rect.K[:2, 2]
            assert p[2] > 0 and 0 <= u < rect.size[0] and 0 <= v < rect.size[1], (x, y)


def test_table_fit_recovers_affine():
    rng = np.random.default_rng(1)
    Z = rng.uniform(0.2, 0.5, (100, 100))
    d = (1 / Z - 0.5) / 2.0
    clutter = np.zeros_like(Z, bool)
    clutter[:20] = True
    d[clutter] += 0.3  # objects/clutter in the "table" region must be rejected
    fit = fit_table_scale(d, Z, np.ones_like(Z, bool))
    assert fit.a == pytest.approx(2.0, rel=1e-3) and fit.b == pytest.approx(0.5, abs=1e-3)
    assert fit.rmse_m < 1e-4
    with pytest.raises(DepthFitError):
        fit_table_scale(d, Z, np.zeros_like(Z, bool))


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_locates_box(scene, noise):
    cfg, kin, cam, rect, q, Z, on_box = scene
    loc = Locator(cam, kin, FakePointer(on_box), FakeSegmenter(on_box), FakeDepth(Z, noise=noise), rectifier=rect)
    img = np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8)
    est, _ = loc.locate(img, q, "box")
    (x0, x1), (y0, y1) = BOX_XY
    assert est.center_base[:2] == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2], abs=0.004)
    assert est.top_z_m == pytest.approx(BOX_H, abs=0.004 + 0.1 * noise)


def test_not_found_raises(scene):
    cfg, kin, cam, rect, q, Z, on_box = scene

    class Blind:
        def point(self, rgb, query):
            return Pointing(None, None, "not here")

    loc = Locator(cam, kin, Blind(), FakeSegmenter(on_box), FakeDepth(Z), rectifier=rect)
    with pytest.raises(PerceptionError):
        loc.locate(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q, "unicorn")


def test_parse_pointing():
    p = parse_pointing('```json\n{"bbox_2d": [100, 200, 300, 400], "point_2d": [200, 300]}\n```', 960, 720)
    assert p.point_px == pytest.approx([192.0, 216.0]) and p.box_px == pytest.approx([96, 144, 288, 288])
    p = parse_pointing('[{"bbox_2d": [300, 400, 100, 200]}]', 1000, 1000)  # swapped corners, no point
    assert p.box_px == pytest.approx([100, 200, 300, 400]) and p.point_px == pytest.approx([200, 300])
    p = parse_pointing('{"bbox_2d": [100, 100, 300, 300], "point_2d": [100, 100]}', 1000, 1000)  # point on a corner
    assert p.point_px == pytest.approx([200, 200])
    assert not parse_pointing('{"bbox_2d": null, "point_2d": null}', 960, 720).found
    assert not parse_pointing("I cannot see it", 960, 720).found
    assert not parse_pointing('{"point_2d": [1, "x"]}', 960, 720).found
