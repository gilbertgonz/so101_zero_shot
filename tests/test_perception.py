"""Perception geometry with fake models: a box on the table must come back at the right place and height."""

from pathlib import Path

import numpy as np
import pytest

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.perception.geometry import FisheyeCamera, Rectifier, to_base
from so101_zero_shot.perception.gripper_mask import GRIPPER_VIEW_PATH, GripperMask, load_gripper_view
from so101_zero_shot.perception.locate import Locator, PerceptionError
from so101_zero_shot.perception.pointer import Pointing, parse_pointing

BOX_XY = ((0.24, 0.28), (-0.03, 0.01))
BOX_H = 0.05


def table_depth(rays_cam, T_base_cam, table_z_m=0.0):
    """Optical-axis depth at which each (x, y, 1) ray meets the plane z = table_z_m (NaN: misses)."""
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dz = rays_cam @ R[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (table_z_m - o[2]) / dz
    Z[~(dz < -1e-6) | ~(Z > 0)] = np.nan
    return Z


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
    """Multi-view depth stand-in: the true depth plus noise (metres); records how many views it got."""

    def __init__(self, Z, noise_m=0.0):
        self.Z = Z + np.random.default_rng(0).normal(0, noise_m, Z.shape)
        self.n_views = None

    def depth(self, ref, others):
        self.n_views = len(others)
        return self.Z.copy(), {}


def views(cam, q):
    """One other view (contents irrelevant to FakeDepth), at the same joints."""
    return [(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q)]


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


@pytest.mark.parametrize("noise_m", [0.0, 0.003])
def test_locates_box(scene, noise_m):
    cfg, kin, cam, rect, q, Z, on_box = scene
    depth = FakeDepth(Z, noise_m)
    loc = Locator(cam, kin, FakePointer(on_box), FakeSegmenter(on_box), depth, rectifier=rect)
    img = np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8)
    est, _ = loc.locate(img, q, "box", views=views(cam, q) * 2)
    (x0, x1), (y0, y1) = BOX_XY
    assert est.center_base[:2] == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2], abs=0.004)
    assert est.top_z_m == pytest.approx(BOX_H, abs=0.004 + noise_m)
    assert depth.n_views == est.n_views == 2


def test_no_other_views_is_an_error(scene):
    cfg, kin, cam, rect, q, Z, on_box = scene
    loc = Locator(cam, kin, FakePointer(on_box), FakeSegmenter(on_box), FakeDepth(Z), rectifier=rect)
    with pytest.raises(PerceptionError, match="no other views"):
        loc.locate(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q, "box")


def _gripper(mask):
    return GripperMask({p: mask for p in range(0, 101, GripperMask.STEP_PCT)})


def test_object_on_gripper_rejected(scene):
    cfg, kin, cam, rect, q, Z, on_box = scene
    loc = Locator(cam, kin, FakePointer(on_box), FakeSegmenter(on_box), FakeDepth(Z), rectifier=rect,
                  gripper_mask=_gripper(on_box.copy()))
    with pytest.raises(PerceptionError, match="own gripper"):
        loc.locate(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q, "box", views=views(cam, q))


def test_object_touching_gripper_kept_minus_gripper_pixels(scene):
    """An object beside or partly behind the fingers is still an object; only the finger pixels go."""
    cfg, kin, cam, rect, q, Z, on_box = scene
    v, _ = np.nonzero(on_box)
    g = np.zeros_like(on_box)
    g[v.max() - 3:] = True  # gripper covers the box's bottom few rows and everything below
    loc = Locator(cam, kin, FakePointer(on_box), FakeSegmenter(on_box), FakeDepth(Z), rectifier=rect,
                  gripper_mask=_gripper(g))
    est, _ = loc.locate(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q, "box", views=views(cam, q))
    assert not (est.mask & g).any()
    assert any("gripper" in w for w in est.warnings)
    (x0, x1), (y0, y1) = BOX_XY
    assert est.center_base[:2] == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2], abs=0.006)


def test_rendered_gripper_mask_matches_real_view():
    """Fitted gripper view: covers the fingertips (bottom centre of the straightened view) and the rods,
    not the table beside the fingers where a cap was wrongly rejected (run grasp_20261125_131428)."""
    if not Path(GRIPPER_VIEW_PATH).exists():
        pytest.skip("gripper view not fitted")
    rect = Rectifier(FisheyeCamera.load())
    gm = GripperMask.render(load_gripper_view(), rect.K, rect.size)
    closed, opened = gm.mask(0), gm.mask(100)
    assert closed[450, 480] and closed[700, 480]  # between the fingertips' base / the rods
    assert not closed[474, 217] and not closed[200, 480]  # the cap beside the left finger; the table ahead
    assert 0.05 < closed.mean() < 0.3
    assert (opened != closed).mean() > 0.01  # the clamps move


def test_not_found_raises(scene):
    cfg, kin, cam, rect, q, Z, on_box = scene

    class Blind:
        def point(self, rgb, query):
            return Pointing(None, None, "not here")

    loc = Locator(cam, kin, Blind(), FakeSegmenter(on_box), FakeDepth(Z), rectifier=rect)
    with pytest.raises(PerceptionError):
        loc.locate(np.zeros((cam.image_size[1], cam.image_size[0], 3), np.uint8), q, "unicorn", views=views(cam, q))


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
