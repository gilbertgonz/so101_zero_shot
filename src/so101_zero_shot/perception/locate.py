"""Prompt -> object in 3D: rectify, point (VLM), mask (SAM), metric depth, base-frame points.

Metric depth comes from other views of the scene with known camera poses (`views`: the previous photo
and frames the camera passed on its way, see skills.Skills.look and perception.multiview)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ..kinematics import SO101Kinematics
from .geometry import FisheyeCamera, Rectifier, to_base
from .gripper_mask import GripperMask
from .multiview import View
from .pointer import Pointing

logger = logging.getLogger(__name__)


class PerceptionError(RuntimeError):
    pass


MAX_MASK_FRAC = 0.25  # a "mask" bigger than this is the table / background, not a graspable object
# Share of a mask on the robot's own gripper (rendered outline, grown by GRIPPER_GROW_PX for fit error)
# above which the mask is the gripper itself, not an object.
MAX_ON_GRIPPER_FRAC = 0.5
GRIPPER_GROW_PX = 6
# While holding: depth readings within HELD_RADIUS_M of the fingertip's line of sight and less than HELD_DEPTH_M
# beyond the fingertip are the held object (it hangs between and below the fingers), not the scene.
HELD_RADIUS_M, HELD_DEPTH_M = 0.06, 0.08


class PointerModel(Protocol):
    def point(self, image_rgb: np.ndarray, query: str) -> Pointing: ...


class SegmenterModel(Protocol):
    def mask(self, image_rgb: np.ndarray, point_px: np.ndarray, box_px: np.ndarray | None = None) -> tuple[np.ndarray, float]: ...


class DepthModel(Protocol):
    def depth(self, ref: View, others: list[View]) -> tuple[np.ndarray, dict]: ...


OtherViews = list[tuple[np.ndarray, np.ndarray]]  # (fisheye BGR, joints) of each other view of the scene


@dataclass
class ObjectEstimate:
    query: str
    pointing: Pointing
    mask: np.ndarray  # (H, W) bool, rectified image
    mask_score: float
    points_base: np.ndarray  # (N, 3) object surface points, base frame (m)
    center_base: np.ndarray  # (3,) centre of the top surface (points within 1 cm of the top); the grasp target
    top_z_m: float  # robust top height above the table
    T_base_cam: np.ndarray
    n_views: int = 0  # other views the depth came from
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "query": self.query,
            "point_px": None if self.pointing.point_px is None else self.pointing.point_px.round(1).tolist(),
            "box_px": None if self.pointing.box_px is None else self.pointing.box_px.round(1).tolist(),
            "vlm_answer": self.pointing.raw,
            "mask_px": int(self.mask.sum()),
            "mask_score": round(self.mask_score, 3),
            "n_points": len(self.points_base),
            "center_base_m": self.center_base.round(4).tolist(),
            "top_z_m": round(self.top_z_m, 4),
            "depth_views": self.n_views,
            "warnings": self.warnings,
        }


class Locator:
    def __init__(
        self,
        cam: FisheyeCamera,
        kin: SO101Kinematics,
        pointer: PointerModel,
        segmenter: SegmenterModel,
        depth: DepthModel,
        rectifier: Rectifier | None = None,
        table_z_m: float = 0.0,
        gripper_mask: GripperMask | None = None,
    ):
        """`depth`: metric depth from several views (perception.multiview). `gripper_mask`: where the
        robot's own gripper is in the image; without it, clicks/answers on the gripper are not caught."""
        self.cam, self.kin = cam, kin
        self.pointer, self.segmenter, self.depth = pointer, segmenter, depth
        self.rect = rectifier or Rectifier(cam)
        self.rays = self.rect.rays()
        self.table_z_m = table_z_m
        self.gripper_mask = gripper_mask
        self.scene_map = None  # SceneMap: when set, every photo's depth is fused into it (Skills sets one per run)
        # The gripper holds something (Skills.look sets it): its pixels are left out of the scene map, or the
        # held object would be mapped as an obstacle floating in front of the camera.
        self.holding = False
        self._tip_cam: np.ndarray | None = None  # the fingertip point in the camera frame (fixed: both on the wrist)
        if gripper_mask is None:
            logger.warning("Locator without a gripper mask: masks on the robot's own fingers are not rejected")

    def _remove_gripper(self, mask: np.ndarray, gripper_pct: float, debug: dict, warnings: list[str]) -> np.ndarray:
        """Reject a mask that is mostly the robot's own gripper; otherwise drop any gripper pixels from it."""
        if self.gripper_mask is None:
            return mask
        g = self.gripper_mask.mask(gripper_pct)
        if g.shape != mask.shape:
            raise PerceptionError(f"gripper mask is {g.shape}, image mask {mask.shape}: rectifier mismatch")
        k = 2 * GRIPPER_GROW_PX + 1
        g = cv2.dilate(g.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        debug["gripper_mask"] = g
        on = float((mask & g).sum() / max(mask.sum(), 1))
        if on > MAX_ON_GRIPPER_FRAC:
            raise PerceptionError("that is the robot's own gripper, or something mostly hidden behind its "
                                  "fingers; pick something else (or move the object out from under the fingers)")
        if on > 0:
            mask = mask & ~g
            if mask.sum() < 50:
                raise PerceptionError("almost all of that mask is on the gripper")
            warnings.append(f"{on:.0%} of the mask was on the gripper and was dropped")
        return mask

    def locate(
        self, fisheye_bgr: np.ndarray, joints_deg: np.ndarray, query: str, gripper_pct: float = 0.0,
        views: OtherViews | None = None,
    ) -> tuple[ObjectEstimate, dict]:
        """Find `query` with the VLM, then mask and measure it. Returns the estimate and debug arrays
        (rectified image, metric depth). Raises PerceptionError if not found / unusable."""
        rect_bgr = self.rect.rectify(fisheye_bgr)
        rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        pointing = self.pointer.point(rgb, query)
        if not pointing.found:
            raise PerceptionError(f"VLM did not find {query!r} (answer: {pointing.raw!r})")
        return self._measure(rect_bgr, rgb, joints_deg, query, pointing, gripper_pct, views)

    def locate_at(
        self, fisheye_bgr: np.ndarray, joints_deg: np.ndarray, point_px: np.ndarray, box_px: np.ndarray | None = None,
        label: str = "selected object", gripper_pct: float = 0.0, views: OtherViews | None = None,
    ) -> tuple[ObjectEstimate, dict]:
        """Mask and measure the object under `point_px` (rectified-image pixels), e.g. a user's click or
        a previously located object projected into a new view. No VLM involved."""
        rect_bgr = self.rect.rectify(fisheye_bgr)
        rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        w, h = self.rect.size
        point_px = np.asarray(point_px, dtype=float)
        if not (0 <= point_px[0] < w and 0 <= point_px[1] < h):
            raise PerceptionError(f"point {np.round(point_px)} is outside the image")
        return self._measure(rect_bgr, rgb, joints_deg, label, Pointing(point_px, box_px, "point prompt"), gripper_pct,
                             views)

    def _measure(
        self, rect_bgr: np.ndarray, rgb: np.ndarray, joints_deg: np.ndarray, query: str, pointing: Pointing,
        gripper_pct: float = 0.0, views: OtherViews | None = None,
    ) -> tuple[ObjectEstimate, dict]:
        T = self.cam.pose_in_base(self.kin, joints_deg)
        debug = {"rect_bgr": rect_bgr}
        mask, score = self.segmenter.mask(rgb, pointing.point_px, pointing.box_px)
        warnings: list[str] = []
        mask = self._clean_mask(mask, query, pointing, gripper_pct, debug, warnings)
        Z = self._depth(rgb, T, views, debug, gripper_pct)
        est = self._estimate(query, pointing, mask, score, Z, T, len(views or []), warnings)
        return est, debug

    def _depth(self, rgb: np.ndarray, T: np.ndarray, views: OtherViews | None, debug: dict,
               gripper_pct: float = 0.0) -> np.ndarray:
        """Metric depth (m along the optical axis, NaN where unknown) on the rectified grid of this photo,
        from the other views; also fused into scene_map if set. Fills debug["depth_m"]."""
        if not views:
            raise PerceptionError("no other views of the scene to measure depth from (photos need the frames "
                                  "from a move to them, or an earlier photo)")
        others = [View(cv2.cvtColor(self.rect.rectify(img), cv2.COLOR_BGR2RGB), self.rect.K,
                       self.cam.pose_in_base(self.kin, q)) for img, q in views]
        try:
            Z, _ = self.depth.depth(View(rgb, self.rect.K, T), others)
        except ValueError as e:
            raise PerceptionError(f"depth failed: {e}") from e
        debug["depth_m"] = Z
        if self.scene_map is not None:
            self._fuse(Z, rgb, T, gripper_pct)
        return Z

    def _near_held(self, Z: np.ndarray) -> np.ndarray:
        """Pixels whose depth puts them where a held object is: in a zone around the fingertips, which sit at
        a fixed point in the camera frame (camera and fingers are both on the wrist)."""
        if self._tip_cam is None:
            q = np.zeros(5)
            self._tip_cam = (np.linalg.inv(self.cam.pose_in_base(self.kin, q)) @ self.kin.fk(q)[:, 3])[:3]
        P = self.rays * np.nan_to_num(Z, nan=np.inf)[..., None]  # camera frame
        lateral = np.linalg.norm(P[..., :2] - self._tip_cam[:2], axis=-1)
        return (P[..., 2] < self._tip_cam[2] + HELD_DEPTH_M) & (lateral < HELD_RADIUS_M)

    def _fuse(self, Z: np.ndarray, rgb: np.ndarray, T: np.ndarray, gripper_pct: float) -> None:
        """Add this photo's depth to the scene map, leaving out the robot's own gripper and the black border
        of the rectified image."""
        valid = rgb.max(axis=2) > 8
        if self.gripper_mask is not None:
            g = self.gripper_mask.mask(gripper_pct)
            k = 4 * GRIPPER_GROW_PX + 1
            valid &= cv2.dilate(g.astype(np.uint8), np.ones((k, k), np.uint8)) == 0
        if self.holding:
            valid &= ~self._near_held(Z)
        try:
            self.scene_map.integrate(Z, self.rect.K, T, rgb, valid)
        except Exception:  # noqa: BLE001 - the map is extra; measuring must not fail because of it
            logger.exception("scene map update failed")

    def _clean_mask(self, mask: np.ndarray, query: str, pointing: Pointing, gripper_pct: float, debug: dict,
                    warnings: list[str]) -> np.ndarray:
        """Reject empty / background masks and masks on the gripper; drop gripper pixels."""
        if mask.sum() < 50:
            raise PerceptionError(f"mask for {query!r} is empty ({mask.sum()} px)")
        if mask.mean() > MAX_MASK_FRAC:
            raise PerceptionError(f"the mask covers {mask.mean():.0%} of the image: that is the table or background, "
                                  "not an object")
        mask = self._remove_gripper(mask, gripper_pct, debug, warnings)
        px, py = np.round(pointing.point_px).astype(int)
        if not mask[py, px]:
            warnings.append("point is outside the object mask")
        return mask

    def _estimate(self, query: str, pointing: Pointing, mask: np.ndarray, score: float, Z: np.ndarray,
                  T: np.ndarray, n_views: int, warnings: list[str]) -> ObjectEstimate:
        """3D points, top height and top-surface centre of one masked object."""
        # Object points from the mask interior (edges mix object and background depth).
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if core.sum() < 20:
            core = mask
        sel = core & np.isfinite(Z)
        if sel.sum() < 20:
            raise PerceptionError("no valid depth inside the object mask")
        P = to_base(self.rays[sel], Z[sel], T)
        P = P[P[:, 2] > self.table_z_m - 0.02]  # depth noise can push a few points under the table
        if len(P) < 20:
            raise PerceptionError("object points all lie below the table; depth is wrong")
        top = float(np.percentile(P[:, 2], 90) - self.table_z_m)
        # Centre from the top surface only: the side walls facing the camera would pull a plain median
        # toward the robot (measured on a can: 2.1 cm -> 1.4 cm distance error).
        top_pts = P[P[:, 2] > top + self.table_z_m - 0.01]
        center = np.median(top_pts if len(top_pts) >= 10 else P, axis=0)
        if top < 0.005:
            warnings.append(f"object top only {top * 1000:.0f} mm above the table (flat object or depth error)")
        for w in warnings:
            logger.warning("%s: %s", query, w)
        return ObjectEstimate(query, pointing, mask, score, P, center, top, T, n_views, warnings)

    def measure_many(
        self, fisheye_bgr: np.ndarray, joints_deg: np.ndarray, prompts: list[tuple[str, Pointing]],
        gripper_pct: float = 0.0, views: OtherViews | None = None,
    ) -> tuple[list[ObjectEstimate | PerceptionError], dict]:
        """Mask and measure several objects in one photo with a single depth pass. Returns one estimate or
        PerceptionError per prompt, plus debug."""
        rect_bgr = self.rect.rectify(fisheye_bgr)
        rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        T = self.cam.pose_in_base(self.kin, joints_deg)
        debug: dict = {"rect_bgr": rect_bgr, "T_base_cam": T, "gripper_pct": gripper_pct}
        cleaned: list[tuple | PerceptionError] = []
        for label, pointing in prompts:
            try:
                mask, score = self.segmenter.mask(rgb, pointing.point_px, pointing.box_px)
                warnings: list[str] = []
                d: dict = {}
                mask = self._clean_mask(mask, label, pointing, gripper_pct, d, warnings)
                debug.setdefault("gripper_mask", d.get("gripper_mask"))
                cleaned.append((label, pointing, mask, score, warnings))
            except PerceptionError as e:
                cleaned.append(e)
        Z = self._depth(rgb, T, views, debug, gripper_pct)
        out: list[ObjectEstimate | PerceptionError] = []
        for c in cleaned:
            if isinstance(c, PerceptionError):
                out.append(c)
                continue
            label, pointing, mask, score, warnings = c
            try:
                out.append(self._estimate(label, pointing, mask, score, Z, T, len(views or []), warnings))
            except PerceptionError as e:
                out.append(e)
        return out, debug


def draw_debug(est: ObjectEstimate, debug: dict) -> np.ndarray:
    """Rectified image with mask, box, point and the 3D result, next to the metric depth map."""
    img = debug["rect_bgr"].copy()
    overlay = img.copy()
    overlay[est.mask] = (0, 200, 0)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    if est.pointing.box_px is not None:
        x1, y1, x2, y2 = est.pointing.box_px.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if debug.get("gripper_mask") is not None:
        cs, _ = cv2.findContours(debug["gripper_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 160, 0), 2)  # robot's own gripper
    cv2.drawMarker(img, tuple(est.pointing.point_px.astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    c = est.center_base * 100
    lines = [f'"{est.query}"', f"centre ({c[0]:.1f}, {c[1]:.1f}) cm  top {est.top_z_m * 100:.1f} cm",
             f"depth from {est.n_views} other views"] + est.warnings
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(img, s, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    Z = debug["depth_m"]
    lo, hi = np.nanpercentile(Z, 2), np.nanpercentile(Z, 98)
    zn = np.clip((np.nan_to_num(Z, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_vis = cv2.applyColorMap((255 * (1 - zn)).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(depth_vis, f"depth {lo * 100:.0f}-{hi * 100:.0f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.hstack([img, depth_vis])
