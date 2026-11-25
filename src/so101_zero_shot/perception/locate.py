"""Prompt -> object in 3D: rectify, point (VLM), mask (SAM), metric depth (table-scaled), base-frame points."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ..kinematics import TIP_FRAME, SO101Kinematics
from .depth import DepthFitError, TableFit, fit_table_scale
from .geometry import FisheyeCamera, Rectifier, table_depth, to_base
from .pointer import Pointing

logger = logging.getLogger(__name__)


class PerceptionError(RuntimeError):
    pass


class PointerModel(Protocol):
    def point(self, image_rgb: np.ndarray, query: str) -> Pointing: ...


class SegmenterModel(Protocol):
    def mask(self, image_rgb: np.ndarray, point_px: np.ndarray, box_px: np.ndarray | None = None) -> tuple[np.ndarray, float]: ...


class DepthModel(Protocol):
    def disparity(self, image_rgb: np.ndarray) -> np.ndarray: ...


@dataclass
class ObjectEstimate:
    query: str
    pointing: Pointing
    mask: np.ndarray  # (H, W) bool, rectified image
    mask_score: float
    points_base: np.ndarray  # (N, 3) object surface points, base frame (m)
    center_base: np.ndarray  # (3,) centre of the top surface (points within 1 cm of the top); the grasp target
    top_z_m: float  # robust top height above the table
    fit: TableFit
    T_base_cam: np.ndarray
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
            "depth_fit": {"a": self.fit.a, "b": self.fit.b, "inlier_frac": round(self.fit.inlier_frac, 3),
                          "table_rmse_m": round(self.fit.rmse_m, 4)},
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
        table_region_m: tuple[tuple[float, float], tuple[float, float]] = ((0.05, 0.40), (-0.25, 0.25)),
        gripper_margin_frac: float = 0.08,
    ):
        self.cam, self.kin = cam, kin
        self.pointer, self.segmenter, self.depth = pointer, segmenter, depth
        self.rect = rectifier or Rectifier(cam)
        self.rays = self.rect.rays()
        self.table_z_m = table_z_m
        self.table_region_m = table_region_m
        self.gripper_margin_frac = gripper_margin_frac

    def gripper_row(self, joints_deg: np.ndarray, T_base_cam: np.ndarray) -> int | None:
        """First image row of the gripper's own fingers (they fill the bottom of the wrist camera's view).

        Table pixels below this row are left out of the depth scale fit: the fingers are 5-12 cm from
        the lens, so the depth model spends most of its range on them and the table pixels between and
        beside them fit badly (measured: mug height 3 cm with them, 10.6 cm without).
        """
        tip = np.linalg.inv(T_base_cam) @ self.kin.fk(joints_deg, TIP_FRAME)[:, 3]
        if tip[2] <= 0:
            return None
        v = self.rect.K[1, 1] * tip[1] / tip[2] + self.rect.K[1, 2]
        row = int(v - self.gripper_margin_frac * self.rect.size[1])
        return row if 0 < row < self.rect.size[1] else None

    def locate(self, fisheye_bgr: np.ndarray, joints_deg: np.ndarray, query: str) -> tuple[ObjectEstimate, dict]:
        """Returns the estimate and debug arrays (rectified image, disparity, metric depth).

        Raises PerceptionError if the object is not found or the geometry is unusable.
        """
        rect_bgr = self.rect.rectify(fisheye_bgr)
        rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        T = self.cam.pose_in_base(self.kin, joints_deg)
        debug = {"rect_bgr": rect_bgr}

        pointing = self.pointer.point(rgb, query)
        if not pointing.found:
            raise PerceptionError(f"VLM did not find {query!r} (answer: {pointing.raw!r})")
        mask, score = self.segmenter.mask(rgb, pointing.point_px, pointing.box_px)
        if mask.sum() < 50:
            raise PerceptionError(f"mask for {query!r} is empty ({mask.sum()} px)")
        warnings = []
        px, py = np.round(pointing.point_px).astype(int)
        if not mask[py, px]:
            warnings.append("VLM point is outside the SAM mask")

        disp = self.depth.disparity(rgb)
        debug["disparity"] = disp
        tz = table_depth(self.rays, T, self.table_z_m)
        # Table pixels: rays that hit the table inside a generous region in front of the robot, away from
        # the object (its base and shadow are not table at the predicted depth).
        with np.errstate(invalid="ignore"):
            hit = to_base(self.rays.reshape(-1, 3), np.nan_to_num(tz.reshape(-1), nan=0.0), T).reshape(tz.shape + (3,))
        (x0, x1), (y0, y1) = self.table_region_m
        in_region = np.isfinite(tz) & (hit[..., 0] > x0) & (hit[..., 0] < x1) & (hit[..., 1] > y0) & (hit[..., 1] < y1)
        grow = max(15, int(0.25 * np.sqrt(mask.sum())))
        near_obj = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
        fit_mask = in_region & ~near_obj
        row = self.gripper_row(joints_deg, T)
        if row is not None:
            fit_mask[row:] = False
        debug["gripper_row"] = row
        try:
            fit = fit_table_scale(disp, tz, fit_mask)
        except DepthFitError as e:
            raise PerceptionError(str(e)) from e
        if fit.inlier_frac < 0.5:
            warnings.append(f"depth/table fit used only {fit.inlier_frac:.0%} of table pixels")
        Z = fit.metric(disp)
        debug["depth_m"] = Z

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
            raise PerceptionError("object points all lie below the table; depth fit is wrong")
        top = float(np.percentile(P[:, 2], 90) - self.table_z_m)
        # Centre from the top surface only: the side walls facing the camera would pull a plain median
        # toward the robot (measured on a can: 2.1 cm -> 1.4 cm distance error).
        top_pts = P[P[:, 2] > top + self.table_z_m - 0.01]
        center = np.median(top_pts if len(top_pts) >= 10 else P, axis=0)
        if top < 0.005:
            warnings.append(f"object top only {top * 1000:.0f} mm above the table (flat object or depth error)")
        est = ObjectEstimate(query, pointing, mask, score, P, center, top, fit, T, warnings)
        for w in warnings:
            logger.warning("%s: %s", query, w)
        return est, debug


def draw_debug(est: ObjectEstimate, debug: dict) -> np.ndarray:
    """Rectified image with mask, box, point and the 3D result, next to the metric depth map."""
    img = debug["rect_bgr"].copy()
    overlay = img.copy()
    overlay[est.mask] = (0, 200, 0)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    if est.pointing.box_px is not None:
        x1, y1, x2, y2 = est.pointing.box_px.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.drawMarker(img, tuple(est.pointing.point_px.astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    c = est.center_base * 100
    lines = [f'"{est.query}"', f"centre ({c[0]:.1f}, {c[1]:.1f}) cm  top {est.top_z_m * 100:.1f} cm",
             f"table fit rmse {est.fit.rmse_m * 1000:.1f} mm ({est.fit.inlier_frac:.0%} inliers)"] + est.warnings
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(img, s, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    Z = debug["depth_m"]
    lo, hi = np.nanpercentile(Z, 2), np.nanpercentile(Z, 98)
    zn = np.clip((np.nan_to_num(Z, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_vis = cv2.applyColorMap((255 * (1 - zn)).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(depth_vis, f"depth {lo * 100:.0f}-{hi * 100:.0f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.hstack([img, depth_vis])
