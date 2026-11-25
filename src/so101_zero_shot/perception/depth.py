"""Monocular depth made metric with the table plane.

Depth Anything V2 (relative) predicts affine-invariant inverse depth (disparity) d. Table pixels have a
known metric depth Z_t from the calibrated camera pose (ray / plane intersection), so fit
    1 / Z  ~=  a * d + b
on table pixels and apply it everywhere. The fit is robust (RANSAC) because the table mask
also contains clutter, shadows and the object's base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DepthFitError(RuntimeError):
    pass


@dataclass
class TableFit:
    a: float
    b: float
    n_used: int
    inlier_frac: float
    rmse_m: float  # depth error on the inlier table pixels

    def metric(self, disparity: np.ndarray) -> np.ndarray:
        inv = self.a * disparity + self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = 1.0 / inv
        Z[~(inv > 1e-6)] = np.nan
        return Z


def fit_table_scale(
    disparity: np.ndarray,
    table_Z: np.ndarray,
    mask: np.ndarray,
    rel_tol: float = 0.02,
    max_samples: int = 20000,
    ransac_iters: int = 300,
    seed: int = 0,
) -> TableFit:
    """Fit 1/Z = a*d + b on pixels where `mask` is set and `table_Z` is finite.

    RANSAC on 2-pixel samples; a pixel is an inlier if the fitted depth is within `rel_tol` (relative)
    of the table depth. Then least squares on the inliers, repeated until the inlier set settles.
    """
    sel = mask & np.isfinite(table_Z) & np.isfinite(disparity)
    d, z = disparity[sel].astype(float), table_Z[sel].astype(float)
    if len(d) < 500:
        raise DepthFitError(f"only {len(d)} table pixels to fit depth scale")
    rng = np.random.default_rng(seed)
    if len(d) > max_samples:
        idx = rng.choice(len(d), max_samples, replace=False)
        d, z = d[idx], z[idx]
    y = 1.0 / z
    tol = rel_tol * y  # |1/Z' - 1/Z| ~= (|dZ| / Z) / Z

    def inliers(a: float, b: float) -> np.ndarray:
        return np.abs(a * d + b - y) < tol

    best, best_n = None, -1
    for _ in range(ransac_iters):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (d[i] - d[j])
        if not a > 0:
            continue  # disparity must grow as depth shrinks
        n = int(inliers(a, y[i] - a * d[i]).sum())
        if n > best_n:
            best, best_n = (a, y[i] - a * d[i]), n
    if best is None or best_n < 0.2 * len(d):
        raise DepthFitError(f"no consistent depth scale on the table ({max(best_n, 0)}/{len(d)} pixels agree)")
    keep = inliers(*best)
    for _ in range(5):
        A = np.stack([d[keep], np.ones(keep.sum())], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        new = inliers(a, b)
        if not a > 0 or new.sum() < 0.2 * len(d):
            a, b = best
            break
        if np.array_equal(new, keep):
            break
        keep = new
    fit = TableFit(float(a), float(b), len(d), float(keep.mean()), 0.0)
    fit.rmse_m = float(np.sqrt(np.nanmean((fit.metric(d[keep]) - z[keep]) ** 2)))
    return fit


class DepthAnything:
    """Depth Anything V2 (relative) via transformers; loaded lazily onto the GPU if available."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def disparity(self, image_rgb: np.ndarray) -> np.ndarray:
        """(H, W) relative inverse depth at the input resolution (larger = closer)."""
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            pred = self.model(**inputs).predicted_depth  # (1, h', w')
        pred = self.torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bicubic", align_corners=False)
        return pred[0, 0].float().cpu().numpy()
