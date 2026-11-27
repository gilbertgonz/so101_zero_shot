"""3D map of the workspace, fused from every photo's depth map during a run (TSDF voxel grid).

Each photo's metric depth (perception/multiview.py) is fused in with the camera pose from the arm's joints:
every 5 mm voxel in front of the robot keeps a weighted average of its signed distance to the nearest
surface along the camera ray (truncated), plus a colour. More views refine it; views that see empty space
where something used to be clear it, so moved objects do not leave ghosts.

Queries: surface point cloud (coloured), height map (top surface per table cell), and whether a spot on
the table is clear. Saved with each run as scene_map.ply (open in MeshLab / CloudCompare / Open3D) and
scene_map_height.png (top view).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VOXEL_M = 0.005
# Workspace box in the base frame (m): in front of the robot, a little below the table to above the objects.
BOUNDS_M = ((0.0, 0.45), (-0.30, 0.30), (-0.02, 0.25))
TRUNC_M = 0.015  # signed distances are clipped to +-this (about 3 voxels)
MAX_WEIGHT = 20.0  # cap, so a voxel can still change when the scene does
NEAR_M = 0.07  # ignore depth closer than this to the camera (fingers, a held object)
SURFACE = 0.4  # |tsdf| below this (in units of TRUNC_M) = on a surface
CLEAR_ABOVE_TABLE_M = 0.012  # a table cell is clear if nothing sticks up more than this
CHANGE_DECAY = 0.1  # a view that strongly disagrees with a voxel keeps only this much of its old weight
CHANGE = 0.8  # "strongly": new and old signed distance differ by more than this (units of TRUNC_M)


class SceneMap:
    def __init__(self, voxel_m: float = VOXEL_M, bounds_m=BOUNDS_M, table_z_m: float = 0.0):
        self.voxel = voxel_m
        self.lo = np.array([b[0] for b in bounds_m], dtype=np.float32)
        self.shape = tuple(round((b[1] - b[0]) / voxel_m) for b in bounds_m)
        self.table_z = table_z_m
        n = int(np.prod(self.shape))
        self.tsdf = np.ones(n, np.float32)
        self.weight = np.zeros(n, np.float32)
        self.color = np.zeros((n, 3), np.float32)  # RGB 0-255
        ix, iy, iz = np.meshgrid(*(np.arange(s) for s in self.shape), indexing="ij")
        self.centers = (np.stack([ix, iy, iz], -1).reshape(-1, 3).astype(np.float32) + 0.5) * voxel_m + self.lo
        self.views = 0

    # ------------------------------------------------------------------ fusion
    def integrate(self, depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, rgb: np.ndarray | None = None,
                  valid: np.ndarray | None = None) -> int:
        """Fuse one depth map (m along the optical axis, NaN = unknown) seen from `T_base_cam` with
        intrinsics K. `valid`: (H, W) bool, pixels to use (e.g. not the robot's own gripper). Returns the
        number of voxels updated."""
        h, w = depth_m.shape
        ok = np.isfinite(depth_m) & (depth_m > NEAR_M)
        if valid is not None:
            ok &= valid
        if not ok.any():
            return 0
        R, t = T_base_cam[:3, :3].astype(np.float32), T_base_cam[:3, 3].astype(np.float32)
        pc = (self.centers - t) @ R  # base -> camera frame (R^T (p - t)), row vectors
        z = pc[:, 2]
        front = z > 0.01
        u = np.full(z.shape, -1, np.int64)
        v = np.full(z.shape, -1, np.int64)
        u[front] = np.round(K[0, 0] * pc[front, 0] / z[front] + K[0, 2]).astype(np.int64)
        v[front] = np.round(K[1, 1] * pc[front, 1] / z[front] + K[1, 2]).astype(np.int64)
        idx = np.flatnonzero(front & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        idx = idx[ok[v[idx], u[idx]]]
        sdf = depth_m[v[idx], u[idx]] - z[idx]
        idx, sdf = idx[sdf > -TRUNC_M], sdf[sdf > -TRUNC_M]  # behind a surface by more than TRUNC: unseen
        if idx.size == 0:
            return 0
        d = np.minimum(sdf / TRUNC_M, 1.0).astype(np.float32)
        w_old = self.weight[idx]
        # Empty where something was, or something where it was empty: the scene changed (an object was moved).
        # Old evidence fades fast, so the map follows within a view or two instead of averaging the new
        # view against every earlier one.
        changed = np.abs(d - self.tsdf[idx]) > CHANGE
        w_old = np.where(changed, w_old * CHANGE_DECAY, w_old)
        w_new = np.minimum(w_old + 1.0, MAX_WEIGHT)
        self.tsdf[idx] = (self.tsdf[idx] * w_old + d) / (w_old + 1.0)
        self.weight[idx] = w_new
        if rgb is not None:
            near = np.abs(d) < 1.0
            ci = idx[near]
            col = rgb[v[ci], u[ci]].astype(np.float32)
            wo = w_old[near][:, None]
            self.color[ci] = (self.color[ci] * wo + col) / (wo + 1.0)
        self.views += 1
        return int(idx.size)

    # ------------------------------------------------------------------ queries
    def _surface(self) -> np.ndarray:
        return (self.weight > 0) & (np.abs(self.tsdf) < SURFACE)

    def _occupied(self) -> np.ndarray:
        """Seen, and on the inside of a surface (negative signed distance)."""
        return (self.weight > 0) & (self.tsdf < 0)

    def points(self, min_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Surface points (N, 3) in the base frame (m) and their colours (N, 3) RGB uint8."""
        sel = self._surface() & (self.weight >= min_weight)
        return self.centers[sel].copy(), np.clip(self.color[sel], 0, 255).astype(np.uint8)

    def height_map(self) -> np.ndarray:
        """(nx, ny) top-surface height (m) per table cell (the highest occupied voxel); NaN where nothing
        was seen."""
        surf = self._occupied().reshape(self.shape)
        z = self.lo[2] + (np.arange(self.shape[2]) + 0.5) * self.voxel
        top = np.where(surf, z[None, None, :], -np.inf).max(axis=2)
        return np.where(np.isfinite(top), top, np.nan).astype(np.float32)

    def cells_in_disk(self, xy: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Grid indices (ix, iy) of table cells whose centre is within radius_m of xy."""
        x = self.lo[0] + (np.arange(self.shape[0]) + 0.5) * self.voxel
        y = self.lo[1] + (np.arange(self.shape[1]) + 0.5) * self.voxel
        X, Y = np.meshgrid(x, y, indexing="ij")
        return np.nonzero((X - xy[0]) ** 2 + (Y - xy[1]) ** 2 <= radius_m ** 2)

    def spot_clear(self, xy: np.ndarray, radius_m: float, ignore: list[tuple[np.ndarray, float]] | None = None
                   ) -> tuple[bool | None, float]:
        """Is the table clear within radius_m of xy? `ignore`: (xy, radius) disks to leave out (e.g. where
        the held object used to stand). Returns (clear, highest point above the table in m); clear is None
        when less than half of the spot was seen."""
        ix, iy = self.cells_in_disk(np.asarray(xy, float), radius_m)
        if ignore:
            x = self.lo[0] + (ix + 0.5) * self.voxel
            y = self.lo[1] + (iy + 0.5) * self.voxel
            keep = np.ones(ix.shape, bool)
            for c, r in ignore:
                keep &= (x - c[0]) ** 2 + (y - c[1]) ** 2 > r ** 2
            ix, iy = ix[keep], iy[keep]
        if ix.size == 0:
            return None, 0.0
        top = self.height_map()[ix, iy]
        seen = np.isfinite(top)
        if seen.mean() < 0.5:
            return None, 0.0
        highest = float(np.nanmax(top) - self.table_z)
        return highest <= CLEAR_ABOVE_TABLE_M, highest

    # ------------------------------------------------------------------ output
    def save(self, out_dir: Path) -> list[Path]:
        """scene_map.ply (coloured surface points) and scene_map_height.png (top view). Returns the paths."""
        out_dir = Path(out_dir)
        pts, cols = self.points()
        ply = out_dir / "scene_map.ply"
        with ply.open("w") as f:
            f.write("ply\nformat ascii 1.0\n"
                    f"comment SceneMap, base frame, m; {self.views} views, voxel {self.voxel * 1000:.0f} mm\n"
                    f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(pts, cols):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        png = out_dir / "scene_map_height.png"
        cv2.imwrite(str(png), self.height_image())
        return [ply, png]

    def height_image(self, scale: int = 3, max_h_m: float = 0.10) -> np.ndarray:
        """Top view, robot at the bottom: height above the table as colour, grey = not seen."""
        hm = self.height_map() - self.table_z
        img = np.full((*hm.shape, 3), 60, np.uint8)
        seen = np.isfinite(hm)
        lvl = (np.clip(np.nan_to_num(hm) / max_h_m, 0, 1) * 255).astype(np.uint8)
        col = cv2.applyColorMap(lvl.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(*hm.shape, 3)
        img[seen] = col[seen]
        # rows = x (far at the top), columns = y (robot's left on the left)
        img = img[::-1, ::-1]
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"height 0-{max_h_m * 100:.0f} cm, grey = unseen ({self.views} views)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img
