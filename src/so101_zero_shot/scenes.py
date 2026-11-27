"""Saved scenes (runs/scene_*): one photo plus the other views its depth is measured with.

Layout: image.jpg (raw fisheye photo), scene.json (joints, camera pose, ...; "views": file + joints of
each other view), views/<i>.jpg (frames taken while the arm moved to the photo pose).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .perception.locate import OtherViews


class SceneError(RuntimeError):
    pass


def save_scene(out: Path, img: np.ndarray, joints_deg: np.ndarray, views: OtherViews, meta: dict) -> None:
    (out / "views").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "image.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    vmeta = []
    for i, (v_img, v_q) in enumerate(views):
        f = f"views/{i:02d}.jpg"
        cv2.imwrite(str(out / f), v_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        vmeta.append({"file": f, "joints_deg": np.asarray(v_q).tolist()})
    (out / "scene.json").write_text(json.dumps({**meta, "joints_deg": np.asarray(joints_deg).tolist(),
                                                "views": vmeta}, indent=1))


def load_scene(d: Path) -> tuple[np.ndarray, np.ndarray, OtherViews, dict]:
    """(fisheye BGR, joints, other views, scene.json). Raises SceneError if anything is missing."""
    try:
        meta = json.loads((d / "scene.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SceneError(f"cannot read {d / 'scene.json'}: {e}") from e
    img = cv2.imread(str(d / "image.jpg"))
    if img is None:
        raise SceneError(f"cannot read {d / 'image.jpg'}")
    if not meta.get("views"):
        raise SceneError(f"{d.name} has no other views for depth (captured before multi-view depth); "
                         "capture it again with scripts/capture_scene.py")
    views = []
    for v in meta["views"]:
        v_img = cv2.imread(str(d / v["file"]))
        if v_img is None:
            raise SceneError(f"cannot read {d / v['file']}")
        views.append((v_img, np.asarray(v["joints_deg"], dtype=float)))
    return img, np.asarray(meta["joints_deg"], dtype=float), views, meta
