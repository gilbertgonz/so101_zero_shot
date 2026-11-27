"""Saved scenes keep the frames their depth is measured with."""

import json

import numpy as np
import pytest

from so101_zero_shot.scenes import SceneError, load_scene, save_scene


def test_round_trip(tmp_path):
    img = np.full((12, 16, 3), 80, np.uint8)
    views = [(np.full((12, 16, 3), 10 * i, np.uint8), np.arange(5.0) + i) for i in range(3)]
    save_scene(tmp_path, img, np.arange(5.0), views, {"pose": "scan"})
    img2, q, views2, meta = load_scene(tmp_path)
    assert img2.shape == img.shape and meta["pose"] == "scan" and np.allclose(q, np.arange(5.0))
    assert len(views2) == 3 and all(np.allclose(v[1], np.arange(5.0) + i) for i, v in enumerate(views2))


def test_old_scene_without_views_is_rejected(tmp_path):
    save_scene(tmp_path, np.zeros((4, 4, 3), np.uint8), np.zeros(5), [], {})
    meta = json.loads((tmp_path / "scene.json").read_text())
    del meta["views"]
    (tmp_path / "scene.json").write_text(json.dumps(meta))
    with pytest.raises(SceneError, match="no other views"):
        load_scene(tmp_path)
