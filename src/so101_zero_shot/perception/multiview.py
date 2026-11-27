"""Metric depth from several wrist-camera views taken at known arm poses.

The camera moves with the arm, so every photo comes with other views of the same scene for free: the
previous photo and the frames the camera passed on its way (skills.motion_frames). With the camera poses
from forward kinematics, MVSAnywhere (zero-shot multi-view stereo) turns them into metric depth on the
reference photo's rectified-pinhole pixel grid (the grid perception.locate works on).
scripts/eval_depth.py measures it against rendered depth in simulation.

MVSAnywhere lives outside this repo (see README "Depth"): a clone of github.com/nianticlabs/mvsanywhere at
MVSANYWHERE_DIR (default ~/data/third_party/mvsanywhere) with weights/mvsanywhere_hero.ckpt.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MVSA_DIR = Path.home() / "data" / "third_party" / "mvsanywhere"


@dataclass
class View:
    """A rectified pinhole image and where its camera was."""
    rgb: np.ndarray  # (H, W, 3) uint8, rectified pinhole (perception.geometry.Rectifier)
    K: np.ndarray  # (3, 3)
    T_base_cam: np.ndarray  # (4, 4), OpenCV camera convention (x right, y down, z forward)


class MvsAnywhereDepth:
    """MVSAnywhere (Niantic, CVPR 2025): zero-shot multi-view stereo with known poses -> metric depth of
    the reference view. Takes any number of other views at any poses (no rectification), so the photos
    and the frames the camera passes on the way to a pose can all be used. Non-commercial license.

    Needs a clone of github.com/nianticlabs/mvsanywhere (MVSANYWHERE_DIR) with weights/mvsanywhere_hero.ckpt."""

    def __init__(self, repo: Path | None = None, size: tuple[int, int] = (480, 640), refinement_steps: int = 1,
                 depth_range_m: tuple[float, float] | None = (0.05, 0.8), device: str = "cuda"):
        """`size`: (H, W) the model runs at. `depth_range_m`: where the scene can be (None: the model
        estimates it)."""
        import importlib
        import types

        import torch

        repo = Path(repo or os.environ.get("MVSANYWHERE_DIR", DEFAULT_MVSA_DIR))
        ckpt = repo / "weights" / "mvsanywhere_hero.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing: clone nianticlabs/mvsanywhere and download its weights")
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))
        # Visualisation / meshing modules imported at package level, never called for inference.
        for name in ("moviepy", "moviepy.editor", "open3d", "pytorch3d", "pytorch3d.renderer", "pytorch3d.utils",
                     "pytorch3d.structures"):
            try:
                importlib.import_module(name)
            except ImportError:
                sys.modules.setdefault(name, types.ModuleType(name))
        import mvsanywhere.options as options
        from mvsanywhere.utils.model_utils import get_model_class, load_model_inference

        # The Lightning checkpoint pickles its config dataclasses; allow exactly those (weights_only load).
        torch.serialization.add_safe_globals([options.Options, options.DataOptions])
        handler = options.OptionsHandler()
        handler.parse_and_merge_options(config_filepaths=str(repo / "configs" / "models" / "mvsanywhere_model.yaml"),
                                       ignore_cl_args=True)  # it would parse our sys.argv
        opts = handler.options
        opts.load_weights_from_checkpoint = str(ckpt)
        self.model = load_model_inference(opts, get_model_class(opts)).to(device).eval()
        self.torch, self.device, self.size = torch, device, size
        self.refinement_steps, self.depth_range_m = refinement_steps, depth_range_m
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    def _image(self, rgb: np.ndarray):
        t = self.torch
        x = t.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        x = t.nn.functional.interpolate(x[None], size=self.size, mode="bilinear", align_corners=False)[0]
        return (x - self.mean) / self.std

    def _K(self, K: np.ndarray, w: int, h: int) -> np.ndarray:
        K4 = np.eye(4)
        K4[:3, :3] = K
        K4[0] *= self.size[1] / w
        K4[1] *= self.size[0] / h
        return K4

    def depth(self, ref: View, others: list[View]) -> tuple[np.ndarray, dict]:
        if not others:
            raise ValueError("MVSAnywhere needs at least one other view")
        t = self.torch
        h, w = ref.rgb.shape[:2]
        T0inv = np.linalg.inv(ref.T_base_cam)  # reference camera = world origin (keeps numbers small)

        def f32(a):
            return t.as_tensor(np.asarray(a), dtype=t.float32, device=self.device)

        cur_img = self._image(ref.rgb)[None]
        src_img = t.stack([self._image(v.rgb) for v in others])[None]
        cur_pose = f32(np.eye(4))[None]
        src_pose = f32(np.stack([T0inv @ v.T_base_cam for v in others]))[None]
        cur_K = f32(self._K(ref.K, w, h))[None]
        src_K = f32(np.stack([self._K(v.K, *v.rgb.shape[1::-1]) for v in others]))[None]
        cur_Km, src_Km = cur_K.clone(), src_K.clone()
        cur_Km[:, :2] /= 4
        src_Km[:, :, :2] /= 4
        cur = {"image_b3hw": cur_img, "cam_T_world_b44": t.linalg.inv(cur_pose), "world_T_cam_b44": cur_pose,
               "K_matching_b44": cur_Km, "invK_matching_b44": t.linalg.inv(cur_Km)}
        src = {"image_b3hw": src_img, "cam_T_world_b44": t.linalg.inv(src_pose), "world_T_cam_b44": src_pose,
               "K_matching_b44": src_Km, "invK_matching_b44": t.linalg.inv(src_Km)}
        if self.depth_range_m is not None:
            cur["min_depth"] = f32([self.depth_range_m[0]])
            cur["max_depth"] = f32([self.depth_range_m[1]])
        with t.inference_mode():
            out = self.model(phase="test", cur_data=cur, src_data=src, return_mask=True,
                             num_refinement_steps=self.refinement_steps,
                             estimate_depth_range=self.depth_range_m is None)
        z = out["depth_pred_s0_b1hw"] if "depth_pred_s0_b1hw" in out else out["depth"]
        z = t.nn.functional.interpolate(z.float().reshape(1, 1, *z.shape[-2:]), size=(h, w), mode="bilinear",
                                        align_corners=False)[0, 0].cpu().numpy()
        z[~(z > 0)] = np.nan
        return z, {"n_views": len(others)}
