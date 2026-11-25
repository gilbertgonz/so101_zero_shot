"""Object mask from a point (and box) prompt with SAM 2.1 via transformers."""

from __future__ import annotations

import numpy as np


class Segmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    def mask(self, image_rgb: np.ndarray, point_px: np.ndarray, box_px: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """(H, W) bool mask of the object under `point_px`, and SAM's predicted IoU score."""
        kw = {"input_points": [[[[float(point_px[0]), float(point_px[1])]]]], "input_labels": [[[1]]]}
        if box_px is not None:
            kw["input_boxes"] = [[[float(v) for v in box_px]]]
        inputs = self.processor(images=image_rgb, return_tensors="pt", **kw).to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # (1, 3, H, W)
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        masks = masks[0].numpy().astype(bool)
        if box_px is not None:
            # Prefer the best-scoring mask that stays inside the (slightly grown) box: the larger
            # multimask outputs are often the table or a group of objects.
            x1, y1, x2, y2 = box_px
            pad = 0.15 * max(x2 - x1, y2 - y1)
            inside = np.zeros_like(masks[0])
            inside[int(max(0, y1 - pad)): int(y2 + pad) + 1, int(max(0, x1 - pad)): int(x2 + pad) + 1] = True
            frac_inside = np.array([(m & inside).sum() / max(m.sum(), 1) for m in masks])
            scores = np.where(frac_inside > 0.9, scores, scores - 1.0)
        i = int(np.argmax(scores))
        return masks[i], float(scores[i])
