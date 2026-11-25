"""Open-vocabulary pointing with a local VLM (Qwen3-VL): prompt -> point + box in the image."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

# Qwen3-VL reports grounding coordinates on a 0..1000 grid regardless of the image size.
COORD_SCALE = 1000.0

PROMPT = (
    'Locate "{query}" in the image. Answer with JSON only, no other text: '
    '{{"bbox_2d": [x1, y1, x2, y2], "point_2d": [x, y]}} where point_2d is a point on the object. '
    'If it is not in the image, answer {{"bbox_2d": null, "point_2d": null}}.'
)


@dataclass
class Pointing:
    point_px: np.ndarray | None  # (2,) x, y in image pixels
    box_px: np.ndarray | None  # (4,) x1, y1, x2, y2
    raw: str

    @property
    def found(self) -> bool:
        return self.point_px is not None


def _vec(v, n: int) -> np.ndarray | None:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    try:
        a = np.array([float(x) for x in v])
    except (TypeError, ValueError):
        return None
    return a if np.all(np.isfinite(a)) else None


def parse_pointing(text: str, width: int, height: int) -> Pointing:
    """Parse the model's answer. Tolerates code fences, a list of objects (takes the first), and either
    field missing (box centre stands in for a missing point). Out-of-range coordinates are clipped."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m is None:
        return Pointing(None, None, text)
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return Pointing(None, None, text)
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict)), {})
    if not isinstance(obj, dict):
        return Pointing(None, None, text)
    scale = np.array([width, height]) / COORD_SCALE
    box = _vec(obj.get("bbox_2d"), 4)
    pt = _vec(obj.get("point_2d"), 2)
    if box is not None:
        box = np.clip(box * np.tile(scale, 2), 0, np.tile([width - 1, height - 1], 2))
        box = np.array([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
    if pt is not None:
        pt = np.clip(pt * scale, 0, [width - 1, height - 1])
    if box is not None:
        # Small models sometimes put the point on a box corner; a point not well inside the box is
        # less trustworthy than the box itself.
        w, h = box[2] - box[0], box[3] - box[1]
        inner = box + np.array([0.15 * w, 0.15 * h, -0.15 * w, -0.15 * h])
        if pt is None or not (inner[0] <= pt[0] <= inner[2] and inner[1] <= pt[1] <= inner[3]):
            pt = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return Pointing(pt, box, text)


class VlmPointer:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str | None = None, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def point(self, image_rgb: np.ndarray, query: str) -> Pointing:
        from PIL import Image

        h, w = image_rgb.shape[:2]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image_rgb)},
            {"type": "text", "text": PROMPT.format(query=query)},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_pointing(text, w, h)
