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


def load_hf_vlm(model_id: str, quantization: str | None = None, device: str | None = None):
    """Load a transformers image-text chat model: returns (processor, model, device).

    quantization: None / "none" (bf16), "8bit" or "4bit" (bitsandbytes, CUDA only): the 4B Qwen3-VL needs
    ~9 GB in bf16 and ~3.5 GB in 4-bit."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    if quantization in (None, "", "none"):
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    elif quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
             if quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, quantization_config=q, device_map=device).eval()
    else:
        raise ValueError(f"quantization must be none, 8bit or 4bit, got {quantization!r}")
    return processor, model, device


class VlmPointer:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str | None = None, max_new_tokens: int = 96,
                 quantization: str | None = None):
        """quantization: see load_hf_vlm. The loaded model can be shared with the agent's reasoner
        (agent.reasoner.HfReasoner.from_loaded)."""
        import torch

        self.torch = torch
        self.model_id = model_id
        self.quantization = None if quantization in (None, "", "none") else quantization
        self.processor, self.model, self.device = load_hf_vlm(model_id, self.quantization, device)
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

    def _ask(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 24) -> str:
        from PIL import Image

        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image_rgb)}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def detect(self, image_rgb: np.ndarray, what: str | None = None, max_new_tokens: int = 512) -> list[Detection]:
        """Every instance of `what` (default: every object on the table) with a box and a short label."""
        h, w = image_rgb.shape[:2]
        prompt = DETECT_PROMPT.format(what=what or DETECT_OBJECTS)
        return parse_detections(self._ask(image_rgb, prompt, max_new_tokens), w, h)

    def describe(self, image_rgb: np.ndarray, mask: np.ndarray) -> str:
        """Short name of the masked object (e.g. "black dry erase marker")."""
        return clean_name(self._ask(describe_crop(image_rgb, mask), DESCRIBE_PROMPT))


# No example names: the 2B model copies them when unsure.
DESCRIBE_PROMPT = ("Name the object outlined in red, in 2 to 6 words (its color and what it is). "
                   "Answer with the name only.")


def describe_crop(image_rgb: np.ndarray, mask: np.ndarray, pad_frac: float = 0.6, min_side: int = 160) -> np.ndarray:
    """Crop around the mask (with context) and outline the mask in red: what the VLM is asked to name."""
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * pad_frac)
    side = max(side, min_side)
    x0, x1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    out = image_rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 0), 2)  # RGB red
    return out[y0:y1, x0:x1]


def clean_name(text: str, max_words: int = 8) -> str:
    """First line of the answer, without quotes, trailing period or a leading article."""
    line = (text.strip().splitlines() or [""])[0].strip().strip('"\'` .').strip()
    words = line.split()
    if words and words[0].lower() in ("a", "an", "the"):
        words = words[1:]
    return " ".join(words[:max_words]) or "unknown object"


DETECT_PROMPT = (
    "Detect {what} in the image. Answer with a JSON list only, no other text: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<color and name, 1 to 4 words>"}}, ...]. '
    "One entry per separate item. Answer [] if there are none."
)
DETECT_OBJECTS = ("every separate object lying on the table (not the table, not the robot's own gripper "
                  "fingers at the bottom of the image, not shadows)")
DETECT_HANDS = "every human hand"


@dataclass
class Detection:
    label: str
    box_px: np.ndarray  # (4,) x1, y1, x2, y2 in image pixels
    point_px: np.ndarray  # box centre

    def as_pointing(self) -> Pointing:
        return Pointing(self.point_px, self.box_px, self.label)


def parse_detections(text: str, width: int, height: int, min_side_px: float = 6.0) -> list[Detection]:
    """Parse a JSON list of {"bbox_2d", "label"} (0..1000 grid). Tolerates code fences, a single object
    instead of a list, and junk entries (skipped). Near-duplicate boxes (IoU > 0.8) are merged."""
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if m is None:
        return []
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        # Long lists sometimes get cut off: keep the complete entries.
        items = re.findall(r"\{[^{}]*\}", m.group(1))
        obj = []
        for it in items:
            try:
                obj.append(json.loads(it))
            except json.JSONDecodeError:
                continue
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return []
    scale = np.tile(np.array([width, height]) / COORD_SCALE, 2)
    out: list[Detection] = []
    for o in obj:
        if not isinstance(o, dict):
            continue
        box = _vec(o.get("bbox_2d"), 4)
        if box is None:
            continue
        box = np.clip(box * scale, 0, np.tile([width - 1, height - 1], 2))
        box = np.array([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
        if min(box[2] - box[0], box[3] - box[1]) < min_side_px:
            continue
        label = clean_name(str(o.get("label") or "object"), max_words=6)
        if any(box_iou(box, d.box_px) > 0.8 for d in out):
            continue
        out.append(Detection(label, box, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
