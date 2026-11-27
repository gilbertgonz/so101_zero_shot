"""The agent's reasoning model, behind one small interface so models can be swapped and compared.

The agent loop only ever asks a Reasoner: "here is the conversation (and the tools), what next?" and gets
back a Reply (its text, and the tool call in it if any). Each backend turns the neutral Messages into its
model's own format and reads its tool calls back. Choose one with a spec string, in config/robot.yaml
(`agent.reasoner`) or on the command line (`scripts/agent.py --reasoner`):

  hf:Qwen/Qwen3-VL-4B-Instruct@4bit    any transformers image-text chat model; @none | @8bit | @4bit

A backend that runs the same model as the perception VLM reuses its loaded weights (no second copy).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import yaml

from ..config import CONFIG_DIR

DEFAULT_SPEC = "hf:Qwen/Qwen3-VL-4B-Instruct@4bit"


@dataclass
class ToolCall:
    name: str
    args: dict
    thought: str  # the model's reason, written before the call
    raw: str


@dataclass
class Message:
    """One chat turn. role: system | user | assistant | tool. image: RGB uint8, at most one per message."""
    role: str
    text: str
    image: np.ndarray | None = None
    tool_call: ToolCall | None = None  # assistant turns that called a tool


@dataclass
class Reply:
    text: str
    tool_call: ToolCall | None
    seconds: float
    tokens: int = 0  # generated tokens, when the backend knows


class Reasoner(Protocol):
    name: str  # backend, model and settings, e.g. "hf:Qwen/Qwen3-VL-4B-Instruct@4bit"; saved with every run
    call_hint: str  # how to ask for a tool call in a prompt, e.g. "in <tool_call></tool_call> tags"

    def step(self, messages: list[Message], tools: list[dict] | None, max_tokens: int) -> Reply:
        """Next reply. tools: OpenAI-style function specs, or None for a plain text answer (the plan)."""
        ...


# ------------------------------------------------------------------ reading tool calls from text
_ARG = re.compile(r"""\s*(?:(\w+)\s*[=:]\s*)?("[^"]*"|'[^']*'|[^,]+?)\s*(?:,|$)""")


def _parse_value(v: str):
    v = v.strip()
    if v[:1] in "\"'" and v[-1:] == v[:1]:
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return float(v) if any(c in v for c in ".eE") else int(v)
    except ValueError:
        return v


def _params(tools: list[dict], name: str) -> list[str]:
    spec = next((t for t in tools if t["function"]["name"] == name), None)
    return list(spec["function"]["parameters"]["properties"]) if spec else []


def _parse_call_text(text: str, tools: list[dict]) -> ToolCall | None:
    """Fallback for replies that name the tool in plain text: `look_around`, `pick_up(id=3)`, `pick_up("3")`,
    `place_at(20, -5)`. Positional values fill the tool's parameters in order."""
    names = [t["function"]["name"] for t in tools]
    if not names:
        return None
    m = re.search(r"\b(" + "|".join(map(re.escape, names)) + r")\b\s*(\(([^()]*)\))?", text)
    if m is None:
        return None
    name, args = m.group(1), {}
    params = _params(tools, name)
    body = (m.group(3) or "").strip()
    if body.startswith("{"):
        try:
            args = json.loads(body)
        except json.JSONDecodeError:
            args = {}
    elif body:
        pos = 0
        for km in _ARG.finditer(body):
            if not km.group(2):
                continue
            key = km.group(1)
            if key is None:
                if pos >= len(params):
                    break
                key, pos = params[pos], pos + 1
            args[key] = _parse_value(km.group(2))
    return ToolCall(name, args if isinstance(args, dict) else {}, text[:m.start()].strip(), text)


def parse_tool_call(text: str, tools: list[dict]) -> ToolCall | None:
    """First tool call in a reply: <tool_call>{"name": ..., "arguments": {...}}</tool_call>, or a bare JSON
    object with "name", or (fallback) a tool named in plain text. The text before it is the model's reason."""
    dec = json.JSONDecoder()
    m = re.search(r"<tool_call>", text)
    starts = [m.end()] if m else []
    starts += [i.start() for i in re.finditer(r"\{", text)]
    for st in starts:
        j = text.find("{", st)
        if j < 0:
            continue
        try:
            obj, _ = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            thought = text[:m.start()] if m else text[:j]
            return ToolCall(obj["name"], args if isinstance(args, dict) else {}, thought.strip(), text)
    return _parse_call_text(text, tools)


# ------------------------------------------------------------------ backends
@dataclass
class HfReasoner:
    """Any transformers image-text chat model, run in this process (greedy decoding)."""
    processor: object
    model: object
    device: str
    name: str
    call_hint: str = "in <tool_call></tool_call> tags"

    @classmethod
    def load(cls, model_id: str, quantization: str | None = None) -> HfReasoner:
        from ..perception.pointer import load_hf_vlm

        processor, model, device = load_hf_vlm(model_id, quantization)
        return cls(processor, model, device, _hf_name(model_id, quantization))

    @classmethod
    def from_loaded(cls, vlm) -> HfReasoner:
        """Share a model already loaded for perception (perception.pointer.VlmPointer)."""
        return cls(vlm.processor, vlm.model, vlm.device, _hf_name(vlm.model_id, vlm.quantization))

    def step(self, messages: list[Message], tools: list[dict] | None, max_tokens: int) -> Reply:
        import torch

        t = time.monotonic()
        kw = {"tools": tools} if tools else {}
        inputs = self.processor.apply_chat_template(
            [_hf_message(m) for m in messages], tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", **kw).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
        return Reply(text, parse_tool_call(text, tools) if tools else None, time.monotonic() - t, int(new.shape[1]))


def _hf_name(model_id: str, quantization: str | None) -> str:
    return f"hf:{model_id}@{quantization or 'none'}"


def _hf_message(m: Message) -> dict:
    from PIL import Image

    content = ([{"type": "image", "image": Image.fromarray(m.image)}] if m.image is not None else [])
    content.append({"type": "text", "text": m.text})
    msg = {"role": m.role, "content": content}
    if m.tool_call is not None:
        msg["tool_calls"] = [{"type": "function", "function": {"name": m.tool_call.name,
                                                                "arguments": m.tool_call.args}}]
    return msg


BACKENDS = {"hf": HfReasoner}


# ------------------------------------------------------------------ choosing one
@dataclass(frozen=True)
class ReasonerSpec:
    backend: str
    model: str
    options: str | None  # hf: quantization

    @classmethod
    def parse(cls, spec: str) -> ReasonerSpec:
        """"backend:model[@options]", e.g. "hf:Qwen/Qwen3-VL-4B-Instruct@4bit"."""
        backend, sep, rest = spec.strip().partition(":")
        if not sep or not rest:
            raise ValueError(f"reasoner spec must look like backend:model[@options], got {spec!r}")
        if backend not in BACKENDS:
            raise ValueError(f"unknown reasoner backend {backend!r}; available: {', '.join(BACKENDS)}")
        model, _, options = rest.partition("@")
        if not model:
            raise ValueError(f"reasoner spec {spec!r} has no model")
        return cls(backend, model, options or None)


def configured_spec() -> str:
    """config/robot.yaml `agent.reasoner`, or the default."""
    cfg = yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()) or {}
    return str((cfg.get("agent") or {}).get("reasoner") or DEFAULT_SPEC)


def build_reasoner(spec: str | None = None, shared_vlm=None) -> Reasoner:
    """Make the reasoner `spec` names (default: the configured one). shared_vlm: the perception VLM; if it
    is the same model with the same settings, its loaded weights are reused."""
    s = ReasonerSpec.parse(spec or configured_spec())
    if s.backend == "hf":
        quant = None if s.options in (None, "none") else s.options
        if shared_vlm is not None and (shared_vlm.model_id, shared_vlm.quantization) == (s.model, quant):
            return HfReasoner.from_loaded(shared_vlm)
        return HfReasoner.load(s.model, quant)
    raise ValueError(f"no builder for backend {s.backend!r}")  # BACKENDS and this function out of sync
