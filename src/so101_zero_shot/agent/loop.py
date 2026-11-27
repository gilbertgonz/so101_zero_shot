"""The agent: a local VLM plans and runs a free-form request with the tools in agent/tools.py.

  1. home, look around (three views), build the numbered scene list
  2. plan: the model writes a short numbered plan, or "CANNOT: why" (the run ends, nothing moves)
  3. the person approves the plan once
  4. loop: the model sees the request, the plan, the latest marked camera view and every tool result so
     far, and calls one tool (one JSON function call). Grasps and moves toward a hand ask for approval.
     It ends with finish(success, message), or when the step / time budget runs out.

Only the latest camera image is kept in the model's context (older ones are replaced by a note), so a
long run does not slow down or run out of memory. Every tool call and result is shown as a step in the
UI and stored in the episode (events + grasp.json "agent" section).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..config import RUNS_DIR
from ..grasping.runner import GraspRunner, RunResult
from ..hardware import MotionStopped
from ..perception.scene import Scene
from ..skills import Session, Skills, add_step, new_run_dir
from .tools import TOOL_SPECS, Toolbox, ToolResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You control a small robot arm (SO-101, 5 joints) with a two-finger gripper and one camera on its wrist. It stands on a table.
Positions are in cm in the robot's base frame: x = forward (away from the robot), y = to the robot's left, z = up; the table is z = 0. Left and right mean as seen in the camera image, which is also the robot's left and right.
Objects in the scene list have numbers: [3] is object 3, [H1] is a person's hand. The camera image shows each number on its object. Sizes and heights are rough (they can be off by about 3 cm); colors and names come from the camera.
The robot can only do what the tools offer: look, pick up one object at a time from above, put it down, give it to a hand, touch, move above something. It cannot open or pour things, push or slide them, stack precisely, carry two things, or reach anything outside about x 10 to 32 cm, y -20 to 20 cm.
Rules:
- Call exactly one tool in each reply. Before the call, write one short sentence saying why.
- Use the numbers from the latest scene list. If what you need is not in the list, use look_around, look_at or find_hands first.
- After a failure, think about the reason and try something different; try the same thing at most twice.
- When the request is done, or cannot be done, call finish with success true or false and a short message for the person."""

PLAN_PROMPT = """Request from the person: "{request}"

Scene now:
{scene}

Tools: {tools}

Write a short numbered plan (at most 6 steps), one tool per step with its arguments, for example:
1. pick_up(id="2")
2. place_next_to(id="4", side="left")
3. finish(success=true, message="Done.")
If what the request needs is not in the scene list, start with look_around or look_at. If the request cannot be done by this robot at all, answer only: CANNOT: <reason in one sentence>"""

GO_PROMPT = ("The person approved the plan. Carry it out now: in each reply write one short sentence, then exactly "
             "one tool call in <tool_call></tool_call> tags.")
IMAGE_REMOVED = "(older camera image removed)"


@dataclass
class AgentConfig:
    max_steps: int = 16  # tool calls
    max_time_s: float = 600.0
    approve_plan: bool = True
    approve_grasp: bool = True
    approve_hand: bool = True
    plan_tokens: int = 320
    step_tokens: int = 320
    look_first: bool = True


@dataclass
class ToolCall:
    name: str
    args: dict
    thought: str
    raw: str


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


def _parse_call_text(text: str, names: list[str]) -> ToolCall | None:
    """Fallback for replies that name the tool in plain text: `look_around`, `pick_up(id=3)`, `pick_up("3")`,
    `place_at(20, -5)`. Positional values fill the tool's parameters in order."""
    pat = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b\s*(\(([^()]*)\))?")
    m = None
    for m in pat.finditer(text):
        break
    if m is None:
        return None
    name, args = m.group(1), {}
    params = list(next(t for t in TOOL_SPECS if t["function"]["name"] == name)["function"]["parameters"]["properties"])
    if m.group(3) and m.group(3).strip():
        body = m.group(3).strip()
        if body.startswith("{"):
            try:
                args = json.loads(body)
            except json.JSONDecodeError:
                args = {}
        else:
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
    return ToolCall(name, args, text[:m.start()].strip(), text)


def parse_tool_call(text: str) -> ToolCall | None:
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
    return _parse_call_text(text, [t["function"]["name"] for t in TOOL_SPECS])


def _image_content(path: Path, max_side: int = 960):
    from PIL import Image

    img = cv2.imread(str(path))
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return {"type": "image", "image": Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))}


@dataclass
class Conversation:
    """Chat history in the transformers format, keeping only the newest image."""
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, text: str, image: Path | None = None, tool_calls: list | None = None) -> None:
        content = []
        if image is not None:
            item = _image_content(image)
            if item is not None:
                for msg in self.messages:  # drop older images
                    msg["content"] = [c if c.get("type") != "image" else {"type": "text", "text": IMAGE_REMOVED}
                                      for c in msg["content"]]
                content.append(item)
        content.append({"type": "text", "text": text})
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def transcript(self) -> list[dict]:
        return [{"role": m["role"], "text": " ".join(c.get("text", "[image]") for c in m["content"]),
                 **({"tool_calls": m["tool_calls"]} if "tool_calls" in m else {})} for m in self.messages]


def agent_steps() -> list[tuple[str, str]]:
    return [("home", "Move to rest"), ("look", "Look around"), ("plan", "Plan"), ("approve", "Approve plan")]


class Agent:
    def __init__(self, runner: GraspRunner, vlm, cfg: AgentConfig | None = None):
        """`runner` supplies the robot parts and the grasp sequence; `vlm` has generate(messages, max_new_tokens,
        tools) (perception.pointer.VlmPointer)."""
        self.runner, self.vlm = runner, vlm
        self.cfg = cfg or AgentConfig()

    @property
    def rep(self):
        return self.runner.rep

    def _generate(self, conv: Conversation, tokens: int, tools: bool = True) -> tuple[str, float]:
        t = time.monotonic()
        text = self.vlm.generate(conv.messages, max_new_tokens=tokens, tools=TOOL_SPECS if tools else None)
        return text.strip(), time.monotonic() - t

    def run(self, arm, camera, request: str, record: bool | None = None) -> RunResult:
        r = self.runner
        out = new_run_dir(r.runs_dir or RUNS_DIR, "agent")
        log: dict = {"query": request, "mode": "agent", "environment": r.environment,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": {"calls": []}}
        s = Session(r.cfg, self.rep, out, log, r.environment, r.speed, r.cam_model)
        sk = r.skills(arm, camera, out, log)
        scene = Scene(r.locator, r.cfg.check_in_workspace)
        tb = Toolbox(r, scene, s, sk, approve_hand=self.cfg.approve_hand, approve_grasp=self.cfg.approve_grasp)
        if not self.rep.confirm("start", f'Agent: "{request}"? It will look around, propose a plan and ask before '
                                         "moving toward anything."):
            log["result"] = "declined"
            s.save()
            return RunResult(out, "declined", log)

        def body(s: Session, sk: Skills) -> None:
            t0 = time.monotonic()
            calls = log["agent"]["calls"]

            def check_stop():
                if getattr(arm, "stop_event", None) is not None and arm.stop_event.is_set():
                    raise MotionStopped("stopped by user")

            s.begin("home", sk.rest)
            sk.home()
            s.rep.step("home", "done")

            s.begin("look")
            res = tb.call("look_around", {}, "look")
            s.rep.step("look", "done" if res.ok else "failed", res.text.split("\n")[0], images=[res.image] if res.image else None)
            check_stop()

            conv = Conversation()
            conv.add("system", SYSTEM_PROMPT)
            s.begin("plan")
            tools = ", ".join(f"{t['function']['name']}({', '.join(t['function']['parameters']['properties'])})"
                              for t in TOOL_SPECS)
            conv.add("user", PLAN_PROMPT.format(request=request, scene=scene.describe(), tools=tools), res.image)
            plan, dt = self._generate(conv, self.cfg.plan_tokens, tools=False)
            log["agent"]["plan"] = plan
            s.event("agent_plan", text=plan)
            check_stop()
            if plan.upper().lstrip("* ").startswith("CANNOT"):
                reason = plan.split(":", 1)[-1].strip() or plan
                s.rep.step("plan", "done", f"cannot do it: {reason}", {"thinking_s": round(dt, 1)})
                log["result"] = f"refused: {reason}"
                log["agent"]["message"] = reason
                return
            s.rep.step("plan", "done", plan, {"thinking_s": round(dt, 1)})
            conv.add("assistant", plan)

            s.begin("approve")
            if self.cfg.approve_plan and not s.rep.confirm("plan", f"Plan for \"{request}\":\n{plan}",
                                                          [res.image] if res.image else None):
                s.rep.step("approve", "failed", "plan declined")
                log["result"] = "aborted: plan declined"
                return
            s.rep.step("approve", "done")
            conv.add("user", GO_PROMPT)

            failed: dict[tuple, int] = {}  # (tool, args) -> failures, to stop the model repeating itself
            for n in range(1, self.cfg.max_steps + 1):
                check_stop()
                if time.monotonic() - t0 > self.cfg.max_time_s:
                    log["result"] = f"gave up: out of time after {n - 1} steps"
                    return
                text, dt = self._generate(conv, self.cfg.step_tokens)
                check_stop()
                call = parse_tool_call(text)
                sid = f"t{n}"
                if call is None:
                    add_step(s.rep, sid, "(no tool call)")
                    s.rep.step(sid, "failed", text[:300])
                    calls.append({"n": n, "raw": text, "error": "no tool call", "thinking_s": round(dt, 1)})
                    conv.add("assistant", text)
                    conv.add("user", "Reply with exactly one tool call (use finish to end).")
                    continue
                label = f"{call.name}({', '.join(f'{k}={v}' for k, v in call.args.items())})"
                add_step(s.rep, sid, label)
                s.begin(sid)
                s.rep.step(sid, "active", call.thought[:300])
                s.event("tool_call", n=n, name=call.name, args=call.args, thought=call.thought)
                conv.add("assistant", call.thought, tool_calls=[{"type": "function", "function": {
                    "name": call.name, "arguments": call.args}}])
                key = (call.name, json.dumps(call.args, sort_keys=True, default=str))
                if failed.get(key, 0) >= 2 and call.name != "finish":
                    res = ToolResult(False, f"{call.name} with these arguments already failed twice; it will not be "
                                            "tried again. Do something different (look around, another object, "
                                            "another tool) or finish with success false.")
                else:
                    res = tb.call(call.name, call.args, sid)
                if not res.ok:
                    failed[key] = failed.get(key, 0) + 1
                calls.append({"n": n, "name": call.name, "args": call.args, "thought": call.thought,
                              "ok": res.ok, "result": res.text, "thinking_s": round(dt, 1)})
                s.event("tool_result", n=n, ok=res.ok, text=res.text)
                s.save()
                s.rep.step(sid, "done" if res.ok else "failed", res.text.split("\n")[0][:300],
                           {**res.detail, "thinking_s": round(dt, 1)}, [res.image] if res.image else None)
                if res.done:
                    log["agent"]["message"] = res.text
                    log["result"] = ("done: " if res.ok else "failed: ") + (res.text or "-")
                    return
                conv.add("tool", ("OK. " if res.ok else "FAILED. ") + res.text, res.image)
            log["result"] = f"gave up: used all {self.cfg.max_steps} steps"

        def task() -> dict:
            return {"type": "agent", "mode": "agent", "query": request, "object": tb.picked_id,
                    "instruction": request, "plan": log["agent"].get("plan"),
                    "tools": [{"name": c.get("name"), "args": c.get("args"), "ok": c.get("ok")}
                              for c in log["agent"]["calls"]]}

        s.execute(arm, camera, sk, r.cfg.recording.enabled if record is None else record, body, task)
        self.rep.log(f"result: {log['result']} ({out.name})")
        return RunResult(out, log["result"], log)
