"""The agent: a local VLM plans and runs a free-form request with the tools in agent/tools.py.

  1. home, look around (three views), build the numbered scene list
  2. plan: the model writes a short numbered plan, or "CANNOT: why" (the run ends, nothing moves)
  3. the person approves the plan once
  4. loop: the model sees the request, the plan, the latest marked camera view and every tool result so
     far, and calls one tool (one JSON function call). Grasps and moves toward a hand ask for approval.
     It ends with finish(success, message), or when the step / time budget runs out. The first
     finish(success=true) is answered with a check instead: the request again and the list of what was
     actually done, so a multi-part request is not called done halfway.

Only the latest camera image is kept in the model's context (older ones are replaced by a note), so a
long run does not slow down or run out of memory. Every tool call and result is shown as a step in the
UI and stored in the episode (events + grasp.json "agent" section).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import RUNS_DIR
from ..grasping.runner import GraspRunner, RunResult
from ..hardware import MotionStopped
from ..perception.scene import Scene
from ..skills import Session, Skills, add_step, new_run_dir, reasoning
from .reasoner import Message, Reasoner, ToolCall
from .tools import TOOL_SPECS, Toolbox, ToolResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You control a small robot arm (SO-101, 5 joints) with a two-finger gripper and one camera on its wrist. It stands on a table.
Positions are in cm in the robot's base frame: x = forward (away from the robot), y = to the robot's left, z = up; the table is z = 0. Left and right mean as seen in the camera image, which is also the robot's left and right.
Objects in the scene list have numbers: [3] is object 3, [H1] is a person's hand. The camera image shows each number on its object. Sizes and heights are rough (they can be off by about 3 cm); colors and names come from the camera.
The robot can look, pick up one object at a time from above, put it down on the table or on top of something, and move its gripper along any path of 3D points, opening or closing it at the end. It reaches about x 10 to 32 cm, y -20 to 20 cm, z 1 to 25 cm, and cannot carry two things at once.
Rules:
- Call exactly one tool in each reply. Before the call, write one short sentence saying why.
- Use the most applicable tool for each step. When no tool does what a step needs, use move_gripper: with the right points it can make almost any motion.
- pick_up and place_at aim by themselves; do not move the gripper first to line them up.
- Use the numbers from the latest scene list. If what you need is not in the list, use look_around, look_at or find_hands first.
- Distances, sides and sizes are already worked out in the scene list (closest first, CLOSEST, FURTHEST, TALLEST); use them instead of guessing from the image.
- After a failure, think about the reason and try something different; try the same thing at most twice.
- If the request names an object that is still not in the scene list after looking around, it is not on the table: call finish with success false and say so. Never use a different object instead.
- When the request is done, or cannot be done, call finish with success true or false and a short message for the person."""

PLAN_PROMPT = """Request from the person: "{request}"

Scene now:
{scene}

Tools: {tools}

Write a short numbered plan (at most 6 steps), one tool per step with its arguments, for example:
1. pick_up(id="2")
2. place_at(x_cm=20, y_cm=5)
3. finish(success=true, message="Done.")
If what the request needs is not in the scene list, start with look_around or look_at. If the request cannot be done by this robot at all, answer only: CANNOT: <reason in one sentence>"""

GO_PROMPT = ("The person approved the plan. Carry it out now: in each reply write one short sentence, then exactly "
             "one tool call {call_hint}.")
IMAGE_REMOVED = "(older camera image removed)"
CHECK_PROMPT = """Before finishing, check the request once more: "{request}"
Done so far:
{done}
Is every part of the request done? If yes, call finish again. If not, do the missing part now."""
ACTION_TOOLS = ("pick_up", "place_at", "move_gripper")


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


def _load_image(path: Path, max_side: int = 960) -> np.ndarray | None:
    """RGB image from disk, shrunk so its longer side is at most max_side."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@dataclass
class Conversation:
    """Chat history, keeping only the newest image."""
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, text: str, image: Path | None = None, tool_call: ToolCall | None = None) -> None:
        rgb = _load_image(image) if image is not None else None
        if rgb is not None:
            for msg in self.messages:  # drop older images
                if msg.image is not None:
                    msg.image, msg.text = None, f"{IMAGE_REMOVED} {msg.text}"
        self.messages.append(Message(role, text, rgb, tool_call))

    def transcript(self) -> list[dict]:
        return [{"role": m.role, "text": m.text,
                 **({"tool_call": {"name": m.tool_call.name, "args": m.tool_call.args}} if m.tool_call else {})}
                for m in self.messages]


def _truthy(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in ("true", "yes", "1")


def agent_steps() -> list[tuple[str, str]]:
    return [("home", "Move to rest"), ("look", "Look around"), ("plan", "Plan"), ("approve", "Approve plan")]


class Agent:
    def __init__(self, runner: GraspRunner, reasoner: Reasoner, cfg: AgentConfig | None = None):
        """`runner` supplies the robot parts and the grasp sequence; `reasoner` plans and picks the tool calls
        (agent.reasoner.build_reasoner)."""
        self.runner, self.reasoner = runner, reasoner
        self.cfg = cfg or AgentConfig()
        self.scene: Scene | None = None  # the latest run's scene list (the UI's 3D map labels it)

    @property
    def rep(self):
        return self.runner.rep

    def run(self, arm, camera, request: str, record: bool | None = None) -> RunResult:
        r = self.runner
        out = new_run_dir(r.runs_dir or RUNS_DIR, "agent")
        log: dict = {"query": request, "mode": "agent", "environment": r.environment,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "agent": {"reasoner": self.reasoner.name, "calls": []}}
        s = Session(r.cfg, self.rep, out, log, r.environment, r.speed, r.cam_model)
        sk = r.skills(arm, camera, out, log)
        scene = self.scene = Scene(r.locator, r.cfg.check_in_workspace)
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
            reasoning(s.rep, "thinking", "writing a plan")
            reply = self.reasoner.step(conv.messages, None, self.cfg.plan_tokens)
            plan, dt = reply.text, reply.seconds
            log["agent"]["plan"] = plan
            s.event("agent_plan", text=plan)
            check_stop()
            if plan.upper().lstrip("* ").startswith("CANNOT"):
                reason = plan.split(":", 1)[-1].strip() or plan
                s.rep.step("plan", "done", f"cannot do it: {reason}", {"thinking_s": round(dt, 1)})
                reasoning(s.rep, "refused", reason)
                log["result"] = f"refused: {reason}"
                log["agent"]["message"] = reason
                return
            s.rep.step("plan", "done", plan, {"thinking_s": round(dt, 1)})
            reasoning(s.rep, "plan", plan)
            conv.add("assistant", plan)

            s.begin("approve")
            if self.cfg.approve_plan and not s.rep.confirm("plan", f"Plan for \"{request}\":\n{plan}",
                                                          [res.image] if res.image else None):
                s.rep.step("approve", "failed", "plan declined")
                log["result"] = "aborted: plan declined"
                return
            s.rep.step("approve", "done")
            conv.add("user", GO_PROMPT.format(call_hint=self.reasoner.call_hint))

            failed: dict[tuple, int] = {}  # (tool, args) -> failures, to stop the model repeating itself
            checked = False  # the completion check was shown
            for n in range(1, self.cfg.max_steps + 1):
                check_stop()
                if time.monotonic() - t0 > self.cfg.max_time_s:
                    log["result"] = f"gave up: out of time after {n - 1} steps"
                    return
                reasoning(s.rep, "thinking", f"step {n}: choosing the next tool")
                reply = self.reasoner.step(conv.messages, TOOL_SPECS, self.cfg.step_tokens)
                text, dt, call = reply.text, reply.seconds, reply.tool_call
                check_stop()
                sid = f"t{n}"
                if call is None:
                    add_step(s.rep, sid, "(no tool call)")
                    reasoning(s.rep, "reply", text)
                    s.rep.step(sid, "failed", text[:300])
                    calls.append({"n": n, "raw": text, "error": "no tool call", "thinking_s": round(dt, 1),
                                  "tokens": reply.tokens})
                    conv.add("assistant", text)
                    conv.add("user", "Reply with exactly one tool call (use finish to end).")
                    continue
                label = f"{call.name}({', '.join(f'{k}={v}' for k, v in call.args.items())})"
                add_step(s.rep, sid, label)
                s.begin(sid)
                s.rep.step(sid, "active", call.thought[:300])
                reasoning(s.rep, "thought", call.thought)
                reasoning(s.rep, "call", label)
                s.event("tool_call", n=n, name=call.name, args=call.args, thought=call.thought)
                conv.add("assistant", call.thought, tool_call=call)
                key = (call.name, json.dumps(call.args, sort_keys=True, default=str))
                if call.name == "finish" and not checked and _truthy(call.args.get("success")):
                    checked = True
                    done = [f"- {c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items())}): "
                            f"{c['result'].splitlines()[0]}" for c in calls if c.get("ok") and c.get("name") in ACTION_TOOLS]
                    res = ToolResult(False, CHECK_PROMPT.format(request=request, done="\n".join(done) or "- nothing yet"))
                    calls.append({"n": n, "name": call.name, "args": call.args, "thought": call.thought,
                                  "ok": False, "result": "completion check", "thinking_s": round(dt, 1),
                                  "tokens": reply.tokens})
                    s.rep.step(sid, "done", "check: is every part of the request done?")
                    conv.add("tool", res.text, None)
                    reasoning(s.rep, "check", res.text)
                    continue
                if failed.get(key, 0) >= 2 and call.name != "finish":
                    res = ToolResult(False, f"{call.name} with these arguments already failed twice; it will not be "
                                            "tried again. Do something different (look around, another object, "
                                            "another tool) or finish with success false.")
                else:
                    res = tb.call(call.name, call.args, sid)
                if not res.ok:
                    failed[key] = failed.get(key, 0) + 1
                calls.append({"n": n, "name": call.name, "args": call.args, "thought": call.thought,
                              "ok": res.ok, "result": res.text, "thinking_s": round(dt, 1),
                              "tokens": reply.tokens})
                s.event("tool_result", n=n, ok=res.ok, text=res.text)
                s.save()
                s.rep.step(sid, "done" if res.ok else "failed", res.text.split("\n")[0][:300],
                           {**res.detail, "thinking_s": round(dt, 1)}, [res.image] if res.image else None)
                if res.done:
                    log["agent"]["message"] = res.text
                    log["result"] = ("done: " if res.ok else "failed: ") + (res.text or "-")
                    reasoning(s.rep, "finished", log["result"])
                    return
                conv.add("tool", ("OK. " if res.ok else "FAILED. ") + res.text, res.image)
                reasoning(s.rep, "result", ("OK. " if res.ok else "FAILED. ") + res.text.split("\nScene now:")[0])
            log["result"] = f"gave up: used all {self.cfg.max_steps} steps"

        def task() -> dict:
            return {"type": "agent", "mode": "agent", "query": request, "reasoner": self.reasoner.name,
                    "object": tb.picked_id,
                    "instruction": request, "plan": log["agent"].get("plan"),
                    "tools": [{"name": c.get("name"), "args": c.get("args"), "ok": c.get("ok")}
                              for c in log["agent"]["calls"]]}

        s.execute(arm, camera, sk, r.cfg.recording.enabled if record is None else record, body, task)
        self.rep.log(f"result: {log['result']} ({out.name})")
        return RunResult(out, log["result"], log)
