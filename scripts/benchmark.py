"""Agent benchmark in the MuJoCo sim: 10 requests that need reasoning about real objects, each run on
3 randomized scenes, scored from the sim's true object positions.

Scenes use scanned household objects (YCB, assets/so101/make_ycb.py) besides the scene file's blocks. A task
names the objects it needs and its distractors (chosen to be plausible wrong answers: other fruit, other
tools, other cutlery); each scene puts them at random spots and turns, adds one random unrelated object,
and changes the lighting, table colour and tower-block colours. A repeat's seed fixes all of that, so every
reasoner sees the same scenes.

The tasks: choosing an object by what it is for (write, screw, soup) or by knowledge about it (calories,
vitamin C), a motion no tool names (knock over), stacking (one object on another, a three-block tower in
size order), a multi-object goal that excludes look-alikes (fruit on the plate, not the strawberry
gelatin), and admitting an object is not there. Each run resets the scene, runs the agent with
everything approved automatically, and scores the result:
  success  the request was done (or correctly refused) and nothing else was knocked around
  partial  a reasonable attempt: the right object was picked up but the goal was not met, part of a
           multi-object goal, or a correct refusal / goal that moved other objects on the way
  fail     anything else (wrong object, claimed success without doing it, ran out of steps, crashed)

Only the reasoning model changes between runs (--reasoner); perception, tools, skills and scenes stay
the same, so the table compares reasoners.

Usage:
  benchmark.py                                   all tasks, 3 scenes each, with the configured reasoner
  benchmark.py --reasoner hf:Qwen/Qwen3-VL-8B-Instruct@4bit
  benchmark.py --tasks calories,tower            only these
  benchmark.py --seeds 1                         only the second scene of each task
  benchmark.py --preview                         only render every task's scenes (scenes.jpg), no agent
  benchmark.py --no-video                        skip the videos (by default: one per task + a mosaic of all)
  benchmark.py --resume runs/sim/benchmark_X     finish an interrupted run (skips what is in its results.jsonl)
  benchmark.py --mosaic runs/sim/benchmark_X     rebuild the mosaic of an earlier run
  benchmark.py --list

Results go to runs/sim/benchmark_<time>/: results.jsonl, summary.md, each task's agent run,
videos/<task>_<repeat>.mp4 (3/4 view + wrist camera, 4x speed) and mosaic.mp4 (one column per task, one row
per repeat, tiles coloured by score).

Each finished run is appended to results.jsonl right away, so a crash or Ctrl-C loses at most the run in
progress: --resume with the same options picks up where it stopped (runs that crashed with an error are
redone), in the same directory, and the summary and mosaic then cover the whole benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import zlib
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from so101_zero_shot.agent import Agent, AgentConfig, build_reasoner
from so101_zero_shot.agent.loop import ACTION_TOOLS
from so101_zero_shot.config import RUNS_DIR
from so101_zero_shot.grasping.runner import GraspRunner
from so101_zero_shot.grasping.setup import build_runner_parts
from so101_zero_shot.mujoco_sim import PLATE, TOWER_BLOCKS
from so101_zero_shot.skills import new_run_dir

LIFT_M = 0.03  # an object counts as picked up when it rose this far
MOVED_M = 0.02  # ... and as moved when its centre shifted this far on the table
ON_TABLE_M = 0.015  # set down = its centre ends within this of its starting height
TIPPED_M = 0.01  # knocked over = its centre ended this much lower (the 7 cm cylinder lying: 1.9 cm lower)
STACK_RISE_M = 0.02  # stacked = its centre ended at least this much higher (on the 3 cm cube: 3 cm higher) ...
STACK_OFF_M = 0.02  # ... and within this of the base's centre (so standing on it, not leaning against it)
PLATE_RISE_M = 0.004  # on the plate = its centre ended this much higher than it started (the plate is 1.2 cm)


# ------------------------------------------------------------------ scoring
@dataclass
class Outcome:
    """What happened in one run, as the sim saw it."""
    status: str  # done | failed | refused | gave up | aborted | error | ...
    start: dict[str, np.ndarray]
    end: dict[str, np.ndarray]
    max_z: dict[str, float]
    hand_contact_n: float
    actions: list[str] = field(default_factory=list)  # the agent's action tool calls (pick_up, place_at, ...)

    def lifted(self, name: str) -> bool:
        return self.max_z[name] - self.start[name][2] >= LIFT_M

    def moved(self, name: str) -> bool:
        return float(np.linalg.norm(self.end[name][:2] - self.start[name][:2])) >= MOVED_M

    def on_table(self, name: str) -> bool:
        return abs(self.end[name][2] - self.start[name][2]) <= ON_TABLE_M

    def disturbed(self, allowed: set[str]) -> list[str]:
        """Other objects that were lifted or pushed."""
        return [n for n in self.start if n not in allowed and (self.lifted(n) or self.moved(n))]


Score = tuple[str, str]  # (success | partial | fail, why)


def score_pick(target: str | Callable[[Outcome], str]) -> Callable[[Outcome], Score]:
    """target: an object name, or a function choosing it from the outcome (e.g. the closest at the start)."""
    def score(o: Outcome) -> Score:
        target_ = target(o) if callable(target) else target
        return _score_pick(o, target_)
    return score


def _score_pick(o: Outcome, target: str) -> Score:
    if not o.lifted(target):
        wrong = [n for n in o.start if o.lifted(n)]
        return "fail", f"{target} not lifted" + (f" (lifted {', '.join(wrong)})" if wrong else "")
    other = o.disturbed({target})
    if o.status != "done":
        return "partial", f"{target} lifted but the run ended {o.status!r}"
    if other:
        return "partial", f"{target} lifted; also disturbed {', '.join(other)}"
    return "success", f"{target} lifted"



def score_knocked_over(obj: str) -> Callable[[Outcome], Score]:
    """obj ends lying on its side (its centre dropped), pushed over rather than picked up and dropped."""
    def score(o: Outcome) -> Score:
        drop = float(o.start[obj][2] - o.end[obj][2])
        down = drop >= TIPPED_M
        other = o.disturbed({obj})
        if not down:
            return "fail", f"{obj} still standing ({drop * 100:.1f} cm lower)"
        if o.lifted(obj):
            return "partial", f"{obj} lying down, but it was picked up and dropped, not knocked over"
        if other or o.status != "done":
            return "partial", f"{obj} knocked over; " + (f"also disturbed {', '.join(other)}" if other
                                                       else f"ended {o.status!r}")
        return "success", f"{obj} knocked over"
    return score


def score_stacked(obj: str, base: str) -> Callable[[Outcome], Score]:
    """obj ends resting on top of base: higher than it started and centred over it, the base not pushed."""
    def score(o: Outcome) -> Score:
        if not o.lifted(obj):
            return "fail", f"{obj} not picked up"
        rise = float(o.end[obj][2] - o.start[obj][2])
        off = float(np.linalg.norm(o.end[obj][:2] - o.end[base][:2]))
        where = f"{rise * 100:+.1f} cm higher than at the start, {off * 100:.1f} cm off {base}'s centre"
        if rise < STACK_RISE_M or off > STACK_OFF_M:
            return "partial", f"{obj} picked up but not on {base} ({where})"
        other = o.disturbed({obj})
        if other or o.status != "done":
            return "partial", f"{obj} on {base}; " + (f"also disturbed {', '.join(other)}" if other
                                                      else f"ended {o.status!r}")
        return "success", f"{obj} on {base} ({where})"
    return score


def _on_top(o: Outcome, obj: str, base: str, rise_m: float) -> tuple[bool, str]:
    """obj ends resting directly on base: centred over it and rise_m (+-25%) higher (end positions, so the
    base may have been moved first)."""
    rise = float(o.end[obj][2] - o.end[base][2])
    off = float(np.linalg.norm(o.end[obj][:2] - o.end[base][:2]))
    ok = 0.75 * rise_m <= rise <= 1.25 * rise_m and off <= STACK_OFF_M
    return ok, f"{obj} {rise * 100:+.1f} cm over {base}, {off * 100:.1f} cm off centre"


def score_tower(blocks: list[str], halves: list[float]) -> Callable[[Outcome], Score]:
    """blocks (bottom first) end stacked in that order, each directly on the one below. halves: their half
    heights (m). Partial: one level right; a tower in another order fails (the order is the reasoning)."""
    def score(o: Outcome) -> Score:
        levels = [_on_top(o, top, base, h0 + h1)
                  for base, top, h0, h1 in zip(blocks, blocks[1:], halves, halves[1:])]
        why = "; ".join(w for _, w in levels)
        other = o.disturbed(set(blocks))
        if all(ok for ok, _ in levels):
            if other or o.status != "done":
                return "partial", why + (f"; also disturbed {', '.join(other)}" if other else f"; ended {o.status!r}")
            return "success", why
        if any(ok for ok, _ in levels):
            return "partial", why
        return "fail", why  # nothing stacked, or stacked in the wrong order
    return score


def score_on_plate(objs: list[str], plate: str, radius_m: float) -> Callable[[Outcome], Score]:
    """Each of objs ends on the plate: within radius_m of its centre and above the table."""
    def score(o: Outcome) -> Score:
        done, notes = [], []
        for obj in objs:
            off = float(np.linalg.norm(o.end[obj][:2] - o.end[plate][:2]))
            up = float(o.end[obj][2] - o.start[obj][2])
            if not o.lifted(obj):
                notes.append(f"{obj} not picked up")
                continue
            ok = off <= radius_m and up >= PLATE_RISE_M
            notes.append(f"{obj} {'on' if ok else 'not on'} the plate ({off * 100:.1f} cm from its centre, "
                         f"{up * 100:+.1f} cm)")
            if ok:
                done.append(obj)
        wrong = [n for n in o.start if n not in objs and n != plate and o.lifted(n)]
        other = [n for n in o.disturbed(set(objs) | {plate}) if n not in wrong]
        why = "; ".join(notes) + (f"; also picked up {', '.join(wrong)}" if wrong else "") + \
            (f"; also disturbed {', '.join(other)}" if other else "")
        if len(done) == len(objs) and not wrong and not other and o.status == "done":
            return "success", why
        if done:
            return "partial", why
        return "fail", why
    return score


def score_refusal(o: Outcome) -> Score:
    """The object is not there: the right answer is to say so without acting. Trying to grab something
    (another object taken for it) is the same mistake whether the grasp works or not, so it fails even when
    the run then ends 'failed'."""
    if o.actions:
        return "fail", f"tried to act instead of saying it is not there ({', '.join(o.actions)}); ended {o.status!r}"
    if o.status not in ("refused", "failed"):
        return "fail", f"should have said it is not there; ended {o.status!r}"
    moved = o.disturbed(set())
    if moved:
        return "partial", f"ended {o.status!r} without acting, but {', '.join(moved)} moved"
    return "success", f"ended {o.status!r} without trying to grab anything"


# ------------------------------------------------------------------ tasks
@dataclass
class Task:
    id: str
    request: str
    tests: str
    score: Callable[[Outcome], Score]
    needs: list[str]  # objects the request is about: placed within easy reach
    distractors: list[str]  # plausible wrong answers, placed anywhere on the table
    no_clutter: tuple[str, ...] = ()  # CLUTTER objects that would make the request ambiguous


# One of these joins every scene, at random: unrelated to every task (except as listed in Task.no_clutter).
CLUTTER = ["sponge", "padlock", "rubiks_cube", "yellow_ball", "green_block"]

TASKS = [
    Task("write", "pick up something I can write with", "choosing an object by what it is for",
         score_pick("marker"), ["marker"], ["spoon", "phillips_screwdriver", "fork", "scissors"]),
    Task("calories", "pick up the food with the most calories", "knowledge: a box of gelatin mix (~300 kcal) "
         "against single fruits (~5-100 kcal)", score_pick("gelatin_box"), ["gelatin_box"],
         ["banana", "strawberry", "apple", "plum"]),
    Task("vitamin_c", "pick up the fruit richest in vitamin C", "knowledge: strawberry (~60 mg/100 g) against "
         "banana, apple, pear, peach (~5-10)", score_pick("strawberry"), ["strawberry"],
         ["banana", "apple", "pear", "peach"]),
    Task("screw", "I need to tighten a screw: pick up the right tool for it", "choosing a tool for a job "
         "among other tools", score_pick("phillips_screwdriver"), ["phillips_screwdriver"],
         ["hammer", "adjustable_wrench", "scissors", "medium_clamp"]),
    Task("soup", "pick up what I should use to eat soup", "choosing cutlery for a job among other cutlery",
         score_pick("spoon"), ["spoon"], ["fork", "knife", "spatula"]),
    Task("knock", "knock over the blue cylinder", "a motion no tool names (the gripper has to push it over)",
         score_knocked_over("blue_cylinder"), ["blue_cylinder"], ["red_cube", "tomato_soup_can"]),
    Task("stack", "place the blue cylinder on top of the red cube", "stacking: setting an object down on another",
         score_stacked("blue_cylinder", "red_cube"), ["blue_cylinder", "red_cube"], ["mustard_bottle", "plum"]),
    Task("tower", "stack the three cubes into a tower, largest at the bottom and smallest on top",
         "ordering by size, then two stacking steps in the right order",
         score_tower(list(TOWER_BLOCKS), list(TOWER_BLOCKS.values())), list(TOWER_BLOCKS), ["apple", "padlock"],
         no_clutter=("rubiks_cube", "green_block")),
    Task("fruit_plate", "put all the fruit on the plate", "a multi-object goal: every fruit, not the "
         "strawberry gelatin box", score_on_plate(["banana", "strawberry"], "plate", PLATE[1] - 0.01),
         ["plate", "banana", "strawberry"], ["gelatin_box", "phillips_screwdriver"]),
    Task("missing", "pick up the purple bottle", "saying an object is not there (a yellow bottle is)",
         score_refusal, [], ["mustard_bottle", "tomato_soup_can", "spoon"]),
]


# ------------------------------------------------------------------ scenes
CELL_M = 0.0025  # layout grid
GRID_LO, GRID_HI = np.array([0.0, -0.30]), np.array([0.45, 0.30])
TABLE_X_M, TABLE_HALF_Y_M = (0.09, 0.42), 0.26  # every object stays inside this ...
BASE_CLEAR_M = 0.12  # ... and this far from the robot's base axis
NEED_ZONE = ((0.18, 0.29), 0.13)  # centres of the objects a task needs: x range, |y| (m). Nearer, the
# gripper hides the table from some of the look-around views.
NEED_REACH_M = 0.29  # ... and at most this far from the base axis (a top-down grasp reaches about 30 cm)
OTHER_ZONE = ((0.12, 0.36), 0.22)  # ... and of everything else
NEED_GAP_M, OTHER_GAP_M = 0.025, 0.012  # free space around needed objects (room for the fingers) / others
# The arm at rest (where every run starts) hangs over the table: nothing under its fingers, and nothing taller
# than ARM_LOW_M under the rest of the gripper. (x range, |y|) in m, measured from the sim at the rest pose.
FINGERS_AT_REST = ((0.195, 0.265), 0.025)
GRIPPER_AT_REST, ARM_LOW_M = ((0.10, 0.265), 0.08), 0.045


def _mask_of(world, name: str, x: float, y: float, yaw_deg: float, grow_m: float) -> np.ndarray:
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = world.footprint_xy(name) @ R.T + [x, y]
    shape = tuple(np.round((GRID_HI - GRID_LO) / CELL_M).astype(int))
    mask = np.zeros(shape[::-1], np.uint8)  # rows = y, columns = x
    cv2.fillConvexPoly(mask, np.round((pts - GRID_LO) / CELL_M).astype(np.int32), 1)
    if grow_m > 0:
        k = 2 * round(grow_m / CELL_M) + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask.astype(bool)


def _allowed_area(height_m: float) -> np.ndarray:
    """Layout grid cells an object of this height may cover."""
    shape = tuple(np.round((GRID_HI - GRID_LO) / CELL_M).astype(int))
    x = GRID_LO[0] + (np.arange(shape[0]) + 0.5) * CELL_M
    y = GRID_LO[1] + (np.arange(shape[1]) + 0.5) * CELL_M
    X, Y = np.meshgrid(x, y)  # rows = y

    def box(zone):
        (x0, x1), hy = zone
        return (X >= x0) & (X <= x1) & (np.abs(Y) <= hy)

    ok = box((TABLE_X_M, TABLE_HALF_Y_M)) & (np.hypot(X, Y) >= BASE_CLEAR_M) & ~box(FINGERS_AT_REST)
    return ok & ~box(GRIPPER_AT_REST) if height_m > ARM_LOW_M else ok


def scene_layout(world, task: Task, rng: np.random.Generator, layouts: int = 50, tries: int = 200
                 ) -> dict[str, tuple[float, float, float]]:
    """Random non-overlapping spots (x, y, yaw deg) for a task's objects, its distractors and one clutter
    object. Draws up to `layouts` layouts and keeps the first with everything placed, else the one that
    placed the most; the needed objects must always fit."""
    clutter = [c for c in CLUTTER if c not in task.no_clutter and c not in task.needs + task.distractors]
    extra = [clutter[int(rng.integers(len(clutter)))]] if clutter else []
    allowed = {False: _allowed_area(0.0), True: _allowed_area(1.0)}  # by: taller than ARM_LOW_M
    # Biggest first, so they still find room; the clutter object last.
    order = sorted(task.needs + task.distractors,
                   key=lambda n: -cv2.contourArea(world.footprint_xy(n).astype(np.float32))) + extra
    best: dict[str, tuple[float, float, float]] = {}
    for _ in range(layouts):
        taken = np.zeros_like(allowed[False])
        out: dict[str, tuple[float, float, float]] = {}
        for name in order:
            need = name in task.needs
            (x0, x1), hy = NEED_ZONE if need else OTHER_ZONE
            gap = NEED_GAP_M if need else OTHER_GAP_M
            area = allowed[world.height_m(name) > ARM_LOW_M]
            for _ in range(tries):
                x, y, yaw = rng.uniform(x0, x1), rng.uniform(-hy, hy), rng.uniform(0.0, 360.0)
                if need and np.hypot(x, y) > NEED_REACH_M:
                    continue
                body = _mask_of(world, name, x, y, yaw, 0.0)
                if (body & ~area).any() or (_mask_of(world, name, x, y, yaw, gap) & taken).any():
                    continue
                out[name] = (x, y, yaw)
                # Needed objects keep their extra room against everything placed after them.
                taken |= _mask_of(world, name, x, y, yaw, NEED_GAP_M - OTHER_GAP_M) if need else body
                break
            else:
                if need:
                    break
        if all(n in out for n in task.needs) and len(out) > len(best):
            best = out
            if len(out) == len(order):
                break
    if not best:
        raise RuntimeError(f"{task.id}: no room for {task.needs} in {layouts} layouts")
    return best


def set_scene(world, task: Task, seed: int) -> dict[str, tuple[float, float, float]]:
    """Reset the sim to task's scene number `seed` (layout and look); returns the layout."""
    rng = np.random.default_rng([seed, zlib.crc32(task.id.encode())])
    world.reset()
    world.hide_hand()
    layout = scene_layout(world, task, rng)
    world.arrange(layout)
    world.randomize_look(rng)
    return layout


# ------------------------------------------------------------------ running
class QuietReporter:
    """Approves everything and prints one line per tool call."""

    def __init__(self):
        self.labels: dict[str, str] = {}

    def add_step(self, step_id, label):
        self.labels[step_id] = label

    def step(self, step_id, status, message="", detail=None, images=None):
        if step_id.startswith("t") and "." not in step_id and status in ("done", "failed"):
            print(f"    {self.labels.get(step_id, step_id).strip()}: {status}" + (f": {message[:120]}" if message else ""))
        elif step_id == "plan" and status == "done":
            print("    plan: " + " | ".join(line.strip() for line in str(message).splitlines() if line.strip()))

    def log(self, message):
        pass

    def confirm(self, kind, prompt, images=None):
        return True

    def choose(self, kind, title, prompt, options, images=None):
        return options[0][0]

    def pick(self, prompt):
        return None


def positions(world) -> dict[str, np.ndarray]:
    return {n: np.array(p, dtype=float) for n, p in world.object_positions().items()}


def run_task(task: Task, agent: Agent, world, cfg, seed: int, video: Path | None = None) -> dict:
    from so101_zero_shot.mujoco_sim import SimVideo, SimWristCamera, make_mujoco_arm

    layout = set_scene(world, task, seed)
    time.sleep(1.0)  # let the objects settle
    world.reset_tracking()
    start = positions(world)
    arm, camera = make_mujoco_arm(cfg, world), SimWristCamera(world)
    t0 = time.monotonic()
    result, out_dir, calls = "error: crashed", None, []
    caption = f'{task.id}: "{task.request}"'
    recorder = SimVideo(world, video, caption) if video is not None else nullcontext()
    try:
        with recorder:
            try:
                res = agent.run(arm, camera, task.request, record=False)
                result, out_dir, calls = res.result, res.out_dir, res.log.get("agent", {}).get("calls", [])
            except Exception as e:  # noqa: BLE001 - one task crashing must not end the benchmark
                result = f"error: {e!r}"
                traceback.print_exc()
            if video is not None:
                recorder.caption = f"{caption} -> {result}"
                time.sleep(0.3)
    finally:
        try:
            arm.disconnect(disable_torque=True)
        except Exception as e:  # noqa: BLE001 - the next task gets a fresh arm anyway
            print(f"  (disconnect failed: {e!r})")
    seconds = time.monotonic() - t0
    time.sleep(0.5)
    actions = [f"{c['name']}({', '.join(f'{k}={v}' for k, v in (c.get('args') or {}).items())})"
               for c in calls if c.get("name") in ACTION_TOOLS]
    o = Outcome(result.partition(":")[0].strip(), start, positions(world), dict(world.max_z), world.hand_contact_n,
                actions)
    verdict, why = task.score(o)
    return {"task": task.id, "request": task.request, "seed": seed, "score": verdict, "why": why,
            "result": result, "seconds": round(seconds, 1), "tool_calls": len(calls),
            "thinking_s": round(sum(c.get("thinking_s", 0.0) for c in calls), 1),
            "run_dir": str(out_dir) if out_dir else None, "video": str(video) if video else None,
            "scene": sorted(layout), "start": {k: v.round(3).tolist() for k, v in start.items() if k in layout},
            "end": {k: v.round(3).tolist() for k, v in o.end.items() if k in layout},
            "hand_contact_n": round(o.hand_contact_n, 2)}


POINTS = {"success": 1.0, "partial": 0.5, "fail": 0.0}


def load_finished(out: Path) -> list[dict]:
    """The finished runs in an earlier benchmark directory's results.jsonl, without runs that crashed with an
    error (they are redone) or a line cut short by the crash. Rewrites the file to just those rows."""
    path = out / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"--resume: no results.jsonl in {out}")
    rows, dropped = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if str(row.get("result", "")).startswith("error"):
            dropped += 1
            continue
        rows.append(row)
    if dropped:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(path)
    print(f"resuming {out}: {len(rows)} runs finished" + (f", {dropped} crashed or cut-off to redo" if dropped else ""))
    return rows


def summary(rows: list[dict], reasoner: str) -> str:
    lines = [f"# Agent benchmark: {reasoner}", "", "## Per task (average over repeats)", "",
             "| task | request | runs | success | partial | fail | score | median s |", "|---|---|---|---|---|---|---|---|"]
    for task in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == task]
        c = {k: sum(r["score"] == k for r in rs) for k in POINTS}
        pts = sum(POINTS[r["score"]] for r in rs)
        lines.append(f"| {task} | {rs[0]['request']} | {len(rs)} | {c['success']} | {c['partial']} | {c['fail']} | "
                     f"{pts / len(rs):.2f} | {np.median([r['seconds'] for r in rs]):.0f} |")
    lines += ["", "## Every run", "", "| task | repeat | score | why | result | time s | calls |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r.get('repeat', 0)} | {r['score']} | {r['why']} | {r['result'][:80]} | "
                     f"{r['seconds']} | {r['tool_calls']} |")
    n = len(rows)
    counts = {k: sum(r["score"] == k for r in rows) for k in POINTS}
    pts = sum(POINTS[r["score"]] for r in rows)
    lines += ["", (f"success {counts['success']}/{n}, partial {counts['partial']}, fail {counts['fail']}; "
                   f"score {pts:.1f}/{n} ({100 * pts / max(n, 1):.0f}%); "
                   f"median time {np.median([r['seconds'] for r in rows]):.0f} s")]
    return "\n".join(lines) + "\n"


SCORE_COLORS = {"success": (60, 180, 60), "partial": (0, 170, 230), "fail": (50, 50, 220)}  # BGR


def make_mosaic(rows: list[dict], path: Path, tile: tuple[int, int] = (480, 360)) -> Path | None:
    """All task videos side by side in one grid video (a column per task, a row per repeat), each tile
    labelled with its task and score. Clips run together; a finished one holds its last frame until the
    longest ends."""
    from so101_zero_shot.mujoco_sim import encode_h264

    clips = [(r, cv2.VideoCapture(r["video"])) for r in rows if r.get("video") and Path(r["video"]).exists()]
    clips = [(r, c) for r, c in clips if c.isOpened()]
    if not clips:
        return None
    fps = clips[0][1].get(cv2.CAP_PROP_FPS) or 40.0
    tw, th = tile
    tasks = list(dict.fromkeys(r["task"] for r, _ in clips))
    repeats = sorted({r.get("repeat", 0) for r, _ in clips})
    cols, rows_n = len(tasks), len(repeats)
    cells = [(repeats.index(r.get("repeat", 0)), tasks.index(r["task"])) for r, _ in clips]
    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * tw, rows_n * th))
    last: list = [None] * len(clips)
    live = [True] * len(clips)
    try:
        while any(live):
            canvas = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
            for i, (r, cap) in enumerate(clips):
                if live[i]:
                    ok, frame = cap.read()
                    if ok:
                        last[i] = cv2.resize(frame, tile, interpolation=cv2.INTER_AREA)
                    else:
                        live[i] = False
                if last[i] is None:
                    continue
                img = last[i].copy()
                color = SCORE_COLORS.get(r["score"], (128, 128, 128))
                cv2.rectangle(img, (0, 0), (tw - 1, th - 1), color, 4)
                cv2.rectangle(img, (0, th - 52), (tw, th), (0, 0, 0), -1)
                cv2.putText(img, f"{r['task']} #{r.get('repeat', 0) + 1}: {r['score'].upper()}", (8, th - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)
                cv2.putText(img, r["request"][:52], (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y, x = cells[i]
                canvas[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = img
            if any(live):
                writer.write(canvas)
    finally:
        writer.release()
        for _, cap in clips:
            cap.release()
    err = encode_h264(raw, path)
    if err:
        print(f"mosaic: {err}")
    return path


def preview(world, tasks: list[Task], seeds: list[int], path: Path, tile: tuple[int, int] = (400, 300)) -> Path:
    """One image of every task's scenes, seen from over the robot's shoulder: a row per task, a column per
    scene. Runs no physics (objects are shown where they are put)."""
    import mujoco

    from so101_zero_shot.viz import LiveRenderer

    rows = []
    for t in tasks:
        row = []
        for seed in seeds:
            set_scene(world, t, seed)
            with world._lock:
                mujoco.mj_forward(world.model, world.data)
                qpos = world.data.qpos.copy()
            r = LiveRenderer(size=tile, model=world.model)
            r.cam.lookat[:] = [0.24, 0.0, 0.0]
            r.cam.distance, r.cam.azimuth, r.cam.elevation = 0.62, 180, -55
            img = r.render_qpos(qpos)
            r.close()
            cv2.rectangle(img, (0, 0), (tile[0], 22), (0, 0, 0), -1)
            cv2.putText(img, f"{t.id} #{seed + 1}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row.append(img)
        strip = np.full((22, tile[0] * len(seeds), 3), 0, np.uint8)
        cv2.putText(strip, f'"{t.request}"', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([strip, np.hstack(row)]))
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoner", default=None, help="backend:model[@options] (default: config agent.reasoner)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", default="", help="comma-separated scene numbers to run instead of 0..repeats-1, "
                    "e.g. 1 to redo only the second scene")
    ap.add_argument("--preview", action="store_true", help="only render the scenes (scenes.jpg), no agent")
    ap.add_argument("--max-steps", type=int, default=AgentConfig.max_steps)
    ap.add_argument("--no-video", action="store_true", help="do not save task videos and the mosaic")
    ap.add_argument("--resume", type=Path, default=None, metavar="DIR",
                    help="continue an interrupted benchmark run directory: skip the runs already in its results.jsonl")
    ap.add_argument("--mosaic", type=Path, default=None, metavar="DIR",
                    help="only rebuild the mosaic video of an earlier benchmark run directory")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.mosaic is not None:
        rows = [json.loads(line) for line in (args.mosaic / "results.jsonl").read_text().splitlines() if line.strip()]
        print(make_mosaic(rows, args.mosaic / "mosaic.mp4") or "no task videos found")
        return
    if args.list:
        for t in TASKS:
            print(f"{t.id:10s} {t.request!r:55s} {t.tests}")
        return
    wanted = [s.strip() for s in args.tasks.split(",") if s.strip()]
    unknown = sorted(set(wanted) - {t.id for t in TASKS})
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; see --list")
    tasks = [t for t in TASKS if not wanted or t.id in wanted]

    from so101_zero_shot.mujoco_sim import SimWorld

    reps = [int(x) for x in args.seeds.split(",") if x.strip()] or list(range(args.repeats))
    if args.preview:
        from so101_zero_shot.config import RobotConfig
        from so101_zero_shot.perception.geometry import FisheyeCamera
        out = new_run_dir(RUNS_DIR / "sim", "scenes")
        out.mkdir(parents=True, exist_ok=True)
        world = SimWorld(RobotConfig.load(), FisheyeCamera.load())
        print(preview(world, tasks, reps, out / "scenes.jpg"))
        return
    done: list[dict] = []
    if args.resume is not None:
        done = load_finished(args.resume)
        earlier = {r["reasoner"] for r in done}
        if len(earlier) > 1:
            raise SystemExit(f"--resume: {args.resume} mixes reasoners {sorted(earlier)}")
        if earlier:
            (name,) = earlier
            if args.reasoner is None:
                args.reasoner = name
            elif args.reasoner != name:
                raise SystemExit(f"--resume: {args.resume} was run with {name!r}, not {args.reasoner!r}")
    todo = [(rep, t) for rep in reps for t in tasks
            if not any(r["task"] == t.id and r["repeat"] == rep for r in done)]
    if args.resume is not None and not todo:
        print("nothing left to run")
        (args.resume / "summary.md").write_text(text := summary(done, done[0]["reasoner"]))
        print(text)
        return
    print("loading models...")
    parts = build_runner_parts(sim=True)
    reasoner = build_reasoner(args.reasoner, shared_vlm=parts["locator"].pointer)
    if args.resume is not None:
        out = args.resume
    else:
        out = new_run_dir(RUNS_DIR / "sim", "benchmark")
        out.mkdir(parents=True, exist_ok=True)
    runner = GraspRunner(**parts, reporter=QuietReporter(), environment="sim", runs_dir=out)
    agent = Agent(runner, reasoner, AgentConfig(max_steps=args.max_steps))
    world = SimWorld(parts["cfg"], parts["cam_model"])
    world.start()
    print(f"reasoner: {reasoner.name}\nresults: {out}\n" + (f"{len(todo)} runs left\n" if done else ""))
    rows: list[dict] = list(done)
    try:
        for rep, t in todo:
            print(f"[{t.id}] {t.request} (scene {rep})")
            row = {"reasoner": reasoner.name, "repeat": rep, **run_task(
                t, agent, world, parts["cfg"], rep, out / "videos" / f"{t.id}_{rep}.mp4" if not args.no_video else None)}
            rows.append(row)
            print(f"  -> {row['score'].upper()}: {row['why']} ({row['result']}; {row['seconds']} s)\n")
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted; summarising what finished")
    finally:
        world.close()
    if rows:
        text = summary(rows, reasoner.name)
        Path(out / "summary.md").write_text(text)
        print(text)
        if not args.no_video:
            print(f"mosaic: {make_mosaic(rows, out / 'mosaic.mp4')}")


if __name__ == "__main__":
    main()
