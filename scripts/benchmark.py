"""Agent benchmark in the MuJoCo sim: 5 hard requests, each run 3 times on a shifted scene, scored from the
sim's true object positions.

The tasks need more than a direct instruction: choosing an object by what it is for, a motion no tool
names (knocking something over, with move_gripper), placing between two objects, a multi-step task that
needs a search first, and admitting an object is not there. Each run
resets the sim scene (objects shifted up to 2 cm and turned, the same shifts for every reasoner), runs the
agent with everything approved automatically, and scores the result:
  success  the request was done (or correctly refused) and nothing else was knocked around
  partial  a reasonable attempt: the right object was picked up but the goal was not met, or a correct
           refusal / goal that moved other objects on the way
  fail     anything else (wrong object, claimed success without doing it, ran out of steps, crashed)

Only the reasoning model changes between runs (--reasoner); perception, tools, skills and scenes stay
the same, so the table compares reasoners.

Usage:
  benchmark.py                                   all tasks, 3 times each, with the configured reasoner
  benchmark.py --reasoner hf:Qwen/Qwen3-VL-8B-Instruct@4bit
  benchmark.py --tasks write,knock           only these
  benchmark.py --repeats 1 --jitter 0            one quick run on the unshifted scene
  benchmark.py --no-video                        skip the videos (by default: one per task + a mosaic of all)
  benchmark.py --mosaic runs/sim/benchmark_X     rebuild the mosaic of an earlier run
  benchmark.py --list

Results go to runs/sim/benchmark_<time>/: results.jsonl, summary.md, each task's agent run,
videos/<task>_<repeat>.mp4 (3/4 view + wrist camera, 4x speed) and mosaic.mp4 (one column per task, one row
per repeat, tiles coloured by score).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from so101_zero_shot.agent import Agent, AgentConfig, build_reasoner
from so101_zero_shot.config import RUNS_DIR
from so101_zero_shot.grasping.runner import GraspRunner
from so101_zero_shot.grasping.setup import build_runner_parts
from so101_zero_shot.skills import new_run_dir

LIFT_M = 0.03  # an object counts as picked up when it rose this far
MOVED_M = 0.02  # ... and as moved when its centre shifted this far on the table
ON_TABLE_M = 0.015  # set down = its centre ends within this of its starting height
TIPPED_M = 0.01  # knocked over = its centre ended this much lower (the 7 cm cylinder lying: 1.9 cm lower)


# ------------------------------------------------------------------ scoring
@dataclass
class Outcome:
    """What happened in one run, as the sim saw it."""
    status: str  # done | failed | refused | gave up | aborted | error | ...
    start: dict[str, np.ndarray]
    end: dict[str, np.ndarray]
    max_z: dict[str, float]
    hand_contact_n: float

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


def score_place(obj: str, goal: Callable[[Outcome], tuple[bool, str]]) -> Callable[[Outcome], Score]:
    def score(o: Outcome) -> Score:
        if not o.lifted(obj):
            return "fail", f"{obj} not picked up"
        if not o.on_table(obj):
            return "partial", f"{obj} picked up but not set down on the table"
        ok, why = goal(o)
        other = o.disturbed({obj})
        if not ok:
            return "partial", f"{obj} placed, but {why}"
        if other or o.status != "done":
            return "partial", f"{why}; " + (f"also disturbed {', '.join(other)}" if other else f"ended {o.status!r}")
        return "success", why
    return score


def next_to(obj: str, ref: str, max_gap_m: float = 0.09) -> Callable[[Outcome], tuple[bool, str]]:
    def goal(o: Outcome) -> tuple[bool, str]:
        d = float(np.linalg.norm(o.end[obj][:2] - o.end[ref][:2]))
        return d <= max_gap_m, f"{d * 100:.1f} cm from {ref} (want <= {max_gap_m * 100:.0f})"
    return goal


def between(obj: str, a: str, b: str, max_off_m: float = 0.03) -> Callable[[Outcome], tuple[bool, str]]:
    """obj ends on the segment from a to b: within max_off_m of it, and in its middle 60%."""
    def goal(o: Outcome) -> tuple[bool, str]:
        pa, pb, p = o.end[a][:2], o.end[b][:2], o.end[obj][:2]
        ab = pb - pa
        t = float(np.dot(p - pa, ab) / max(np.dot(ab, ab), 1e-9))
        off = float(np.linalg.norm(p - (pa + np.clip(t, 0, 1) * ab)))
        ok = 0.2 <= t <= 0.8 and off <= max_off_m
        return ok, f"{off * 100:.1f} cm off the line from {a} to {b}, {100 * t:.0f}% of the way (want 20-80%)"
    return goal


def score_place_all(objs: list[str], goal: Callable[[Outcome, str], tuple[bool, str]]) -> Callable[[Outcome], Score]:
    """Several objects must each be picked up, set down on the table and meet `goal`."""
    def score(o: Outcome) -> Score:
        done, notes = [], []
        for obj in objs:
            if not o.lifted(obj):
                notes.append(f"{obj} not picked up")
            elif not o.on_table(obj):
                notes.append(f"{obj} not set down on the table")
            else:
                ok, why = goal(o, obj)
                notes.append(f"{obj} {why}")
                if ok:
                    done.append(obj)
        other = o.disturbed(set(objs))
        why = "; ".join(notes) + (f"; also disturbed {', '.join(other)}" if other else "")
        if len(done) == len(objs) and not other and o.status == "done":
            return "success", why
        if done or any(o.lifted(obj) for obj in objs):
            return "partial", why
        return "fail", why
    return score


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


def score_refusal(o: Outcome) -> Score:
    moved = o.disturbed(set())
    if o.status not in ("refused", "failed"):
        return "fail", f"should have refused or reported failure; ended {o.status!r}"
    if moved:
        return "partial", f"ended {o.status!r} but moved {', '.join(moved)}"
    return "success", f"ended {o.status!r}, nothing moved"


# ------------------------------------------------------------------ tasks
@dataclass
class Task:
    id: str
    request: str
    tests: str
    score: Callable[[Outcome], Score]
    setup: Callable[[object], None] = field(default=lambda world: None)


# "Next to" as centre distance: about a 5 cm gap or less. The 13 cm marker's centre cannot get as close.
NEXT_TO_M = {"red_cube": 0.10, "marker": 0.13}


def _marker_off_to_the_left(world) -> None:
    world.set_object("marker", [0.16, 0.18, 0.009], yaw_deg=40.0)


TASKS = [
    Task("write", "pick up something I can write with", "choosing an object by what it is for", score_pick("marker")),
    Task("knock", "knock over the blue cylinder", "a motion no tool names (the gripper has to push it over)",
         score_knocked_over("blue_cylinder")),
    Task("between", "put the yellow ball between the red cube and the blue cylinder", "placing relative to two objects",
         score_place("yellow_ball", between("yellow_ball", "red_cube", "blue_cylinder"))),
    Task("tidy", "put the marker and the red cube next to the blue cylinder", "multi-step after a search (the marker "
         "starts far off to the left)",
         score_place_all(["marker", "red_cube"], lambda o, obj: next_to(obj, "blue_cylinder", NEXT_TO_M[obj])(o)),
         _marker_off_to_the_left),
    Task("missing", "pick up the purple bottle", "saying an object is not there", score_refusal),
]


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


def run_task(task: Task, agent: Agent, world, cfg, seed: int | None, jitter_m: float,
             video: Path | None = None) -> dict:
    from so101_zero_shot.mujoco_sim import SimVideo, SimWristCamera, make_mujoco_arm

    world.reset(jitter_m=jitter_m, seed=seed)
    world.hide_hand()
    task.setup(world)
    time.sleep(0.5)  # let moved objects settle
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
    o = Outcome(result.partition(":")[0].strip(), start, positions(world), dict(world.max_z), world.hand_contact_n)
    verdict, why = task.score(o)
    return {"task": task.id, "request": task.request, "seed": seed, "score": verdict, "why": why,
            "result": result, "seconds": round(seconds, 1), "tool_calls": len(calls),
            "thinking_s": round(sum(c.get("thinking_s", 0.0) for c in calls), 1),
            "run_dir": str(out_dir) if out_dir else None, "video": str(video) if video else None,
            "start": {k: v.round(3).tolist() for k, v in start.items()},
            "end": {k: v.round(3).tolist() for k, v in o.end.items()}, "hand_contact_n": round(o.hand_contact_n, 2)}


POINTS = {"success": 1.0, "partial": 0.5, "fail": 0.0}


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
    import cv2

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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoner", default=None, help="backend:model[@options] (default: config agent.reasoner)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=0.02,
                    help="shift objects up to this many m and turn them (repeat i uses seed i)")
    ap.add_argument("--seeds", default="", help="comma-separated repeat numbers to run instead of 0..repeats-1, "
                    "e.g. 1 to redo only the second scene")
    ap.add_argument("--max-steps", type=int, default=AgentConfig.max_steps)
    ap.add_argument("--no-video", action="store_true", help="do not save task videos and the mosaic")
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

    print("loading models...")
    parts = build_runner_parts(sim=True)
    reasoner = build_reasoner(args.reasoner, shared_vlm=parts["locator"].pointer)
    out = new_run_dir(RUNS_DIR / "sim", "benchmark")
    out.mkdir(parents=True, exist_ok=True)
    runner = GraspRunner(**parts, reporter=QuietReporter(), environment="sim", runs_dir=out)
    agent = Agent(runner, reasoner, AgentConfig(max_steps=args.max_steps))
    world = SimWorld(parts["cfg"], parts["cam_model"])
    world.start()
    print(f"reasoner: {reasoner.name}\nresults: {out}\n")
    rows: list[dict] = []
    try:
        reps = [int(x) for x in args.seeds.split(",") if x.strip()] or list(range(args.repeats))
        for rep in reps:
            for t in tasks:
                seed = rep if args.jitter > 0 else None
                print(f"[{t.id}] {t.request}" + (f" (seed {seed})" if seed is not None else ""))
                row = {"reasoner": reasoner.name, "repeat": rep, **run_task(
                    t, agent, world, parts["cfg"], seed, args.jitter,
                    out / "videos" / f"{t.id}_{rep}.mp4" if not args.no_video else None)}
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
