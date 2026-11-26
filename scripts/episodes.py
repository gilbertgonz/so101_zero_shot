"""List, filter, label and index the recorded episodes in runs/ (see src/so101_zero_shot/recording.py).

Usage (add --sim before the command for the simulated runs in runs/sim/):
  episodes.py list [--status held] [--mode click] [--object marker] [--success yes|no|unknown]
                   [--attempted] [--since 2026-11-25]
  episodes.py label RUN --success yes|no [--note TEXT]     human verdict (overrides the automatic check)
  episodes.py index                                        write runs/episodes.parquet, one row per episode
  episodes.py frames RUN OUT.mp4                           contact video of the stored frames, for a look

"success" is the human label if set, else the automatic held check; "unknown" = no grasp was attempted
(aborted, stopped, ...) or not decided.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from so101_zero_shot.config import RUNS_DIR
from so101_zero_shot.recording import (IMAGE_KEY, META_FILE, decode_image, effective_success, iter_episodes,
                                       load_episode)


def flat(run_dir: Path, m: dict) -> dict:
    task, out, label = m.get("task") or {}, m.get("outcome") or {}, m.get("label") or {}
    return {
        "episode_id": m.get("episode_id", run_dir.name),
        "path": str(run_dir),
        "start_time": m.get("start_time"),
        "duration_s": m.get("duration_s"),
        "num_frames": m.get("num_frames"),
        "fps": m.get("fps"),
        "mode": task.get("mode"),
        "query": task.get("query"),
        "object": task.get("object"),
        "instruction": task.get("instruction"),
        "status": out.get("status"),
        "result": out.get("result"),
        "grasp_attempted": out.get("grasp_attempted"),
        "failed_step": out.get("failed_step"),
        "success_auto": out.get("success"),
        "success_label": label.get("success"),
        "success": effective_success(m),
        "label_note": label.get("note"),
        "git_commit": (m.get("software") or {}).get("commit"),
    }


def select(args) -> list[dict]:
    rows = []
    for run_dir, m in iter_episodes(args.runs_dir):
        r = flat(run_dir, m)
        if args.status and r["status"] != args.status:
            continue
        if args.mode and r["mode"] != args.mode:
            continue
        if args.object and args.object.lower() not in (r["object"] or "").lower():
            continue
        if args.attempted and not r["grasp_attempted"]:
            continue
        if args.since and (r["start_time"] or "") < args.since:
            continue
        if args.success and {"yes": True, "no": False, "unknown": None}[args.success] != r["success"]:
            continue
        rows.append(r)
    return rows


def cmd_list(args) -> None:
    rows = select(args)
    yn = {True: "yes", False: "no", None: "-"}
    print(f"{'episode':<24} {'mode':<6} {'object':<28} {'status':<9} {'success':<8} {'frames':>6} {'secs':>6}")
    for r in rows:
        succ = yn[r["success"]] + ("*" if r["success_label"] is not None else "")
        print(f"{r['episode_id']:<24} {r['mode'] or '':<6} {(r['object'] or '')[:28]:<28} {r['status'] or '':<9} "
              f"{succ:<8} {r['num_frames'] or 0:>6} {r['duration_s'] or 0:>6.0f}")
    attempted = [r for r in rows if r["grasp_attempted"]]
    ok = sum(r["success"] is True for r in attempted)
    print(f"\n{len(rows)} episodes, {len(attempted)} grasp attempts, {ok} succeeded (* = human label)")


def cmd_label(args) -> None:
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = args.runs_dir / args.run
    path = run_dir / META_FILE
    if not path.exists():
        sys.exit(f"no {META_FILE} in {run_dir}")
    m = json.loads(path.read_text())
    m["label"] = {"success": args.success == "yes", "note": args.note,
                  "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.write_text(json.dumps(m, indent=1))
    auto = (m.get("outcome") or {}).get("success")
    print(f"{run_dir.name}: success={args.success} (automatic check said {auto})")


def cmd_index(args) -> None:
    rows = [flat(d, m) for d, m in iter_episodes(args.runs_dir)]
    out = args.runs_dir / "episodes.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"wrote {out} ({len(rows)} episodes)")


def cmd_frames(args) -> None:
    run_dir = Path(args.run) if Path(args.run).is_dir() else args.runs_dir / args.run
    meta, table = load_episode(run_dir)
    images, phases, ts = table[IMAGE_KEY].to_pylist(), table["phase"].to_pylist(), table["timestamp"].to_pylist()
    writer = None
    for cell, phase, t in zip(images, phases, ts):
        img = decode_image(cell)
        if img is None:
            continue
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), meta.get("fps", 10), (w, h))
        cv2.putText(img, f"{t:6.1f}s  {phase or ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        writer.write(img)
    if writer is None:
        sys.exit("no frames with images")
    writer.release()
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulated runs (runs/sim/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", help="held | missed | aborted | stopped | estopped | error")
    ls.add_argument("--mode", choices=["prompt", "click"])
    ls.add_argument("--object", help="substring of the object name")
    ls.add_argument("--success", choices=["yes", "no", "unknown"])
    ls.add_argument("--attempted", action="store_true", help="only episodes where the gripper closed")
    ls.add_argument("--since", help="ISO date/time, e.g. 2026-11-25")
    lb = sub.add_parser("label")
    lb.add_argument("run")
    lb.add_argument("--success", choices=["yes", "no"], required=True)
    lb.add_argument("--note")
    sub.add_parser("index")
    fr = sub.add_parser("frames")
    fr.add_argument("run")
    fr.add_argument("out")
    args = ap.parse_args()
    args.runs_dir = RUNS_DIR / "sim" if args.sim else RUNS_DIR
    {"list": cmd_list, "label": cmd_label, "index": cmd_index, "frames": cmd_frames}[args.cmd](args)


if __name__ == "__main__":
    main()
