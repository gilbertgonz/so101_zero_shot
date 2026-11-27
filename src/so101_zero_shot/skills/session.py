"""One robot run from torque-on to torque-off: shared by the grasp runner and the agent.

Session.execute() enables torque, calls the run's body, and whatever happens (done, SkillFailed, Stop,
crash) returns the arm to rest and disables torque; after an E-stop it never moves again. It also starts
and finishes the episode recording and keeps the run's JSON log (<run>/grasp.json) saved.

Stopping: SO101Arm.request_stop() from any thread interrupts the current move (MotionStopped), after
which the arm returns to rest; a stop pressed during that return cuts torque where the arm is.
SO101Arm.emergency_torque_off() cuts torque; the session then does not try to move again.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import yaml

from .. import recording
from ..config import CONFIG_DIR, RobotConfig
from ..hardware import MotionStopped
from .core import SkillFailed, Skills

logger = logging.getLogger(__name__)


class Reporter(Protocol):
    def step(self, step_id: str, status: str, message: str = "", detail: dict | None = None,
             images: list[Path] | None = None) -> None:
        """status: active | done | failed | skipped | stopped"""

    def log(self, message: str) -> None: ...

    def confirm(self, kind: str, prompt: str, images: list[Path] | None = None) -> bool:
        """kind: 'start' (before torque), 'grasp' (after planning), 'plan' (agent plan), ... True = go ahead."""

    def choose(self, kind: str, title: str, prompt: str, options: list[tuple[str, str]],
               images: list[Path] | None = None) -> str:
        """Ask the user to pick one of `options` [(value, label)]; return the value ('cancel' if stopped)."""

    def pick(self, prompt: str) -> np.ndarray | None:
        """Wait for the user to click a point in the live (rectified) camera view; None if cancelled."""


def add_step(rep, step_id: str, label: str) -> None:
    """Append a step to the reporter's plan (agent tool calls); reporters without a plan ignore it."""
    fn = getattr(rep, "add_step", None)
    if fn is not None:
        fn(step_id, label)


def jsonable(o):
    return np.asarray(o).tolist()


class RecordingReporter:
    """Passes everything to the real reporter and notes steps and operator answers in the episode."""

    def __init__(self, rep: Reporter, rec: recording.EpisodeRecorder):
        self.rep, self.rec = rep, rec

    def add_step(self, step_id, label):
        add_step(self.rep, step_id, label)

    def step(self, step_id, status, message="", detail=None, images=None):
        if status == "active":
            self.rec.set_phase(step_id)
        self.rec.event("step", step=step_id, status=status, message=message or None)
        self.rep.step(step_id, status, message, detail, images)

    def log(self, message):
        self.rec.event("log", message=message)
        self.rep.log(message)

    def confirm(self, kind, prompt, images=None):
        answer = self.rep.confirm(kind, prompt, images)
        self.rec.event("confirm", question=kind, answer=bool(answer))
        return answer

    def choose(self, kind, title, prompt, options, images=None):
        answer = self.rep.choose(kind, title, prompt, options, images)
        self.rec.event("choose", question=kind, answer=answer)
        return answer

    def pick(self, prompt):
        click = self.rep.pick(prompt)
        self.rec.event("click", px=None if click is None else [round(float(v), 1) for v in click])
        return click

    def event(self, kind, **data):
        self.rec.event(kind, **data)


class Session:
    """State of one run: its folder, JSON log, reporter, current step and how to get home safely."""

    def __init__(self, cfg: RobotConfig, rep: Reporter, out: Path, log: dict, environment: str, speed: float,
                 cam_model=None):
        self.cfg, self.rep, self.out, self.log = cfg, rep, out, log
        self.environment, self.speed, self.cam_model = environment, speed, cam_model
        self.current: str | None = None
        # Go up through the scan pose on the way home, unless the body knows a direct path is clear
        # (grasp plans check their lift -> rest path).
        self.home_via_scan = True

    def save(self) -> None:
        (self.out / "grasp.json").write_text(json.dumps(self.log, indent=1, default=jsonable))

    def begin(self, step_id: str, target_deg: np.ndarray | None = None) -> None:
        """Mark a step active; `target_deg` (the pose it moves to) lets a UI draw a ghost of it."""
        self.current = step_id
        self.rep.step(step_id, "active", detail=None if target_deg is None
                      else {"target_deg": [round(float(v), 1) for v in target_deg]})

    def event(self, kind: str, **data) -> None:
        """Note something in the episode (no-op when not recording)."""
        fn = getattr(self.rep, "event", None)
        if fn is not None:
            fn(kind, **data)

    # ------------------------------------------------------------------ recording
    def _start_recording(self, arm, camera, record: bool) -> recording.EpisodeRecorder | None:
        """Start the episode recorder; a failure is reported but does not block the run."""
        self.log["recorded"] = record
        if not record:
            return None
        rec = recording.EpisodeRecorder(self.out, arm, camera, self.cfg.recording)
        try:
            rec.start()
        except Exception as e:  # noqa: BLE001
            self.log["recording_error"] = repr(e)
            self.rep.log(f"WARNING: episode recording failed to start, this run is not recorded: {e!r}")
            logger.exception("episode recorder failed to start")
            return None
        return rec

    def _finish_recording(self, rec: recording.EpisodeRecorder, task: dict) -> None:
        try:
            rec.stop()
            perception = (yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()) or {}).get("perception") or {}
            rec.finish({
                "environment": self.environment,
                "task": task,
                "outcome": recording.outcome_of(self.log.get("result", "error: no result"), rec.events),
                "label": {"success": None, "note": None, "labeled_at": None},
                "grasp": self.log,
                "robot": {"type": "so101_follower", "gripper": "roboninecom SO-ARM100-101-102 parallel gripper",
                          "joint_offsets_deg": self.cfg.joint_offsets_deg, "tip_xyz_m": self.cfg.tip_xyz_m,
                          "gripper_open_gap_m": self.cfg.gripper_open_gap_m, "workspace_m": self.cfg.workspace,
                          "speed_deg_s": self.speed,
                          "poses_deg": {"rest": self.cfg.poses["rest"], "scan": self.cfg.poses["scan"]}},
                "perception": perception,
                "software": recording.git_version(),
            }, cam_model=self.cam_model)
            self.rep.log(f"recorded {rec.num_frames} samples ({rec.frames_with_image} with images) to "
                         f"{rec.out_dir.name}/{recording.EPISODE_FILE}"
                         + (f"; WRITE ERROR {rec.write_error}" if rec.write_error else ""))
        except Exception as e:  # noqa: BLE001 - the run itself is over; report and move on
            self.rep.log(f"WARNING: saving the episode failed: {e!r}")
            logger.exception("saving the episode failed")

    # ------------------------------------------------------------------ run
    def execute(self, arm, camera, sk: Skills, record: bool, body: Callable[[Session, Skills], None],
                task: Callable[[], dict]) -> None:
        """Torque on, body(session, skills), then home and torque off whatever happened. The body sets
        log["result"]; SkillFailed -> "aborted: ...", Stop -> "stopped", crash -> "error: ...". `task()`
        gives the episode's task metadata (called at the end, so it can use what the run found)."""
        rec = self._start_recording(arm, camera, record)
        rep = self.rep
        if rec is not None:
            self.rep = RecordingReporter(rep, rec)
        try:
            arm.clear_stop()
            arm.hold()
            arm.bus.enable_torque()
            try:
                try:
                    body(self, sk)
                except SkillFailed as e:
                    self.rep.step(self.current, "failed", str(e))
                    self.log["result"] = f"aborted: {e}"
                except (MotionStopped, KeyboardInterrupt):
                    if getattr(arm, "estopped", False):
                        raise
                    self.rep.step(self.current, "stopped", "stopped by user")
                    self.log["result"] = "stopped"
                except Exception as e:
                    if getattr(arm, "estopped", False):
                        raise MotionStopped("emergency stop") from e
                    self.rep.step(self.current, "failed", repr(e))
                    self.log["result"] = f"error: {e!r}"
                    self.home_via_scan = True  # unknown state: go up first
                    logger.exception("run failed")
                self.save()
                # Return to rest. A stop request that ended the run above is consumed here, so only a stop
                # pressed *during* the return cuts torque where the arm is. After an E-stop, do not move again.
                if getattr(arm, "estopped", False):
                    raise MotionStopped("emergency stop")
                arm.clear_stop()
                at_rest = self.current == "home"
                self.begin("return", sk.rest)
                sk.return_home(via_scan=self.home_via_scan and not at_rest)
                self.rep.step("return", "done")
            except (MotionStopped, KeyboardInterrupt):
                est = getattr(arm, "estopped", False)
                self.log["result"] = "estopped" if est else self.log.get("result", "stopped")
                if self.current:
                    self.rep.step(self.current, "stopped", "E-STOP: torque off" if est else
                                  "stopped again: torque off where the arm is")
                self.rep.log("E-STOP: torque off" if est else "stopped while returning: torque disabled where the arm is")
            finally:
                try:
                    arm.bus.disable_torque(num_retry=5)
                except Exception as e:  # noqa: BLE001 - report, the caller still disconnects
                    self.rep.log(f"failed to disable torque: {e!r}")
                try:
                    sk.save_map()
                except Exception as e:  # noqa: BLE001 - the map is extra; the run log must still be saved
                    logger.exception("saving the scene map failed")
                    self.log["scene_map_error"] = repr(e)
                self.save()
        finally:
            self.rep = rep
            if rec is not None:
                self._finish_recording(rec, task())


def new_run_dir(runs_dir: Path, prefix: str) -> Path:
    out = runs_dir / time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
    n = 1
    while out.exists():  # two runs in the same second (tests, benchmark)
        n += 1
        out = runs_dir / time.strftime(f"{prefix}_%Y%m%d_%H%M%S_{n}")
    out.mkdir(parents=True)
    return out
