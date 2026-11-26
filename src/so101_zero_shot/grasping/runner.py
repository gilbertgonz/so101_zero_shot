"""The grasp sequence, shared by the CLI (scripts/grasp.py) and the UI (scripts/ui.py).

Steps (STEPS) are reported as they start and finish through a Reporter, together with the numbers and
images behind each decision; approvals go through Reporter.confirm. Stopping: SO101Arm.request_stop()
from any thread interrupts the current move (MotionStopped), after which the arm returns to rest.
SO101Arm.emergency_torque_off() cuts torque; the runner then does not try to move again.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from ..config import CONFIG_DIR, RUNS_DIR, RobotConfig
from ..hardware import MotionStopped
from ..kinematics import SO101Kinematics
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import Locator, ObjectEstimate, PerceptionError, draw_debug
from .. import recording
from .feedback import is_holding, read_grip
from .topdown import GraspPlan, GraspPlanningError, TopDownPlanner, grasp_height, object_axis_xy

logger = logging.getLogger(__name__)

_GRASP_STEPS: list[tuple[str, str]] = [
    ("relook", "Look closer"),
    ("plan", "Plan grasp"),
    ("approve", "Approval"),
    ("pregrasp", "Open gripper, move above"),
    ("descend", "Descend"),
    ("close", "Close gripper"),
    ("lift", "Lift and check"),
    ("place", "Put it back"),
    ("return", "Return to rest"),
]
# Prompt mode: the VLM finds the object named in the prompt.
STEPS: list[tuple[str, str]] = [("home", "Move to rest"), ("coarse", "Look from scan pose"), *_GRASP_STEPS]
# Click mode: the user clicks the object in the live view; the VLM only names it for confirmation.
STEPS_CLICK: list[tuple[str, str]] = [
    ("home", "Move to rest"), ("scan", "Move to scan pose"), ("pick", "Click the object"),
    ("identify", "Identify and confirm"), *_GRASP_STEPS,
]
RELOCATE_MAX_JUMP_M = 0.04  # closer look: a VLM answer further than this from the first estimate is another object

OPEN_PCT = 100.0  # = the user's max safe opening (gripper calibration range_min)
DESCEND_SPEED = 15.0


class Reporter(Protocol):
    def step(self, step_id: str, status: str, message: str = "", detail: dict | None = None,
             images: list[Path] | None = None) -> None:
        """status: active | done | failed | skipped | stopped"""

    def log(self, message: str) -> None: ...

    def confirm(self, kind: str, prompt: str, images: list[Path] | None = None) -> bool:
        """kind: 'start' (before torque) or 'grasp' (after planning). Return True to go ahead."""

    def choose(self, kind: str, title: str, prompt: str, options: list[tuple[str, str]],
               images: list[Path] | None = None) -> str:
        """Ask the user to pick one of `options` [(value, label)]; return the value ('cancel' if stopped)."""

    def pick(self, prompt: str) -> np.ndarray | None:
        """Wait for the user to click a point in the live (rectified) camera view; None if cancelled."""


class Abort(Exception):
    pass


@dataclass
class RunResult:
    out_dir: Path
    result: str  # held | missed | aborted: ... | stopped | estopped | error: ...
    log: dict


def _jsonable(o):
    return np.asarray(o).tolist()


class _RecordingReporter:
    """Passes everything to the real reporter and notes steps and operator answers in the episode."""

    def __init__(self, rep: Reporter, rec: recording.EpisodeRecorder):
        self.rep, self.rec = rep, rec

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


def overlay_grasp(img: np.ndarray, est: ObjectEstimate, rect: Rectifier, target: np.ndarray,
                  closing_xy: np.ndarray | None, axis_xy: np.ndarray | None) -> np.ndarray:
    """Planned fingertip point (yellow), jaw closing line (red) and object long axis (blue)."""
    Tinv = np.linalg.inv(est.T_base_cam)

    def px(p):
        c = Tinv @ np.append(p, 1.0)
        return tuple(int(v) for v in rect.K[:2, :2] @ (c[:2] / c[2]) + rect.K[:2, 2])

    for vec, color, half in ((axis_xy, (255, 120, 0), 0.04), (closing_xy, (0, 0, 255), 0.025)):
        if vec is not None:
            d = np.append(vec, 0.0) * half
            cv2.line(img, px(target - d), px(target + d), color, 3)
    cv2.circle(img, px(target), 6, (0, 255, 255), -1)
    return img


def render_plan(poses: list[tuple[str, np.ndarray, float]]) -> np.ndarray:
    from ..viz import render

    rows = []
    for name, q, g in poses:
        img = render(q, gripper_pct=g, size=(480, 360),
                     cameras=((90, -5, "side"), (135, -25, "3/4"), (90, -89, "top (arm faces right)")))
        cv2.putText(img, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        rows.append(img)
    return np.vstack(rows)


class GraspRunner:
    def __init__(self, cfg: RobotConfig, kin: SO101Kinematics, cam_model: FisheyeCamera, rect: Rectifier,
                 planner: TopDownPlanner, locator: Locator, reporter: Reporter, speed_deg_s: float = 30.0):
        self.cfg, self.kin, self.cam_model, self.rect = cfg, kin, cam_model, rect
        self.planner, self.locator, self.rep = planner, locator, reporter
        self.speed = speed_deg_s
        self._gripper_pct = 0.0  # opening when the last photo was taken (dry runs: closed)
        self.scan, self.rest = cfg.poses["scan"], cfg.poses["rest"]

    # ------------------------------------------------------------------ helpers
    def _new_run_dir(self, prefix: str = "grasp") -> Path:
        out = RUNS_DIR / time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=False)
        return out

    def _locate(self, img: np.ndarray, q: np.ndarray, query: str, out: Path, tag: str) -> tuple[ObjectEstimate, dict]:
        cv2.imwrite(str(out / f"{tag}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        try:
            est, dbg = self.locator.locate(img, q, query, gripper_pct=self._gripper_pct)
        except PerceptionError as e:
            raise Abort(str(e)) from e
        cv2.imwrite(str(out / f"{tag}_locate.png"), draw_debug(est, dbg))
        return est, dbg

    def _est_detail(self, est: ObjectEstimate) -> dict:
        c = est.center_base * 100
        return {"centre_cm": [round(float(c[0]), 1), round(float(c[1]), 1)], "top_cm": round(est.top_z_m * 100, 1),
                "table_fit_mm": round(est.fit.rmse_m * 1000, 1), "warnings": est.warnings}

    def _plan(self, est: ObjectEstimate, from_q: np.ndarray, dbg: dict, out: Path, tag: str) -> tuple[GraspPlan, dict]:
        top = est.points_base[est.points_base[:, 2] > est.top_z_m - 0.01]
        axis = object_axis_xy(top) if len(top) else None
        target = np.array([est.center_base[0], est.center_base[1], grasp_height(est.top_z_m)])
        try:
            self.cfg.check_in_workspace(target)
            plan = self.planner.plan_grasp(target, axis, from_q)
        except (ValueError, GraspPlanningError) as e:
            raise Abort(f"cannot grasp: {e}") from e
        cv2.imwrite(str(out / f"{tag}_grasp_overlay.jpg"),
                    overlay_grasp(dbg["rect_bgr"].copy(), est, self.rect, target, plan.closing_xy, axis))
        detail = {
            "fingertips_cm": [round(float(v) * 100, 1) for v in target],
            "jaw_alignment": "object not elongated, any wrist angle" if axis is None
            else f"{plan.misalignment_deg:.0f} deg from across the long axis",
            "wrist_roll_deg": round(float(plan.grasp_deg[4]), 0),
            "pregrasp_deg": [round(float(v), 1) for v in plan.pregrasp_deg],
            "grasp_deg": [round(float(v), 1) for v in plan.grasp_deg],
            "lift_deg": [round(float(v), 1) for v in plan.lift_deg],
        }
        raw = {"target_m": target, "object_axis_xy": axis, "closing_xy": plan.closing_xy,
               "misalignment_deg": plan.misalignment_deg, "pregrasp_deg": plan.pregrasp_deg,
               "grasp_deg": plan.grasp_deg, "lift_deg": plan.lift_deg}
        return plan, {"detail": detail, "raw": raw}

    def _project(self, est: ObjectEstimate, T_base_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Previous estimate's top-surface centre and padded point-cloud box, in a new rectified view."""
        Tinv = np.linalg.inv(T_base_cam)
        P = np.c_[est.points_base, np.ones(len(est.points_base))] @ Tinv.T
        c = Tinv @ np.append(est.center_base, 1.0)
        if c[2] <= 0 or np.any(P[:, 2] <= 0):
            return None
        uv = P[:, :2] / P[:, 2:] @ self.rect.K[:2, :2].T + self.rect.K[:2, 2]
        pt = self.rect.K[:2, :2] @ (c[:2] / c[2]) + self.rect.K[:2, 2]
        lo, hi = uv.min(0), uv.max(0)
        pad = 0.25 * (hi - lo) + 10
        w, h = self.rect.size
        box = np.clip(np.r_[lo - pad, hi + pad], 0, [w - 1, h - 1, w - 1, h - 1])
        return pt, box

    def _relocate(self, img: np.ndarray, q: np.ndarray, name: str, prev: ObjectEstimate, out: Path, tag: str):
        """Find the same object again from a new pose: VLM by name, accepted only if it lands near the
        previous 3D estimate; otherwise SAM at the previous estimate projected into this view."""
        cv2.imwrite(str(out / f"{tag}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        how = "VLM"
        try:
            est, dbg = self.locator.locate(img, q, name, gripper_pct=self._gripper_pct)
            jump = float(np.linalg.norm(est.center_base[:2] - prev.center_base[:2]))
            if jump > RELOCATE_MAX_JUMP_M:
                raise PerceptionError(f"VLM answer is {jump * 100:.0f} cm from the first estimate")
        except PerceptionError as e:
            proj = self._project(prev, self.cam_model.pose_in_base(self.kin, q))
            if proj is None:
                raise Abort(f"lost the object: {e}") from e
            try:
                est, dbg = self.locator.locate_at(img, q, proj[0], proj[1], label=name,
                                                  gripper_pct=self._gripper_pct)
            except PerceptionError as e2:
                raise Abort(f"lost the object: {e}; projection: {e2}") from e2
            how = f"projected first estimate ({e})"
        cv2.imwrite(str(out / f"{tag}_locate.png"), draw_debug(est, dbg))
        return est, dbg, how

    def _identify_image(self, dbg: dict, est: ObjectEstimate, click: np.ndarray, out: Path) -> Path:
        from ..perception.pointer import describe_crop

        rgb = cv2.cvtColor(dbg["rect_bgr"], cv2.COLOR_BGR2RGB)
        crop = cv2.cvtColor(describe_crop(rgb, est.mask), cv2.COLOR_RGB2BGR)
        full = dbg["rect_bgr"].copy()
        ov = full.copy()
        ov[est.mask] = (0, 200, 0)
        full = cv2.addWeighted(ov, 0.45, full, 0.55, 0)
        cv2.drawMarker(full, tuple(int(v) for v in click), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        h = full.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * h / crop.shape[0]), h))
        path = out / "identify.jpg"
        cv2.imwrite(str(path), np.hstack([full, crop]))
        return path

    # ------------------------------------------------------------------ dry run
    def dry_run(self, scene_dir: Path, query: str) -> RunResult:
        """Plan from a saved scene (runs/scene_*): coarse locate, re-look pose, grasp plan, render. No arm."""
        out = self._new_run_dir()
        log: dict = {"query": query, "dry_run": str(scene_dir)}
        try:
            q = np.asarray(json.loads((scene_dir / "scene.json").read_text())["joints_deg"])
            img = cv2.imread(str(scene_dir / "image.jpg"))
            if img is None:
                raise Abort(f"cannot read {scene_dir / 'image.jpg'}")
            self.rep.step("coarse", "active")
            est, dbg = self._locate(img, q, query, out, "coarse")
            self.rep.step("coarse", "done", detail=self._est_detail(est), images=[out / "coarse_locate.png"])
            self.rep.step("relook", "active")
            relook = self.planner.plan_relook(est.center_base * [1, 1, 0], q)
            self.rep.step("relook", "done", "planned only (dry run)", {"joints_deg": [round(float(v), 1) for v in relook]})
            self.rep.step("plan", "active")
            plan, p = self._plan(est, relook, dbg, out, "coarse")
            cv2.imwrite(str(out / "plan.png"), render_plan([
                ("1 scan", q, 0.0), ("2 re-look", relook, 0.0), ("3 pregrasp (open)", plan.pregrasp_deg, OPEN_PCT),
                ("4 grasp", plan.grasp_deg, OPEN_PCT), ("5 lift (closed)", plan.lift_deg, 0.0)]))
            self.rep.step("plan", "done", detail=p["detail"], images=[out / "coarse_grasp_overlay.jpg", out / "plan.png"])
            log.update(relook_deg=relook, coarse_plan=p["raw"], result="dry run")
        except (Abort, GraspPlanningError) as e:
            self.rep.log(f"ABORT: {e}")
            log["result"] = f"aborted: {e}"
        (out / "grasp.json").write_text(json.dumps(log, indent=1, default=_jsonable))
        return RunResult(out, log["result"], log)

    def _pick_and_identify(self, arm, camera, snap, out: Path, log: dict, begin):
        """Click mode: wait for a click, mask + measure + name the object, ask the user to confirm.
        Loops on 'pick again' and on clicks that are not graspable objects. Returns (estimate, name, q)."""
        message = "Click the object to grasp in the camera view"
        attempt = 0
        while True:
            begin("pick")
            self.rep.step("pick", "active", message)
            click = self.rep.pick(message)
            if click is None:
                raise Abort("no object picked")
            attempt += 1
            self.rep.step("pick", "done", f"clicked ({click[0]:.0f}, {click[1]:.0f})")
            begin("identify")
            img, q = snap(None, f"pick{attempt}")
            cv2.imwrite(str(out / f"pick{attempt}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            try:
                est, dbg = self.locator.locate_at(img, q, click, gripper_pct=self._gripper_pct)
                rgb = cv2.cvtColor(dbg["rect_bgr"], cv2.COLOR_BGR2RGB)
                name = self.locator.pointer.describe(rgb, est.mask)
            except PerceptionError as e:
                self.rep.step("identify", "failed", str(e))
                message = f"{e}. Click again."
                continue
            img_path = self._identify_image(dbg, est, click, out)
            c = est.center_base * 100
            detail = {"looks_like": name, **self._est_detail(est)}
            self.rep.step("identify", "active", "waiting for confirmation", detail, [img_path])
            log.setdefault("picks", []).append({"click_px": click, "name": name, "center_m": est.center_base,
                                                "top_m": est.top_z_m})
            height = (f"about {est.top_z_m * 100:.1f} cm tall" if est.top_z_m >= 0.005
                      else "flat or height unclear (checked again on the closer look)")
            text = f'This looks like "{name}" at ({c[0]:.1f}, {c[1]:.1f}) cm from the robot base, {height}.'
            options = [("grasp", "Yes, grasp it"), ("retry", "Pick again"), ("cancel", "Cancel")]
            try:
                self.cfg.check_in_workspace(np.array([est.center_base[0], est.center_base[1], grasp_height(est.top_z_m)]))
                text += " Grasp it?"
            except ValueError as e:
                text += f" It is outside the reachable area ({e}); move it closer or pick another object."
                options = options[1:]
                detail["reachable"] = False
            answer = self.rep.choose("identify", "Is this the object?", text, options, [img_path])
            if answer == "grasp":
                self.rep.step("identify", "done", f'confirmed: "{name}"', detail)
                return est, name, q
            if answer == "retry":
                self.rep.step("identify", "skipped", "picking again")
                message = "Click the object to grasp in the camera view"
                continue
            raise Abort("cancelled at identification")

    # ------------------------------------------------------------------ recording
    def _start_recording(self, out: Path, arm, camera, log: dict) -> recording.EpisodeRecorder | None:
        """Start the episode recorder; a failure is reported but does not block the run."""
        if not self.cfg.recording.enabled:
            return None
        rec = recording.EpisodeRecorder(out, arm, camera, self.cfg.recording)
        try:
            rec.start()
        except Exception as e:  # noqa: BLE001
            log["recording_error"] = repr(e)
            self.rep.log(f"WARNING: episode recording failed to start, this run is not recorded: {e!r}")
            logger.exception("episode recorder failed to start")
            return None
        return rec

    def _finish_recording(self, rec: recording.EpisodeRecorder, log: dict, query: str | None) -> None:
        try:
            rec.stop()
            obj = log.get("identified_as") or query
            perception = (yaml.safe_load((CONFIG_DIR / "robot.yaml").read_text()) or {}).get("perception") or {}
            rec.finish({
                "task": {"type": "pick", "mode": log.get("mode"), "query": query, "object": obj,
                         "instruction": f"pick up the {obj}" if obj else None},
                "outcome": recording.outcome_of(log.get("result", "error: no result"), rec.events),
                "label": {"success": None, "note": None, "labeled_at": None},
                "grasp": log,
                "robot": {"type": "so101_follower", "gripper": "roboninecom SO-ARM100-101-102 parallel gripper",
                          "joint_offsets_deg": self.cfg.joint_offsets_deg, "tip_xyz_m": self.cfg.tip_xyz_m,
                          "gripper_open_gap_m": self.cfg.gripper_open_gap_m, "workspace_m": self.cfg.workspace,
                          "speed_deg_s": self.speed, "poses_deg": {"rest": self.rest, "scan": self.scan}},
                "perception": perception,
                "software": recording.git_version(),
            }, cam_model=self.cam_model)
            self.rep.log(f"recorded {rec.num_frames} samples ({rec.frames_with_image} with images) to "
                         f"{rec.out_dir.name}/{recording.EPISODE_FILE}"
                         + (f"; WRITE ERROR {rec.write_error}" if rec.write_error else ""))
        except Exception as e:  # noqa: BLE001 - the run itself is over; report and move on
            self.rep.log(f"WARNING: saving the episode failed: {e!r}")
            logger.exception("saving the episode failed")

    # ------------------------------------------------------------------ live
    def run(self, arm, camera, query: str | None, keep: bool = False) -> RunResult:
        """Full grasp on the arm. `query` names the object (prompt mode); None = the user clicks it in the
        live view (click mode, see STEPS_CLICK). Returns with the arm at rest and torque off (after an
        E-stop: torque off wherever the arm is)."""
        out = self._new_run_dir()
        log: dict = {"query": query, "mode": "prompt" if query else "click",
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        current = {"step": None}
        home_via_scan = True  # until a plan exists; plans are checked for direct relook/lift -> rest paths

        def save():
            (out / "grasp.json").write_text(json.dumps(log, indent=1, default=_jsonable))

        def begin(step_id: str, target_deg: np.ndarray | None = None):
            """Mark a step active; `target_deg` (the pose it moves to) lets a UI draw a ghost of it."""
            current["step"] = step_id
            self.rep.step(step_id, "active", detail=None if target_deg is None
                          else {"target_deg": [round(float(v), 1) for v in target_deg]})

        def snap(q_cmd: np.ndarray | None, tag: str) -> tuple[np.ndarray, np.ndarray]:
            if q_cmd is not None:
                arm.move_to(q_cmd, speed_deg_s=self.speed, settle_timeout_s=3.0)
                time.sleep(1.0)  # auto exposure settles after the move
            img = camera.read_fresh(0.3)
            q = arm.read_joints()
            self._gripper_pct = float(arm.read_gripper())  # the gripper's outline in the image depends on it
            log[f"{tag}_joints_deg"] = q
            log[f"{tag}_gripper_pct"] = self._gripper_pct
            return img, q

        what = f'"{query}"' if query else "the object you click"
        if not self.rep.confirm("start", f"Find and grasp {what}? rest -> scan -> closer look; asks again before "
                                         "grasping. Hand near the power switch."):
            log["result"] = "declined"
            save()
            return RunResult(out, "declined", log)

        rec = self._start_recording(out, arm, camera, log)
        rep = self.rep
        if rec is not None:
            self.rep = _RecordingReporter(rep, rec)
        try:
            arm.clear_stop()
            arm.hold()
            arm.bus.enable_torque()
            try:
                try:
                    begin("home", self.rest)
                    arm.move_to(self.rest, gripper=0.0, speed_deg_s=self.speed)
                    self.rep.step("home", "done")

                    if query:
                        begin("coarse", self.scan)
                        img, q = snap(self.scan, "coarse")
                        est, _ = self._locate(img, q, query, out, "coarse")
                        name = query
                        self.rep.step("coarse", "done", detail=self._est_detail(est), images=[out / "coarse_locate.png"])
                    else:
                        begin("scan", self.scan)
                        arm.move_to(self.scan, speed_deg_s=self.speed, settle_timeout_s=3.0)
                        self.rep.step("scan", "done")
                        est, name, q = self._pick_and_identify(arm, camera, snap, out, log, begin)
                        log["identified_as"] = name

                    begin("relook")
                    try:
                        relook = self.planner.plan_relook(est.center_base * [1, 1, 0], q)
                    except GraspPlanningError as e:
                        raise Abort(str(e)) from e
                    self.rep.step("relook", "active", detail={"target_deg": [round(float(v), 1) for v in relook]})
                    img, q = snap(relook, "relook")
                    est, dbg, how = self._relocate(img, q, name, est, out, "relook")
                    self.rep.step("relook", "done", detail={**self._est_detail(est), "found_by": how},
                                  images=[out / "relook_locate.png"])

                    begin("plan")
                    plan, p = self._plan(est, q, dbg, out, "relook")
                    cv2.imwrite(str(out / "plan.png"), render_plan([
                        ("pregrasp (open)", plan.pregrasp_deg, OPEN_PCT), ("grasp", plan.grasp_deg, OPEN_PCT),
                        ("lift (closed)", plan.lift_deg, 0.0)]))
                    log["plan"] = p["raw"]
                    save()
                    images = [out / "relook_grasp_overlay.jpg", out / "plan.png"]
                    self.rep.step("plan", "done", detail=p["detail"], images=images)
                    home_via_scan = False

                    begin("approve")
                    if not self.rep.confirm("grasp", "Grasp? Yellow = fingertip point, red = jaw closing line, "
                                                     "blue = object axis.", images):
                        raise Abort("grasp declined")
                    self.rep.step("approve", "done")

                    begin("pregrasp", plan.pregrasp_deg)
                    arm.move_to(plan.pregrasp_deg, gripper=OPEN_PCT, speed_deg_s=self.speed, settle_timeout_s=3.0)
                    self.rep.step("pregrasp", "done")

                    begin("descend", plan.grasp_deg)
                    arm.move_to(plan.grasp_deg, speed_deg_s=DESCEND_SPEED, settle_timeout_s=3.0)
                    tip = self.kin.fk(arm.read_joints())[:3, 3]
                    log["grasp_tip_measured_m"] = tip
                    self.rep.step("descend", "done", detail={"fingertips_cm": [round(float(v) * 100, 1) for v in tip]})

                    begin("close")
                    arm.set_gripper(0.0, duration_s=1.0)
                    cpos, cload = read_grip(arm)
                    self.rep.step("close", "done", detail={"opening_pct": round(cpos, 1), "load": cload,
                                                           "squeezing": is_holding(cpos, cload)})

                    begin("lift", plan.lift_deg)
                    arm.move_to(plan.lift_deg, speed_deg_s=DESCEND_SPEED, settle_timeout_s=3.0)
                    lpos, lload = read_grip(arm)
                    held = is_holding(lpos, lload)
                    cv2.imwrite(str(out / "after_lift.jpg"), camera.read_fresh(0.3))
                    log.update(gripper_after_close={"open_pct": cpos, "load": cload},
                               gripper_after_lift={"open_pct": lpos, "load": lload}, held=held)
                    save()
                    verdict = "HOLDING the object" if held else "MISSED (jaws closed on nothing)"
                    self.rep.step("lift", "done" if held else "failed", verdict,
                                  {"opening_pct": round(lpos, 1), "load": lload, "held": held}, [out / "after_lift.jpg"])

                    if held and not keep:
                        begin("place", plan.grasp_deg)
                        arm.move_to(plan.grasp_deg, speed_deg_s=DESCEND_SPEED, settle_timeout_s=3.0)
                        arm.set_gripper(OPEN_PCT, duration_s=0.8)
                        arm.move_to(plan.lift_deg, speed_deg_s=DESCEND_SPEED, settle_timeout_s=3.0)
                        arm.set_gripper(0.0, duration_s=0.6)
                        self.rep.step("place", "done")
                    else:
                        self.rep.step("place", "skipped", "kept in the gripper" if held else "nothing to put back")
                    if not held:
                        arm.set_gripper(0.0, duration_s=0.6)
                    log["result"] = "held" if held else "missed"
                except Abort as e:
                    self.rep.step(current["step"], "failed", str(e))
                    log["result"] = f"aborted: {e}"
                except (MotionStopped, KeyboardInterrupt):
                    if getattr(arm, "estopped", False):
                        raise
                    self.rep.step(current["step"], "stopped", "stopped by user")
                    log["result"] = "stopped"
                except Exception as e:
                    if getattr(arm, "estopped", False):
                        raise MotionStopped("emergency stop") from e
                    self.rep.step(current["step"], "failed", repr(e))
                    log["result"] = f"error: {e!r}"
                    home_via_scan = True  # unknown state: go up first
                    logger.exception("grasp run failed")
                save()
                # Return to rest. A stop request that ended the run above is consumed here, so only a stop
                # pressed *during* the return cuts torque where the arm is. After an E-stop, do not move again.
                if getattr(arm, "estopped", False):
                    raise MotionStopped("emergency stop")
                arm.clear_stop()
                begin("return", self.rest)
                if home_via_scan and current["step"] != "home":
                    arm.move_to(self.scan, speed_deg_s=min(self.speed, 30.0), settle_timeout_s=3.0)
                arm.move_to(self.rest, speed_deg_s=min(self.speed, 30.0), settle_timeout_s=3.0)
                self.rep.step("return", "done")
            except (MotionStopped, KeyboardInterrupt):
                est = getattr(arm, "estopped", False)
                log["result"] = "estopped" if est else log.get("result", "stopped")
                if current["step"]:
                    self.rep.step(current["step"], "stopped", "E-STOP: torque off" if est else
                                  "stopped again: torque off where the arm is")
                self.rep.log("E-STOP: torque off" if est else "stopped while returning: torque disabled where the arm is")
            finally:
                try:
                    arm.bus.disable_torque(num_retry=5)
                except Exception as e:  # noqa: BLE001 - report, the caller still disconnects
                    self.rep.log(f"failed to disable torque: {e!r}")
                save()
        finally:
            self.rep = rep
            if rec is not None:
                self._finish_recording(rec, log, query)
        self.rep.log(f"result: {log['result']} ({out.name})")
        return RunResult(out, log["result"], log)
