"""The grasp sequence, shared by the CLI (scripts/grasp.py) and the UI (scripts/ui.py).

Built from skills (src/so101_zero_shot/skills): this module only decides the order and reports each step
(STEPS) as it starts and finishes through a Reporter, with the numbers and images behind each decision;
approvals go through Reporter.confirm. Torque, stopping, the way home and episode recording are handled
by skills.Session (see there for Stop / E-stop behaviour).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import RUNS_DIR, RobotConfig
from ..kinematics import SO101Kinematics
from ..perception.geometry import FisheyeCamera, Rectifier
from ..perception.locate import Locator, ObjectEstimate, PerceptionError
from ..skills import (OPEN_PCT, Held, Reporter, Session, SkillFailed, Skills, deg_list, est_detail, jsonable,
                      new_run_dir, overlay_grasp)
from ..skills.core import DESCEND_SPEED, RELOCATE_MAX_JUMP_M  # noqa: F401 - re-exported
from ..scenes import SceneError, load_scene
from ..skills.motion_frames import LookSettings
from .topdown import GraspPlanningError, TopDownPlanner, grasp_height

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

Abort = SkillFailed  # older name
__all__ = ["Abort", "GraspRunner", "OPEN_PCT", "Reporter", "RunResult", "STEPS", "STEPS_CLICK", "overlay_grasp",
           "render_plan"]


@dataclass
class RunResult:
    out_dir: Path
    result: str  # held | missed | aborted: ... | stopped | estopped | error: ...
    log: dict


def render_plan(poses: list[tuple[str, np.ndarray, float]]) -> np.ndarray:
    from ..viz import render

    rows = []
    for name, q, g in poses:
        img = render(q, gripper_pct=g, size=(480, 360),
                     cameras=((90, -5, "side"), (135, -25, "3/4"), (90, -89, "top (arm faces right)")))
        cv2.putText(img, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        rows.append(img)
    return np.vstack(rows)


def identify_image(dbg: dict, est: ObjectEstimate, click: np.ndarray, out: Path) -> Path:
    """The clicked object's mask on the full view, next to the crop the VLM was asked to name."""
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


class GraspRunner:
    def __init__(self, cfg: RobotConfig, kin: SO101Kinematics, cam_model: FisheyeCamera, rect: Rectifier,
                 planner: TopDownPlanner, locator: Locator, reporter: Reporter, speed_deg_s: float = 30.0,
                 environment: str = "real", runs_dir: Path | None = None,
                 look_settings: LookSettings | None = None):
        """environment: real | sim (MuJoCo), stored with each episode; runs_dir:
        where run folders go (default runs/)."""
        self.cfg, self.kin, self.cam_model, self.rect = cfg, kin, cam_model, rect
        self.environment, self.runs_dir = environment, runs_dir
        self.planner, self.locator, self.rep = planner, locator, reporter
        self.speed = speed_deg_s
        self.look_settings = look_settings or LookSettings()

    def skills(self, arm, camera, out: Path, log: dict) -> Skills:
        return Skills(self.cfg, self.kin, self.cam_model, self.rect, self.planner, self.locator, arm, camera, out,
                      log, self.speed, look_settings=self.look_settings)

    def _new_run_dir(self, prefix: str = "grasp") -> Path:
        return new_run_dir(self.runs_dir or RUNS_DIR, prefix)

    # ------------------------------------------------------------------ dry run
    def dry_run(self, scene_dir: Path, query: str) -> RunResult:
        """Plan from a saved scene (runs/scene_*): coarse locate, re-look pose, grasp plan, render. No arm."""
        out = self._new_run_dir()
        log: dict = {"query": query, "dry_run": str(scene_dir)}
        sk = self.skills(None, None, out, log)
        try:
            try:
                img, q, views, _ = load_scene(scene_dir)
            except SceneError as e:
                raise SkillFailed(str(e)) from e
            self.rep.step("coarse", "active")
            est, dbg = sk.locate(img, q, query, "coarse", views)
            self.rep.step("coarse", "done", detail=est_detail(est), images=[out / "coarse_locate.png"])
            self.rep.step("relook", "active")
            relook = sk.plan_relook(est, q)
            self.rep.step("relook", "done", "planned only (dry run)", {"joints_deg": deg_list(relook)})
            self.rep.step("plan", "active")
            plan, p = sk.plan_grasp(est, relook, dbg, "coarse")
            cv2.imwrite(str(out / "plan.png"), render_plan([
                ("1 scan", q, 0.0), ("2 re-look", relook, 0.0), ("3 pregrasp (open)", plan.pregrasp_deg, OPEN_PCT),
                ("4 grasp", plan.grasp_deg, OPEN_PCT), ("5 lift (closed)", plan.lift_deg, 0.0)]))
            self.rep.step("plan", "done", detail=p["detail"], images=[out / "coarse_grasp_overlay.jpg", out / "plan.png"])
            log.update(relook_deg=relook, coarse_plan=p["raw"], result="dry run")
        except (SkillFailed, GraspPlanningError) as e:
            self.rep.log(f"ABORT: {e}")
            log["result"] = f"aborted: {e}"
        (out / "grasp.json").write_text(json.dumps(log, indent=1, default=jsonable))
        return RunResult(out, log["result"], log)

    # ------------------------------------------------------------------ click mode
    def _pick_and_identify(self, s: Session, sk: Skills):
        """Click mode: wait for a click, mask + measure + name the object, ask the user to confirm.
        Loops on 'pick again' and on clicks that are not graspable objects. Returns (estimate, name, q)."""
        rep, out = s.rep, s.out
        message = "Click the object to grasp in the camera view"
        attempt = 0
        while True:
            s.begin("pick")
            rep.step("pick", "active", message)
            click = rep.pick(message)
            if click is None:
                raise SkillFailed("no object picked")
            attempt += 1
            rep.step("pick", "done", f"clicked ({click[0]:.0f}, {click[1]:.0f})")
            s.begin("identify")
            img, q, views = sk.look(None, f"pick{attempt}")
            sk.save_photo(img, f"pick{attempt}")
            try:
                est, dbg = sk.locate_at(img, q, click, views=views)
                rgb = cv2.cvtColor(dbg["rect_bgr"], cv2.COLOR_BGR2RGB)
                name = self.locator.pointer.describe(rgb, est.mask)
            except PerceptionError as e:
                rep.step("identify", "failed", str(e))
                message = f"{e}. Click again."
                continue
            img_path = identify_image(dbg, est, click, out)
            c = est.center_base * 100
            detail = {"looks_like": name, **est_detail(est)}
            rep.step("identify", "active", "waiting for confirmation", detail, [img_path])
            s.log.setdefault("picks", []).append({"click_px": click, "name": name, "center_m": est.center_base,
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
            answer = rep.choose("identify", "Is this the object?", text, options, [img_path])
            if answer == "grasp":
                rep.step("identify", "done", f'confirmed: "{name}"', detail)
                return est, name, q
            if answer == "retry":
                rep.step("identify", "skipped", "picking again")
                message = "Click the object to grasp in the camera view"
                continue
            raise SkillFailed("cancelled at identification")

    # ------------------------------------------------------------------ grasp steps (shared with the agent)
    def grasp_steps(self, s: Session, sk: Skills, est: ObjectEstimate, name: str, q: np.ndarray, keep: bool,
                    approve: bool = True, place_back: bool = True) -> bool:
        """Look closer, plan, approve, open + pregrasp, descend, close, lift; then put it back unless `keep`.
        Reports the _GRASP_STEPS (except 'return'). Returns whether the object is held. On a hold with
        keep=True, sk.held says what is in the gripper."""
        rep, out, log = s.rep, s.out, s.log
        s.begin("relook")
        relook = sk.plan_relook(est, q)
        rep.step("relook", "active", detail={"target_deg": deg_list(relook)})
        img, q, views = sk.look(relook, "relook")  # other views: the photo est came from + frames on the way
        est, dbg, how = sk.relocate(img, q, name, est, "relook", views)
        rep.step("relook", "done", detail={**est_detail(est), "found_by": how}, images=[out / "relook_locate.png"])

        s.begin("plan")
        plan, p = sk.plan_grasp(est, q, dbg, "relook")
        cv2.imwrite(str(out / "plan.png"), render_plan([
            ("pregrasp (open)", plan.pregrasp_deg, OPEN_PCT), ("grasp", plan.grasp_deg, OPEN_PCT),
            ("lift (closed)", plan.lift_deg, 0.0)]))
        log["plan"] = p["raw"]
        s.save()
        images = [out / "relook_grasp_overlay.jpg", out / "plan.png"]
        rep.step("plan", "done", detail=p["detail"], images=images)
        s.home_via_scan = False

        s.begin("approve")
        if approve and not rep.confirm("grasp", "Grasp? Yellow = fingertip point, red = jaw closing line, "
                                                "blue = object axis.", images):
            raise SkillFailed("grasp declined")
        rep.step("approve", "done" if approve else "skipped")

        s.begin("pregrasp", plan.pregrasp_deg)
        sk.pregrasp(plan)
        rep.step("pregrasp", "done")

        s.begin("descend", plan.grasp_deg)
        tip = sk.descend(plan)
        log["grasp_tip_measured_m"] = tip
        rep.step("descend", "done", detail={"fingertips_cm": [round(float(v) * 100, 1) for v in tip]})

        s.begin("close")
        c = sk.close()
        rep.step("close", "done", detail={"opening_pct": round(c.open_pct, 1), "load": c.load,
                                          "squeezing": c.squeezing})

        s.begin("lift", plan.lift_deg)
        g = sk.lift(plan)
        held = g.squeezing
        cv2.imwrite(str(out / "after_lift.jpg"), sk.camera.read_fresh(0.3))
        log.update(gripper_after_close={"open_pct": c.open_pct, "load": c.load},
                   gripper_after_lift={"open_pct": g.open_pct, "load": g.load}, held=held)
        s.save()
        verdict = "HOLDING the object" if held else "MISSED (jaws closed on nothing)"
        rep.step("lift", "done" if held else "failed", verdict,
                 {"opening_pct": round(g.open_pct, 1), "load": g.load, "held": held}, [out / "after_lift.jpg"])
        if held:
            sk.held = Held(name, plan, est.center_base.copy(), est.top_z_m)

        if held and not keep and place_back:
            s.begin("place", plan.grasp_deg)
            sk.place_back(plan)
            rep.step("place", "done")
        elif place_back:
            rep.step("place", "skipped", "kept in the gripper" if held else "nothing to put back")
        if not held:
            sk.arm.set_gripper(0.0, duration_s=0.6)
        return held

    # ------------------------------------------------------------------ live
    def run(self, arm, camera, query: str | None, keep: bool = False, record: bool | None = None) -> RunResult:
        """Full grasp on the arm. `query` names the object (prompt mode); None = the user clicks it in the
        live view (click mode, see STEPS_CLICK). `record`: save an episode (None = config
        recording.enabled). Returns with the arm at rest and torque off (after an E-stop: torque off
        wherever the arm is)."""
        out = self._new_run_dir()
        log: dict = {"query": query, "mode": "prompt" if query else "click", "environment": self.environment,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        s = Session(self.cfg, self.rep, out, log, self.environment, self.speed, self.cam_model)

        what = f'"{query}"' if query else "the object you click"
        if not self.rep.confirm("start", f"Find and grasp {what}? rest -> scan -> closer look; asks again before "
                                         "grasping. Hand near the power switch."):
            log["result"] = "declined"
            s.save()
            return RunResult(out, "declined", log)

        def body(s: Session, sk: Skills) -> None:
            s.begin("home", sk.scan)
            sk.home()
            s.rep.step("home", "done")
            if query:
                s.begin("coarse", sk.scan)
                img, q, views = sk.look(sk.scan, "coarse")
                est, _ = sk.locate(img, q, query, "coarse", views)
                name = query
                s.rep.step("coarse", "done", detail=est_detail(est), images=[out / "coarse_locate.png"])
            else:
                s.begin("scan", sk.scan)
                sk.go_scan()
                s.rep.step("scan", "done")
                est, name, q = self._pick_and_identify(s, sk)
                log["identified_as"] = name
            held = self.grasp_steps(s, sk, est, name, q, keep)
            log["result"] = "held" if held else "missed"

        def task() -> dict:
            obj = log.get("identified_as") or query
            return {"type": "pick", "mode": log.get("mode"), "query": query, "object": obj,
                    "instruction": f"pick up the {obj}" if obj else None}

        s.execute(arm, camera, self.skills(arm, camera, out, log),
                  self.cfg.recording.enabled if record is None else record, body, task)
        self.rep.log(f"result: {log['result']} ({out.name})")
        return RunResult(out, log["result"], log)
