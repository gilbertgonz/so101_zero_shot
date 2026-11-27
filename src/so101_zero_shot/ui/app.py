"""Local web UI: live wrist camera, live arm view, prompt, step-by-step plan panel, approvals, stop.

Threads: models load in the background at startup; the arm monitor reads joints at 10 Hz (the bus is
locked, so this is safe during motion); the robot view renders the current pose (plus a ghost of the
pose being moved to) at 10 Hz; each grasp runs in its own thread. The page gets state over
server-sent events and video as MJPEG.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..config import RUNS_DIR, RobotConfig
from ..grasping.runner import STEPS, STEPS_CLICK, GraspRunner

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
FRAME_PERIOD_S = 1 / 15


# ---------------------------------------------------------------------------------------------- state
@dataclass
class UiState:
    status: dict = field(default_factory=lambda: {"models": "loading", "arm": "connecting", "camera": "connecting"})
    busy: bool = False
    query: str = ""
    run_dir: str = ""
    result: str = ""
    steps: list = field(default_factory=list)
    logs: deque = field(default_factory=lambda: deque(maxlen=200))
    confirm: dict | None = None
    pick: dict | None = None  # waiting for a click in the camera view: {"id", "prompt"}
    mode: str = "prompt"
    target_deg: list | None = None
    joints_deg: list | None = None
    gripper_pct: float | None = None
    torque: bool = False
    environment: str = "real"  # real | sim (MuJoCo)
    recording: bool = True  # record an episode for the next run
    version: int = 0

    def snapshot(self) -> dict:
        return {k: (list(v) if isinstance(v, deque) else v) for k, v in self.__dict__.items()}


class Hub:
    """Thread-safe UiState plus change notification for the SSE stream."""

    def __init__(self):
        self.state = UiState()
        self.cond = threading.Condition()
        self.reset_steps()

    def update(self, **kw) -> None:
        with self.cond:
            for k, v in kw.items():
                setattr(self.state, k, v)
            self.state.version += 1
            self.cond.notify_all()

    def mutate(self, fn) -> None:
        with self.cond:
            fn(self.state)
            self.state.version += 1
            self.cond.notify_all()

    def reset_steps(self, steps: list[tuple[str, str]] = STEPS) -> None:
        self.mutate(lambda s: setattr(s, "steps", [
            {"id": i, "label": lbl, "status": "pending", "message": "", "detail": {}, "images": [], "t0": None, "t1": None}
            for i, lbl in steps]))

    def log(self, msg: str) -> None:
        self.mutate(lambda s: s.logs.append(f"{time.strftime('%H:%M:%S')} {msg}"))

    def wait_change(self, version: int, timeout: float) -> dict:
        with self.cond:
            self.cond.wait_for(lambda: self.state.version != version, timeout=timeout)
            return self.state.snapshot()


def file_url(p: Path) -> str:
    rel = Path(p).resolve().relative_to(RUNS_DIR.resolve())
    return f"/files/{rel.as_posix()}?v={int(Path(p).stat().st_mtime * 1000)}"


class UiReporter:
    def __init__(self, hub: Hub, stop_flag: threading.Event):
        self.hub, self.stop_flag = hub, stop_flag
        self._lock = threading.Lock()
        self._pending: tuple | None = None

    def add_step(self, step_id, label):
        """Append a step (agent tool calls and their parts) unless it is already in the plan."""
        def fn(s: UiState):
            if not any(st["id"] == step_id for st in s.steps):
                s.steps.append({"id": step_id, "label": label, "status": "pending", "message": "", "detail": {},
                                "images": [], "t0": None, "t1": None})
        self.hub.mutate(fn)

    def step(self, step_id, status, message="", detail=None, images=None):
        urls = [file_url(p) for p in images or [] if Path(p).exists()]

        def fn(s: UiState):
            for st in s.steps:
                if st["id"] != step_id:
                    continue
                st["status"], st["message"] = status, message
                if detail:
                    st["detail"] = {**st["detail"], **{k: v for k, v in detail.items() if k != "target_deg"}}
                if urls:
                    st["images"] = urls
                if status == "active":
                    st["t0"] = st["t0"] or time.time()
                else:
                    st["t1"] = time.time()
            if detail and "target_deg" in detail:
                s.target_deg = detail["target_deg"]
            if status != "active" and step_id in ("return", "home") and status == "done":
                s.target_deg = None

        self.hub.mutate(fn)
        if status in ("failed", "stopped"):
            self.hub.log(f"{step_id}: {status}" + (f": {message}" if message else ""))

    def log(self, message):
        self.hub.log(message)

    def _wait(self, field_name: str, payload: dict):
        """Publish `payload` under state.<field_name> and block until answer() or a stop. Returns the answer."""
        cid = f"{field_name}-{time.time():.3f}"
        ev, box = threading.Event(), {"value": None}
        with self._lock:
            self._pending = (cid, ev, box)
        self.hub.update(**{field_name: {"id": cid, **payload}})
        try:
            while not ev.wait(0.1):
                if self.stop_flag.is_set():
                    break
        finally:
            with self._lock:
                self._pending = None
            self.hub.update(**{field_name: None})
        return box["value"]

    def answer(self, cid: str, value) -> bool:
        with self._lock:
            pending = self._pending
        if not pending or pending[0] != cid:
            return False
        pending[2]["value"] = value
        pending[1].set()
        return True

    CONFIRM_TEXT = {  # kind -> (title, approve label, decline label)
        "grasp": ("Approve grasp?", "Approve grasp", "Cancel"),
        "plan": ("Approve the agent's plan?", "Approve plan", "Cancel (return to rest)"),
        "hand": ("Move toward a hand?", "Go ahead", "Don't"),
    }

    def confirm(self, kind, prompt, images=None):
        if kind == "start":
            return True  # pressing Run is the consent to enable torque
        title, yes, no = self.CONFIRM_TEXT.get(kind, ("Approve?", "Approve", "Cancel"))
        value = self.choose(kind, title, prompt, [("yes", yes), ("no", no)], images)
        return value == "yes"

    def choose(self, kind, title, prompt, options, images=None):
        value = self._wait("confirm", {
            "kind": kind, "title": title, "prompt": prompt,
            "images": [file_url(p) for p in images or [] if Path(p).exists()],
            "options": [{"value": v, "label": lbl} for v, lbl in options],
        })
        return value if value in {v for v, _ in options} else "cancel"

    def pick(self, prompt):
        value = self._wait("pick", {"prompt": prompt})
        return None if value is None else np.asarray(value, dtype=float)


# ---------------------------------------------------------------------------------------------- app
class RunRequest(BaseModel):
    query: str = ""
    keep: bool = False
    mode: str = "prompt"  # prompt | click | agent


class ConfirmRequest(BaseModel):
    id: str
    answer: str


class ClickRequest(BaseModel):
    id: str
    x: float  # fraction of the displayed image width (0..1)
    y: float
    view: str = "rect"  # rect | raw


class RecordingRequest(BaseModel):
    enabled: bool


class SimResetRequest(BaseModel):
    shuffle: bool = False


class ViewRequest(BaseModel):
    view: str


ROBOT_VIEWS = {"3/4": (135.0, -25.0), "side": (90.0, -5.0), "top": (90.0, -89.0), "front": (180.0, -15.0)}


class Backend:
    SIM_SHUFFLE_M = 0.03  # "shuffle" moves each object up to this far and turns it

    def __init__(self, sim: bool = False):
        self.sim = sim
        self.environment = "sim" if sim else "real"
        self.hub = Hub()
        self.cfg = RobotConfig.load()
        self.hub.update(environment=self.environment, recording=self.cfg.recording.enabled)
        self.world = None  # SimWorld in sim mode
        self.arm = None
        self.camera = None
        self.runner: GraspRunner | None = None
        self.agent = None  # agent.Agent once the models are loaded
        self.rect = None
        self.stop_flag = threading.Event()
        self.reporter = UiReporter(self.hub, self.stop_flag)
        self.robot_jpeg: bytes | None = None
        self.robot_view = "3/4"
        self._running = True
        # Streams (camera, arm view, events) never end on their own; uvicorn waits for open connections
        # before running the app shutdown, so they must be stopped first (see close_streams / scripts/ui.py).
        self._streaming = True
        self._run_thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        threading.Thread(target=self._connect_hardware, daemon=True, name="hw").start()
        threading.Thread(target=self._load_models, daemon=True, name="models").start()
        self._loops = [threading.Thread(target=self._monitor_arm, daemon=True, name="arm-monitor"),
                       threading.Thread(target=self._render_robot, daemon=True, name="robot-view")]
        for th in self._loops:
            th.start()

    def close_streams(self) -> None:
        """End every open stream within ~1 s (called on Ctrl-C, before the server waits for connections)."""
        self._streaming = False

    def shutdown(self) -> None:
        self._streaming = False
        self._running = False
        if self.arm is not None:
            if self.hub.state.busy:
                self.arm.request_stop()
                if self._run_thread:
                    self._run_thread.join(timeout=20)
            try:
                self.arm.disconnect(disable_torque=True)
            except Exception as e:  # noqa: BLE001
                logger.error("arm disconnect: %s", e)
        if self.camera is not None:
            self.camera.disconnect()
        if self.world is not None:
            self.world.close()
        for th in getattr(self, "_loops", []):  # they poll _running every 0.1 s
            th.join(timeout=2.0)

    def _connect_hardware(self) -> None:
        if self.sim:
            self._start_sim()
            return
        try:
            cam = self.cfg.make_camera()
            cam.connect()
            cam.set_exposure(None)
            self.camera = cam
            self.hub.mutate(lambda s: s.status.update(camera="ok"))
        except Exception as e:  # noqa: BLE001
            self.hub.mutate(lambda s: s.status.update(camera=f"error: {e}"))
        try:
            arm = self.cfg.make_arm()
            arm.connect(enable_torque=False)
            self.arm = arm
            self.hub.mutate(lambda s: s.status.update(arm="ok"))
        except Exception as e:  # noqa: BLE001
            self.hub.mutate(lambda s: s.status.update(arm=f"error: {e}"))

    def _start_sim(self) -> None:
        try:
            from ..mujoco_sim import SimWorld, SimWristCamera, make_mujoco_arm
            from ..perception.geometry import FisheyeCamera

            world = SimWorld(self.cfg, FisheyeCamera.load())
            world.start()
            self.world = world
            self.camera = SimWristCamera(world)
            self.arm = make_mujoco_arm(self.cfg, world)
            self.hub.mutate(lambda s: s.status.update(arm="ok (MuJoCo sim)", camera="ok (MuJoCo sim)"))
        except Exception as e:  # noqa: BLE001
            logger.exception("simulation failed to start")
            self.hub.mutate(lambda s: s.status.update(arm=f"error: {e}", camera=f"error: {e}"))

    def sim_reset(self, shuffle: bool) -> None:
        if self.world is None:
            raise HTTPException(400, "not running in simulation")
        if self.hub.state.busy:
            raise HTTPException(409, "a run is in progress")
        self.world.reset(jitter_m=self.SIM_SHUFFLE_M if shuffle else 0.0)
        self.hub.log("sim: objects " + ("shuffled" if shuffle else "back in place") + ", arm at rest")

    def _load_models(self) -> None:
        try:
            from ..grasping.setup import build_runner_parts

            parts = build_runner_parts(sim=self.sim)
            self.rect = parts["rect"]
            # Simulated runs (and their episodes) go to runs/sim/, away from the real-robot data.
            self.runner = GraspRunner(**parts, reporter=self.reporter, environment=self.environment,
                                      runs_dir=None if self.environment == "real" else RUNS_DIR / "sim")
            from ..agent import Agent, build_reasoner
            self.agent = Agent(self.runner, build_reasoner(shared_vlm=parts["locator"].pointer))
            self.hub.mutate(lambda s: s.status.update(models="ok"))
        except Exception as e:  # noqa: BLE001
            logger.exception("model loading failed")
            self.hub.mutate(lambda s: s.status.update(models=f"error: {e}"))

    def _monitor_arm(self) -> None:
        last_err = ""
        while self._running:
            time.sleep(0.1)
            arm = self.arm
            if arm is None:
                continue
            try:
                q = arm.read_joints()
                g = arm.read_gripper()
                torque = bool(arm.bus.read("Torque_Enable", "shoulder_pan"))
                self.hub.update(joints_deg=[round(float(v), 1) for v in q], gripper_pct=round(g, 1), torque=torque)
                last_err = ""
            except Exception as e:  # noqa: BLE001 - shown in the UI; keep polling
                if str(e) != last_err:
                    self.hub.log(f"arm read failed: {e}")
                    last_err = str(e)

    def _render_robot(self) -> None:
        from ..viz import LiveRenderer

        try:
            if self.sim:  # show the simulation itself: arm and objects
                from ..mujoco_sim import SIM_SCENE_PATH
                r = LiveRenderer(size=(480, 360), open_gap_m=self.cfg.gripper_open_gap_m, scene_path=SIM_SCENE_PATH)
            else:
                r = LiveRenderer(size=(480, 360), open_gap_m=self.cfg.gripper_open_gap_m)
        except Exception as e:  # noqa: BLE001
            self.hub.log(f"robot view unavailable: {e}")
            return
        tint = np.array([255, 220, 0], dtype=np.float32)  # BGR: cyan ghost
        last_err = ""
        while self._running:
            time.sleep(0.1)
            s = self.hub.state
            if s.joints_deg is None:
                continue
            try:
                r.set_view(*ROBOT_VIEWS.get(self.robot_view, ROBOT_VIEWS["3/4"]))
                g = s.gripper_pct or 0.0
                world = self.world
                img = (r.render_qpos(world.qpos(), *world.mocap_state()) if world is not None
                       else r.render(np.array(s.joints_deg), g))
                target = s.target_deg
                if target is not None and s.busy and np.max(np.abs(np.subtract(target, s.joints_deg))) > 2.0:
                    mask = r.arm_mask(np.array(target), g)
                    img[mask] = (0.55 * img[mask] + 0.45 * tint).astype(np.uint8)
                    cv2.putText(img, "cyan ghost = pose it is moving to", (10, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 220, 0), 1)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.robot_jpeg = buf.tobytes()
                last_err = ""
            except Exception as e:  # noqa: BLE001 - a bad frame must not kill the live view
                if repr(e) != last_err:
                    logger.exception("robot view render failed")
                    self.hub.log(f"robot view render failed: {e!r}")
                    last_err = repr(e)
        r.close()  # in this thread: the EGL context belongs to it

    # ---------------------------------------------------------------- actions
    def start_run(self, query: str, keep: bool, mode: str = "prompt") -> None:
        query = query.strip()
        if mode not in ("prompt", "click", "agent"):
            raise HTTPException(400, "mode must be prompt, click or agent")
        if mode in ("prompt", "agent") and not query:
            raise HTTPException(400, "empty prompt")
        if self.hub.state.busy:
            raise HTTPException(409, "a run is already in progress")
        if self.runner is None or self.arm is None or self.camera is None:
            raise HTTPException(503, f"not ready: {self.hub.state.status}")
        record = bool(self.hub.state.recording)  # fixed for this run, whatever the toggle does meanwhile
        self.stop_flag.clear()
        from ..agent import agent_steps

        self.hub.reset_steps({"prompt": STEPS, "click": STEPS_CLICK, "agent": agent_steps()}[mode])
        self.hub.update(busy=True, query=query if mode != "click" else "", mode=mode, result="", run_dir="",
                        target_deg=None)
        self.hub.log({"prompt": f'run: "{query}"', "click": "run: pick by clicking", "agent": f'agent: "{query}"'}[mode]
                     + (" (keep)" if keep and mode != "agent" else "") + ("" if record else " (not recorded)"))

        def work():
            try:
                if mode == "agent":
                    res = self.agent.run(self.arm, self.camera, query, record=record)
                else:
                    res = self.runner.run(self.arm, self.camera, query if mode == "prompt" else None, keep=keep,
                                          record=record)
                self.hub.update(result=res.result, run_dir=res.out_dir.name)
            except Exception as e:  # noqa: BLE001 - surface in the UI
                logger.exception("run crashed")
                self.hub.update(result=f"error: {e!r}")
                self.hub.log(f"run crashed: {e!r}")
            finally:
                self.hub.update(busy=False, confirm=None, pick=None, target_deg=None)

        self._run_thread = threading.Thread(target=work, daemon=True, name="grasp-run")
        self._run_thread.start()

    def click(self, req: "ClickRequest") -> None:
        """Map a click on the displayed stream (fractions of its size) to rectified-image pixels."""
        if not (0 <= req.x <= 1 and 0 <= req.y <= 1):
            raise HTTPException(400, "click outside the image")
        if self.rect is None:
            raise HTTPException(503, "models not loaded")
        w, h = self.rect.size
        if req.view == "raw":
            fw, fh = self.rect.cam.image_size
            pt = np.array([[[req.x * fw, req.y * fh]]], dtype=np.float64)
            uv = cv2.fisheye.undistortPoints(pt, self.rect.cam.K, self.rect.cam.D, R=np.eye(3), P=self.rect.K)[0, 0]
            if not (0 <= uv[0] < w and 0 <= uv[1] < h):
                raise HTTPException(400, "that point is outside the calibrated view; click nearer the centre")
        else:
            uv = np.array([req.x * w, req.y * h])
        if not self.reporter.answer(req.id, [float(uv[0]), float(uv[1])]):
            raise HTTPException(409, "not waiting for a click")
        self.hub.log(f"clicked ({uv[0]:.0f}, {uv[1]:.0f}) in the straightened view")

    def stop(self) -> None:
        self.stop_flag.set()
        if self.arm is not None:
            self.arm.request_stop()
        self.hub.log("STOP requested: holding, then returning to rest (press Stop again while returning to cut torque)")

    def estop(self) -> None:
        self.stop_flag.set()
        if self.arm is not None:
            try:
                self.arm.emergency_torque_off()
                self.hub.log("E-STOP: torque disabled")
            except Exception as e:  # noqa: BLE001
                self.hub.log(f"E-STOP failed to reach the servos: {e!r}. Use the power switch!")

    # ---------------------------------------------------------------- streams
    def camera_frames(self, view: str):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while self._streaming:
            cam = self.camera
            if cam is None:
                time.sleep(0.2)
                continue
            t0 = time.monotonic()
            try:
                frame, _ = cam.read(timeout_s=2.0)
            except Exception:  # noqa: BLE001 - camera reconnecting
                time.sleep(0.2)
                continue
            if view == "rect" and self.rect is not None:
                frame = self.rect.rectify(frame)
            else:
                frame = cv2.resize(frame, (960, 720))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield boundary + buf.tobytes() + b"\r\n"
            time.sleep(max(0.0, FRAME_PERIOD_S - (time.monotonic() - t0)))

    def robot_frames(self):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        while self._streaming:
            jpg = self.robot_jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield boundary + jpg + b"\r\n"
            time.sleep(0.05)

    def events(self):
        version, last_sent = -1, 0.0
        while self._streaming:
            snap = self.hub.wait_change(version, timeout=1.0)  # short, so close_streams takes effect quickly
            if snap["version"] == version and time.monotonic() - last_sent < 15.0:
                continue
            version, last_sent = snap["version"], time.monotonic()
            yield f"data: {json.dumps(snap, default=str)}\n\n"


def create_app(sim: bool = False) -> FastAPI:
    from contextlib import asynccontextmanager

    backend = Backend(sim=sim)

    @asynccontextmanager
    async def lifespan(_app):
        backend.start()
        yield
        backend.shutdown()

    app = FastAPI(title="so101_zero_shot", lifespan=lifespan)
    app.state.backend = backend

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/state")
    def state():
        return json.loads(json.dumps(backend.hub.state.snapshot(), default=str))

    @app.get("/api/events")
    def events():
        return StreamingResponse(backend.events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/stream/camera.mjpg")
    def camera(view: str = "rect"):
        return StreamingResponse(backend.camera_frames(view), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/robot.mjpg")
    def robot():
        return StreamingResponse(backend.robot_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/run")
    def run(req: RunRequest):
        backend.start_run(req.query, req.keep, req.mode)
        return {"ok": True}

    @app.post("/api/click")
    def click(req: ClickRequest):
        backend.click(req)
        return {"ok": True}

    @app.post("/api/confirm")
    def confirm(req: ConfirmRequest):
        if not backend.reporter.answer(req.id, req.answer):
            raise HTTPException(409, "no such pending confirmation")
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        backend.stop()
        return {"ok": True}

    @app.post("/api/estop")
    def estop():
        backend.estop()
        return {"ok": True}

    @app.post("/api/recording")
    def set_recording(req: RecordingRequest):
        if backend.hub.state.busy:
            raise HTTPException(409, "a run is in progress; change recording before the next run")
        backend.hub.update(recording=req.enabled)
        backend.hub.log("episode recording " + ("on" if req.enabled else "off") + " for the next runs")
        return {"ok": True}

    @app.post("/api/sim/reset")
    def sim_reset(req: SimResetRequest):
        backend.sim_reset(req.shuffle)
        return {"ok": True}

    @app.post("/api/robot_view")
    def robot_view(req: ViewRequest):
        if req.view not in ROBOT_VIEWS:
            raise HTTPException(400, f"view must be one of {list(ROBOT_VIEWS)}")
        backend.robot_view = req.view
        return {"ok": True}

    @app.get("/files/{path:path}")
    def files(path: str):
        p = (RUNS_DIR / path).resolve()
        if RUNS_DIR.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    return app
