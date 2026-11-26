"""Episode recording: every run on the arm saved as a time series of wrist-camera frames and robot state.

Each run folder (runs/grasp_*) gets:
  episode.parquet   one row per sample (default 10 Hz), columns in EPISODE_COLUMNS
  episode.json      what was attempted and how it went (task, outcome, grasp plan, step events,
                    camera calibration, code version), for filtering without opening the parquet

Column names follow the LeRobot conventions (observation.state, action, observation.images.*) and the
image column uses the Hugging Face `datasets` Image layout (struct of JPEG bytes + path), so the files
load directly with pyarrow / pandas / polars / `datasets`, and convert to a LeRobot dataset without
re-recording.

Recording never stops or fails a run: sampling errors are counted and the sample is kept with NaN state
or a null image.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import REPO_ROOT, RecordingConfig

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EPISODE_FILE = "episode.parquet"
META_FILE = "episode.json"
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMAGE_KEY = "observation.images.wrist"

_VEC6 = pa.list_(pa.float32(), 6)
EPISODE_SCHEMA = pa.schema([
    pa.field("frame_index", pa.int64(), nullable=False),
    pa.field("timestamp", pa.float64(), nullable=False),  # s since episode start, when the state was read
    pa.field("wall_time", pa.float64(), nullable=False),  # unix time of the same moment
    pa.field("image_timestamp", pa.float64()),  # s since episode start, when the frame was captured
    pa.field(IMAGE_KEY, pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("observation.state", _VEC6),  # measured: 5 joints (deg, URDF convention) + gripper (% open)
    pa.field("observation.load", _VEC6),  # Present_Load per motor (0.1 % of max torque, signed)
    pa.field("action", _VEC6),  # last goal sent to each motor (same units as the state); NaN = none yet
    pa.field("phase", pa.string()),  # runner step active at this moment (home, coarse, descend, ...)
    pa.field("state_ok", pa.bool_(), nullable=False),  # False: the bus read failed, state/load are NaN
])
EPISODE_COLUMNS = EPISODE_SCHEMA.names

# Lets `datasets.load_dataset("parquet", ...)` decode the image column as images.
_VALUE = {"_type": "Value"}
_HF_FEATURES = {
    "frame_index": {**_VALUE, "dtype": "int64"},
    "timestamp": {**_VALUE, "dtype": "float64"},
    "wall_time": {**_VALUE, "dtype": "float64"},
    "image_timestamp": {**_VALUE, "dtype": "float64"},
    IMAGE_KEY: {"_type": "Image"},
    **{k: {"_type": "Sequence", "feature": {**_VALUE, "dtype": "float32"}, "length": 6}
       for k in ("observation.state", "observation.load", "action")},
    "phase": {**_VALUE, "dtype": "string"},
    "state_ok": {**_VALUE, "dtype": "bool"},
}
EPISODE_SCHEMA = EPISODE_SCHEMA.with_metadata({"huggingface": json.dumps({"info": {"features": _HF_FEATURES}})})


def git_version() -> dict:
    """Commit and dirty flag of the code that recorded the episode (None if git is unavailable)."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def outcome_of(result: str, events: list[dict]) -> dict:
    """Summarise a runner result string (held | missed | aborted: why | stopped | estopped | error: why)."""
    status, _, reason = result.partition(":")
    status = status.strip()
    attempted = any(e.get("step") == "close" and e.get("status") == "active" for e in events)
    last = next((e for e in reversed(events) if e.get("kind") == "step"
                 and e.get("status") in ("failed", "stopped")), None)
    return {
        "result": result,
        "status": status,
        # Did the jaws close on the object? Only known when a grasp was actually tried (auto, from the
        # gripper's position/load; see grasping/feedback.py). Human labels go in "label".
        "success": {"held": True, "missed": False}.get(status),
        "grasp_attempted": attempted,
        "failed_step": last["step"] if last else None,
        "reason": reason.strip() or None,
    }


class EpisodeRecorder:
    """Samples camera + arm in a background thread and streams rows to episode.parquet.

    Usage: start(); set_phase()/event() while the run goes; stop(); finish(meta).
    """

    FLUSH_ROWS = 50

    def __init__(self, out_dir: Path, arm, camera, cfg: RecordingConfig):
        self.out_dir, self.arm, self.camera, self.cfg = Path(out_dir), arm, camera, cfg
        self.path = self.out_dir / EPISODE_FILE
        self.events: list[dict] = []
        self._phase: str | None = None
        self._lock = threading.Lock()  # guards _phase / events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: pq.ParquetWriter | None = None
        self._rows: list[dict] = []
        self.t0 = self.wall0 = 0.0
        self.num_frames = self.frames_with_image = self.state_errors = 0
        self.raw_size: tuple[int, int] | None = None
        self.stored_size: tuple[int, int] | None = None
        self.write_error: str | None = None

    # ------------------------------------------------------------------ control (runner thread)
    def start(self) -> None:
        self.t0, self.wall0 = time.monotonic(), time.time()
        self._writer = pq.ParquetWriter(self.path, EPISODE_SCHEMA, compression="zstd")
        self._thread = threading.Thread(target=self._loop, name="episode-recorder", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str | None) -> None:
        with self._lock:
            self._phase = phase

    def event(self, kind: str, **data) -> None:
        """Timestamped event (step changes, operator answers) stored in episode.json."""
        with self._lock:
            self.events.append({"t": round(time.monotonic() - self.t0, 3), "kind": kind, **data})

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.error("episode recorder did not stop; the episode file may be incomplete")
                return  # the thread still owns the writer
        self._flush()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as e:  # noqa: BLE001
                self.write_error = self.write_error or repr(e)
            self._writer = None

    def finish(self, meta: dict, cam_model=None) -> Path:
        """Write episode.json: `meta` (task, outcome, grasp, ...) plus what the recorder knows; `cam_model`
        (FisheyeCamera) adds the lens calibration scaled to the stored frames."""
        meta = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.out_dir.name,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.wall0)),
            "duration_s": round(time.monotonic() - self.t0, 2),
            "num_frames": self.num_frames,
            "frames_with_image": self.frames_with_image,
            "fps": self.cfg.fps,
            "state_read_errors": self.state_errors,
            "write_error": self.write_error,
            "files": {"episode": EPISODE_FILE if self.path.exists() else None},
            "columns": {"state_names": STATE_NAMES, "state_units": "deg x5, gripper % open (0 closed, 100 max safe)",
                        "load_units": "0.1 % of max torque, signed", "image": f"{IMAGE_KEY}: JPEG, raw fisheye resized",
                        "raw_image_size": self.raw_size, "stored_image_size": self.stored_size},
            **({"camera": camera_meta(cam_model, self.stored_size)} if cam_model is not None else {}),
            **meta,
            "events": self.events,
        }
        path = self.out_dir / META_FILE
        path.write_text(json.dumps(meta, indent=1, default=_jsonable))
        return path

    # ------------------------------------------------------------------ sampling (recorder thread)
    def _loop(self) -> None:
        period = 1.0 / self.cfg.fps
        t_next = time.monotonic()
        last_stamp = None
        while not self._stop.is_set():
            try:
                last_stamp = self._sample(last_stamp, period)
            except Exception as e:  # noqa: BLE001 - never take the run down; keep sampling
                log.warning("episode recorder: %r", e)
            if len(self._rows) >= self.FLUSH_ROWS:
                self._flush()
            t_next += period
            delay = t_next - time.monotonic()
            if delay < -period:  # fell behind (slow bus/camera): skip ahead instead of bursting
                t_next = time.monotonic()
            self._stop.wait(max(0.0, delay))

    def _sample(self, last_stamp: float | None, period: float) -> float | None:
        image, img_t = None, None
        try:
            frame, stamp = self.camera.read(newer_than=last_stamp, timeout_s=period)
            if stamp != last_stamp:
                image, img_t, last_stamp = self._encode(frame), stamp - self.t0, stamp
        except Exception as e:  # noqa: BLE001 - no new frame in time (dropout): keep the state stream going
            log.debug("episode recorder: no frame (%r)", e)
        now, wall = time.monotonic(), time.time()
        try:
            pos, load = self.arm.read_state()
            ok = True
        except Exception as e:  # noqa: BLE001 - bus hiccup; keep the row so time stays continuous
            pos = load = np.full(6, np.nan)
            ok = False
            self.state_errors += 1
            log.debug("episode recorder: state read failed (%r)", e)
        with self._lock:
            phase = self._phase
        self._rows.append({
            "frame_index": self.num_frames,
            "timestamp": now - self.t0,
            "wall_time": wall,
            "image_timestamp": img_t,
            IMAGE_KEY: None if image is None else {"bytes": image, "path": None},
            "observation.state": pos.astype(np.float32),
            "observation.load": load.astype(np.float32),
            "action": self.arm.last_command().astype(np.float32),
            "phase": phase,
            "state_ok": ok,
        })
        self.num_frames += 1
        self.frames_with_image += image is not None
        return last_stamp

    def _encode(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        self.raw_size = (w, h)
        if w != self.cfg.image_width:
            size = (self.cfg.image_width, round(h * self.cfg.image_width / w))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        self.stored_size = (frame.shape[1], frame.shape[0])
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def _flush(self) -> None:
        rows, self._rows = self._rows, []
        if not rows or self._writer is None or self.write_error:
            return
        try:
            cols = {k: [r[k] for r in rows] for k in EPISODE_COLUMNS}
            for k in ("observation.state", "observation.load", "action"):
                cols[k] = pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(cols[k]), pa.float32()), 6)
            self._writer.write_table(pa.table(cols, schema=EPISODE_SCHEMA))
        except Exception as e:  # noqa: BLE001 - e.g. disk full: stop writing, report in episode.json
            self.write_error = repr(e)
            log.error("episode recorder: cannot write %s: %r", self.path, e)


def camera_meta(cam_model, stored_size: tuple[int, int] | None) -> dict:
    """Fisheye calibration for the stored frames (intrinsics scaled from the calibrated resolution)."""
    w0, h0 = cam_model.image_size
    s = 1.0 if stored_size is None else stored_size[0] / w0
    K = np.array(cam_model.K, dtype=float)
    K[:2] *= s
    return {"model": "opencv_fisheye", "calibrated_size": [w0, h0], "stored_size": stored_size,
            "K_stored": K, "D": np.asarray(cam_model.D).ravel(), "hand_frame": cam_model.hand_frame,
            "T_hand_camera": cam_model.T_hand_camera}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return np.asarray(o).tolist()


# ---------------------------------------------------------------------- reading
def iter_episodes(runs_dir: Path):
    """(run folder, episode.json contents) for every recorded episode, oldest first."""
    for meta_path in sorted(Path(runs_dir).glob(f"*/{META_FILE}")):
        try:
            yield meta_path.parent, json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.warning("skipping %s: %s", meta_path, e)


def effective_success(meta: dict) -> bool | None:
    """Human label if there is one, else the automatic held check."""
    label = (meta.get("label") or {}).get("success")
    return label if label is not None else (meta.get("outcome") or {}).get("success")


def load_episode(run_dir: Path) -> tuple[dict, pa.Table]:
    """(episode.json, frames table). Decode a frame with decode_image(table[IMAGE_KEY][i].as_py())."""
    run_dir = Path(run_dir)
    return json.loads((run_dir / META_FILE).read_text()), pq.read_table(run_dir / EPISODE_FILE)


def decode_image(cell: dict | None) -> np.ndarray | None:
    """BGR image from an image cell ({'bytes': ..., 'path': ...}), None if the sample has no frame."""
    if not cell or not cell.get("bytes"):
        return None
    return cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR)
