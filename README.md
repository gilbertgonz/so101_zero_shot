# so101_zero_shot

## Summary

Zero-shot, language-conditioned grasping on an SO-101 arm with a single wrist-mounted fisheye camera:
VLM points at the object from a prompt, SAM masks it, monocular depth lifts it to 3D, and a top-down IK
grasp picks it up. No demonstrations, no training. All models run locally.
Instead of typing a prompt you can also click the object in the live camera view; the robot names what
it found and where, and asks before grasping.

https://github.com/user-attachments/assets/5e1c48a6-4354-42b8-9a45-cb0b77081d4b

## Setup
```
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -e '.[sim,dev,perception,ui]'
.venv/bin/pytest
```

## Use
- `scripts/ui.py`: web UI at http://127.0.0.1:8000: live wrist camera, live arm view (cyan ghost = pose it
  is moving to), prompt, step-by-step plan with images, approve/cancel, Stop, E-STOP.
  `--fake` runs it against a simulated arm and a saved scene image.
  **Pick by clicking**: click an object in the camera view (either view); the robot masks and
  measures it, names it, and shows what it found and where before asking to grasp it.
- `scripts/grasp.py "object"`: the same grasp from the terminal (`--dry-run SCENE` plans from a saved image).

## Recorded episodes
Every run on the arm (UI or terminal, prompt or click) is recorded from the start confirmation until torque
is off, into its run folder:
- `episode.parquet`: one row per sample (10 Hz, `recording:` in `config/robot.yaml`): `timestamp`,
  `observation.images.wrist` (raw fisheye frame, JPEG, 800x600), `observation.state` (5 joints in deg +
  gripper % open), `observation.load` (servo load), `action` (last commanded goal), `phase` (current step),
  `state_ok`. LeRobot-style column names; loads with pyarrow, pandas or `datasets.load_dataset("parquet")`.
- `episode.json`: what was attempted and how it went: task (mode, query, object, instruction), outcome
  (status, success, whether the gripper closed, failed step, reason), an optional human label, the grasp plan,
  timestamped step events and operator answers, lens calibration for the stored frames, code commit.

`scripts/episodes.py list --attempted --success no` filters them, `label RUN --success yes|no` records a
human verdict (overrides the automatic held check), `index` writes `runs/episodes.parquet`, `frames RUN out.mp4`
makes a quick video. Reading in Python: `so101_zero_shot.recording.load_episode(run_dir)`.

## Scripts
- `scripts/capture_calibration.py`, `scripts/solve_calibration.py`: wrist camera calibration.
- `scripts/capture_scene.py`, `scripts/locate.py`: take a picture from the scan pose; locate objects in it.
- `scripts/gripper_probe.py`: gripper position/load when closing on nothing vs an object.
- `scripts/upload_runs.py`: push runs/ (including episodes) to the Hugging Face dataset.
- `scripts/episodes.py`: list, filter, label and index recorded episodes.
- `scripts/calibrate_arm.py [--from-eeprom]`: create the arm calibration file.
- `scripts/read_joints.py`: live joint angles + FK tip position, torque off.
- `scripts/render_pose.py`: render the arm in a joint pose or top-down tip target (for "put the arm like this").

Data (runs, frames, logs) is stored on Hugging Face (`gilbertgonz/so101_zero_shot`, dataset repo), not in git.
See `PROGRESS.md` for status.
