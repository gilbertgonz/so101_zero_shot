# so101_zero_shot

Zero-shot, language-conditioned grasping on an SO-101 arm with a single wrist-mounted fisheye camera:
VLM points at the object from a prompt, SAM masks it, monocular depth lifts it to 3D, and a top-down IK
grasp picks it up. No demonstrations, no training. All models run locally.

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

## Scripts
- `scripts/capture_calibration.py`, `scripts/solve_calibration.py`: wrist camera calibration.
- `scripts/capture_scene.py`, `scripts/locate.py`: take a picture from the scan pose; locate objects in it.
- `scripts/gripper_probe.py`: gripper position/load when closing on nothing vs an object.
- `scripts/upload_runs.py`: push runs/ to the Hugging Face dataset.
- `scripts/calibrate_arm.py [--from-eeprom]`: create the arm calibration file.
- `scripts/read_joints.py`: live joint angles + FK tip position, torque off.
- `scripts/render_pose.py`: render the arm in a joint pose or top-down tip target (for "put the arm like this").

Data (runs, frames, logs) is stored on Hugging Face (`gilbertgonz/so101_zero_shot`, dataset repo), not in git.
See `PROGRESS.md` for status.
