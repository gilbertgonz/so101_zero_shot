# so101_zero_shot

Zero-shot, language-conditioned grasping on an SO-101 arm with a single wrist-mounted fisheye camera:
VLM points at the object from a prompt, SAM masks it, monocular depth lifts it to 3D, and a top-down IK
grasp picks it up. No demonstrations, no training. All models run locally.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -e '.[sim,dev]'
.venv/bin/pytest
```

## Scripts
- `scripts/calibrate_arm.py [--from-eeprom]`: create the arm calibration file.
- `scripts/read_joints.py`: live joint angles + FK tip position, torque off.
- `scripts/render_pose.py`: render the arm in a joint pose or top-down tip target (for "put the arm like this").

Data (runs, frames, logs) is stored on Hugging Face (`gilbertgonz/so101_zero_shot`, dataset repo), not in git.
See `PROGRESS.md` for status.
