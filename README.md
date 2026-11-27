# so101_zero_shot

## Summary

Zero-shot, language-conditioned grasping on an SO-101 arm with a single wrist-mounted fisheye camera:
VLM points at the object from a prompt, SAM masks it, multi-view depth from the photos and video frames
the arm collects on its way lifts it to 3D, and a top-down IK grasp picks it up. No demonstrations, no training. All models run locally.
Instead of typing a prompt you can also click the object in the live camera view; the robot names what
it found and where, and asks before grasping.

https://github.com/user-attachments/assets/5e1c48a6-4354-42b8-9a45-cb0b77081d4b

## Setup
```
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -e '.[sim,perception,quant,multiview,ui]'
```
Check the whole pipeline end to end in MuJoCo (scan, locate, re-look, plan, grasp, put back; should end with `result: held`):
```
.venv/bin/python scripts/grasp.py "red cube" --sim
```

### Depth
Metric depth comes from views the arm gets anyway, with no extra motion. While the arm moves to a photo
pose (rest -> scan, scan -> closer look), a background thread keeps a video frame every ~1 cm of camera
travel, tagged with the joint angles at its exposure time (`skills/motion_frames.py`). Each photo is then
measured with [MVSAnywhere](https://github.com/nianticlabs/mvsanywhere) (zero-shot multi-view stereo,
202M parameters, non-commercial license) from the photo, the previous photo and 6 of those frames spread
along the path; camera poses come from forward kinematics. The closer-look pose is chosen at least 10 deg
off the scan view's line of sight so the views differ enough. Setup (outside this repo):
```
git clone https://github.com/nianticlabs/mvsanywhere ~/data/third_party/mvsanywhere
.venv/bin/pip install -e ~/data/third_party/mvsanywhere --no-deps
curl -L -o ~/data/third_party/mvsanywhere/weights/mvsanywhere_hero.ckpt \
  https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt
```
`MVSANYWHERE_DIR` overrides the location. A photo with no other views (no move to it, no earlier
photo) cannot be measured and fails with a clear error. Saved scenes (`scripts/capture_scene.py`) keep the
frames from their move in `views/`, so `scripts/locate.py` and `grasp.py --dry-run` measure the same way.

`scripts/eval_depth.py` measures it on simulated runs (photos and frames on the way) against rendered
ground truth: `capture`, `run METHOD`, `report`. 24 samples (4 scan, 20 closer looks), closer looks:

| other views | median depth error | target top error: median / 90% | time | GPU memory |
|---|---|---|---|---|
| previous photo + 2 frames | 3.9 mm | 5.0 mm / - | 0.53 s | 1.2 GB |
| previous photo + 4 frames | 3.5 mm | 4.2 / 18 mm | 0.60 s | 1.4 GB |
| **previous photo + 6 frames (default)** | **3.3 mm** | **4.2 / 18 mm** | **0.66 s** | **1.5 GB** |
| previous photo + 8 frames | 3.5 mm | 4.5 mm / - | 0.79 s | 1.7 GB |

More frames did not help (a move rarely gives more than 8 frames 1 cm apart); larger model inputs within
the 960x720 rectified camera image (576x768: 3.9 mm, 720x960: 5.1 mm) were worse and slower. About 1 in 5
objects gets its height wrong by 1-2 cm (the 90% column). On the real arm the frames are taken while
moving, so `perception.look.camera_latency_s` (exposure to timestamp delay) needs measuring.

## Use
- `scripts/ui.py`: web UI at http://127.0.0.1:8000: live wrist camera, live arm view (cyan ghost = pose it
  is moving to), prompt, step-by-step plan with images, approve/cancel, Stop, E-STOP.
  **Pick by clicking**: click an object in the camera view (either view); the robot masks and
  measures it, names it, and shows what it found and where before asking to grasp it.
- `scripts/grasp.py "object"`: the same grasp from the terminal (`--dry-run SCENE` plans from a saved scene).

### Simulation (no arm needed)
`scripts/ui.py --sim` (or `scripts/grasp.py --sim "red cube"`) runs the whole stack on a MuJoCo arm on a
table with five objects (red cube, blue can, yellow ball, black marker, green block;
`assets/so101/sim_scene.xml`). The real arm code drives simulated servos, the wrist camera is rendered
through the calibrated fisheye model, and grasps are physics. The UI gets **Reset scene** / **Shuffle objects**
buttons. Simulated runs and episodes go to `runs/sim/` (not uploaded); `scripts/episodes.py --sim list`.

## Recorded episodes
Runs on the arm (UI or terminal, prompt or click) are recorded from the start confirmation until torque
is off, into their run folder. The UI's **record** checkbox (next to Run) turns it off for the next runs;
`scripts/grasp.py --no-record` for the terminal; `recording.enabled` in `config/robot.yaml` is the default:
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
