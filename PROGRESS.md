# Progress log

## 2026-11-25: Phase 0 (standalone control), in progress

Done
- Vendored and trimmed the control stack from lerobot-playground / LeRobot. No lerobot dependency:
  `hardware/feetech.py` (STS3215 bus), `hardware/so101.py` (arm, calibration, deg conversion, safe moves),
  `hardware/camera.py` (threaded fresh-frame wrist camera), `kinematics/` (placo FK/IK).
- URDF, meshes, MuJoCo scene copied to `assets/so101/` (Apache-2.0, TheRobotStudio SO-ARM100).
- Calibration rebuilt from servo EEPROM (old LeRobot calibration was still on the motors):
  `config/calibration/so101_follower.json`.
- Offline tests pass (`pytest`): encoding, FK/IK round trips, top-down reachability.

Findings
- Wrist camera: 5MP USB fisheye, ~180 deg, circular image. Use 4:3 (1600x1200 @ 20 fps) to keep the full
  circle; 16:9 modes crop it. Clear plastic dome removed from the lens (it caused reflections).
- URDF joint limits are tighter than the real arm. Soft limits now come from the calibrated range minus 4 deg
  (e.g. wrist_flex +-104.5 vs +-95 in the URDF).
- Known-pose check (arm placed by hand, torque off, renders in docs/poses/):
  - Pose A (URDF zero) read elbow_flex = 12.7 deg; pose B (tip straight down on table) needed an 11.2 deg
    elbow correction to be vertical. Both agree: `joint_offsets_deg.elbow_flex = 11.5` in config.
    Other joints within ~3 deg (hand-placement noise).
  - With the offset, pose B's model tip was still 3.4 cm above the table: the parallel-jaw gripper
    (roboninecom SO-ARM100-101-102-Parallel-Gripper) has longer fingers than the URDF's stock jaw.
    Provisional `tool.tip_extension_m = 0.034`; to confirm with a ruler measurement.
  - Parallel gripper reads max ticks when closed: `gripper_inverted: true` (0 = closed, 100 = open).
- GRIPPER SAFETY: the parallel gripper's gear rack jumps off its rail if opened past raw tick ~2210
  (48.9% of the old 427..3917 range). Gripper range is now 2260 (max safe open, ~4 deg margin) .. 4020
  (closed), written to both the calibration file and the servo EEPROM position limits, so the servo itself
  refuses to open further. 100% = max safe open. scripts/calibrate_arm.py never re-homes or sweeps the gripper.
- Poses A and B alone only constrain the SUM of shoulder_lift + elbow_flex + wrist_flex offsets (tool pitch).
  The pose-B ruler check (pan axis to fingertips, docs/poses/pose_b_measure.png) measured ~7 in = 17.8 cm vs
  model 17.7 cm, so the elbow offset + 3.4 cm tip extension are consistent to ~1 cm (ruler precision).
  (A first reading of 9.5 in was taken from the wrong base point.) Phase 1 marker touches will refine this.
- Base sits directly on the table, so table plane is z = 0 in the base frame.
- Top-down workspace (with parallel gripper and calibrated limits), x = 0.10..0.30 m, |y| <= 0.20 m:
  exactly vertical for tip height 0..6 cm; at 12 cm the gripper must tilt 15..25 deg; at 18 cm 35..40 deg.
  Plan: hover high with tilt, finish the last few cm vertical.
- IK: position is a hard constraint, approach axis soft (best achievable tilt is reported);
  infeasible targets fall back to a soft solve and report failure. wrist_roll pinned (seed or
  commanded), multi-seed for joint-limit local minima. ~2 ms per solve.
- Sign convention: +shoulder_pan turns toward -y (right side when looking forward).

- First powered move (rest -> scan -> rest, 30 deg/s): smooth, looked right by eye. Steady-state gravity sag
  on the elbow of 5.1 deg in the scan pose (tip 2.2 cm low); other joints <= 0.5 deg. Cause: low P gain (16).
  Fix: `move_to` now re-commands target + measured error a few times (capped at 10 deg per joint so it
  cannot shove an obstacle); `compensate=False` for moves meant to end in contact. Tested on a fake bus.
- Setup (photo 2026-11-25): base near the desk's back-right corner, the edge is behind the base (outside the
  workspace). The table is cleared before every session. Workspace box added to config.
- Sag compensation verified on hardware: scan pose errors <= 0.5 deg (elbow 5.1 -> 0.3), tip 3 mm from target.
  PHASE 0 DONE.

Next
- Phase 1: fisheye intrinsics + hand-eye calibration.
