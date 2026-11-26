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

## 2026-11-25: Phase 1 (camera calibration), in progress

- No printer: ChArUco board shown full screen on a phone lying flat (docs/calibration/charuco_phone.png,
  4x6 squares, caliper: 6 squares = 88.6 mm -> 14.767 mm squares).
- Attempts 1-2: board visible from the scan pose but 0 corners. Auto exposure is set for the dim desk,
  so the screen is blown out and the thin black marker borders wash out. Fix: exposure sweep at the scan
  pose (keep the best), per-view retry at half/double exposure; phone image now has a dark surround.
- Attempt 3: the camera dropped off USB (errno 19) repeatedly, at 2592x1944 and later at 1600x1200, even
  with the arm still; intermittent. It shares an unpowered hub with the servo adapter. Camera wrapper now
  auto-reconnects and re-applies exposure; back to 1600x1200. Hardware suspect: power or connector.
- Attempt 4 (calib_20261125_091657): exposure sweep works (auto 0/15, manual 25 -> 15/15 corners). Captured
  3 views, then the camera connector unseated at a steep wrist angle. Also found a safety bug: a non-Ctrl-C
  error skipped the return-to-rest and went straight to torque-off (arm goes limp at the viewpoint). Fixed:
  any error now returns scan -> rest first. Capture now parks at scan on disconnect, waits 20 s for the user
  to reseat the connector, retries the view (skips after 2 disconnects), and `--resume` continues a session.
  Exposure: each view starts from the last setting that worked (half/double retry if needed); a run or
  resume tries the session's last good exposure at scan and only does the full sweep if it fails.
- Resume of 091657: views 4-9 captured, then the arm bumped the desk monitor at view 10 (base 47 deg, tip
  16 cm out to the side) and the servo bus reset (no status packet; arm limp). 8 usable frames kept.
- Compact plan: tip within x 14-26 cm, |y| <= 8 cm, z 8-16 cm, base <= 25 deg, 10 views (was 40).
  Per-view timing added: moves take 1-4 s; the earlier 10-30 s per view came from exposure retries on
  views where the board is partly visible (one grab took 17 s), not from the arm.
- Attempt 5 (calib_20261125_094009): 10/10 views, 9 usable, no disconnects, ~1 min.
- Solver (scripts/solve_calibration.py, calibration/solve.py): cv2.fisheye intrinsics, then camera-in-
  gripper_link + one board pose per session, refined through the arm FK. Both sessions used (17 frames).
  Result (config/calibration/wrist_camera.yaml):
  - fx 484.5, fy 484.3, c (806, 607), small distortion terms, rms 0.11 px.
  - Camera 6.2 cm forward / 4.9 cm below gripper_link; optical axis 30 deg off the approach axis
    (which is why "aim at the board" views put the board off-centre).
  - Corners predicted from joint angles alone: rms 6.6 px (~3 mm at 25 cm), worst view 16 px.
    This is the arm's own accuracy and bounds what grasping can expect from the wrist camera.
  - Sanity: both sessions put the board at the same place (29, 0, 1) cm within 1 cm (phone did not
    move), 3-4 deg from flat.
  - Limits: the board corners only reach ~49 deg off-axis (413 px of ~1000 px), so the lens model is
    extrapolated near the image edges. On synthetic data, intrinsics-only has ~2% focal bias; refining the
    intrinsics through the arm FK fixes that on synthetic data but overfits arm error on the real arm
    (fx/fy split 3%, board tilt 7 deg), so it is opt-in (--refine-intrinsics).
  PHASE 1 DONE (good enough for centre-of-image grasping; revisit if edge accuracy matters).

- Runs uploaded to the private HF dataset gilbertgonz/so101_zero_shot (all 6 calib sessions + README
  card in runs/README.md). Future uploads: `scripts/upload_runs.py` (checks the HF account first).

## 2026-11-25: Phase 2 (perception), in progress

- Models (smallest that work; all local): Qwen3-VL-2B-Instruct (pointing; 4B kept as fallback),
  SAM 2.1 hiera-small (mask), Depth Anything V2 Small (relative depth). Load ~11 s, ~1-2 s per query.
- Pipeline (perception/locate.py): fisheye -> pinhole (100 deg, inside the calibrated region) -> VLM box +
  point -> SAM mask -> depth made metric by fitting it to the table (true table depth per pixel is known
  from the calibrated camera pose) -> object points in the base frame, top height, top-surface centre.
- Look pose = scan: the straightened view covers the whole workspace (docs: 5 cm table grid overlay).
- Scene 1 (scene_20261125_102546: mug, Planters can, marker half under the gripper): all three found and
  masked by the 2B model.
- Lessons:
  - The gripper fingers fill the bottom of the view, 5-12 cm from the lens; including the table between
    them in the depth fit squashed heights (mug 3 cm). Rows below the projected fingertip are now
    excluded from the fit -> mug 11.1 cm.
  - The 2B model sometimes puts its point on a box corner: fall back to the box centre.
  - Depth Anything V2 Base was worse than Small on this scene (can height 3 cm); staying with Small.
- Ruler check (user): can lid 30.7 cm from the pan axis (est. 29.3 with top-surface centre; 28.6 before),
  can height 8.3 cm (est. 10.4), mug height 10.2 cm (est. 11.1). Errors ~1-2 cm, object tops come out
  too close to the camera. Objects under/between the fingers get no reliable height (marker: -1.6 cm).
  Plan: absorb the remaining error in phase 3 by re-looking from directly above the target before the
  grasp (object near image centre, short range) instead of chasing monocular depth accuracy.

## 2026-11-25: Phase 3 (top-down grasp), in progress

- scripts/grasp.py QUERY: scan -> coarse locate -> re-look (camera ~20 cm from the target, sampled IK,
  safe paths checked) -> re-locate -> plan -> confirm -> open, pregrasp, descend 15 deg/s, close, lift ->
  put back -> rest. `--dry-run SCENE` plans and renders from a saved scene without moving.
- Jaw closing direction = camera x axis (fingers are left/right in the image), from the calibration; the
  stock MuJoCo jaw swings along a different axis, so the model is not used for this.
- Straight-down is only reachable with the fingertips low (<= ~6 cm at x ~ 0.25 m): pregrasp 5 cm and
  lift 7 cm above the grasp point, allowed to tilt up to 20 deg.
- Wrist roll is chosen so the jaws close across the object's long axis (PCA of the top-surface points).
- Grasp 1 (grasp_20261125_111100, "black dry erase marker" at (30.3, 4.7) cm): FIRST ZERO-SHOT GRASP,
  held and lifted cleanly (after_lift.jpg), jaws 3 deg from ideal, fingertips 8.8 mm above the table.
  Logged as "missed" because the held check used gripper position: the parallel jaws close to 0.7 %
  around a marker. gripper_probe.py: empty 0.2 % / |load| 28, marker 0.9 % / |load| 72. Held is now
  |load| >= 50 or opening >= 4 % (grasping/feedback.py). One sample each; recheck on more objects.

- Web UI (scripts/ui.py, FastAPI + one HTML page, 127.0.0.1 only): live camera (straightened/raw), live arm
  render from joint readings at 10 Hz with a cyan ghost of the pose being moved to, prompt, plan panel with
  per-step numbers and images, approval dialog, Stop, E-STOP. `--fake` = real arm code on a simulated bus.
- Grasp sequence moved to grasping/runner.py, shared by the CLI and the UI.
- Stopping: SO101Arm.request_stop() interrupts a move at the next control tick (hold + MotionStopped);
  emergency_torque_off() from any thread; the bus is now locked per transaction. Found and fixed with the
  simulated UI: a Stop pressed at the approval dialog used to also abort the return home, cutting torque
  with the arm in the air (regression test in tests/test_grasp_runner.py).
- Click to pick (UI "Pick by clicking"): the arm goes to scan, the user clicks the object, SAM masks it at
  the click (no VLM pointing), depth gives its position, then the VLM names the outlined crop. A card shows
  the name, position, height and mask before anything moves: grasp / pick again / cancel. Clicks on the
  gripper (mask mostly below the fingertip row) are rejected; out-of-reach objects can only be re-picked or
  cancelled. Raw-view clicks are mapped through the fisheye model into the straightened view. After the
  re-look the object is re-found by name and kept only if within 4 cm of the clicked estimate, otherwise
  SAM is re-run at the projected clicked position. Checked in --fake mode; not yet on the real arm.
- Note on --fake: the sim arm sits exactly at the scan pose, but the saved image was taken with the elbow
  2.5 deg off, so object positions in the sim are off by several cm (mug 39 cm vs 32 cm). Real runs use
  the measured joints.
- Parallel gripper modelled (assets/so101/so101_new_calib.xml): the stock fixed + moving jaw are replaced
  by the roboninecom repo's meshes (@42b4e72): link5_1 (frame, servo, rods, camera holder) on the
  wrist-roll flange and two clamps on slide joints (`gripper`, `gripper_left`, metres open each), driven
  from gripper % by gripper_model.set_gripper_qpos (100 % = 46 mm gap, estimated from the CAD pinion;
  tool.gripper_open_gap_m). Live view, plan renders and collision checks use it.
  - Placement checks: CAD fingertips 13.2 cm from the flange = measured tip length; CAD camera holder
    within 1 cm of the hand-eye camera position.
  - Tool point is now the CAD fingertip centre (tool.tip_xyz_m = [0, 0, -0.1318], replacing
    tip_extension_m): same length, but on the wrist-roll axis instead of 8 mm off it (the stock jaw's
    offset). Re-planning grasp 1 moves the pan by 1.7 deg.
  - Rendered from the hand-eye camera into the straightened view, the clamps only overlap the real
    fingers by IoU 0.30. Fitting the camera pose to the finger silhouette alone gives IoU 0.85 (shape
    matches) but the camera moves ~2 cm / 10 deg and the board reprojection goes 6.6 -> 27 px; a joint fit
    cannot satisfy both. Mounting rotation about the roll axis does not explain it (IoU <= 0.36). So the
    hand-eye result is an "effective" pose that absorbs arm/FK error and is not the physical camera mount;
    keep it for 3D geometry, use a separate silhouette-fitted camera pose for an image mask of the fingers.
  - Tried to reconcile them: per-joint offsets (shoulder lift, elbow, wrist flex, wrist roll) with the
    camera fixed at the finger pose leave 22.8 px on the boards; a joint fit of boards + finger contour +
    offsets stops at ~17 px; free camera + offsets reaches 5.6 px only with +-15 deg offsets (overfit).
    The fingers are 12-39 deg off-axis, inside the calibrated lens region, so not the lens model either.
    Unresolved; a proper fix needs more data (e.g. board views at several gripper-near distances, or
    fingertip touch points). Practical split kept: hand-eye for 3D, finger-fit pose for the mask.
- Gripper mask (perception/gripper_mask.py, config/calibration/gripper_view.yaml from
  scripts/fit_gripper_view.py: SAM on the real fingers, IoU fit from a grid of starts, 0.835).
  The gripper's outline only depends on the jaw opening, so masks are pre-rendered per 5 % and the runner
  passes the measured opening with each photo. Geometry nearer than 3 cm (camera holder rim) is clipped.
  - Click/answer rejection: > 50 % of the mask on the (6 px grown) gripper = the gripper, otherwise the
    gripper pixels are just dropped. Replaces the "below the fingertip row" rule, which rejected a cap
    sitting beside the left finger (grasp_20261125_131428); it is now "white plastic cap" at (22.8, 15.5).
  - Depth fit: leaves out the fingers + 25 px instead of every row below the fingertips, so table beside
    the fingers counts. Can/mug unchanged (33.0 / 33.2 cm, tops 10.6 / 10.9 cm); the cap no longer comes out
    below the table, but its height (~0 cm) is still not trustworthy this close to the fingers.
  - Scene 102546's marker (mostly hidden behind the fingers) is now refused instead of measured at -1.6 cm.

## 2026-11-26: Episode recording

- Every run on the arm is now recorded (recording.py, `recording:` in robot.yaml), from the start
  confirmation until torque is off, including aborted, stopped and E-stopped runs (declined-at-start and
  dry runs have no motion and are not recorded). Per run folder: `episode.parquet` (10 Hz rows: raw fisheye
  frame as 800x600 JPEG, measured joints + gripper, servo loads, last commanded goal, current step) and
  `episode.json` (task, outcome, human label slot, grasp plan, step events and operator answers, lens
  calibration scaled to the stored frames, git commit). LeRobot-style column names, `datasets` Image layout
  (checked: `load_dataset("parquet")` decodes the frames). scripts/episodes.py lists/filters/labels/indexes.
- The recorder thread reads the bus with no retries (two sync reads per sample) so it cannot hold the bus
  for long; a failed read keeps the row with NaN state (`state_ok` false). Never fails the run.
- About 0.5-0.9 MB/s, ~60 MB per 90 s grasp. Not yet run on the real arm.
- The automatic `success` is the gripper load/position check; `scripts/episodes.py label` overrides it.
- Earlier runs (before this) have stills and grasp.json only.

Next
- More grasps (different objects, positions, orientations) to measure the success rate.
- VLM check of the after-lift image as a second opinion on "held".
