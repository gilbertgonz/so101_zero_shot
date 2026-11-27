"""MuJoCo stand-in for the arm and the wrist camera, for running the whole stack without hardware.

`scripts/ui.py --sim`: the arm sits on a table with a few objects (assets/so101/sim_scene.xml). The real
SO101Arm code runs on top of MujocoBus (goal/position/load registers backed by the physics), and
SimWristCamera renders the wrist view through the calibrated fisheye model, so perception, planning,
grasping, the UI and episode recording all run unchanged.

Differences from the real setup worth knowing:
- The camera is rendered from the hand-eye pose (wrist_camera.yaml), the same pose perception uses for
  3D, so 3D estimates have no calibration error; the gripper mask is rendered from that pose too.
- Servo "load" is the actuator force as 0.1 % of its force limit; torque off = actuators disabled
  (the arm sags onto its joint limits, like the real one drops onto its stops).
- Arm links do not collide with each other (only with the table and objects).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .config import ASSETS_DIR, RobotConfig  # noqa: E402
from .hardware.so101 import ARM_JOINTS, GRIPPER, MOTOR_IDS, SO101Arm, load_calibration  # noqa: E402
from .hardware.so101 import joint_limits_from_calibration  # noqa: E402
from .perception.geometry import FisheyeCamera  # noqa: E402
from .perception.gripper_mask import set_pinhole_camera  # noqa: E402

log = logging.getLogger(__name__)

SIM_SCENE_PATH = ASSETS_DIR / "sim_scene.xml"
STOP_MARGIN_DEG = 4.0  # soft limits are the calibrated range minus this; the sim joint range is the full range
OBJECT_GAP_M = 0.005  # shifted objects (reset with jitter) keep at least this much space between them
_ARM_CONTYPE, _WORLD_CONTYPE = 2, 1  # arm geoms collide with the world (table, objects), not each other
# MuJoCo collides meshes as their convex hulls; a clamp's hull is a solid 5 x 7 x 9 cm block that stops the
# jaws ~4 mm early on each side. Each clamp's collision is replaced by a box over its finger: the last 35 mm
# before the fingertip, measured from pg_clamp_*_visual (clamp body frame: centre, half sizes, metres).
_FINGER_PADS = {
    "clamp_right": ((0.05865, 0.10575, 0.18600), (0.00625, 0.01745, 0.01480)),
    "clamp_left": ((-0.07850, 0.10795, 0.18355), (0.00630, 0.01745, 0.01485)),
}


class SimWorld:
    """Physics + wrist-camera rendering in one background thread, stepped in real time.

    Thread safety: bus-facing methods and the stepping share a lock; the camera frame is rendered from a
    scene captured under the lock and drawn outside it.
    """

    def __init__(self, cfg: RobotConfig, cam_model: FisheyeCamera, camera_fps: float = 10.0,
                 render_px: int = 1800, render_hfov_deg: float = 130.0, near_clip_m: float = 0.03):
        self.cfg, self.cam_model = cfg, cam_model
        self.camera_fps, self.near_clip_m = camera_fps, near_clip_m
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.open_gap_m = cfg.gripper_open_gap_m
        self._lock = threading.RLock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._frame_cv = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

        m = self.model
        self._jid = {j: m.joint(j).id for j in ARM_JOINTS}
        self._qadr = {j: int(m.jnt_qposadr[i]) for j, i in self._jid.items()}
        self._aid = {j: m.actuator(j).id for j in (*ARM_JOINTS, GRIPPER)}
        self._grip_qadr = int(m.joint("gripper").qposadr[0])
        self._grip_left_qadr = int(m.joint("gripper_left").qposadr[0])
        self._gripper_body = m.body("gripper").id
        self._widen_joint_ranges()
        self._filter_collisions()
        self._objects = [m.body(i).name for i in range(m.nbody)
                         if m.body_jntnum[i] == 1 and m.jnt_type[m.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE]
        # Top-view radius of each object (largest over its geoms' boxes), for keeping shifted objects apart.
        self._radius_xy = np.array([max(float(np.linalg.norm(np.abs(m.geom_pos[g][:2]) + np.abs(m.geom_aabb[g][:2])) +
                                              np.linalg.norm(m.geom_aabb[g][3:5]))
                                        for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(n).id)
                                    for n in self._objects])
        self._hand_body = m.body("hand").id if "hand" in {m.body(i).name for i in range(m.nbody)} else None
        self._hand_mocap = int(m.body_mocapid[self._hand_body]) if self._hand_body is not None else None
        self._hand_park = m.body_pos[self._hand_body].copy() if self._hand_body is not None else None
        self._robot_bodies = {i for i in range(1, m.nbody) if m.body_rootid[i] == m.body("base").id}
        # Ground truth for tests and the benchmark, updated every physics step (see track()).
        self.max_z: dict[str, float] = {}
        self.hand_contact_n = 0.0  # largest arm-hand contact force since the last reset_tracking()
        self._home = self.data.qpos.copy()  # object layout from the XML
        self.torque = False
        self.reset()

        # Pinhole render (square, wide) remapped into the calibrated fisheye image.
        self.render_px = render_px
        f = (render_px / 2.0) / np.tan(np.radians(render_hfov_deg) / 2.0)
        self.K_render = np.array([[f, 0, (render_px - 1) / 2.0], [0, f, (render_px - 1) / 2.0], [0, 0, 1.0]])
        self._map_x, self._map_y = self._fisheye_maps(np.radians(render_hfov_deg) / 2.0 - np.radians(2.0))
        m.vis.global_.offwidth = m.vis.global_.offheight = render_px

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _build_model() -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(SIM_SCENE_PATH))
        for body_name, (centre, half) in _FINGER_PADS.items():
            body = spec.body(body_name)
            for g in body.geoms:
                if g.contype or g.conaffinity:
                    g.contype = g.conaffinity = 0
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(centre), group=3,
                          friction=[1.5, 0.02, 0.002], condim=4, contype=1, conaffinity=1)
        return spec.compile()

    def _widen_joint_ranges(self) -> None:
        """The URDF/MJCF limits are tighter than the real arm (rest pose is outside them): use the
        calibrated range (soft limits + margin), so every pose the real arm can reach works here."""
        cal = load_calibration(self.cfg.calibration_file)
        soft = {**joint_limits_from_calibration(cal, self.cfg.joint_offsets_deg), **self.cfg.joint_limits_deg}
        for j in ARM_JOINTS:
            lo, hi = (np.radians(v) for v in (soft[j][0] - STOP_MARGIN_DEG, soft[j][1] + STOP_MARGIN_DEG))
            self.model.jnt_range[self._jid[j]] = (lo, hi)
            self.model.actuator_ctrlrange[self._aid[j]] = (lo, hi)
        self.model.actuator_ctrlrange[self._aid[GRIPPER]] = (0.0, self.open_gap_m / 2.0)

    def _filter_collisions(self) -> None:
        m = self.model
        robot = {i for i in range(1, m.nbody) if m.body_rootid[i] == m.body("base").id}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue  # visual only
            if m.geom_bodyid[g] in robot:
                m.geom_contype[g], m.geom_conaffinity[g] = _ARM_CONTYPE, _WORLD_CONTYPE
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = _WORLD_CONTYPE, _WORLD_CONTYPE | _ARM_CONTYPE

    def _fisheye_maps(self, max_theta: float) -> tuple[np.ndarray, np.ndarray]:
        """For each fisheye pixel, where it falls in the pinhole render (-1 = outside: black)."""
        w, h = self.cam_model.image_size
        K, D = self.cam_model.K, self.cam_model.D
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        theta_d = np.hypot((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1])
        pts = np.stack([u, v], -1).reshape(-1, 1, 2)
        xy = cv2.fisheye.undistortPoints(pts, K, D).reshape(h, w, 2)
        theta = np.arctan(np.hypot(xy[..., 0], xy[..., 1]))
        bad = (theta_d > max_theta + 0.1) | (theta > max_theta) | ~np.isfinite(xy).all(-1)
        Kr = self.K_render
        mx = (Kr[0, 0] * xy[..., 0] + Kr[0, 2]).astype(np.float32)
        my = (Kr[1, 1] * xy[..., 1] + Kr[1, 2]).astype(np.float32)
        mx[bad] = my[bad] = -1.0
        return mx, my

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mujoco-sim", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def reset(self, jitter_m: float = 0.0, seed: int | None = None) -> None:
        """Objects back to the XML layout (optionally shifted by up to `jitter_m` and turned), arm at
        rest, torque off."""
        rng = np.random.default_rng(seed)
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._home
            adrs = [int(self.model.jnt_qposadr[self.model.body(n).jntadr[0]]) for n in self._objects]
            for _ in range(100):  # redraw shifts that make objects overlap (physics would throw them apart)
                self.data.qpos[:] = self._home
                for adr in adrs:
                    if jitter_m > 0:
                        self.data.qpos[adr:adr + 2] += rng.uniform(-jitter_m, jitter_m, 2)
                        yaw = rng.uniform(-np.pi, np.pi)
                        q = np.empty(4)
                        mujoco.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                                           self.data.qpos[adr + 3:adr + 7].copy())
                        self.data.qpos[adr + 3:adr + 7] = q
                xy = np.array([self.data.qpos[a:a + 2] for a in adrs])
                gap = np.linalg.norm(xy[:, None] - xy[None], axis=-1) - self._radius_xy[:, None] - self._radius_xy[None]
                if jitter_m <= 0 or (gap + np.eye(len(adrs)) * 1.0).min() >= OBJECT_GAP_M:
                    break
            for j, q in zip(ARM_JOINTS, self.cfg.poses["rest"]):
                self.data.qpos[self._qadr[j]] = np.radians(q)
                self.data.ctrl[self._aid[j]] = np.radians(q)
            self.data.qpos[self._grip_qadr] = self.data.qpos[self._grip_left_qadr] = 0.0
            self.data.ctrl[self._aid[GRIPPER]] = 0.0
            self._set_torque(False)
            if self._hand_mocap is not None:
                self.data.mocap_pos[self._hand_mocap] = self._hand_park
                self.data.mocap_quat[self._hand_mocap] = (1, 0, 0, 0)
            mujoco.mj_forward(self.model, self.data)
            self.reset_tracking()

    # ------------------------------------------------------------------ props and ground truth
    # Ready-made hand placement (palm centre m, yaw deg) for scripts/agent.py --hand-down.
    HAND_PLACEMENTS = {
        "down": ([0.24, 0.08, 0.035], 150.0),  # lying palm down on the table
    }

    def set_hand(self, pos_m, yaw_deg: float = 0.0) -> None:
        """Put the hand prop palm down at `pos_m` (palm centre, base frame; fingers along +x), turned by
        yaw_deg about z."""
        if self._hand_mocap is None:
            raise RuntimeError("this scene has no hand prop")
        h = np.radians(yaw_deg) / 2.0
        with self._lock:
            self.data.mocap_pos[self._hand_mocap] = np.asarray(pos_m, dtype=float)
            self.data.mocap_quat[self._hand_mocap] = (np.cos(h), 0.0, 0.0, np.sin(h))
            mujoco.mj_forward(self.model, self.data)

    def hide_hand(self) -> None:
        if self._hand_mocap is not None:
            with self._lock:
                self.data.mocap_pos[self._hand_mocap] = self._hand_park

    def set_object(self, name: str, pos_m, yaw_deg: float = 0.0) -> None:
        """Place a free object (resting pose: give z = its half height)."""
        if name not in self._objects:
            raise ValueError(f"no object {name!r}; objects: {self._objects}")
        with self._lock:
            adr = int(self.model.jnt_qposadr[self.model.body(name).jntadr[0]])
            home = self._home[adr + 3:adr + 7].copy()
            q = np.empty(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)]), home)
            self.data.qpos[adr:adr + 3] = pos_m
            self.data.qpos[adr + 3:adr + 7] = q
            dadr = int(self.model.jnt_dofadr[self.model.body(name).jntadr[0]])
            self.data.qvel[dadr:dadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def reset_tracking(self) -> None:
        with self._lock:
            self.max_z = {n: float(self.data.body(n).xpos[2]) for n in self._objects}
            self.hand_contact_n = 0.0

    def _track(self) -> None:
        """Highest point each object reached and the strongest arm-hand contact (called under the lock)."""
        for n in self._objects:
            z = float(self.data.body(n).xpos[2])
            if z > self.max_z.get(n, -np.inf):
                self.max_z[n] = z
        if self._hand_body is None:
            return
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]
            if self._hand_body in (b1, b2) and ({b1, b2} & self._robot_bodies):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                self.hand_contact_n = max(self.hand_contact_n, float(abs(f6[0])))

    # ------------------------------------------------------------------ bus-facing (any thread)
    def _set_torque(self, on: bool) -> None:
        if on:
            self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        else:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.torque = on

    def set_torque(self, on: bool) -> None:
        with self._lock:
            self._set_torque(on)

    def set_goal(self, motor: str, value: float) -> None:
        """Arm joints in deg (URDF convention), gripper in % open."""
        with self._lock:
            a = self._aid[motor]
            if motor == GRIPPER:
                target = float(np.clip(value, 0.0, 100.0)) / 100.0 * self.open_gap_m / 2.0
            else:
                target = np.radians(value)
            lo, hi = self.model.actuator_ctrlrange[a]
            self.data.ctrl[a] = min(max(target, lo), hi)

    def position(self, motor: str) -> float:
        with self._lock:
            if motor == GRIPPER:
                return 100.0 * float(self.data.qpos[self._grip_qadr]) / (self.open_gap_m / 2.0)
            return float(np.degrees(self.data.qpos[self._qadr[motor]]))

    def load(self, motor: str) -> int:
        """Actuator force as 0.1 % of its limit (like Present_Load); 0 with torque off."""
        with self._lock:
            a = self._aid[motor]
            limit = max(abs(self.model.actuator_forcerange[a]).max(), 1e-6)
            return int(round(1000.0 * float(self.data.actuator_force[a]) / limit)) if self.torque else 0

    def qpos(self) -> np.ndarray:
        with self._lock:
            return self.data.qpos.copy()

    def mocap_state(self, hide_parked: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(mocap_pos, mocap_quat) of the props; a parked hand is moved under the floor (out of every view)."""
        with self._lock:
            pos, quat = self.data.mocap_pos.copy(), self.data.mocap_quat.copy()
        if hide_parked and self._hand_mocap is not None and np.allclose(pos[self._hand_mocap], self._hand_park):
            pos[self._hand_mocap] = [0.0, 0.0, -5.0]
        return pos, quat

    def object_positions(self) -> dict[str, list[float]]:
        with self._lock:
            return {n: [round(float(v), 3) for v in self.data.body(n).xpos] for n in self._objects}

    def render_truth(self, T_base_cam: np.ndarray, K: np.ndarray, size: tuple[int, int],
                     near_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Exact view from a pinhole camera (OpenCV convention) at T_base_cam: (RGB, depth along the optical
        axis in m). Ground truth for evaluating depth estimation; uses its own renderer in the calling
        thread. Also sets self.last_robot_mask: pixels showing the robot itself."""
        w, h = size
        renderer = mujoco.Renderer(self.model, height=h, width=w)
        try:
            with self._lock:
                renderer.update_scene(self.data, camera=mujoco.MjvCamera())
            set_pinhole_camera(renderer.scene, T_base_cam, K, (w, h), near_m or self.near_clip_m)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            with self._lock:
                renderer.update_scene(self.data, camera=mujoco.MjvCamera())
            set_pinhole_camera(renderer.scene, T_base_cam, K, (w, h), near_m or self.near_clip_m)
            depth = renderer.render().astype(np.float64)
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            with self._lock:
                renderer.update_scene(self.data, camera=mujoco.MjvCamera())
            set_pinhole_camera(renderer.scene, T_base_cam, K, (w, h), near_m or self.near_clip_m)
            seg = renderer.render()
        finally:
            renderer.close()
        geom = seg[..., 0]
        objtype = seg[..., 1]
        robot_geoms = np.array([self.model.geom_bodyid[g] in self._robot_bodies for g in range(self.model.ngeom)])
        self.last_robot_mask = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom >= 0) & \
            robot_geoms[np.clip(geom, 0, self.model.ngeom - 1)]
        # The renderer linearises the depth buffer with the model's near/far planes; the camera above uses
        # its own near plane, so undo that and linearise again with it.
        m = self.model
        near_model, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
        near = near_m or self.near_clip_m
        with np.errstate(divide="ignore", invalid="ignore"):
            zbuf = (1.0 - near_model / depth) / (1.0 - near_model / far)
            depth = near / (1.0 - zbuf * (1.0 - near / far))
        depth[~np.isfinite(depth) | (depth <= 0) | (depth > 0.99 * far)] = np.nan
        return rgb, depth

    def object_geometry(self) -> dict[str, dict]:
        """Each free object's centre and highest point (m) right now, from its geoms' bounding boxes."""
        out = {}
        with self._lock:
            for n in self._objects:
                b = self.model.body(n).id
                tops = []
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] != b:
                        continue
                    R = self.data.geom_xmat[g].reshape(3, 3)
                    half = self.model.geom_aabb[g][3:]
                    tops.append(float(self.data.geom_xpos[g][2] + np.abs(R[2]) @ half))
                out[n] = {"center": self.data.body(n).xpos.copy(), "top_z": max(tops)}
        return out

    # ------------------------------------------------------------------ camera (any thread)
    def last_frame(self) -> np.ndarray | None:
        """The newest wrist-camera frame (BGR), without waiting."""
        with self._frame_cv:
            return self._frame

    def read_frame(self, newer_than: float | None, timeout_s: float) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + timeout_s
        with self._frame_cv:
            while self._frame is None or (newer_than is not None and self._stamp <= newer_than):
                left = deadline - time.monotonic()
                if left <= 0 or not self._running:
                    raise TimeoutError(f"no new sim camera frame within {timeout_s}s" +
                                       (f" ({self.error})" if self.error else ""))
                self._frame_cv.wait(left)
            return self._frame.copy(), self._stamp

    # ------------------------------------------------------------------ thread
    def _camera_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self._gripper_body].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self._gripper_body]
        return T @ self.cam_model.T_hand_camera

    def _loop(self) -> None:
        try:
            renderer = mujoco.Renderer(self.model, height=self.render_px, width=self.render_px)
        except Exception as e:  # noqa: BLE001
            self.error = f"renderer: {e!r}"
            log.error("sim camera unavailable: %s", self.error)
            renderer = None
        dt = self.model.opt.timestep
        wall0, sim0 = time.perf_counter(), self.data.time
        next_frame = 0.0
        try:
            while self._running:
                now = time.perf_counter()
                with self._lock:
                    if self.data.time < sim0:  # reset() rewound the clock
                        wall0, sim0 = now, self.data.time
                    target = sim0 + (now - wall0)
                    steps = 0
                    while self.data.time < target and steps < 100:
                        mujoco.mj_step(self.model, self.data)
                        steps += 1
                    if steps:
                        self._track()
                    if self.data.time < target - 0.1:  # cannot keep up: run slower than real time
                        wall0, sim0 = now, self.data.time
                    draw = renderer is not None and now >= next_frame
                    if draw:
                        renderer.update_scene(self.data, camera=mujoco.MjvCamera())
                        set_pinhole_camera(renderer.scene, self._camera_pose(), self.K_render,
                                           (self.render_px, self.render_px), self.near_clip_m)
                if draw:
                    next_frame = now + 1.0 / self.camera_fps
                    rgb = renderer.render()
                    fish = cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self._map_x, self._map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    with self._frame_cv:
                        self._frame, self._stamp = fish, time.monotonic()
                        self._frame_cv.notify_all()
                time.sleep(max(0.0, dt - (time.perf_counter() - now)))
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            log.exception("simulation stopped")
        finally:
            if renderer is not None:
                renderer.close()
            with self._frame_cv:
                self._frame_cv.notify_all()


class MujocoBus:
    """The FeetechBus calls SO101Arm makes, answered by the simulation (raw ticks in and out)."""

    def __init__(self, world: SimWorld, arm: SO101Arm):
        self.world, self.arm = world, arm
        self._lock = threading.RLock()

    @property
    def torque(self) -> bool:
        return self.world.torque

    def read(self, register: str, motor: str, num_retry: int = 2) -> int:
        if register == "Present_Position":
            v = self.world.position(motor)
            return self.arm.gripper_to_ticks(v) if motor == GRIPPER else self.arm.deg_to_ticks(motor, v)
        if register == "Present_Load":
            return self.world.load(motor)
        if register == "Torque_Enable":
            return int(self.world.torque)
        return 0

    def sync_read(self, register: str, motors=None, num_retry: int = 2) -> dict[str, int]:
        return {m: self.read(register, m) for m in (motors or MOTOR_IDS)}

    def sync_write(self, register: str, values: dict[str, int], num_retry: int = 1) -> None:
        if register != "Goal_Position":
            return
        for m, ticks in values.items():
            v = self.arm.ticks_to_gripper(ticks) if m == GRIPPER else self.arm.ticks_to_deg(m, ticks)
            self.world.set_goal(m, v)

    def write(self, register: str, motor: str, value: int, num_retry: int = 2) -> None:
        if register == "Torque_Enable":
            self.world.set_torque(bool(value))

    def enable_torque(self, motors=None, num_retry: int = 2) -> None:
        self.world.set_torque(True)

    def disable_torque(self, motors=None, num_retry: int = 2) -> None:
        self.world.set_torque(False)

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.world.set_torque(False)


def make_mujoco_arm(cfg: RobotConfig, world: SimWorld) -> SO101Arm:
    """The real SO101Arm (limits, interpolation, stopping, sag compensation) on the simulated bus."""
    arm = SO101Arm("/dev/null", cfg.calibration_file, cfg.joint_limits_deg or None, cfg.max_speed_deg_s,
                   joint_offsets_deg=cfg.joint_offsets_deg, gripper_inverted=cfg.gripper_inverted)
    arm.calibration = load_calibration(cfg.calibration_file)
    arm.joint_limits = {**joint_limits_from_calibration(arm.calibration, arm.joint_offsets), **cfg.joint_limits_deg}
    arm.bus = MujocoBus(world, arm)
    arm.connect = lambda enable_torque=True: None  # type: ignore[method-assign]
    return arm


class SimWristCamera:
    """WristCamera interface over SimWorld's rendered fisheye frames."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.reconnects = 0

    def connect(self, warmup_s: float = 0.0) -> None:
        pass

    def set_exposure(self, value) -> None:
        pass

    def read(self, newer_than: float | None = None, timeout_s: float = 8.0) -> tuple[np.ndarray, float]:
        return self.world.read_frame(newer_than, timeout_s)

    def read_fresh(self, settle_s: float = 0.15) -> np.ndarray:
        t = time.monotonic() + settle_s
        time.sleep(settle_s)
        return self.read(newer_than=t)[0]

    def wait_until_streaming(self, timeout_s: float) -> bool:
        try:
            self.read(newer_than=time.monotonic(), timeout_s=timeout_s)
            return True
        except TimeoutError:
            return False

    def disconnect(self) -> None:
        pass


class SimVideo:
    """Record a run in the sim as an mp4: the whole scene from a 3/4 view with the wrist camera inset and
    a caption, sped up `speedup` times (labelled). A context manager; frames are drawn in its own thread.

        with SimVideo(world, "run.mp4", caption="pick up the red cube") as v:
            ...  # run the robot
            v.caption = "done"
    """

    # Camera: from the front left, a little above, framing the arm and the objects in front of it.
    LOOKAT, DISTANCE, AZIMUTH, ELEVATION = (0.2, 0.0, 0.05), 0.6, 150.0, -35.0

    def __init__(self, world: SimWorld, path, caption: str = "", fps: float = 10.0, speedup: float = 4.0,
                 size: tuple[int, int] = (800, 600), hold_s: float = 2.0):
        self.world, self.path, self.caption = world, Path(path), caption
        self.fps, self.speedup, self.size, self.hold_s = fps, speedup, size, hold_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def __enter__(self) -> SimVideo:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.error:
            log.warning("sim video %s: %s", self.path, self.error)

    def _frame(self, r) -> np.ndarray:
        w, h = self.size
        img = r.render_qpos(self.world.qpos(), *self.world.mocap_state())
        wrist = self.world.last_frame()
        if wrist is not None:  # inset, top right
            iw = w // 3
            ih = int(wrist.shape[0] * iw / wrist.shape[1])
            img[8:8 + ih, w - iw - 8:w - 8] = cv2.resize(wrist, (iw, ih), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (w - iw - 8, 8), (w - 8, 8 + ih), (255, 255, 255), 1)
            cv2.putText(img, "wrist camera", (w - iw - 4, 24 + ih), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(img, self.caption[:90], (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"{self.speedup:g}x speed", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        return img

    def _run(self) -> None:
        from .viz import LiveRenderer

        raw = self.path.with_suffix(".raw.mp4")
        writer = None
        try:
            r = LiveRenderer(size=self.size, open_gap_m=self.world.open_gap_m, scene_path=SIM_SCENE_PATH)
            r.cam.lookat[:] = self.LOOKAT
            r.cam.distance, r.cam.azimuth, r.cam.elevation = self.DISTANCE, self.AZIMUTH, self.ELEVATION
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), self.fps * self.speedup, self.size)
            next_t = time.monotonic()
            while True:
                frame = self._frame(r)
                writer.write(frame)
                if self._stop.is_set():
                    for _ in range(int(self.hold_s * self.fps * self.speedup)):  # last frame stays on screen
                        writer.write(frame)
                    break
                next_t += 1.0 / self.fps
                self._stop.wait(max(0.0, next_t - time.monotonic()))
            r.close()
        except Exception as e:  # noqa: BLE001 - a broken video must not break the run
            self.error = repr(e)
        finally:
            if writer is not None:
                writer.release()
        err = encode_h264(raw, self.path)
        if err:
            self.error = err


def encode_h264(raw: Path, path: Path) -> str | None:
    """Re-encode `raw` as H.264 at `path` (OpenCV's mp4v does not play in browsers or on GitHub) and delete
    `raw`. If ffmpeg is missing or fails, `raw` is moved to `path` as it is. Returns an error message or None."""
    import shutil
    import subprocess

    raw, path = Path(raw), Path(path)
    if not raw.exists():
        return f"{raw} was not written"
    err = "ffmpeg not found"
    if shutil.which("ffmpeg"):
        res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt",
                              "yuv420p", "-crf", "26", str(path)], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            raw.unlink()
            return None
        err = f"ffmpeg: {res.stderr.strip()[:200]}"
    raw.replace(path)
    return err
