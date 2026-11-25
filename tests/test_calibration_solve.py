"""The calibration solver recovers a known camera mount and board pose from synthetic observations."""

import json

import cv2
import numpy as np
import pytest

from so101_zero_shot.calibration.board import CharucoTarget
from so101_zero_shot.calibration.solve import HAND_FRAME, Observation, board_sanity, calibrate
from so101_zero_shot.config import REPO_ROOT, RobotConfig

SIZE = (1600, 1200)
K = np.array([[484.0, 0, 805.0], [0, 484.0, 607.0], [0, 0, 1]])
D = np.array([[-0.02], [0.026], [-0.028], [0.008]])


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    T[:3, 3] = t
    return T


@pytest.fixture(scope="module")
def setup():
    cfg = RobotConfig.load()
    kin = cfg.make_kinematics()
    target = CharucoTarget(4, 6, 0.014767, 0.75, "DICT_5X5_100")
    # Camera mount and board roughly like the real solution, board flat on the table.
    X = _T([0.0, -0.5, 0.1], [0.06, 0.015, -0.05]) @ _T([np.pi, 0, 0], [0, 0, 0])
    boards = {"a": _T([0, 0, 0.1], [0.27, -0.04, 0.01]), "b": _T([0, 0, -0.2], [0.26, -0.03, 0.012])}
    rng = np.random.default_rng(1)
    obs = []
    for k, s in enumerate(boards):
        tries = 0
        while sum(o.session == s for o in obs) < 9:
            tries += 1
            assert tries < 5000
            q = np.array([0, -45, 8, 100, 0]) + rng.uniform([-20, -25, -25, -15, -60], [20, 25, 25, 15, 60])
            T_cb = np.linalg.inv(kin.fk(q, HAND_FRAME) @ X) @ boards[s]
            if T_cb[2, 3] < 0.08:
                continue
            pts, _ = cv2.fisheye.projectPoints(target.corner_points.reshape(-1, 1, 3), cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, D)
            pts = pts.reshape(-1, 2)
            cam_pts = (T_cb[:3, :3] @ target.corner_points.T).T + T_cb[:3, 3]
            ok = (cam_pts[:, 2] > 0) & (np.linalg.norm(pts - K[:2, 2], axis=1) < 450)
            if ok.sum() < 15:
                continue
            obs.append(Observation(s, f"{len(obs)}.jpg", q, pts + rng.normal(0, 0.2, pts.shape), target.corner_points.astype(float)))
    return kin, target, X, boards, obs


def test_default_is_close(setup):
    """Intrinsics from cv2.fisheye.calibrate alone: a few percent focal bias, mount within ~1 cm."""
    kin, _, X, _, obs = setup
    res = calibrate(obs, kin, SIZE)
    assert abs(res.K[0, 0] / K[0, 0] - 1) < 0.05 and res.robot_rms_px < 1.5
    assert np.linalg.norm(res.T_hand_camera[:3, 3] - X[:3, 3]) < 0.015


def test_joint_refinement_recovers_mount_and_board(setup):
    kin, target, X, boards, obs = setup
    res = calibrate(obs, kin, SIZE, refine_intrinsics=True)
    assert res.intrinsics_rms_px < 0.5 and res.robot_rms_px < 0.5
    assert np.allclose(res.K, K, atol=3.0)
    assert np.linalg.norm(res.T_hand_camera[:3, 3] - X[:3, 3]) < 0.002
    dR = res.T_hand_camera[:3, :3].T @ X[:3, :3]
    assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 0.5
    for s, T in boards.items():
        assert np.linalg.norm(res.T_base_board[s][:3, 3] - T[:3, 3]) < 0.003
        assert board_sanity(res.T_base_board[s], target.size_m)["tilt_deg"] < 0.5


def test_rejects_too_few_frames(setup):
    kin, _, _, _, obs = setup
    with pytest.raises(ValueError):
        calibrate([o for o in obs if o.session == "a"][:2] + [o for o in obs if o.session == "b"], kin, SIZE)
