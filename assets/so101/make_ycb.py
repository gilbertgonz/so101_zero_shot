"""Download and convert the YCB objects the sim benchmark uses (assets/so101/ycb/, git-ignored).

YCB (https://www.ycbbenchmarks.com, CC BY 4.0): real household objects scanned with textures. For each object
in OBJECTS this fetches the google_16k scan, turns it into its resting pose (the given rotation, then let
it settle on a table in MuJoCo so the pose is really stable; base on z = 0, centred in x/y), halves the
texture to 1024 px, and writes ycb/<key>/mesh.obj + texture.png. The
list with names, masses and sizes goes to ycb/manifest.json, which mujoco_sim.py reads to add the objects.

Run once (about 200 MB of downloads, cached in ycb/_downloads/):  .venv/bin/python assets/so101/make_ycb.py
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import cv2
import mujoco
import numpy as np

OUT = Path(__file__).parent / "ycb"
URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{id}_google_16k.tgz"
TEXTURE_PX = 1024
# YCB's plastic strawberry is 4.5 cm wide, bigger than a real one and too wide for the gripper's 4.6 cm;
# its banana is 3.6-4 cm thick, too close to it (scaled: a 13 cm banana; still hard: light and round, it
# rolls away from fingers that brush it).
SCALE = {"strawberry": 0.8, "banana": 0.75}

# key: (YCB id, what the object is called in requests, mass kg (YCB), rotation to its resting pose
# (x, y, z Euler deg, applied to the scan), kind). Boxes lie on a narrow side: upright they are too tall for a
# straight-down grasp, flat too wide for the 46 mm gripper.
OBJECTS: dict[str, tuple[str, str, float, tuple[float, float, float], str]] = {
    # food
    "sugar_box": ("004_sugar_box", "box of sugar", 0.514, (90, 0, 0), "food"),
    "pudding_box": ("008_pudding_box", "box of chocolate pudding mix", 0.187, (0, 0, 0), "food"),
    "gelatin_box": ("009_gelatin_box", "box of strawberry gelatin", 0.097, (90, 90, 0), "food"),
    "banana": ("011_banana", "banana", 0.066, (0, 0, 0), "fruit"),
    "strawberry": ("012_strawberry", "strawberry", 0.018, (0, 0, 0), "fruit"),
    "apple": ("013_apple", "apple", 0.068, (0, 0, 0), "fruit"),
    "peach": ("015_peach", "peach", 0.033, (0, 0, 0), "fruit"),
    "pear": ("016_pear", "pear", 0.049, (0, 0, 0), "fruit"),
    "plum": ("018_plum", "plum", 0.025, (0, 0, 0), "fruit"),
    # tools and utensils
    "phillips_screwdriver": ("043_phillips_screwdriver", "screwdriver", 0.097, (0, 0, 0), "tool"),
    "hammer": ("048_hammer", "hammer", 0.665, (0, 0, 0), "tool"),
    "adjustable_wrench": ("042_adjustable_wrench", "adjustable wrench", 0.252, (0, 0, 0), "tool"),
    "scissors": ("037_scissors", "scissors", 0.082, (0, 0, 0), "tool"),
    "medium_clamp": ("050_medium_clamp", "clamp", 0.059, (0, 0, 0), "tool"),
    "spatula": ("033_spatula", "spatula", 0.052, (0, 0, 0), "utensil"),
    "spoon": ("031_spoon", "spoon", 0.030, (0, 0, 0), "utensil"),
    "fork": ("030_fork", "fork", 0.034, (0, 0, 0), "utensil"),
    "knife": ("032_knife", "knife", 0.031, (0, 0, 0), "utensil"),
    # other things (distractors)
    "mustard_bottle": ("006_mustard_bottle", "mustard bottle", 0.603, (0, 0, 0), "other"),
    "tomato_soup_can": ("005_tomato_soup_can", "can of tomato soup", 0.349, (0, 0, 0), "food"),
    "padlock": ("038_padlock", "padlock", 0.208, (0, 0, 0), "other"),
    "sponge": ("026_sponge", "sponge", 0.006, (0, 0, 0), "other"),
    "rubiks_cube": ("077_rubiks_cube", "Rubik's cube", 0.094, (0, 0, 0), "other"),
}


def _rot(euler_deg) -> np.ndarray:
    rx, ry, rz = np.radians(euler_deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _fetch(ycb_id: str) -> Path:
    cache = OUT / "_downloads" / f"{ycb_id}.tgz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(id=ycb_id), tmp)
        tmp.rename(cache)
    return cache


def _write_obj(path: Path, lines: list[str], v: np.ndarray, R: np.ndarray) -> None:
    """The scan's OBJ with vertices v and normals turned by R (texture coordinates and faces unchanged)."""
    out, vi = [], 0
    for ln in lines:
        if ln.startswith("v "):
            out.append("v {:.6f} {:.6f} {:.6f}".format(*v[vi]))
            vi += 1
        elif ln.startswith("vn "):
            out.append("vn {:.6f} {:.6f} {:.6f}".format(*(R @ np.array(ln.split()[1:4], float))))
        elif ln.startswith(("vt ", "f ")):
            out.append(ln)
    path.write_text("\n".join(out) + "\n")


def _settled_rotation(mesh: Path, mass: float, seconds: float = 3.0) -> np.ndarray:
    """Drop the mesh (base on z = 0) onto a plane and let it come to rest: the rotation it ended up in."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    spec.add_mesh(name="m", file=str(mesh))
    b = spec.worldbody.add_body(name="obj", pos=[0, 0, 0.001])
    b.add_freejoint()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m", mass=mass, friction=[1.5, 0.02, 0.002], condim=4)
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(round(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, d.body("obj").xquat)
    return R.reshape(3, 3)


def _convert(key: str, ycb_id: str, euler_deg, mass: float, scale: float = 1.0) -> dict:
    with tarfile.open(_fetch(ycb_id)) as tar:
        obj_txt = tar.extractfile(f"{ycb_id}/google_16k/textured.obj").read().decode()
        tex = cv2.imdecode(np.frombuffer(tar.extractfile(f"{ycb_id}/google_16k/texture_map.png").read(), np.uint8),
                           cv2.IMREAD_COLOR)
    R = _rot(euler_deg)
    lines = obj_txt.splitlines()
    v0 = scale * np.array([ln.split()[1:4] for ln in lines if ln.startswith("v ")], float)
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)

    def place(R):  # turned by R, base on z = 0, centred in x/y
        v = v0 @ R.T
        lo, hi = v.min(0), v.max(0)
        return v + [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]], hi - lo

    v, _ = place(R)
    _write_obj(d / "mesh.obj", lines, v, R)
    R = _settled_rotation(d / "mesh.obj", mass) @ R
    v, size = place(R)
    _write_obj(d / "mesh.obj", lines, v, R)
    cv2.imwrite(str(d / "texture.png"), cv2.resize(tex, (TEXTURE_PX, TEXTURE_PX), interpolation=cv2.INTER_AREA))
    return {"size_m": [round(float(x), 4) for x in size]}


def main() -> None:
    manifest = {}
    for key, (ycb_id, label, mass, euler, kind) in OBJECTS.items():
        info = _convert(key, ycb_id, euler, mass, SCALE.get(key, 1.0))
        manifest[key] = {"ycb_id": ycb_id, "label": label, "mass_kg": mass, "kind": kind, **info}
        print(f"{key:22s} {label:32s} size {np.round(np.array(info['size_m']) * 100, 1)} cm")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} objects to {OUT}")


if __name__ == "__main__":
    main()
