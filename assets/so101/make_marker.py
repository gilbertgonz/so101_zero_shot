"""Make the sim marker's visual mesh and label texture (assets/marker_barrel.obj, assets/marker_label.png).

A plain black capsule did not read as a writing tool to the VLM. This draws a permanent marker: a white
barrel printed "PERMANENT MARKER" with a black band, used by sim_scene.xml together with a black cap,
pocket clip and end plug (primitives there). The barrel axis is the mesh's z, centred on 0.

Run once after changing it:  .venv/bin/python assets/so101/make_marker.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "assets"
RADIUS_M = 0.009
Z0_M, Z1_M = -0.045, 0.030  # barrel from the end plug to the cap
SEGMENTS = 48
TEX_W, TEX_H = 1024, 256  # u along the barrel (0 at the cap end), v around it


def label() -> np.ndarray:
    img = np.full((TEX_H, TEX_W, 3), 235, np.uint8)  # off-white barrel
    cv2.rectangle(img, (TEX_W - 90, 0), (TEX_W, TEX_H), (20, 20, 20), -1)  # black band at the plug end (u = 1)
    for row in (TEX_H // 4, 3 * TEX_H // 4):  # printed twice around, so one side always shows it
        cv2.putText(img, "PERMANENT MARKER", (40, row + 22), cv2.FONT_HERSHEY_DUPLEX, 2.0, (15, 15, 15), 5,
                    cv2.LINE_AA)
    return img


def barrel_obj() -> str:
    lines = ["# marker barrel, z = axis (m), uv: u along z, v around"]
    for k in range(SEGMENTS + 1):  # last column repeats the first with v = 1 (texture seam)
        a = 2 * np.pi * k / SEGMENTS
        for z in (Z0_M, Z1_M):
            lines.append(f"v {RADIUS_M * np.cos(a):.6f} {RADIUS_M * np.sin(a):.6f} {z:.6f}")
            lines.append(f"vt {(Z1_M - z) / (Z1_M - Z0_M):.6f} {k / SEGMENTS:.6f}")  # u = 0 at the cap: reads right
            lines.append(f"vn {np.cos(a):.6f} {np.sin(a):.6f} 0")
    for k in range(SEGMENTS):
        a0, a1, b0, b1 = 2 * k + 1, 2 * k + 2, 2 * k + 3, 2 * k + 4  # (column, end), 1-based
        lines.append(f"f {a0}/{a0}/{a0} {b0}/{b0}/{b0} {b1}/{b1}/{b1}")
        lines.append(f"f {a0}/{a0}/{a0} {b1}/{b1}/{b1} {a1}/{a1}/{a1}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "marker_label.png"), label())
    (OUT / "marker_barrel.obj").write_text(barrel_obj())
    print(f"wrote {OUT / 'marker_label.png'} and {OUT / 'marker_barrel.obj'}")


if __name__ == "__main__":
    main()
