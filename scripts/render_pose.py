"""Render the SO-101 in a given joint pose (deg) to a PNG, from a side and a 3/4 view.

Used to show a human how to place the arm, e.g.:
  render_pose.py --pose 0 0 0 0 0 --out docs/poses/zero.png --title "Pose A"
  render_pose.py --tip 0.20 0 0.0 --out docs/poses/down.png   (top-down IK to a tip position)
"""

import argparse
import cv2
import numpy as np

from so101_zero_shot.config import RobotConfig
from so101_zero_shot.viz import render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pose", type=float, nargs=5, metavar="DEG")
    g.add_argument("--tip", type=float, nargs=3, metavar="M", help="top-down IK target in base frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if args.tip is not None:
        res = RobotConfig.load().make_kinematics().ik(args.tip, np.array([0, -20, 20, 60, 0.0]))
        if not res.success:
            raise SystemExit(f"IK failed for {args.tip}: {res}")
        q = res.joints_deg
    else:
        q = np.array(args.pose)
    img = render(q)
    if args.title:
        cv2.putText(img, args.title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 160), 2)
    cv2.imwrite(args.out, img)
    print("joints (deg):", np.round(q, 1), "->", args.out)


if __name__ == "__main__":
    main()
