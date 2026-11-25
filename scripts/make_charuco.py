"""Generate a printable ChArUco calibration board as an exact-scale PDF.

Print at 100% / "Actual size" (NOT "fit to page"), then measure one square with a ruler and put the
measured value in config/robot.yaml (calibration_board.square_mm) if it differs from the nominal.
"""

import argparse

import cv2
import numpy as np
from PIL import Image

from so101_zero_shot.config import CONFIG_DIR, REPO_ROOT

PAPER_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", choices=sorted(PAPER_MM), default="letter")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--square-mm", type=float, default=35.0)
    ap.add_argument("--marker-mm", type=float, default=26.0)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "calibration" / "charuco_board.pdf"))
    args = ap.parse_args()

    paper_w, paper_h = PAPER_MM[args.paper]
    board_w, board_h = args.cols * args.square_mm, args.rows * args.square_mm
    if board_w > paper_w - 12 or board_h > paper_h - 12:
        raise SystemExit(f"board {board_w}x{board_h} mm does not fit on {args.paper} with margins")

    px_per_mm = args.dpi / 25.4
    board = cv2.aruco.CharucoBoard(
        (args.cols, args.rows),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    bw, bh = round(board_w * px_per_mm), round(board_h * px_per_mm)
    board_img = board.generateImage((bw, bh), marginSize=0, borderBits=1)

    page = np.full((round(paper_h * px_per_mm), round(paper_w * px_per_mm)), 255, np.uint8)
    y0, x0 = (page.shape[0] - bh) // 2, (page.shape[1] - bw) // 2
    page[y0 : y0 + bh, x0 : x0 + bw] = board_img
    label = (
        f"ChArUco {args.cols}x{args.rows}, square {args.square_mm} mm, marker {args.marker_mm} mm, DICT_5X5_100. "
        "Print at 100% (actual size). Check: one square = {:.0f} mm.".format(args.square_mm)
    )
    cv2.putText(page, label, (x0, y0 + bh + round(6 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, args.dpi / 300 * 0.9, 0, 2)

    Image.fromarray(page).save(args.out, "PDF", resolution=float(args.dpi))
    print(f"wrote {args.out} ({args.paper}, board {board_w:.0f}x{board_h:.0f} mm)")
    print(f"config: {CONFIG_DIR / 'robot.yaml'} -> calibration_board")


if __name__ == "__main__":
    main()
