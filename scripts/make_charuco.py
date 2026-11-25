"""Generate a ChArUco calibration board: an exact-scale PDF for printing, or a PNG for a phone screen.

PDF: print at 100% / "Actual size" (NOT "fit to page"), then measure one square with a ruler and put the
measured value in config/robot.yaml (calibration_board.square_mm) if it differs from the nominal.

Phone (--phone): a full-screen PNG. Its physical size depends on the phone and viewer zoom, so after
opening it full screen, measure the board's long edge and set square_mm = measured_mm / rows.
Keep the same zoom for the whole session.
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--phone", action="store_true", help="full-screen PNG for a phone instead of a PDF")
    ap.add_argument("--phone-px", type=int, nargs=2, default=(1170, 2532), metavar=("W", "H"))
    args = ap.parse_args()

    if args.phone:
        make_phone_png(args)
        return
    args.out = args.out or str(REPO_ROOT / "docs" / "calibration" / "charuco_board.pdf")

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


def make_phone_png(args) -> None:
    """Board as large as the screen allows, black border so the edge is easy to measure."""
    w, h = args.phone_px
    margin = round(0.04 * w)  # unchanged from v1 so the on-screen square size (and the caliper measurement) stays the same
    square = min((w - 2 * margin) // args.cols, (h - 2 * margin) // args.rows)
    board = cv2.aruco.CharucoBoard(
        (args.cols, args.rows),
        1.0,
        args.marker_mm / args.square_mm,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    bw, bh = square * args.cols, square * args.rows
    # Dark surround with a thin white quiet zone: a mostly white screen blooms in the camera and
    # washes out the thin black marker borders (seen on the first capture attempt).
    img = np.full((h, w), 40, np.uint8)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    q = square // 6  # white quiet zone, > 1 marker cell
    img[y0 - q : y0 + bh + q, x0 - q : x0 + bw + q] = 255
    img[y0 : y0 + bh, x0 : x0 + bw] = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    out = args.out or str(REPO_ROOT / "docs" / "calibration" / "charuco_phone.png")
    cv2.imwrite(out, img)
    print(f"wrote {out}: {args.cols}x{args.rows} squares of {square}px; measure the long edge ({args.rows} squares)")


if __name__ == "__main__":
    main()
