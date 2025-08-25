from __future__ import annotations
import argparse
import cv2
from .core import MotionExtractor, DiffParams, Mog2Params
from .io import FrameSource, FrameSink

def main() -> None:
    p = argparse.ArgumentParser(description="Extract motion mask from video or image sequence.")
    p.add_argument("-i", "--input", required=True,
                   help="Input video path, webcam index (e.g. 0), directory, glob, or printf pattern (e.g. img_%06d.png)")
    p.add_argument("-o", "--output", help="Output video path (e.g. out.mp4). Not required if using --out-seq/--out-dir.")
    p.add_argument("--method", choices=["diff", "mog2"], default="diff")

    # Diff params
    p.add_argument("--thresh", type=int, default=25)
    p.add_argument("--blur", type=int, default=5)
    p.add_argument("--morph", type=int, default=3)
    p.add_argument("--iters", type=int, default=1)

    # MOG2 params
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--varthr", type=float, default=16.0)
    p.add_argument("--no-shadows", action="store_true")

    # I/O extras
    p.add_argument("--in-fps", type=float, default=None,
                   help="FPS hint for image sequences or sources that report 0.")
    p.add_argument("--out-fps", type=float, default=None,
                   help="Output FPS (defaults to input FPS).")
    p.add_argument("--out-seq", type=str, default=None,
                   help='Image sequence template, e.g. "out/frame_%06d.png"')
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory to write frame_000000.png, frame_000001.png, ...")

    args = p.parse_args()

    diff_params = DiffParams(
        thresh=args.thresh,
        blur_kernel=(args.blur, args.blur) if args.blur > 0 else (0, 0),
        morph_kernel=(args.morph, args.morph) if args.morph > 0 else (0, 0),
        morph_iters=args.iters,
    )
    mog2_params = Mog2Params(
        history=args.history, var_threshold=args.varthr, detect_shadows=not args.no_shadows
    )

    extractor = MotionExtractor(method=args.method, diff_params=diff_params, mog2_params=mog2_params)

    # Input
    try:
        src = int(args.input)  # webcam index if numeric
    except ValueError:
        src = args.input
    fs = FrameSource(src, in_fps=args.in_fps)
    info = fs.info

    # Output
    out_fps = args.out_fps or info.fps
    sink = FrameSink(
        output=args.output,
        frame_size=(info.width, info.height),
        fps=out_fps,
        out_seq=args.out_seq,
        out_dir=args.out_dir,
    )

    while True:
        ok, frame = fs.read()
        if not ok:
            break
        mask = extractor.process(frame)
        sink.write(mask)

    fs.release()
    sink.release()
