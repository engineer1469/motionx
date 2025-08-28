from __future__ import annotations
import argparse
from .core import MotionExtractor, DiffParams, Mog2Params, WeightedParams
from .io import FrameSource, FrameSink

def main() -> None:
    p = argparse.ArgumentParser(description="Extract motion representation from video or image sequence.")
    p.add_argument("-i", "--input", required=True,
                   help="Input video path, webcam index (e.g. 0), directory, glob, or printf pattern (e.g. img_%%06d.png)")
    p.add_argument("-o", "--output",
                   help="Output video path (.mp4/.avi). Not required when using --out-seq or --out-dir.")
    p.add_argument("--method", choices=["diff", "mog2", "weighted", "diffgray"], default="diff",
                   help="Motion extraction method.")

    # Diff params
    p.add_argument("--thresh", type=int, default=25, help="Binary threshold (diff).")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd) for preprocessing.")
    p.add_argument("--morph", type=int, default=3, help="Morphology kernel size (diff).")
    p.add_argument("--iters", type=int, default=1, help="Morphology iterations (diff).")

    # MOG2 params
    p.add_argument("--history", type=int, default=500, help="History for MOG2 background model.")
    p.add_argument("--varthr", type=float, default=16.0, help="Variance threshold for MOG2.")
    p.add_argument("--no-shadows", action="store_true", help="Disable shadow detection for MOG2.")

    # Weighted params
    p.add_argument("--alpha", type=float, default=0.5, help="Blending weight for current frame (weighted).")
    p.add_argument("--beta", type=float, default=0.5, help="Blending weight for inverted previous frame (weighted).")
    p.add_argument("--offset", type=int, default=1, help="Frame offset for weighted motion.")
    p.add_argument("--gain", type=float, default=1.0, help="Contrast gain around mid-gray for weighted output.")

    # I/O extras
    p.add_argument("--in-fps", type=float, default=None, help="FPS hint for image sequences or sources that report 0.")
    p.add_argument("--out-fps", type=float, default=None, help="Output FPS (defaults to input FPS).")
    p.add_argument("--out-seq", type=str, default=None, help='Image sequence template, e.g. "out/frame_%%06d.png"')
    p.add_argument("--out-dir", type=str, default=None, help="Directory to write image sequence frames.")

    # Multithreading
    p.add_argument("--workers", type=int, default=1, help="Worker threads for data-parallel processing (diff/diffgray/weighted).")
    p.add_argument("--max-futures", type=int, default=32, help="Max in-flight tasks when --workers > 1.")

    args = p.parse_args()

    # normalize blur kernel to be odd or zero
    def _odd_or_zero(n: int) -> int:
        if n <= 0:
            return 0
        return n if n % 2 == 1 else n + 1

    blur = _odd_or_zero(args.blur)
    morph = args.morph if args.morph > 0 else 0

    diff_params = DiffParams(
        thresh=args.thresh,
        blur_kernel=(blur, blur) if blur > 0 else (0, 0),
        morph_kernel=(morph, morph) if morph > 0 else (0, 0),
        morph_iters=max(args.iters, 0),
    )
    mog2_params = Mog2Params(
        history=args.history,
        var_threshold=args.varthr,
        detect_shadows=not args.no_shadows,
    )
    weighted_params = WeightedParams(
        alpha=args.alpha,
        beta=args.beta,
        offset=args.offset,
        gain=args.gain,
    )

    # Input
    try:
        src = int(args.input)
    except ValueError:
        src = args.input
    fs = FrameSource(src, in_fps=args.in_fps)
    info = fs.info

    # Output
    out_fps = args.out_fps or info.fps
    is_color = (args.method == "weighted")
    sink = FrameSink(
        output=args.output,
        frame_size=(info.width, info.height),
        fps=out_fps,
        out_seq=args.out_seq,
        out_dir=args.out_dir,
        is_color=is_color,
    )

    # --------------------
    # Sequential vs MT run
    # --------------------
    if args.workers > 1 and args.method in ("diff", "diffgray", "weighted"):
        from .mt import run_parallel
        try:
            run_parallel(
                method=args.method,
                fs=fs,
                sink=sink,
                diff_params=diff_params,
                weighted_params=weighted_params,
                workers=args.workers,
                max_futures=args.max_futures,
            )
        finally:
            fs.release()
            sink.release()
        return

    # Fallback: sequential (also used for mog2)
    extractor = MotionExtractor(
        method=args.method,
        diff_params=diff_params,
        mog2_params=mog2_params,
        weighted_params=weighted_params,
    )

    while True:
        ok, frame = fs.read()
        if not ok:
            break
        processed = extractor.process(frame)
        sink.write(processed)

    fs.release()
    sink.release()
