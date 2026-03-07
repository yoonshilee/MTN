import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced screenshots from a video and save them under ./logs/."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for screenshots. Default: ./logs/video_screenshots/<video_stem>",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of screenshots to extract. Default: 4.",
    )
    parser.add_argument(
        "--prefix",
        default="shot",
        help="Filename prefix for saved screenshots. Default: shot.",
    )
    parser.add_argument(
        "--skip-first-last",
        action="store_true",
        help="Avoid sampling the very first and last frame.",
    )
    return parser.parse_args()


def build_indices(frame_count: int, count: int, skip_first_last: bool):
    if frame_count <= 0:
        return []

    count = max(1, min(count, frame_count))

    if count == 1:
        return [frame_count // 2]

    if skip_first_last and frame_count > 2:
        start = 1
        end = frame_count - 2
    else:
        start = 0
        end = frame_count - 1

    if end < start:
        start = 0
        end = frame_count - 1

    if count == 1:
        return [start + (end - start) // 2]

    step = (end - start) / (count - 1) if count > 1 else 0
    indices = [round(start + i * step) for i in range(count)]

    deduped = []
    seen = set()
    for idx in indices:
        idx = max(0, min(frame_count - 1, idx))
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)

    probe = 0
    while len(deduped) < count and probe < frame_count:
        if probe not in seen:
            deduped.append(probe)
            seen.add(probe)
        probe += 1

    return sorted(deduped)


def main():
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.output_dir is None:
        output_dir = video_path.parent.parent.parent / "logs" / "video_screenshots" / video_path.stem
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    indices = build_indices(frame_count, args.count, args.skip_first_last)

    if not indices:
        cap.release()
        raise RuntimeError(f"No frames available in video: {video_path}")

    saved_files = []
    for order, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        out_path = output_dir / f"{args.prefix}_{order:02d}_frame{frame_idx:05d}.png"
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"Failed to save screenshot: {out_path}")
        saved_files.append((out_path, frame_idx))

    cap.release()

    if not saved_files:
        raise RuntimeError("Failed to extract any screenshots from the video.")

    summary_path = output_dir / "_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"video={video_path}\n")
        f.write(f"fps={fps}\n")
        f.write(f"frame_count={frame_count}\n")
        f.write(f"saved_count={len(saved_files)}\n")
        for out_path, frame_idx in saved_files:
            timestamp = frame_idx / fps if fps > 0 else -1
            f.write(f"{out_path.name},frame={frame_idx},time_sec={timestamp:.3f}\n")

    print(f"Saved {len(saved_files)} screenshots to {output_dir}")
    for out_path, frame_idx in saved_files:
        timestamp = frame_idx / fps if fps > 0 else -1
        print(f"- {out_path} | frame={frame_idx} | time_sec={timestamp:.3f}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
