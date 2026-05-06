#!/usr/bin/env python3
"""Create a EuRoC-style dataset whose PNGs are decoded from CI-style MP4s.

The script mirrors Basalt's CI video preprocessing:
  1. Read each mav0/cam*/data.csv in file order.
  2. Build an ffmpeg concat file pointing at the original PNGs.
  3. Encode mav0/cam*/data.mp4 with the same two-pass libx264 defaults.
  4. Decode data.mp4 sequentially with OpenCV and save the decoded frames as
     PNGs using the original timestamped filenames.

The PNG writing itself is lossless; the intended artifacts are introduced by
the intermediate lossy MP4 encode/decode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2


DEFAULT_X264OPTS = (
    "partitions=p8x8,p4x4,i8x8:keyint=1000:me=umh:merange=64:subme=6:bframes=0:ref=1"
)

MISMATCH_ALLOWLIST_DATASETS = {"V2_03_difficult", "V1_02_medium"}


def run_checked(cmd: list[str]) -> None:
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def get_euroc_cam_dirs(dataset_path: Path) -> list[Path]:
    mav0_path = dataset_path / "mav0"
    if not mav0_path.is_dir():
        raise FileNotFoundError(f"Expected EuRoC mav0 directory under {dataset_path}")

    cam_dirs = sorted(p for p in mav0_path.glob("cam*") if p.is_dir())
    if not cam_dirs:
        raise FileNotFoundError(f"No cam* directories found under {mav0_path}")
    return cam_dirs


def read_data_csv(data_csv: Path) -> tuple[list[str], list[tuple[str, str]]]:
    headers: list[str] = []
    rows: list[tuple[str, str]] = []

    with data_csv.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                headers.append(line)
                continue

            parts = stripped.split(",", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected EuRoC data.csv row in {data_csv}: {line!r}"
                )
            rows.append((parts[0], parts[1]))

    if not rows:
        raise ValueError(f"No image rows found in {data_csv}")
    return headers, rows


def write_data_csv(
    data_csv: Path, headers: list[str], rows: list[tuple[str, str]]
) -> None:
    with data_csv.open("w", encoding="utf-8") as f:
        f.writelines(headers)
        for timestamp, filename in rows:
            f.write(f"{timestamp},{filename}\n")


def compute_framerate(data_csv: Path) -> int:
    _, rows = read_data_csv(data_csv)
    if len(rows) < 2:
        raise ValueError(f"{data_csv} has fewer than 2 rows; cannot compute framerate")
    first = int(rows[0][0])
    last = int(rows[-1][0])
    avg_dt_ns = (last - first) / (len(rows) - 1)
    return round(1e9 / avg_dt_ns)


def load_encoding_options(config_path: Path | None) -> dict[str, object]:
    options: dict[str, object] = {
        "use_dataset_framerate": False,
        "framerate": 30,
        "encoder": "libx264",
        "bitrate": "500k",
        "x264opts": DEFAULT_X264OPTS,
        "pix_fmt": "yuv420p",
    }

    if config_path is None:
        return options

    with config_path.open("r", encoding="utf-8") as f:
        value0 = json.load(f).get("value0", {})

    options["use_dataset_framerate"] = bool(
        value0.get(
            "config.ffmpeg_use_dataset_framerate", options["use_dataset_framerate"]
        )
    )
    options["framerate"] = int(
        value0.get("config.ffmpeg_framerate", options["framerate"])
    )
    options["encoder"] = str(value0.get("config.ffmpeg_encoder", options["encoder"]))
    options["bitrate"] = value0.get("config.ffmpeg_x264_bitrate", options["bitrate"])
    options["pix_fmt"] = str(value0.get("config.ffmpeg_pix_fmt", options["pix_fmt"]))

    x264opts = value0.get("config.ffmpeg_x264opts", options["x264opts"])
    options["x264opts"] = None if x264opts in (None, "") else str(x264opts)
    options["bitrate"] = (
        None if options["bitrate"] in (None, "") else str(options["bitrate"])
    )

    return options


def copy_euroc_skeleton(src_dataset: Path, dst_dataset: Path) -> None:
    def ignore_cam_data_dirs(src: str, names: list[str]) -> set[str]:
        src_path = Path(src)
        if src_path.name.startswith("cam") and src_path.parent.name == "mav0":
            return {"data"} & set(names)
        return set()

    if dst_dataset.exists():
        remove_tree(dst_dataset)
    shutil.copytree(src_dataset, dst_dataset, ignore=ignore_cam_data_dirs)
    make_tree_user_writable(dst_dataset)


def remove_tree(path: Path) -> None:
    """Remove a tree that may contain read-only files from a previous run."""
    make_tree_user_writable(path)

    def chmod_and_retry(func, failed_path, exc_info):
        failed = Path(failed_path)
        try:
            failed.chmod(failed.stat().st_mode | 0o700)
        except OSError:
            pass
        func(failed_path)

    shutil.rmtree(path, onerror=chmod_and_retry)


def make_tree_user_writable(path: Path) -> None:
    """Make copied output metadata writable even if the source dataset is read-only."""
    for item in path.rglob("*"):
        mode = item.stat().st_mode
        if item.is_dir():
            item.chmod(mode | 0o700)
        else:
            item.chmod(mode | 0o600)
    path.chmod(path.stat().st_mode | 0o700)


def maybe_filter_shared_timestamps(
    src_dataset: Path,
    dst_dataset: Path,
    src_cam_dirs: list[Path],
    allow_any_mismatch: bool,
) -> dict[str, list[tuple[str, str]]]:
    csv_data: dict[str, tuple[list[str], list[tuple[str, str]]]] = {}
    timestamp_sets: dict[str, set[str]] = {}

    for cam_dir in src_cam_dirs:
        headers, rows = read_data_csv(cam_dir / "data.csv")
        csv_data[cam_dir.name] = (headers, rows)
        timestamp_sets[cam_dir.name] = {timestamp for timestamp, _ in rows}

    all_equal = (
        len({frozenset(timestamps) for timestamps in timestamp_sets.values()}) == 1
    )
    if all_equal:
        return {cam_name: rows for cam_name, (_, rows) in csv_data.items()}

    counts = {
        cam_name: len(timestamps) for cam_name, timestamps in timestamp_sets.items()
    }
    if src_dataset.name not in MISMATCH_ALLOWLIST_DATASETS and not allow_any_mismatch:
        raise RuntimeError(
            f"Timestamp mismatch across cameras for dataset '{src_dataset.name}': {counts}. "
            "This matches Basalt CI behavior. Pass --allow-filter-mismatched-timestamps "
            "to filter any mismatched dataset to shared timestamps."
        )

    print(f"WARNING: timestamp mismatch across cameras: {counts}")
    print("Filtering output data.csv files to timestamps shared by all cameras")

    shared_timestamps = set.intersection(*timestamp_sets.values())
    if not shared_timestamps:
        raise RuntimeError(f"No shared timestamps across all cameras for {src_dataset}")

    filtered: dict[str, list[tuple[str, str]]] = {}
    for cam_name, (headers, rows) in csv_data.items():
        kept_rows = [
            (timestamp, filename)
            for timestamp, filename in rows
            if timestamp in shared_timestamps
        ]
        write_data_csv(dst_dataset / "mav0" / cam_name / "data.csv", headers, kept_rows)
        print(f"  {cam_name}: {len(kept_rows)} rows kept")
        filtered[cam_name] = kept_rows

    return filtered


def build_ffmpeg_concat_file(
    rows: list[tuple[str, str]],
    original_data_dir: Path,
    output_path: Path,
    framerate: int,
) -> None:
    frame_duration = 1.0 / framerate
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n\n")
        for _, filename in rows:
            png_path = (original_data_dir / filename).resolve()
            if not png_path.exists():
                raise FileNotFoundError(f"PNG listed in data.csv not found: {png_path}")
            f.write(f"file '{png_path}'\n")
            f.write(f"duration {frame_duration}\n")


def encode_video(
    concat_txt: Path,
    output_video: Path,
    framerate: int,
    options: dict[str, object],
) -> None:
    passlogfile = str(output_video.parent / "ffmpeg2pass")

    base_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-r",
        str(framerate),
        "-i",
        str(concat_txt),
        "-c:v",
        str(options["encoder"]),
    ]

    if options["bitrate"] is not None:
        base_cmd += ["-b:v", str(options["bitrate"])]

    base_cmd += [
        "-pix_fmt",
        str(options["pix_fmt"]),
        "-r",
        str(framerate),
        "-vf",
        "setpts=PTS-STARTPTS",
    ]

    if options["x264opts"] is not None and options["encoder"] == "libx264":
        base_cmd += ["-x264opts", str(options["x264opts"])]

    run_checked(
        base_cmd
        + ["-passlogfile", passlogfile, "-pass", "1", "-an", "-f", "null", "/dev/null"]
    )
    run_checked(
        base_cmd
        + [
            "-passlogfile",
            passlogfile,
            "-pass",
            "2",
            "-an",
            "-f",
            "mp4",
            str(output_video),
        ]
    )


def decode_video_to_pngs(
    output_video: Path, rows: list[tuple[str, str]], output_data_dir: Path
) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(output_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {output_video}")

    try:
        for frame_idx, (_, filename) in enumerate(rows):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended early for {output_video}: expected {len(rows)} frames, got {frame_idx}"
                )

            out_path = output_data_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write decoded PNG: {out_path}")

        ok, _ = cap.read()
        if ok:
            print(
                f"WARNING: {output_video} contains extra decoded frames; ignored after {len(rows)} expected frames"
            )
    finally:
        cap.release()


def process_dataset(args: argparse.Namespace) -> None:
    src_dataset = args.euroc_dataset.resolve()
    dst_dataset = args.save_path.resolve()
    options = load_encoding_options(args.config)

    src_cam_dirs = get_euroc_cam_dirs(src_dataset)
    copy_euroc_skeleton(src_dataset, dst_dataset)
    rows_by_cam = maybe_filter_shared_timestamps(
        src_dataset, dst_dataset, src_cam_dirs, args.allow_filter_mismatched_timestamps
    )

    for src_cam_dir in src_cam_dirs:
        cam_name = src_cam_dir.name
        if cam_name not in rows_by_cam:
            continue

        dst_cam_dir = dst_dataset / "mav0" / cam_name
        output_data_dir = dst_cam_dir / "data"
        output_video = dst_cam_dir / "data.mp4"
        concat_txt = dst_cam_dir / "input.txt"

        if options["use_dataset_framerate"]:
            framerate = compute_framerate(dst_cam_dir / "data.csv")
        else:
            framerate = int(options["framerate"])

        rows = rows_by_cam[cam_name]
        build_ffmpeg_concat_file(rows, src_cam_dir / "data", concat_txt, framerate)
        print(
            f"Written concat file: {concat_txt} ({len(rows)} entries, {framerate} fps)"
        )
        print(
            f"Encoding {len(rows)} frames from {src_cam_dir / 'data'} into {output_video}"
        )
        encode_video(concat_txt, output_video, framerate, options)

        print(f"Decoding {output_video} to timestamped PNGs in {output_data_dir}")
        decode_video_to_pngs(output_video, rows, output_data_dir)

    print(f"Done: {dst_dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "euroc_dataset",
        type=Path,
        help="Input EuRoC dataset directory, e.g. MH_01_easy",
    )
    parser.add_argument(
        "save_path", type=Path, help="Output EuRoC dataset directory to create"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Basalt config JSON. If omitted, Basalt CI ffmpeg defaults are used.",
    )
    parser.add_argument(
        "--allow-filter-mismatched-timestamps",
        action="store_true",
        help="For non-CI-allowlisted timestamp mismatches, filter all cameras to shared timestamps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
