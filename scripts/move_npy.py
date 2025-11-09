#!/usr/bin/env python3
"""Move all .npy files from one folder to another.

Usage:
    python scripts/move_npy.py /path/to/src /path/to/dst

This script searches recursively under the source directory for files ending with
.npy and moves them into the destination directory. If a filename collision occurs
in the destination, a numeric suffix will be appended (e.g., file_1.npy).
"""
import argparse
import shutil
from pathlib import Path
import sys


def move_npy(src_dir: Path, dst_dir: Path, recursive: bool = True) -> int:
    if not src_dir.exists():
        print(f"Source directory does not exist: {src_dir}")
        return 1
    dst_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.npy" if recursive else "*.npy"
    files = list(src_dir.glob(pattern))
    if not files:
        print(f"No .npy files found under {src_dir}")
        return 0

    moved = 0
    for src in files:
        if not src.is_file():
            continue
        dest = dst_dir / src.name
        # handle name collisions
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1
        try:
            shutil.move(str(src), str(dest))
            print(f"Moved: {src} -> {dest}")
            moved += 1
        except Exception as e:
            print(f"Failed to move {src}: {e}")

    print(f"Done. Total moved: {moved}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Move all .npy files from src to dst")
    parser.add_argument("src", help="Source directory to search for .npy files")
    parser.add_argument("dst", help="Destination directory to move files into")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false",
                        help="Do not search recursively (only top-level files)")
    args = parser.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    return move_npy(src, dst, recursive=args.recursive)


if __name__ == "__main__":
    sys.exit(main())
