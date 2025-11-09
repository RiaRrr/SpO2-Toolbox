#!/usr/bin/env python3
"""Copy ROI and CSV files into processed data folders using a filelist.

Usage example:
  python scripts/copy_files_from_rois.py \
      --filelist /data1/disk/3/jjt/SpO2Dataset/PreprocessVideo/filelist_stats.csv \
      --roi-root /data1/disk/3/experiment_processed_roi_v2 \
      --processed-root /data1/disk/3/experiment_processed/data \
      --exp-root /data1/disk/3/experiment/data

This script will for each entry in filelist's `file_path` column:
- determine a subpath like 070200/v01
- copy segment.avi -> RGBA_ROI.avi and segment_corrected.avi -> RGBA__ROI_corrected.avi from roi-root/subpath
- copy all .csv files from processed-root/subpath into processed-root/subpath
- copy frames_timestamp.csv and missed_frames.csv from exp-root/subpath into processed-root/subpath

It preserves the folder organization and supports --dry-run and --overwrite flags.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from typing import List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Copy ROI and CSV files into processed data folders using a filelist")
    p.add_argument("--filelist", required=True, help="CSV file containing file_path column")
    p.add_argument("--roi-root", required=True, help="Root of ROI processed folders (contains segment.avi)")
    p.add_argument("--processed-root", required=True, help="Root of processed data folders (source of processed CSVs)")
    p.add_argument("--dest-root", required=False, help="Optional destination root where all copied files will be saved. If omitted, --processed-root is used as destination.")
    p.add_argument("--exp-root", required=True, help="Root of original experiment data folders (contains frames_timestamp.csv)")
    p.add_argument("--roi-h264-root", required=False, help="Optional root of H264 ROI folders (contains segment.avi/segment_corrected.avi)")
    p.add_argument("--col", default="\ufefffile_path", help="Column name in CSV with the relative path (default: file_path)")
    p.add_argument("--dry-run", action="store_true", help="Don't actually copy, only print what would be done")
    p.add_argument("--overwrite", action="store_true", help="Overwrite files at destination if present")
    return p.parse_args()


def extract_subpath(path: str) -> str:
    """Extract subpath like 070200/v01 from a path string.

    Heuristic: find token that starts with 'v' followed by digits (e.g. v01, v1) and take the previous token
    plus that token. If not found, return the cleaned path.
    """
    s = path.strip().replace('\\\\', '/').replace('\\', '/')
    tokens = [t for t in s.split('/') if t]
    for i, t in enumerate(tokens):
        if re.match(r'v\d{1,3}$', t, flags=re.IGNORECASE):
            if i >= 1:
                return os.path.join(tokens[i - 1], t)
            return t
    # fallback: if path already looks like two-level e.g. 070200/v01 or one-level 070200
    if len(tokens) >= 2 and re.match(r'\d{6}$', tokens[0]):
        return os.path.join(tokens[0], tokens[1]) if len(tokens) >= 2 else tokens[0]
    return s.strip('/')


def read_file_paths(csv_path: str, col: str) -> List[str]:
    paths: List[str] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if col not in reader.fieldnames:
            raise ValueError(f"Column '{col}' not found in {csv_path}. Available columns: {reader.fieldnames}")
        for row in reader:
            val = row.get(col)
            if val:
                sub = extract_subpath(val)
                if sub:
                    paths.append(sub)
    return paths


def ensure_dir(path: str, dry_run: bool = False):
    if not os.path.exists(path):
        if dry_run:
            print(f"[DRY-RUN] Would create dir: {path}")
        else:
            os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str, dry_run: bool = False, overwrite: bool = False) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.exists(dst) and not overwrite:
        print(f"Skip (exists): {dst}")
        return True
    if dry_run:
        print(f"[DRY-RUN] Copy {src} -> {dst}")
        return True
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
    return True


def process_subpath(subpath: str, roi_root: str, processed_src_root: str, dest_root: str, exp_root: str, roi_h264_root: str | None, dry_run: bool, overwrite: bool) -> Tuple[int,int,int,int,int]:
    """Process a single subpath.

    Returns counts: (copied_rgba_roi, copied_h264_roi, copied_csv_processed, copied_frames, missing)
    """
    copied_rgba = copied_h264 = copied_csv_processed = copied_frames = missing = 0

    src_roi = os.path.join(roi_root, subpath) if roi_root else None
    src_h264 = os.path.join(roi_h264_root, subpath) if roi_h264_root else None
    src_processed = os.path.join(processed_src_root, subpath)
    src_exp = os.path.join(exp_root, subpath)
    dest = os.path.join(dest_root, subpath)

    ensure_dir(dest, dry_run=dry_run)

    # Copy RGBA ROI videos from roi_root
    if src_roi:
        seg = os.path.join(src_roi, 'segment.avi')
        seg_corr = os.path.join(src_roi, 'segment_corrected.avi')
        dst_seg = os.path.join(dest, 'RGBA_ROI.avi')
        dst_seg_corr = os.path.join(dest, 'RGBA_ROI_corrected.avi')

        if os.path.exists(seg):
            if copy_file(seg, dst_seg, dry_run=dry_run, overwrite=overwrite):
                copied_rgba += 1
        else:
            print(f"Missing ROI segment: {seg}")
            missing += 1

        if os.path.exists(seg_corr):
            if copy_file(seg_corr, dst_seg_corr, dry_run=dry_run, overwrite=overwrite):
                copied_rgba += 1
        else:
            print(f"Missing ROI segment_corrected: {seg_corr}")
            missing += 1

    # Copy H264 ROI videos from roi_h264_root (optional)
    if src_h264:
        seg_h = os.path.join(src_h264, 'segment.avi')
        seg_corr_h = os.path.join(src_h264, 'segment_corrected.avi')
        dst_seg_h = os.path.join(dest, 'H264_ROI.avi')
        dst_seg_corr_h = os.path.join(dest, 'H264_ROI_corrected.avi')

        if os.path.exists(seg_h):
            if copy_file(seg_h, dst_seg_h, dry_run=dry_run, overwrite=overwrite):
                copied_h264 += 1
        else:
            print(f"Missing H264 ROI segment: {seg_h}")
            missing += 1

        if os.path.exists(seg_corr_h):
            if copy_file(seg_corr_h, dst_seg_corr_h, dry_run=dry_run, overwrite=overwrite):
                copied_h264 += 1
        else:
            print(f"Missing H264 ROI segment_corrected: {seg_corr_h}")
            missing += 1

    # Copy all .csv from processed (source) to dest
    if os.path.exists(src_processed):
        for fname in os.listdir(src_processed):
            if fname.lower().endswith('.csv'):
                srcf = os.path.join(src_processed, fname)
                dstf = os.path.join(dest, fname)
                if copy_file(srcf, dstf, dry_run=dry_run, overwrite=overwrite):
                    copied_csv_processed += 1
    else:
        print(f"Processed source folder missing: {src_processed}")
        missing += 1

    # Copy frames_timestamp.csv and missed_frames.csv from experiment root
    for fname in ('frames_timestamp.csv', 'missed_frames.csv'):
        srcf = os.path.join(src_exp, fname)
        if os.path.exists(srcf):
            dstf = os.path.join(dest, fname)
            if copy_file(srcf, dstf, dry_run=dry_run, overwrite=overwrite):
                copied_frames += 1
        else:
            # not fatal, just note if absent
            print(f"Missing (optional) {fname} in {src_exp}")

    return copied_rgba, copied_h264, copied_csv_processed, copied_frames, missing


def main():
    args = parse_args()

    try:
        subpaths = read_file_paths(args.filelist, args.col)
    except Exception as e:
        print(f"Error reading filelist: {e}")
        sys.exit(2)

    seen = set()
    summary = {'copied_rgba':0, 'copied_h264':0, 'copied_csv_processed':0, 'copied_frames':0, 'missing':0, 'processed_entries':0}

    dest_root = args.dest_root if getattr(args, 'dest_root', None) else args.processed_root

    for sub in subpaths:
        if sub in seen:
            continue
        seen.add(sub)
        print(f"\nProcessing: {sub}")
        copied_rgba, copied_h264, copied_csv_processed, copied_frames, missing = process_subpath(
            sub, args.roi_root, args.processed_root, dest_root, args.exp_root, args.roi_h264_root, args.dry_run, args.overwrite
        )
        summary['copied_rgba'] += copied_rgba
        summary['copied_h264'] += copied_h264
        summary['copied_csv_processed'] += copied_csv_processed
        summary['copied_frames'] += copied_frames
        summary['missing'] += missing
        summary['processed_entries'] += 1

    print('\nSummary:')
    print(f"Entries processed: {summary['processed_entries']}")
    print(f"RGBA ROI files copied: {summary['copied_rgba']}")
    print(f"H264 ROI files copied: {summary['copied_h264']}")
    print(f"Processed CSVs copied: {summary['copied_csv_processed']}")
    print(f"Frames/missed files copied: {summary['copied_frames']}")
    print(f"Missing items reported: {summary['missing']}")


if __name__ == '__main__':
    main()
