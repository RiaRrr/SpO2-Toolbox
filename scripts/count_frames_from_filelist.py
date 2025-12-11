import os
import sys
import csv
import numpy as np

import cv2
cv2.setNumThreads(1)


# Hard-coded file list CSV path (can be changed as needed)
FILELIST_CSV = "/root/jjt/SpO2Dataset/PreprocessVideo/filelist_stats.csv"

# Expected columns in filelist:
#   file_path: relative path under DATA_PATH, e.g. "070203/v01" or similar
FILELIST_COLUMN = "\ufefffile_path"

# Video type names and corresponding avi file names we want to count
VIDEO_TYPES = {
    "RGBA_ROI": "RGBA_ROI.avi",
    "H264_ROI": "H264_ROI.avi",
    "RGBA_ROI_corrected": "RGBA_ROI_corrected.avi",
    "H264_ROI_corrected": "H264_ROI_corrected.avi",
}


def read_video_frames(video_file, frame_skip=1):
    """Read a video and return number of frames after frame_skip sampling.

    This mirrors SPO2Loader.read_video: every `frame_skip`-th frame is kept.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_file}")
        return 0

    count = 0
    i = 0
    success, frame = cap.read()
    while success:
        if i % frame_skip == 0:
            count += 1
        success, frame = cap.read()
        i += 1
    cap.release()
    return count


def main():
    if not os.path.exists(FILELIST_CSV):
        print(f"[ERROR] Filelist CSV not found: {FILELIST_CSV}")
        sys.exit(1)

    # You can change this to match your actual DATA_PATH used in configs
    # e.g. /root/shared/SpO2-Dataset
    data_path = "/root/shared/SpO2-Dataset"
    print(f"Using DATA_PATH = {data_path}")

    # Read file list
    with open(FILELIST_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"[ERROR] Empty filelist: {FILELIST_CSV}")
        sys.exit(1)
    print("rows:", rows)
    if FILELIST_COLUMN not in rows[0]:
        # If the expected column is not there, fallback to first column name
        first_col = list(rows[0].keys())[0]
        print("first_col:", first_col)
        print(f"[WARN] Column '{FILELIST_COLUMN}' not found, fallback to '{first_col}'")
        col_name = first_col
    else:
        col_name = FILELIST_COLUMN

    summary = []

    for idx, row in enumerate(rows):
        entry = row[col_name].strip()
        if not entry:
            continue

        # Construct the directory containing the videos, e.g. DATA_PATH/070203/v01
        video_dir = os.path.join(data_path, entry)
        if not os.path.isdir(video_dir):
            print(f"[WARN] Not a directory, skip: {video_dir}")
            continue

        result_row = {
            "index_in_filelist": idx,
            "entry": entry,
        }

        # For each video type we care about, look for the specific file name
        for col_label, avi_name in VIDEO_TYPES.items():
            video_file = os.path.join(video_dir, avi_name)
            if os.path.exists(video_file):
                num_frames = read_video_frames(video_file, frame_skip=1)
            else:
                num_frames = 0
            result_row[col_label + "_frame"] = num_frames

        summary.append(result_row)

    # Build output table
    if not summary:
        print("[WARN] No valid entries found in filelist.")
        return

    # Determine all columns dynamically
    all_keys = set()
    for r in summary:
        all_keys.update(r.keys())
    all_keys = ["index_in_filelist", "entry"] + sorted(k for k in all_keys if k not in {"index_in_filelist", "entry"})

    out_csv = os.path.join(os.path.dirname(FILELIST_CSV), "filelist_frame_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in summary:
            writer.writerow(r)

    print(f"[INFO] Saved frame summary to {out_csv}")


if __name__ == "__main__":
    main()
