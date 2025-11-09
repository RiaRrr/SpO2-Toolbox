import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)   # 禁止 OpenCV 内部多线程，防止 CPU 爆满
import glob
import os
from scipy.interpolate import interp1d
from dataset.data_loader.BaseLoader import BaseLoader

class SPO2Loader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a  dataloader.
            Args:
                data_path (str): Path to a folder containing raw video and BVP data.
                For example, data_path should be "data" for the following structure:
                -----------------
                     Data/
                    |   |-- 070200/
                    |       |-- v01
                    |           |-- BVP.csv
                    |           |-- frames_timestamp.csv
                    |           |-- SpO2.csv                    
                    |           |-- video_ZIP_H264.avi
                    |           |-- video_RAW_RGBA.avi
                    |       |-- v02
                    |   |-- 060201/
                    |       |-- v01
                    |           |-- BVP.csv
                    |           |-- frames_timestamp.csv
                    |           |-- SpO2.csv                    
                    |           |-- RR.csv(some)
                    |           |-- video_ZIP_H264.avi
                    |           |-- video_RAW_RGBA.avi
                    |       |-- v02
                    |       |...
                    |...
                    |   |-- 0602mn/
                    |       |-- v01
                    |       |-- v02
                    |       |...
                -----------------
                name (str): Name of the dataloader.
                config_data (CfgNode): Data settings (ref: config.py).
        """
        self.info = config_data.INFO
        print(data_path)
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories in the specified path.

        This implementation supports :
        - If a CSV exists at self.file_list_path, interpret each row as a raw-entry
          (either prefix like '070200_v01' or a relative/absolute path to the video file)
          and map it to the expected data_dirs dicts.
        """
        print(f"[SPO2Loader] get_raw_data scanning: {data_path}")

        dirs = []

        # 1) If user provided a raw-filelist under config (TRAIN.DATA.FILE_LIST), use it to pick which videos to process.
        #    We only use the `file_path` column (relative or absolute paths) to select videos; we do NOT change
        #    the canonical `file_list_path` used by the framework for writing generated lists.
        filelist_cfg = None
        try:
            # config may expose FILE_LIST as a string path
            filelist_cfg = getattr(self.config_data, 'FILE_LIST', None) or getattr(self.config_data, 'file_list', None)
        except Exception:
            filelist_cfg = None

        if filelist_cfg:
            # resolve the filelist_cfg path (it may be relative to cwd or relative to data_path)
            cand_paths = [filelist_cfg, os.path.join(data_path, filelist_cfg)]
            filelist_path = None
            for p in cand_paths:
                if p and os.path.exists(p):
                    filelist_path = p
                    break

            if filelist_path:
                try:
                    df = pd.read_csv(filelist_path)
                    # Prefer column named 'file_path' (user-provided). If missing, try first column.
                    if 'file_path' in df.columns:
                        col = 'file_path'
                    else:
                        col = df.columns[0]

                    for entry in df[col].astype(str).tolist():
                        entry = entry.strip()
                        if not entry:
                            continue

                        # resolve entry to an absolute path under data_path if not absolute
                        if os.path.isabs(entry):
                            candidate = entry
                        else:
                            candidate = os.path.join(data_path, entry)

                        # If candidate is a file, use it directly. If it's a directory, find an avi inside.
                        if os.path.isfile(candidate):
                            video_file = candidate
                        elif os.path.isdir(candidate):
                            video_dir = candidate
                            avi_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.avi')]
                            if not avi_files:
                                continue
                            chosen = None
                            for f in avi_files:
                                if 'raw' in f.lower():
                                    chosen = f
                                    break
                            if chosen is None:
                                chosen = avi_files[0]
                            video_file = os.path.join(video_dir, chosen)
                        else:
                            # try globbing
                            matches = glob.glob(candidate)
                            found = False
                            for m in matches:
                                if os.path.isdir(m):
                                    video_dir = m
                                    avi_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.avi')]
                                    if not avi_files:
                                        continue
                                    chosen = None
                                    for f in avi_files:
                                        if 'raw' in f.lower():
                                            chosen = f
                                            break
                                    if chosen is None:
                                        chosen = avi_files[0]
                                    video_file = os.path.join(video_dir, chosen)
                                    found = True
                                    break
                            if not found:
                                continue

                        # derive subject and session
                        video_dir = os.path.dirname(video_file)
                        session = os.path.basename(os.path.dirname(video_file))
                        subj = os.path.basename(os.path.dirname(os.path.dirname(video_file)))
                        dirs.append({
                            'index': session[1:] if session.startswith('v') else session,
                            'path': video_file,
                            'subject': subj,
                            'type': os.path.splitext(os.path.basename(video_file))[0].split('_')[-1].lower()
                        })

                    if dirs:
                        return dirs
                except Exception:
                    # if reading user-provided filelist fails, fall through to existing behavior
                    pass
        return dirs
        

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        # Split according to tags v01, v02, v03, v04
        
        data_info = dict()
        for data in data_dirs:
            # index = data['index']
            # data_dir = data['path']
            subject = data['subject']
            # type = data['type'] # face or finger
            # Create a data directory dictionary indexed by subject number
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = list(data_info.keys())  # Get all subject numbers
        subj_list = sorted(subj_list)  # Sort subject numbers
        
        num_subjs = len(subj_list)  # Total number of subjects      
        
        # Get data set split (according to start/end ratio)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('Subjects ID used for split:', [subj_list[i] for i in subj_range])

        # Add file paths that meet the split range to the new list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
        
        print(data_dirs_new)
        return data_dirs_new            


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,  file_list_dict):
        
        # Read video frames
        video_file = data_dirs[i]['path']
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)

        # Extract subject ID and experiment ID from the directory path
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID
        print(f"subject_id: {subject_id}, experiment_id: {experiment_id}")
        # Get BVP, frame timestamps
        bvp_file = os.path.join(video_dir, "BVP.csv")
        timestamp_file = os.path.join(video_dir, "frames_timestamp.csv")

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Resample BVP data to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)

        # Process frames, BVP signals, and SpO2 signals according to the configuration
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = resampled_bvp

        # Label once here
        if "face" in video_file:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            filename = f"{subject_id}_{experiment_id}"
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, filename)
            file_list_dict[i] = input_name_list
        

    def load_preprocessed_data(self):
        """Load preprocessed data listed in the file list."""

        file_list_path = self.file_list_path   # Get file list path
        file_list_df = pd.read_csv(file_list_path)  # Read file list
        inputs_temp = file_list_df['input_files'].tolist()  # Get input file list
        inputs_face = [] 
        
        # v01 v02 v03 v04 face configuration information
        for each_input in inputs_temp:
            inputs_face.append(each_input) 
       
        inputs_face = sorted(inputs_face)
        labels_bvp = [input_file.replace("input", "label") for input_file in inputs_face]
           
        self.inputs = inputs_face    
        self.labels = labels_bvp
        self.preprocessed_data_len = len(inputs_face)

    @staticmethod
    def read_bvp(bvp_file):
        """Reads a BVP signal file with timestamps."""
        data = pd.read_csv(bvp_file)
        timestamps = data['timestamp'].values
        bvp_values = data['bvp'].values
        return timestamps, bvp_values

    @staticmethod
    def read_frame_timestamps(timestamp_file):
        """Reads timestamps for each video frame."""
        data = pd.read_csv(timestamp_file)
        return data['timestamp'].values

    @staticmethod
    def synchronize_and_resample(timestamps_data, data_values, timestamps_frames):
        """Synchronize and resample data to match video frame timestamps."""
        interpolator = interp1d(timestamps_data, data_values, bounds_error=False, fill_value="extrapolate")
        resampled_data = interpolator(timestamps_frames)
        return resampled_data

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.array(frames)