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

        # Detect raw-on-the-fly mode: if user does not want preprocessing and
        # the provided data_path contains the original Data/subject/vXX/ structure
        # with video files and BVP.csv, enter raw mode and skip BaseLoader init.
        self.raw_mode = False
        try:
            want_raw = (not config_data.DO_PREPROCESS)
        except Exception:
            want_raw = False

        def _looks_like_raw_folder(p):
            # detect at least one subject/vXX folder containing .avi and BVP.csv
            try:
                for subj in os.listdir(p):
                    subj_p = os.path.join(p, subj)
                    if not os.path.isdir(subj_p):
                        continue
                    for v in os.listdir(subj_p):
                        v_p = os.path.join(subj_p, v)
                        if not os.path.isdir(v_p):
                            continue
                        files = os.listdir(v_p)
                        if any(f.lower().endswith('.avi') for f in files) and 'BVP.csv' in files:
                            return True
            except Exception:
                return False
            return False

        if want_raw and _looks_like_raw_folder(data_path):
            # initialize minimal attributes and build raw data index
            self.raw_mode = True
            self.inputs = []
            self.labels = []
            self.dataset_name = name
            self.raw_data_path = data_path
            self.cached_path = data_path
            self.file_list_path = None
            self.preprocessed_data_len = 0
            self.data_format = config_data.DATA_FORMAT
            self.do_preprocess = False
            self.config_data = config_data

            # Build raw data list using get_raw_data (which expects the THUS-style structure)
            # Fallback: a more permissive scan if get_raw_data returns empty
            try:
                raw_dirs = self.get_raw_data(data_path)
            except Exception:
                raw_dirs = []
                for subj in os.listdir(data_path):
                    subj_p = os.path.join(data_path, subj)
                    if not os.path.isdir(subj_p):
                        continue
                    for v in os.listdir(subj_p):
                        v_p = os.path.join(subj_p, v)
                        if not os.path.isdir(v_p):
                            continue
                        avi = None
                        for f in os.listdir(v_p):
                            if f.lower().endswith('.avi'):
                                avi = os.path.join(v_p, f)
                                break
                        if avi and os.path.exists(os.path.join(v_p, 'BVP.csv')):
                            raw_dirs.append({'index': v[1:] if v.startswith('v') else v,
                                             'path': avi,
                                             'subject': subj,
                                             'type': 'face'})
            self.raw_data_dirs = raw_dirs
            print(f'Entering raw-on-the-fly mode: found {len(self.raw_data_dirs)} videos under {data_path}')
            return

        # default: use BaseLoader flow (cached / preprocessed mode)
        print(data_path)
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """
        Returns data directories in the specified path (suitable for the THUSPO2 dataset).
        Automatically detects all subject folders (e.g., 060200, 060201, 070200, 0602mn...).

        Each subject folder should contain v01, v02... subfolders, which include:
            - BVP.csv
            - frames_timestamp.csv
            - one or more .avi files
        """
        print(f"[SPO2Loader] Scanning raw data path: {data_path}")

        # 匹配六位（数字或字母组合）的目录名，例如 060200, 070200, 0602mn
        data_dirs = [os.path.join(data_path, d)
                    for d in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, d)) and len(d) == 6]
        if not data_dirs:
            raise ValueError(f"{self.dataset_name} Data path is empty or malformed! ({data_path})")

        dirs = []
        for data_dir in sorted(data_dirs):
            subject_name = os.path.split(data_dir)[-1]
            d_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"[SPO2Loader] Found subject {subject_name}, sessions: {d_dirs}")

            # 遍历每个 v01/v02/v03 子目录
            for session in sorted(d_dirs):
                session_path = os.path.join(data_dir, session)
                items = os.listdir(session_path)

                for item in items:
                    if item.lower().endswith('.avi'):
                        dirs.append({
                            'index': session[1:] if session.startswith('v') else session,
                            'path': os.path.join(session_path, item),
                            'subject': subject_name,
                            'type': item.split('_')[-1].split('.')[0] if '_' in item else 'raw'
                        })

        print(f"[SPO2Loader] Total {len(dirs)} video entries found.")
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

        # RR and SpO2 files may or may not exist in some recordings. Read if present.
        rr_file = os.path.join(video_dir, "RR.csv")
        spo2_file = os.path.join(video_dir, "SpO2.csv")
        rr_timestamps = None
        rr_values = None
        spo2_values = None
        if os.path.exists(rr_file):
            try:
                rr_df = pd.read_csv(rr_file)
                # attempt to infer timestamp and rr column names
                if 'timestamp' in rr_df.columns and 'rr' in rr_df.columns:
                    rr_timestamps = rr_df['timestamp'].values
                    rr_values = rr_df['rr'].values
                else:
                    # fallback: use first two numeric columns
                    numeric_cols = rr_df.select_dtypes(include=[float, int]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        rr_timestamps = rr_df[numeric_cols[0]].values
                        rr_values = rr_df[numeric_cols[1]].values
                    elif len(numeric_cols) == 1:
                        rr_values = rr_df[numeric_cols[0]].values
                        rr_timestamps = None
                    else:
                        rr_values = None
                        rr_timestamps = None
            except Exception:
                print(f"⚠️ Failed to read RR file: {rr_file}. Continuing without RR.")
                rr_timestamps = None
                rr_values = None
        else:
            # Not all datasets include RR.csv — this is acceptable
            # print a debug message for visibility
            # (kept as print to avoid adding heavy logging dependencies here)
            print(f"ℹ️ RR file not found for {video_dir}; continuing without RR.")

        if os.path.exists(spo2_file):
            try:
                spo2_df = pd.read_csv(spo2_file)
                # assume first numeric column is the SpO2 values
                numeric_cols = spo2_df.select_dtypes(include=[float, int]).columns.tolist()
                if len(numeric_cols) >= 1:
                    spo2_values = spo2_df[numeric_cols[0]].values
                else:
                    spo2_values = None
            except Exception:
                print(f"⚠️ Failed to read SpO2 file: {spo2_file}. Ignoring SpO2 for this sample.")

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
        
    def __len__(self):
        # In raw-on-the-fly mode, length equals number of raw videos discovered
        if getattr(self, 'raw_mode', False):
            return len(self.raw_data_dirs)
        return super().__len__()

    def __getitem__(self, index):
        # Raw-on-the-fly behaviour: read raw video + csvs, preprocess and return first chunk
        if getattr(self, 'raw_mode', False):
            entry = self.raw_data_dirs[index]
            video_file = entry['path']
            frames = self.read_video(video_file)
            video_dir = os.path.dirname(video_file)

            # timestamps and bvp
            timestamp_file = os.path.join(video_dir, 'frames_timestamp.csv')
            bvp_file = os.path.join(video_dir, 'BVP.csv')
            frame_timestamps = None
            try:
                frame_timestamps = self.read_frame_timestamps(timestamp_file)
            except Exception:
                pass
            bvp_timestamps, bvp_values = (None, None)
            try:
                bvp_timestamps, bvp_values = self.read_bvp(bvp_file)
            except Exception:
                pass

            if frame_timestamps is not None and bvp_timestamps is not None and bvp_values is not None:
                resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)
            else:
                # If timestamps missing, fall back to zeros with matching length
                resampled_bvp = np.zeros(frames.shape[0], dtype=np.float32)

            # Use preprocess settings provided by config (user may set DO_CHUNK=False, DO_CROP_FACE=False etc.)
            frames_clips, bvps_clips = self.preprocess(frames, resampled_bvp, self.config_data.PREPROCESS)

            # take first clip by default
            data = frames_clips[0]
            label = bvps_clips[0]

            # align format with BaseLoader.__getitem__ expectations
            if self.data_format == 'NDCHW':
                data = np.transpose(data, (0, 3, 1, 2))
            elif self.data_format == 'NCDHW':
                data = np.transpose(data, (3, 0, 1, 2))
            elif self.data_format == 'NDHWC':
                pass

            data = np.float32(data)
            label = np.float32(label)

            filename = f"{entry['subject']}_{entry['index']}"
            chunk_id = '0'
            return data, label, filename, chunk_id

        return super().__getitem__(index)


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
    def read_video(video_file, frame_skip=1):  # ← 每x帧取1帧
        cap = cv2.VideoCapture(video_file)
        frames = []
        i = 0
        success, frame = cap.read()
        while success:
            if i % frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            success, frame = cap.read()
            i += 1
        cap.release()
        return np.array(frames, dtype=np.uint8)
