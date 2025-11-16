import numpy as np
import pandas as pd
import cv2
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


                    video_type_cfg = getattr(self.config_data, "VIDEO_TYPE", "").strip()
                    if video_type_cfg:
                        wanted_name = video_type_cfg + ".avi"
                    else:
                        wanted_name = None

                    for entry in df[col].astype(str).tolist():
                        entry = entry.strip()
                        if not entry:
                            continue

                        # 拼完整路径（形如 /data1/disk/3/jjt/SpO2-Dataset/070203/v01）
                        video_dir = os.path.join(data_path, entry)
                        if not os.path.isdir(video_dir):
                            print(f"[WARN] Skipping non-dir path: {video_dir}")
                            continue

                        # 在该目录下找目标视频
                        avi_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".avi")]
                        if not avi_files:
                            print(f"[WARN] No .avi found in {video_dir}")
                            continue

                        # 精确匹配 VIDEO_TYPE
                        chosen = None
                        if wanted_name:
                            for f in avi_files:
                                if f.lower() == wanted_name.lower():
                                    chosen = f
                                    break

                        # 如果没指定或没匹配到，就跳过
                        if not chosen:
                            print(f"[WARN] No matching {wanted_name} in {video_dir}")
                            continue

                        video_file = os.path.join(video_dir, chosen)

                        # 提取 subject / session 信息
                        session = os.path.basename(video_dir)
                        subj = os.path.basename(os.path.dirname(video_dir))

                        dirs.append({
                            "index": session[1:] if session.startswith("v") else session,
                            "path": video_file,
                            "subject": subj,
                            "type": os.path.splitext(os.path.basename(video_file))[0].split("_")[-1].lower()
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

        # Decide label by TASK (default HR)
        task = getattr(self.config_data, 'TASK', 'HR').upper() if isinstance(getattr(self.config_data, 'TASK', 'HR'), str) else 'HR'

        # Prepare RR and SpO2 resampled signals if present
        resampled_rr = None
        if rr_values is not None:
            try:
                if rr_timestamps is not None:
                    resampled_rr = self.synchronize_and_resample(rr_timestamps, rr_values, frame_timestamps)
                else:
                    # Fallback: interpolate RR values over their index to match frames
                    rr_idx = np.linspace(0, len(rr_values) - 1, num=len(rr_values))
                    frame_idx = np.linspace(0, len(rr_values) - 1, num=frames.shape[0])
                    resampled_rr = np.interp(frame_idx, rr_idx, rr_values)
            except Exception as e:
                print(f"⚠️ Failed to resample RR for {video_dir}: {e}")
                resampled_rr = None

        resampled_spo2 = None
        if spo2_values is not None:
            try:
                sp_idx = np.linspace(0, len(spo2_values) - 1, num=len(spo2_values))
                frame_idx = np.linspace(0, len(spo2_values) - 1, num=frames.shape[0])
                resampled_spo2 = np.interp(frame_idx, sp_idx, spo2_values)
            except Exception as e:
                print(f"⚠️ Failed to resample SpO2 for {video_dir}: {e}")
                resampled_spo2 = None

        # If RR task but RR missing/invalid, drop this sample from file list
        if task == 'RR' and (resampled_rr is None or (hasattr(resampled_rr, '__len__') and len(resampled_rr) == 0)):
            print(f"⚠️ RR task selected but RR.csv missing/invalid in {video_dir}. Dropping this sample.")
            file_list_dict[i] = []
            return

        # Select label signal by task
        if task == 'HR':
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                label_signal = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                label_signal = resampled_bvp
        elif task == 'RR':
            label_signal = resampled_rr
        elif task == 'SPO2':
            if resampled_spo2 is None:
                print(f"ℹ️ SpO2 task selected but SpO2.csv missing or invalid in {video_dir}. Using zeros as labels.")
                label_signal = np.zeros(frames.shape[0], dtype=np.float32)
            else:
                label_signal = resampled_spo2
        else:
            print(f"ℹ️ Unknown TASK '{task}', defaulting to HR label (BVP).")
            label_signal = resampled_bvp

        frames_clips, label_clips = self.preprocess(frames, label_signal, config_preprocess)
        filename = f"{subject_id}_{experiment_id}"
        input_name_list, label_name_list = self.save_multi_process(frames_clips, label_clips, filename)
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
