import numpy as np
import torch
from types import SimpleNamespace as ns

# Import the function under test
from evaluation.metrics import calculate_metrics

# Build a minimal config matching what calculate_metrics expects
config = ns()
config.TASK = 'RR'  # test RR branch
config.TOOLBOX_MODE = 'only_test'
config.INFERENCE = ns()
config.INFERENCE.EVALUATION_METHOD = 'FFT'
config.INFERENCE.EVALUATION_WINDOW = ns()
config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
config.TEST = ns()
config.TEST.DATA = ns()
config.TEST.DATA.FS = 30
config.TEST.DATA.PREPROCESS = ns()
config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Raw'
config.TEST.DEBUG_METRICS = True
config.TEST.METRICS = ['MAE','RMSE','MAPE','Pearson','SNR','MACC']
config.INFERENCE.MODEL_PATH = 'model.pth'
config.TRAIN = ns()
config.TRAIN.MODEL_FILE_NAME = 'model'
config.LOG = ns()
config.LOG.PATH = '.'

# Create synthetic signals: one video, 1 channel waveform per-frame
fs = config.TEST.DATA.FS
n_seconds = 20
n_frames = fs * n_seconds
t = np.arange(n_frames) / fs

# Ground truth RR: 12 rpm -> 0.2 Hz
rr_rpm = 12.0
f_rr = rr_rpm / 60.0
label_signal = 0.5 * np.sin(2 * np.pi * f_rr * t)
# Predicted signal: slightly noisy and phase shifted
pred_signal = 0.5 * np.sin(2 * np.pi * f_rr * t + 0.1) + 0.05 * np.random.randn(n_frames)

# Pack into the expected dict-of-dicts structure. Values should be torch tensors used by _reform_data_from_dict
predictions = {'video1': {0: torch.tensor(pred_signal, dtype=torch.float32).unsqueeze(0)}}
labels = {'video1': {0: torch.tensor(label_signal, dtype=torch.float32).unsqueeze(0)}}

# Call calculate_metrics
metrics = calculate_metrics(predictions, labels, config)
print('\nReturned metrics dict:')
for k,v in metrics.items():
    if k.startswith('_'):
        print(k, 'len=', len(v))
    else:
        print(k, v)
