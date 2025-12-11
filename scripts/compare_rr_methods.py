import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram, find_peaks, detrend
from datetime import datetime

# Ensure project root is on PYTHONPATH so we can import evaluation.post_process
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.post_process import calculate_metric_per_video

# adjust this to the same CSV the analyzer used
INPUT_CSV = os.path.expanduser("~/shared/SpO2-Dataset/080103/v01/RR.csv")
BAND_LOW = 0.1
BAND_HIGH = 0.5
WINDOW_SEC = 600

if not os.path.exists(INPUT_CSV):
    print(f"Input CSV not found: {INPUT_CSV}")
    raise SystemExit(1)

print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
if not {"rr","timestamp"}.issubset(df.columns):
    print("CSV missing required columns")
    raise SystemExit(1)

timestamps = df['timestamp'].values.astype(float)
signal = df['rr'].values.astype(float)

dt = np.median(np.diff(timestamps))
fs = 1.0 / dt
print(f"Estimated fs = {fs}")

window_samples = int(max(1, round(WINDOW_SEC * fs)))
start_idx = 0
end_idx = min(window_samples, len(signal))
win_sig = signal[start_idx:end_idx]
win_ts = timestamps[start_idx:end_idx]

print(f"Window samples: {len(win_sig)} (idx {start_idx}:{end_idx})")

# Analyzer-style processing (scipy.detrend + butter order=2)
def bandpass_filter(sig, fs, low, high, order=2):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype='bandpass')
    return filtfilt(b, a, sig)

sig_dt = detrend(win_sig)
try:
    sig_f = bandpass_filter(sig_dt, fs, BAND_LOW, BAND_HIGH, order=2)
except Exception as e:
    print('Analyzer bandpass failed:', e)
    sig_f = sig_dt.copy()

# Analyzer FFT
N = 1 << ((len(sig_f)-1).bit_length())
f, pxx = periodogram(sig_f, fs=fs, nfft=N, detrend=False)
mask = (f >= BAND_LOW) & (f <= BAND_HIGH)
if np.any(mask):
    f_mask = f[mask]
    pxx_mask = pxx[mask]
    idx = np.argmax(pxx_mask)
    rr_fft_an = f_mask[idx] * 60.0
else:
    rr_fft_an = np.nan

# Analyzer peak
peaks, _ = find_peaks(sig_f)
if len(peaks) >= 2:
    rr_peak_an = 60.0 / (np.mean(np.diff(peaks)) / fs)
else:
    rr_peak_an = np.nan

print('Analyzer-style results:')
print('  rr_fft_an =', rr_fft_an)
print('  rr_peak_an =', rr_peak_an)

# Metrics-style: use calculate_metric_per_video (it returns hr_label, hr_pred, SNR, macc)
# We pass same signal for predictions and labels to focus on method differences
from evaluation.post_process import _detrend as _pp_detrend

# Use calculate_metric_per_video with diff_flag=False and hr_method=FFT and Peak
hr_label_fft, hr_pred_fft, SNR, macc = calculate_metric_per_video(win_sig.copy(), win_sig.copy(), fs=fs, diff_flag=False, use_bandpass=True, hr_method='FFT', low_pass=BAND_LOW, high_pass=BAND_HIGH)
hr_label_peak, hr_pred_peak, SNR2, macc2 = calculate_metric_per_video(win_sig.copy(), win_sig.copy(), fs=fs, diff_flag=False, use_bandpass=True, hr_method='Peak', low_pass=BAND_LOW, high_pass=BAND_HIGH)

print('\nMetrics.calculate_metric_per_video results:')
print('  FFT label (hr_label) =', hr_label_fft)
print('  FFT pred  (hr_pred)  =', hr_pred_fft)
print('  Peak label (hr_label) =', hr_label_peak)
print('  Peak pred  (hr_pred)  =', hr_pred_peak)
print('  SNRs:', SNR, SNR2)
print('  MACC:', macc, macc2)

# Compare numeric differences
print('\nDifferences (Analyzer - Metrics):')
print('  FFT diff (rpm):', rr_fft_an - hr_pred_fft)
print('  Peak diff (rpm):', rr_peak_an - hr_pred_peak)

# Save quick summary
out = {
    'fs': fs,
    'rr_fft_an': rr_fft_an,
    'rr_peak_an': rr_peak_an,
    'hr_pred_fft_metrics': hr_pred_fft,
    'hr_pred_peak_metrics': hr_pred_peak,
}

print('\nDone.')
