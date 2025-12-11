#!/usr/bin/env python3
"""
Analyze respiration signal from a CSV and produce per-window plots + RR estimates.

Expected CSV format (hardcoded below):
    rr,timestamp
    1,1023.0
    2,1023.01

This script hardcodes the input filename, window length (seconds), and band-pass
low/high (Hz). For each window it:
  - plots raw and filtered waveform
  - computes and plots periodogram, marks dominant frequency in band
  - computes RR (rpm) by FFT (dominant freq * 60) and by peak detection
  - saves per-window PNG and writes a summary CSV

Run: python scripts/analyze_rr_windows.py
"""
import os
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, periodogram, find_peaks, detrend

# ensure project root on PYTHONPATH so we can import evaluation.post_process
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.post_process import calculate_metric_per_video


# ----------------- Hardcoded parameters (edit here) -----------------
INPUT_CSV = "~/shared/SpO2-Dataset/080103/v01/RR.csv"  # <-- put your CSV path here
WINDOW_SEC = 30.0                       # window length in seconds
BAND_LOW = 0.1                          # Hz (e.g., 0.1 for 6 rpm)
BAND_HIGH = 0.5                         # Hz (e.g., 0.5 for 30 rpm)
MIN_SAMPLES_PER_WINDOW = 9
OUTPUT_DIR_ROOT = "logs/rr_analysis"
# --------------------------------------------------------------------


def next_power_of_two(x: int) -> int:
    return 1 if x == 0 else 2 ** ((x - 1).bit_length())


def bandpass_filter(sig, fs, low, high, order=2):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, sig)


def compute_fft_rr(sig, fs, low, high):
    # compute periodogram with an NFFT near power of two
    N = next_power_of_two(len(sig))
    f, pxx = periodogram(sig, fs=fs, nfft=N, detrend=False)
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return np.nan, f, pxx
    f_masked = f[mask]
    pxx_masked = pxx[mask]
    idx = np.argmax(pxx_masked)
    dom_freq = float(f_masked[idx])
    rr_fft = dom_freq * 60.0
    return rr_fft, f, pxx


def compute_peak_rr(sig, fs):
    # find peaks and compute mean peak-to-peak interval
    peaks, _ = find_peaks(sig)
    if len(peaks) < 2:
        return np.nan, peaks
    mean_diff_samples = np.mean(np.diff(peaks))
    rr_peak = 60.0 / (mean_diff_samples / fs)
    return rr_peak, peaks


def ensure_output_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def analyze_file(input_csv, window_sec, band_low, band_high, out_root):
    df = pd.read_csv(input_csv)
    if not {"rr", "timestamp"}.issubset(df.columns):
        raise ValueError("CSV must contain columns 'rr' and 'timestamp'")

    timestamps = df["timestamp"].values.astype(float)
    signal = df["rr"].values.astype(float)

    if len(timestamps) < 2:
        raise ValueError("Not enough timestamps to compute sampling rate")

    # estimate sampling frequency from timestamps (seconds)
    dt = np.median(np.diff(timestamps))
    if dt <= 0:
        raise ValueError("Non-increasing timestamps found")
    fs = 1.0 / dt

    window_samples = int(max(1, round(window_sec * fs)))

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"rr_analysis_{run_ts}")
    ensure_output_dir(out_dir)

    summary = []

    num_windows = int(np.ceil(len(signal) / window_samples))
    for w in range(num_windows):
        start_idx = w * window_samples
        end_idx = min((w + 1) * window_samples, len(signal))
        win_sig = signal[start_idx:end_idx]
        win_ts = timestamps[start_idx:end_idx]

        if len(win_sig) < MIN_SAMPLES_PER_WINDOW:
            print(f"Skipping small window {w} (len={len(win_sig)})")
            continue

        # detrend + bandpass (use scipy.detrend then butterfilt)
        sig_dt = detrend(win_sig)
        try:
            sig_f = bandpass_filter(sig_dt, fs, band_low, band_high, order=2)
        except Exception as e:
            print(f"Bandpass filtering failed for window {w}: {e}")
            sig_f = sig_dt.copy()

        # FFT-based RR
        rr_fft, f, pxx = compute_fft_rr(sig_f, fs, band_low, band_high)

        # Peak-based RR (operate on filtered signal)
        rr_peak, peaks = compute_peak_rr(sig_f, fs)

        # Save plot for the window
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        # time axis in seconds relative to window start
        t_rel = win_ts - win_ts[0]
        axes[0].plot(t_rel, win_sig, label="raw")
        axes[0].plot(t_rel, sig_f, label="detrended+bandpassed")
        if len(peaks) > 0:
            axes[0].plot(t_rel[peaks], sig_f[peaks], "x", label="peaks")
        axes[0].set_title(f"Window {w}: time-domain (samples {start_idx}:{end_idx})")
        axes[0].set_xlabel("time (s)")
        axes[0].legend()

        # frequency / periodogram
        axes[1].semilogy(f, pxx, label="PSD")
        axes[1].set_xlim(0, fs / 2)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Power")
        # highlight band
        axes[1].axvspan(band_low, band_high, color="orange", alpha=0.2)
        if not np.isnan(rr_fft):
            dom_freq_hz = rr_fft / 60.0
            axes[1].axvline(dom_freq_hz, color="r", linestyle="--", label=f"dom {dom_freq_hz:.3f} Hz ({rr_fft:.2f} rpm)")
        axes[1].legend()

        fig_name = os.path.join(out_dir, f"window_{start_idx}_{end_idx}.png")
        fig.suptitle(f"Window {w}: FFT RR={rr_fft:.2f} rpm, Peak RR={rr_peak if not np.isnan(rr_peak) else 'NaN'} rpm")
        fig.savefig(fig_name)
        plt.close(fig)

        summary.append({
            "window_idx": int(w),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_ts": float(win_ts[0]),
            "end_ts": float(win_ts[-1]),
            "window_len_samples": int(len(win_sig)),
            "fs": float(fs),
            "rr_fft_rpm": float(rr_fft) if not np.isnan(rr_fft) else np.nan,
            "rr_peak_rpm": float(rr_peak) if not np.isnan(rr_peak) else np.nan,
            # Also compute the toolbox/metrics-style RR values (use same window signal for pred & label)
            # This helps directly compare the analyzer's peak/FFT results with the evaluation pipeline.
            # calculate_metric_per_video returns (hr_label, hr_pred, SNR, macc)
            **({
                "rr_fft_metrics_rpm": float(calculate_metric_per_video(win_sig.copy(), win_sig.copy(), fs=fs, diff_flag=False, hr_method='FFT', low_pass=band_low, high_pass=band_high)[1])
            } if True else {}),
            **({
                "rr_peak_metrics_rpm": float(calculate_metric_per_video(win_sig.copy(), win_sig.copy(), fs=fs, diff_flag=False, hr_method='Peak', low_pass=band_low, high_pass=band_high)[1])
            } if True else {}),
            "figure": fig_name,
        })

        print(f"Window {w}: start={win_ts[0]:.3f} len={len(win_sig)} samples, RR_fft={rr_fft:.2f}, RR_peak={rr_peak}")

    summary_df = pd.DataFrame(summary)
    # Save summary inside the analysis output directory. If you want a custom
    # destination, set an absolute path (or a path containing ~ which will be
    # expanded). Avoid joining an absolute/custom path into out_dir which may
    # create invalid nested paths.
    summary_csv = os.path.join(out_dir, "rr_windows_summary.csv")
    ensure_output_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved window figures to: {out_dir}")


if __name__ == "__main__":
    print("RR window analyzer")
    print(f"Input CSV (hardcoded): {INPUT_CSV}")
    analyze_file(INPUT_CSV, WINDOW_SEC, BAND_LOW, BAND_HIGH, OUTPUT_DIR_ROOT)
