import os
import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("Calculating metrics!")
    # Determine bandpass based on TASK
    task = getattr(config, 'TASK', 'HR') if hasattr(config, 'TASK') else 'HR'
    task = str(task).upper()
    if task == 'RR':
        band_low, band_high = 0.1, 0.5  # Hz (6-30 rpm)
    else:
        band_low, band_high = 0.75, 3.0  # Hz (45-180 bpm)
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            # RR task evaluation should not invert diff-normalized signals; RR labels are not treated as
            # 1st-derivative waveforms. Force no cumsum/detrend inversion in RR mode.
            if task == 'RR':
                diff_flag_test = False
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak', low_pass=band_low, high_pass=band_high)
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT', low_pass=band_low, high_pass=band_high)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all, dtype=float)
        predict_hr_fft_all = np.array(predict_hr_fft_all, dtype=float)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        # Collect metrics for CSV
        csv_metrics = {}
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
                csv_metrics["MAE"] = float(MAE_FFT)
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
                csv_metrics["RMSE"] = float(RMSE_FFT)
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
                csv_metrics["MAPE"] = float(MAPE_FFT)
            elif metric == "Pearson":
                # Guard against all-constant / NaN arrays to avoid NaN Pearson
                valid_mask = np.isfinite(predict_hr_fft_all) & np.isfinite(gt_hr_fft_all)
                x = predict_hr_fft_all[valid_mask]
                y = gt_hr_fft_all[valid_mask]
                if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    correlation_coefficient = 0.0
                    standard_error = 0.0
                else:
                    Pearson_FFT = np.corrcoef(x, y)
                    correlation_coefficient = float(Pearson_FFT[0][1])
                    standard_error = float(np.sqrt((1 - correlation_coefficient**2) / max(len(x) - 2, 1)))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
                csv_metrics["Pearson"] = float(correlation_coefficient)
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
                csv_metrics["SNR"] = float(SNR_FFT)
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
                csv_metrics["MACC"] = float(MACC_avg)
            elif "AU" in metric:
                pass
            elif "BA" in metric:  
                # Bland-Altman can fail if data are degenerate (all same, NaNs) and KDE covariance is not PD.
                # Add a small jitter and drop NaNs; if still problematic, skip plots gracefully.
                try:
                    valid_mask = np.isfinite(gt_hr_fft_all) & np.isfinite(predict_hr_fft_all)
                    ba_gt = gt_hr_fft_all[valid_mask]
                    ba_pred = predict_hr_fft_all[valid_mask]
                    if len(ba_gt) < 3:
                        print("[BA] Not enough valid points for Bland-Altman plot, skipping.")
                    else:
                        compare = BlandAltman(ba_gt, ba_pred, config, averaged=True)
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                            file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                            file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
                except Exception as e:
                    print(f"[BA] Skipping Bland-Altman plots due to error: {e}")
            else:
                raise ValueError("Wrong Test Metric Type")
        # Write CSV of test metrics if logging is configured
        try:
            log_dir = os.environ.get('SPO2_LOG_DIR', None)
            run_ts = os.environ.get('SPO2_RUN_TS', None)
            if log_dir and run_ts:
                os.makedirs(log_dir, exist_ok=True)
                csv_path = os.path.join(log_dir, f"test_result_{run_ts}.csv")
                # Persist only the metrics requested
                if len(csv_metrics) > 0:
                    pd.DataFrame([csv_metrics]).to_csv(csv_path, index=False)
                    print(f"Saved test metrics CSV to: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to save test metrics CSV: {e}")
        # Return metrics to allow trainer to also append into unified CSV
        return csv_metrics
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all, dtype=float)
        predict_hr_peak_all = np.array(predict_hr_peak_all, dtype=float)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        # Collect metrics for CSV
        csv_metrics = {}
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
                csv_metrics["MAE"] = float(MAE_PEAK)
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
                csv_metrics["RMSE"] = float(RMSE_PEAK)
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
                csv_metrics["MAPE"] = float(MAPE_PEAK)
            elif metric == "Pearson":
                valid_mask = np.isfinite(predict_hr_peak_all) & np.isfinite(gt_hr_peak_all)
                x = predict_hr_peak_all[valid_mask]
                y = gt_hr_peak_all[valid_mask]
                if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    correlation_coefficient = 0.0
                    standard_error = 0.0
                else:
                    Pearson_PEAK = np.corrcoef(x, y)
                    correlation_coefficient = float(Pearson_PEAK[0][1])
                    standard_error = float(np.sqrt((1 - correlation_coefficient**2) / max(len(x) - 2, 1)))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
                csv_metrics["Pearson"] = float(correlation_coefficient)
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (PEAK Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
                csv_metrics["SNR"] = float(SNR_PEAK)
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("PEAK MACC (PEAK Label): {0} +/- {1}".format(MACC_avg, standard_error))
                csv_metrics["MACC"] = float(MACC_avg)
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                try:
                    valid_mask = np.isfinite(gt_hr_peak_all) & np.isfinite(predict_hr_peak_all)
                    ba_gt = gt_hr_peak_all[valid_mask]
                    ba_pred = predict_hr_peak_all[valid_mask]
                    if len(ba_gt) < 3:
                        print("[BA] Not enough valid points for Bland-Altman plot (peak), skipping.")
                    else:
                        compare = BlandAltman(ba_gt, ba_pred, config, averaged=True)
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                            file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                            file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
                except Exception as e:
                    print(f"[BA] Skipping Bland-Altman plots (peak) due to error: {e}")
            else:
                raise ValueError("Wrong Test Metric Type")
        # Write CSV of test metrics if logging is configured
        try:
            log_dir = os.environ.get('SPO2_LOG_DIR', None)
            run_ts = os.environ.get('SPO2_RUN_TS', None)
            if log_dir and run_ts:
                os.makedirs(log_dir, exist_ok=True)
                csv_path = os.path.join(log_dir, f"test_result_{run_ts}.csv")
                if len(csv_metrics) > 0:
                    pd.DataFrame([csv_metrics]).to_csv(csv_path, index=False)
                    print(f"Saved test metrics CSV to: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to save test metrics CSV: {e}")
        return csv_metrics
    else:
        raise ValueError("Inference evaluation method name wrong!")
