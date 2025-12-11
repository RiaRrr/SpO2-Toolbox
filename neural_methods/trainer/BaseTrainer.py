import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import csv
import pickle
import numpy as np


class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def __init__(self):
        # CSV unified logger path; set by _init_csv_logger if environment is configured
        self.csv_path = None
        self._csv_header_written = False
        self._csv_columns = [
            'mode','epoch','batch','lr','a_coeff','b_coeff','sharp',
            'NegPearson','fre_CEloss','kl_loss','hr_mae','loss',
            'valid_loss','best_epoch_so_far','min_valid_loss',
            'MAE','RMSE','MAPE','Pearson','SNR','MACC',
            # Extended (multi-task / auxiliary)
            'AU_AvgF1','AU_AvgPrec','AU_AvgAcc',
            'Resp_MAE','Resp_RMSE','Resp_MAPE','Resp_Pearson','Resp_SNR'
        ]

    def _init_csv_logger(self):
        """Prepare a unified CSV file capturing train/valid/test metrics for the current run.
        Requires main to have set SPO2_LOG_DIR and SPO2_RUN_TS.
        """
        try:
            log_dir = os.environ.get('SPO2_LOG_DIR')
            run_ts = os.environ.get('SPO2_RUN_TS')
            if not log_dir or not run_ts:
                return
            self.csv_path = os.path.join(log_dir, f"test_result_{run_ts}.csv")
            header_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            if not header_exists:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                    writer.writeheader()
            self._csv_header_written = True
        except Exception as e:
            print(f"WARN: CSV logger init failed: {e}")
            self.csv_path = None

    def _append_csv_row(self, row_dict):
        """Append a row to the unified CSV if initialized. Missing columns are filled with blanks."""
        # Always try to log to CSV (if initialized) and optionally to WandB
        row = {c: row_dict.get(c, '') for c in self._csv_columns}

        if self.csv_path:
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                    writer.writerow(row)
            except Exception as e:
                print(f"WARN: failed to append CSV row: {e}")

        # WandB logging (batch-wise & epoch summaries)
        if os.environ.get('SPO2_USE_WANDB') == '1':
            try:
                import wandb  # local import to avoid hard dependency
                log_payload = {}
                # Map a subset of columns to wandb metrics
                metric_keys = [
                    'loss','NegPearson','fre_CEloss','kl_loss','hr_mae',
                    'valid_loss','min_valid_loss','MAE','RMSE','MAPE','Pearson','SNR','MACC',
                    'AU_AvgF1','AU_AvgPrec','AU_AvgAcc','Resp_MAE','Resp_RMSE','Resp_MAPE','Resp_Pearson','Resp_SNR'
                ]
                for k in metric_keys:
                    v = row.get(k, '')
                    if isinstance(v, (int, float)) or (isinstance(v, str) and v != ''):
                        # attempt cast to float where possible
                        try:
                            log_payload[k] = float(v)
                        except Exception:
                            log_payload[k] = v
                # Add context tags
                if row.get('mode'):
                    log_payload['mode'] = row['mode']
                if row.get('epoch') not in ['', None]:
                    try:
                        log_payload['epoch'] = int(row['epoch'])
                    except Exception:
                        pass
                if row.get('batch') not in ['', None]:
                    try:
                        log_payload['batch'] = int(row['batch'])
                    except Exception:
                        pass
                if row.get('lr') not in ['', None]:
                    try:
                        log_payload['lr'] = float(row['lr'])
                    except Exception:
                        pass
                if log_payload:
                    wandb.log(log_payload)
            except Exception as e:
                print(f"WARN: wandb logging skipped: {e}")

    def _resolve_model_dir(self):
        """Return the directory to save model checkpoints.
        Prefers SPO2_RUN_ROOT/PreTrainedModels when available; falls back to self.model_dir.
        """
        run_root = os.environ.get('SPO2_RUN_ROOT', None)
        if run_root:
            target = os.path.join(run_root, 'PreTrainedModels')
            try:
                os.makedirs(target, exist_ok=True)
            except Exception:
                pass
            return target
        return getattr(self, 'model_dir', '.')

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def save_test_outputs(self, predictions, labels, config, metrics_dict=None):
        """Save final predictions and labels to disk.

        - 主 CSV: 逐帧波形 (prediction, label)，便于还原时域信号
        - {TASK}_prediction.csv: 按窗口的 FFT 结果 (index, label, prediction)，每行都是标量
        """
        import pandas as pd

        # Prefer unified run root/saved_test_outputs if provided
        run_root = os.environ.get('SPO2_RUN_ROOT', None)
        output_dir = os.path.join(run_root, 'saved_test_outputs') if run_root else config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

        # Ensure numpy arrays on CPU (waveform export)
        import numpy as _np

        def _to_1d_array(x):
            if isinstance(x, (list, tuple)):
                x = _np.asarray(x)
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            elif hasattr(x, 'cpu'):
                x = x.cpu().numpy()
            x = _np.asarray(x).reshape(-1)
            return x

        pred_arr = _to_1d_array(predictions)
        label_arr = _to_1d_array(labels)
        n = min(len(pred_arr), len(label_arr))
        pred_arr = pred_arr[:n]
        label_arr = label_arr[:n]

        df = pd.DataFrame({
            'index': _np.arange(n),
            'prediction': pred_arr,
            'label': label_arr,
            'label_type': [config.TEST.DATA.PREPROCESS.LABEL_TYPE] * n,
            'fs': [config.TEST.DATA.FS] * n,
        })

        # 1) 模型+数据集命名的结果 CSV（原有逻辑）
        csv_path = os.path.join(output_dir, filename_id + '_outputs.csv')
        df.to_csv(csv_path, index=False)
        print('Saving outputs to CSV:', csv_path)

        # 2) 额外导出一个按 TASK 命名的 CSV：{TASK}_prediction.csv，内容为按窗口 FFT 得到的标量
        try:
            task_name = getattr(config, 'TASK', 'TASK').upper()
        except Exception:
            task_name = 'TASK'

        # 只有在 FFT 评估、且 metrics_dict 提供了窗口级输出时才导出
        if metrics_dict is not None and \
           "_fft_window_labels" in metrics_dict and "_fft_window_predictions" in metrics_dict:
            fft_labels = _np.asarray(metrics_dict["_fft_window_labels"], dtype=float).reshape(-1)
            fft_preds = _np.asarray(metrics_dict["_fft_window_predictions"], dtype=float).reshape(-1)
            m = min(len(fft_labels), len(fft_preds))
            fft_labels = fft_labels[:m]
            fft_preds = fft_preds[:m]

            df_fft = pd.DataFrame({
                'index': _np.arange(m),
                'label': fft_labels,
                'prediction': fft_preds,
            })

            task_csv_path = os.path.join(output_dir, f"{task_name}_prediction.csv")
            df_fft.to_csv(task_csv_path, index=False)
            print('Saving task-level FFT prediction CSV to:', task_csv_path)

    def save_best_model(self):
        """Save a snapshot of the current model as best_model.pth in self.model_dir."""
        try:
            if not hasattr(self, 'model'):
                return
            target_dir = self._resolve_model_dir()
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            best_path = os.path.join(target_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
            print('Saved Best Model Path: ', best_path)
        except Exception as e:
            print(f"WARN: failed to save best model: {e}")

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):
        """Plot training/validation losses and LR schedule.
        - Prefers saving under SPO2_RUN_ROOT/plots if available.
        - Flattens LR lists (e.g., OneCycleLR returns a list per step) to scalars.
        - Robust to empty/mismatched inputs.
        """
        # Prefer unified run root if available
        run_root = os.environ.get('SPO2_RUN_ROOT')
        output_dir = os.path.join(run_root, 'plots') if run_root else os.path.join(
            config.LOG.PATH, getattr(config.TRAIN.DATA, 'EXP_DATA_NAME', ''), 'plots'
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = getattr(self, 'model_file_name', 'model')
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

        # Guard inputs
        train_loss = list(train_loss or [])
        valid_loss = list(valid_loss or [])

        # Plot training and validation losses
        if len(train_loss) == 0:
            print('WARN: No training loss values to plot. Skipping loss plot.')
        else:
            plt.figure(figsize=(10, 6))
            epochs = range(0, len(train_loss))
            plt.plot(epochs, train_loss, label='Training Loss')
            if len(valid_loss) > 0:
                # If lengths mismatch, align to min length to avoid shape errors
                if len(valid_loss) != len(train_loss):
                    m = min(len(valid_loss), len(train_loss))
                    print(f'INFO: Aligning valid/train loss lengths ({len(valid_loss)} vs {len(train_loss)}) to {m}.')
                    plt.plot(range(m), valid_loss[:m], label='Validation Loss')
                else:
                    plt.plot(epochs, valid_loss, label='Validation Loss')
            else:
                print('INFO: Validation loss list is empty. Only training loss will be plotted.')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{filename_id} Losses')
            plt.legend()
            plt.xticks(epochs)
            ax = plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))
            loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
            plt.savefig(loss_plot_filename, dpi=300)
            plt.close()

        # Flatten/normalize LR list for plotting
        flat_lrs = []
        for v in (lrs or []):
            try:
                # Common case: list/tuple/ndarray from get_last_lr()
                if isinstance(v, (list, tuple)):
                    if len(v) > 0:
                        flat_lrs.append(float(v[0]))
                elif 'numpy' in str(type(v)):
                    arr = np.array(v).reshape(-1)
                    if arr.size > 0:
                        flat_lrs.append(float(arr[0]))
                elif isinstance(v, torch.Tensor):
                    flat_lrs.append(float(v.flatten()[0].item()))
                else:
                    flat_lrs.append(float(v))
            except Exception:
                continue

        if len(flat_lrs) == 0:
            print('WARN: No learning rate values to plot. Skipping LR plot.')
        else:
            plt.figure(figsize=(6, 4))
            scheduler_steps = range(0, len(flat_lrs))
            plt.plot(scheduler_steps, flat_lrs, label='Learning Rate')
            plt.xlabel('Scheduler Step')
            plt.ylabel('Learning Rate')
            plt.title(f'{filename_id} LR Schedule')
            plt.legend()
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
            plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
            plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)
