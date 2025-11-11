import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import csv
import pickle


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
        if not self.csv_path:
            return
        try:
            row = {c: row_dict.get(c, '') for c in self._csv_columns}
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                writer.writerow(row)
        except Exception as e:
            print(f"WARN: failed to append CSV row: {e}")

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def save_test_outputs(self, predictions, labels, config):
    
        output_dir = config.TEST.OUTPUT_SAVE_DIR
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
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)

    def save_best_model(self):
        """Save a snapshot of the current model as best_model.pth in self.model_dir."""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'model_dir'):
                return
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
            print('Saved Best Model Path: ', best_path)
        except Exception as e:
            print(f"WARN: failed to save best model: {e}")

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):

        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a single plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        if len(valid_loss) > 0:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("The list of validation losses is empty. The validation loss will not be plotted!")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # Force scientific notation

        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)
