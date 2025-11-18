from collections import defaultdict
import yaml
import pathlib
import csv

import numpy as np


class NumpyLogger:

    def __init__(self, save_dir: str, version: str):
        self.save_dir = save_dir
        self.version = version
        self.log_path = f"{save_dir}/{version}"

        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)

        self.current_storage: dict = defaultdict(list)
        self.storage: dict = defaultdict(list)

        self.csv_file = f"{self.log_path}/metrics.csv"
        self._fieldnames = ['epoch', 'step']
        self._rows: list = []
        self._csv_initialized = False

    def __getitem__(self, key):
        return self.storage[key]

    def log(self, name, value, epoch: int, step: int, on_epoch: bool = True):
        if on_epoch:
            self.current_storage[name].append(value)
        else:
            self.storage[name].append(value)
            row = {
                'epoch': epoch,
                'step': step,
                name: value
            }
            self._write_csv_row(row)
    
    def compute_on_epoch(self, epoch, step):
        for key, values in self.current_storage.items():
            value = np.mean(values)
            self.storage[key].append(value)

            row = {
                'epoch': epoch,
                'step': step,
                key: value
            }
            self._write_csv_row(row)

    def _write_csv_row(self, row: dict):
        
        new_keys = [k for k in row.keys() if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv_with_new_headers()

        self._rows.append(row)
        file_exists = pathlib.Path(self.csv_file).exists()

        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if not file_exists or not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def _rewrite_csv_with_new_headers(self):
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def reset(self):
        self.current_storage.clear()

    def save_hyperparameters(self, hparams: dict):
        filepath = f"{self.save_dir}/{self.version}/hparams.yaml"

        with open(filepath, 'w+') as f:
            yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)

        self.hparams = hparams
