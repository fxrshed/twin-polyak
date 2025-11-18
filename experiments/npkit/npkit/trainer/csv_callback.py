import pathlib
import csv
import datetime
import json

from .trainer import NumpyBaseTrainingModule
from .callback import BaseCallback


class CSVLoggerCallback(BaseCallback):

    def __init__(self, trainer: NumpyBaseTrainingModule, db_path: str = 'np_experiments_db.csv'):

        super().__init__(trainer, db_path)

        self.base_fields: list = ['start_time', 'end_time', 'seed',
                             'dataset', 'batch_size', 'task',
                             'model', 'max_epochs', 'optimizer',
                             'lmd', 'optimizer_hparams', 'metrics_path']

        self.fields = self.base_fields.copy()

        if not pathlib.Path(db_path).exists():
            with open(db_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def on_fit_start(self):
        self.start_time = datetime.datetime.now().isoformat() 

    def on_fit_end(self):
        end_time = datetime.datetime.now().isoformat()

        hparams = self.trainer.logger.hparams
        config = hparams.get('config', {})
        optimizer_hparams = config.get('optimizer_hparams', {})
        flat_opt_hparams = {f"optimizer_hp_{k}": str(v) for k, v in optimizer_hparams.items()}

        metrics_path = self.trainer.logger.log_path

        row = {
            'start_time': self.start_time,
            'end_time': end_time,
            'max_epochs': config.get('max_epochs', -1),
            'seed': config.get('seed', -1),
            'dataset': hparams.get('dataset', 'unknown'),
            'batch_size': config.get('batch_size', -1),
            'task': hparams.get('task', 'unknown'),
            'model': hparams.get('model', 'unknown'),
            'optimizer': str(config.get('optimizer', 'unknown')),
            'optimizer_hparams': json.dumps(optimizer_hparams),
            'lmd': config.get('lmd', 0.0),
            'metrics_path': metrics_path,
        }

        row.update(flat_opt_hparams)
        self.update_csv_header_if_needed(row.keys())

        with open(self.db_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        
        def is_match(existing_row):
            return (
                existing_row['dataset'] == row['dataset'] and
                existing_row['model'] == row['model'] and
                existing_row['optimizer'] == row['optimizer'] and
                existing_row['lmd'] == row['lmd'] and
                existing_row['seed'] == str(row['seed']) and
                existing_row['optimizer_hparams'] == row['optimizer_hparams']
            )
        
        updated_rows = [r for r in existing_rows if not is_match(r)]
        updated_rows.append(row)

        with open(self.db_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            for r in updated_rows:
                writer.writerow(r)

    def update_csv_header_if_needed(self, new_fields):

        with open(self.db_path, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_fields = next(reader)

        new_fieldnames = existing_fields.copy()
        for field in new_fields:
            if field not in existing_fields:
                new_fieldnames.append(field)

        if new_fieldnames != existing_fields:
            with open(self.db_path, 'r') as f:
                rows = list(csv.DictReader(f))

            with open(self.db_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            self.fields = new_fieldnames
        else:
            self.fields = existing_fields