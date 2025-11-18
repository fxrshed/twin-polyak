import os
import csv
import datetime
import uuid
import json

from lightning import Callback


class DBLogger(Callback):

    def __init__(self, db_path: str = 'experiments_db.csv'):
        super().__init__()
        self.db_path: str = db_path

        self.run_id = str(uuid.uuid4())
        self.base_fields: list = ['run_id', 'start_time', 'end_time', 'seed',
                                  'dataset', 'batch_size', 'task', 'model',
                                  'max_epochs', 'optimizer',
                                  'optimizer_hparams', 'metrics_path']
        self.fields = self.base_fields.copy()

        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def on_fit_start(self, trainer, pl_module):
        self.start_time = datetime.datetime.now().isoformat()

    def on_fit_end(self, trainer, pl_module):
        end_time = datetime.datetime.now().isoformat()

        hparams = trainer.model.hparams
        config = hparams.get('config', {})
        optimizer_hparams = config.get('optimizer_hparams', {})
        flat_opt_hparams = self.flatten_hparams(optimizer_hparams)

        metrics_path = trainer.logger[0].log_dir if isinstance(trainer.logger, list) else trainer.logger.log_dir

        row = {
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': end_time,
            'max_epochs': config.get('max_epochs', -1),
            'seed': config.get('seed', -1),
            'dataset': hparams.get('dataset', 'unknown'),
            'batch_size': config.get('batch_size', -1),
            'task': hparams.get('task', 'unknown'),
            'model': hparams.get('model', 'unknown'),
            'optimizer': config.get('optimizer', 'unknown'),
            'optimizer_hparams': json.dumps(optimizer_hparams),
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

    def flatten_hparams(self, hparams: dict) -> dict:
        return {f"hp_{k}": v for k, v in hparams.items()}

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
