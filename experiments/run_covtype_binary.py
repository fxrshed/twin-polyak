import os
import datetime
import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from utils import SimpleDataset, parse_optimizer_hparams

import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing

import scipy

import lightning as L
from lightning.pytorch import seed_everything, loggers
import torchmetrics

import neptune

from custom_logger import DBLogger
from base_module import BaseTrainingModule

from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing:"
            )
        )


neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)


class CovtypeBinaryDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = os.getenv("LIBSVM_DIR"), batch_size: int = 32):
        super().__init__()

        data, target = sklearn.datasets.load_svmlight_file(f'{data_dir}/covtype.libsvm.binary.scale')
        data = sklearn.preprocessing.normalize(data, norm='l2', axis=1)
        target = (target - 1).astype(np.float32)

        assert np.all(np.sort(np.unique(target)) == [0.0, 1.0])

        self.train_data, self.val_data, self.train_target, self.val_target = sklearn.model_selection.train_test_split(data, target, test_size=0.2, random_state=0)

        self.batch_size: int = batch_size
        self.num_features: int = 54
        self.num_labels: int = 2

    def setup(self, stage: str):

        if stage  == 'fit':
            self.train_dataset = SimpleDataset(self.train_data, self.train_target)
            self.val_dataset = SimpleDataset(self.val_data, self.val_target)
        if stage == 'test':
            self.val_dataset = SimpleDataset(self.val_data, self.val_target)
        if stage == 'predict':
            self.val_dataset = SimpleDataset(self.val_data, self.val_target)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._sparse_collate, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._sparse_collate, num_workers=2, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._sparse_collate, num_workers=2, shuffle=False)

    def _sparse_coo_to_tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        s = torch.Size(shape)

        return torch.sparse_coo_tensor(i, v, s)

    def _sparse_collate(self, batch):
        xs, ys = zip(*batch)

        xs = scipy.sparse.vstack(xs).tocoo()
        xs = self._sparse_coo_to_tensor(xs)

        return xs, torch.tensor(ys, dtype=torch.float32).unsqueeze(1)


class LinearBinaryModel(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class CovtypeBinaryClassifier(BaseTrainingModule):

    def __init__(self, input_dim: int, config: dict):

        self.input_dim = input_dim

        self.save_hyperparameters(
            {
                'dataset': 'covtype.binary',
                'task': 'binary-classification',
                'model': 'linear',
                'config': config,
            }
        )

        super().__init__(config)

    def build_model(self):
        return LinearBinaryModel(self.input_dim)

    def define_loss_fn(self):
        return nn.BCEWithLogitsLoss()

    def define_val_acc_metric(self):
        return torchmetrics.classification.BinaryAccuracy()

    def unpack_batch(self, batch):
        x, y = batch
        return x.to_dense(), y.to_dense()


def run_experiment(config: dict) -> None:

    data_module = CovtypeBinaryDataModule(batch_size=config['batch_size'])
    data_module.setup('fit')

    seed_everything(config['seed'], workers=True)

    model = CovtypeBinaryClassifier(input_dim=data_module.num_features, config=config)

    db_logger_callback = DBLogger()
    csv_logger = loggers.CSVLogger(
        save_dir=f"logs/{model.hparams['dataset']}",
        version=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    neptune_run = neptune.init_run(
        mode='async',
        tags=[model.hparams['task']],
    )

    neptune_run['dataset/name'] = model.hparams['dataset']
    neptune_run['model'] = model.hparams['model']
    neptune_run['config'] = config

    print(config)

    neptune_logger = loggers.NeptuneLogger(
        run=neptune_run
    )

    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        logger=[csv_logger, neptune_logger],
        callbacks=[db_logger_callback],
        accelerator='cpu',
        devices=1,
        log_every_n_steps=min(len(data_module.train_dataloader()), 50)
        )

    trainer.fit(model=model, datamodule=data_module)

    neptune_run.stop()


def main(args):

    config = {
        'seed': args.seed,
        'max_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'optimizer_hparams': args.optimizer_hparams,
    }

    print(args)

    if 'lr' in args.optimizer_hparams and args.optimizer_hparams.get('lr') == -1:
        print("[INFO]: Learning rate sweep is enabled.")
        lrs = [10**x for x in range(-10, 6)]

        for lr in lrs:
            config['optimizer_hparams']['lr'] = lr
            run_experiment(config)
    elif 'eta_max' in args.optimizer_hparams and args.optimizer_hparams.get('eta_max') == -1:
        print("[INFO]: Learning rate sweep is enabled.")
        lrs = [10**x for x in range(-10, 6)]

        for lr in lrs:
            config['optimizer_hparams']['eta_max'] = lr
            run_experiment(config)
    else:
        run_experiment(config=config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide additional arguments for parser, such as optimizer hyperparameters, after required arguments.")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args, unknown = parser.parse_known_args()

    args.optimizer_hparams = parse_optimizer_hparams(unknown)

    main(args)