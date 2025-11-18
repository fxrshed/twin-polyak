import os
import datetime
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split

import scipy

import lightning as L
from lightning.pytorch import seed_everything, loggers

from utils import SimpleDataset, parse_optimizer_hparams
from custom_logger import DBLogger
from base_module import BaseTrainingModule

from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class AbaloneDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = os.getenv("LIBSVM_DIR"), batch_size: int = 32):
        super().__init__()

        data, target = sklearn.datasets.load_svmlight_file(f'{data_dir}/abalone_scale')
        data = sklearn.preprocessing.normalize(data, norm='l2', axis=1)
        self.train_data, self.val_data, self.train_target, self.val_target = train_test_split(data, target, test_size=0.2, random_state=0)

        self.batch_size: int = batch_size
        self.num_features: int = data.shape[1]

    def setup(self, stage: str):

        if stage  == 'fit':
            self.train_dataset = SimpleDataset(self.train_data, self.train_target)
            self.val_dataset = SimpleDataset(self.val_data, self.val_target)
        if stage in ('test', 'predict'):
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

        return xs, torch.tensor(ys, dtype=torch.float32)


class RegressionModel(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class AbaloneRegressor(BaseTrainingModule):

    def __init__(self, input_dim: int, config: dict):

        self.input_dim = input_dim

        self.save_hyperparameters(
            {
                'dataset': 'abalone_scale',
                'task': 'regression',
                'model': 'linear',
                'config': config,
            }
        )

        super().__init__(config)

    def build_model(self):
        return RegressionModel(self.input_dim)

    def define_loss_fn(self):
        return F.mse_loss

    def define_val_acc_metric(self):
        return None

    def unpack_batch(self, batch):
        x, y = batch
        return x.to_dense(), y


def run_experiment(config):

    data_module = AbaloneDataModule(batch_size=config['batch_size'])
    data_module.setup('fit')

    seed_everything(config['seed'], workers=True)

    model = AbaloneRegressor(input_dim=data_module.num_features, config=config)

    db_logger_callback = DBLogger()
    csv_logger = loggers.CSVLogger(
        save_dir=f"logs/{model.hparams['dataset']}",
        version=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        logger=[csv_logger],
        callbacks=[db_logger_callback],
        accelerator='cpu',
        devices=1,
        log_every_n_steps=min(len(data_module.train_dataloader()), 50)
        )

    trainer.fit(model=model, datamodule=data_module)


def main(args):

    config = {
        'seed': args.seed,
        'max_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'optimizer_hparams': args.optimizer_hparams,
    }

    if 'lr' in args.optimizer_hparams and args.optimizer_hparams.get('lr') == -1:
        print("[INFO]: Learning rate sweep is enabled.")
        # lrs = [10**x for x in range(-10, 6)]
        lrs = [10**x for x in range(6, 10)]

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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args, unknown = parser.parse_known_args()

    args.optimizer_hparams = parse_optimizer_hparams(unknown)

    main(args)