import os
import datetime
import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.transforms import v2

import lightning as L
import torchmetrics
from lightning.pytorch import seed_everything, loggers

import neptune

from custom_logger import DBLogger
from base_module import BaseTrainingModule
from utils import parse_optimizer_hparams

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


class USPSDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = os.getenv("TORCHVISION_DATASETS_DIR"), batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size: int = batch_size
        self.num_features: int = 256  # 16 * 16
        self.num_labels: int = 10

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.get_default_dtype(), scale=True),
            v2.Normalize(
                (0.5, ), (0.5, ))
        ])

    def setup(self, stage: str):

        if stage  == 'fit':
            self.train_dataset = tv.datasets.USPS(self.data_dir, train=True, download=True, transform=self.transforms)
            self.val_dataset = tv.datasets.USPS(self.data_dir, train=False, download=True, transform=self.transforms)
        if stage == 'test':
            self.val_dataset = tv.datasets.USPS(self.data_dir, train=False, download=True, transform=self.transforms)
        if stage == 'predict':
            self.val_dataset = tv.datasets.USPS(self.data_dir, train=False, download=True, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


class MLP(nn.Module):

    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_labels)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return self.fc2(x)


class USPSClassifier(BaseTrainingModule):

    def __init__(self, input_dim: int, num_labels: int, config: dict):

        self.input_dim = input_dim
        self.num_labels = num_labels

        self.save_hyperparameters(
            {
                'dataset': 'usps',
                'task': 'multi-class-classification',
                'model': 'mlp',
                'config': config,
            }
        )

        super().__init__(config)

    def build_model(self):
        return MLP(input_dim=self.input_dim, num_labels=self.num_labels)

    def define_loss_fn(self):
        return nn.CrossEntropyLoss()

    def define_val_acc_metric(self):
        return torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_labels)

    def unpack_batch(self, batch):
        x, y = batch
        return x.to_dense(), y.to_dense()


def run_experiment(config: dict) -> None:

    data_module = USPSDataModule(batch_size=config['batch_size'])
    data_module.setup('fit')

    seed_everything(config['seed'], workers=True)

    model = USPSClassifier(input_dim=data_module.num_features,
                           num_labels=data_module.num_labels,
                           config=config)

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
        accelerator='gpu',
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
        'reg': args.reg,
        'optimizer': args.optimizer,
        'optimizer_hparams': args.optimizer_hparams,
    }

    print(args)

    if 'lr' in args.optimizer_hparams and args.optimizer_hparams.get('lr') == -1:
        print("[INFO]: Learning rate sweep is enabled.")
        # lrs = [10**x for x in range(-10, 6)]
        lrs = [10**x for x in range(-5, 3)]

        for lr in lrs:
            config['optimizer_hparams']['lr'] = lr
            run_experiment(config)
    elif 'eta_max' in args.optimizer_hparams and args.optimizer_hparams.get('eta_max') == -1:
        print("[INFO]: Learning rate sweep is enabled.")
        # lrs = [10**x for x in range(-10, 6)]
        lrs = [10**x for x in range(-5, 3)]

        for lr in lrs:
            config['optimizer_hparams']['eta_max'] = lr
            run_experiment(config)
    else:
        run_experiment(config=config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide additional arguments to parser, such as optimizer hyperparameters, after required arguments.")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args, unknown = parser.parse_known_args()

    args.optimizer_hparams = parse_optimizer_hparams(unknown)

    main(args)