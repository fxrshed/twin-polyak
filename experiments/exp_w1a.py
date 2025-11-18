import os 
import argparse

import numpy as np

import sklearn.datasets
import sklearn.preprocessing

import utils

from npkit.data import NumpyDataLoader

from libsvm_binary_classifier import NumpyLibSVMBinaryClassifier


from dotenv import load_dotenv
load_dotenv()


class W1ANumpyDataModule:
    def __init__(self, data_dir: str = os.getenv("LIBSVM_DIR"), batch_size: int = 64):
        super().__init__()

        self.dataset = 'w1a'
        self.batch_size: int = batch_size
        self.num_samples: int = 2_477
        self.num_features: int = 300
        self.num_labels: int = 2
        
        train_data, train_target = sklearn.datasets.load_svmlight_file(f'{data_dir}/{self.dataset}', n_features=self.num_features)
        train_data = sklearn.preprocessing.normalize(train_data, norm='l2', axis=1)
        train_target = utils.map_classes_to(train_target, [-1.0, 1.0])

        val_data, val_target = sklearn.datasets.load_svmlight_file(f'{data_dir}/{self.dataset}.t', n_features=self.num_features)
        val_data = sklearn.preprocessing.normalize(val_data, norm='l2', axis=1)
        val_target = utils.map_classes_to(val_target, [-1.0, 1.0])
        
        assert np.all(np.sort(np.unique(train_target)) == [-1.0, 1.0])
        assert np.all(np.sort(np.unique(val_target)) == [-1.0, 1.0])

        self.train_dataloader = NumpyDataLoader(train_data, train_target, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.val_dataloader = NumpyDataLoader(val_data, val_target, batch_size=self.num_samples, shuffle=False, drop_last=False) # `batch_size` == dataset size because it is a small dataset


def run_experiment(config):

    np.random.seed(config['seed'])
    datamodule = W1ANumpyDataModule(batch_size=config['batch_size'])

    classifier = NumpyLibSVMBinaryClassifier(dataset_name=datamodule.dataset, 
                                            input_dim=datamodule.num_features,
                                            config=config, 
                                            max_epochs=config['max_epochs'])
    classifier.fit(datamodule=datamodule)


def main(args):

    config = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'max_epochs': args.n_epochs,
        'optimizer': args.optimizer,
        'optimizer_hparams': args.optimizer_hparams
    }

    if args.seed == -1:
        seeds = [0, 1, 2, 3, 4]
        for seed in seeds:
            config['seed'] = seed
            run_experiment(config)
    else:
        run_experiment(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide additional arguments to parser, such as optimizer hyperparameters, after required arguments.")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args, unknown = parser.parse_known_args()

    args.optimizer_hparams = utils.parse_optimizer_hparams(unknown)

    main(args)