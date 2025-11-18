import pytest

import numpy as np

import sklearn.datasets
import sklearn.model_selection

from npkit.data import NumpyDataLoader


class DummyNumpyDataModule:
    def __init__(self, batch_size: int = 10):
        super().__init__()

        self.batch_size: int = batch_size
        self.num_samples: int = 100 
        self.num_features: int = 20 

        data, target = sklearn.datasets.make_classification(n_samples=self.num_samples, n_features=self.num_features, n_redundant=0, n_clusters_per_class=1, class_sep=3.0, random_state=0)
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=0.2, random_state=0)

        self.train_dataloader = NumpyDataLoader(train_data, train_target, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = NumpyDataLoader(test_data, test_target, batch_size=self.batch_size, shuffle=False, drop_last=False)


def test_dataloader_reproducibility_1():

    np.random.seed(0)
    data_module_1 = DummyNumpyDataModule()
    d1 = []

    for x, y in data_module_1.train_dataloader:
        d1.append(x)

    np.random.seed(0)
    data_module_2 = DummyNumpyDataModule()
    d2 = []

    for x, y in data_module_2.train_dataloader:
        d2.append(x)

    np.testing.assert_array_equal(d1, d2)


def test_dataloader_reproducibility_2():

    np.random.seed(0)
    data_module_1 = DummyNumpyDataModule()
    d1 = []

    for x, y in data_module_1.train_dataloader:
        d1.append(x)

    np.random.seed(1)
    data_module_2 = DummyNumpyDataModule()
    d2 = []

    for x, y in data_module_2.train_dataloader:
        d2.append(x)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(d1, d2)