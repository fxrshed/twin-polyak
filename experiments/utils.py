import os
import argparse
import re
import pickle
import datetime

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import numpy as np

import svmlight_loader

from dotenv import load_dotenv
load_dotenv()

def save_results(results, model_name, dataset_name, scale, batch_size,
                 n_epochs, optimizer, lr, seed):

    results_path = os.getenv("RESULTS_DIR")

    directory = f"{results_path}/{model_name}/{dataset_name}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{n_epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(f"{directory}/summary.p", "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {directory}")

def load_results(dataset_name: str, model_name: str, scale: int, batch_size: int, n_epochs: int, optimizer: str, lr: float, seed: int) -> dict:
    
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{model_name}/{dataset_name}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{n_epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    
    assert os.path.exists(directory), f"Results {directory} do not exist."

    # results = torch.load(f"{directory}/summary.p", map_location=torch.device('cpu'))
    
    with open(f"{directory}/summary.p", "rb") as f:
        results = pickle.load(f)

        
    return results 

def make_synthetic_binary_classification(
    n_samples: int, 
    n_features: int, 
    classes: tuple[int, int] = (-1, 1),
    symmetric: bool = False, 
    seed: int = 0):
    
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    
    if symmetric:
        assert n_samples == n_features, f"`n_samples` must be equal to `n_features` to get symmetric matrix. " \
            f"Currently `n_samples={n_samples}` and `n_features={n_features}`."
        data = (data + data.T) / 2
    w_star = np.random.randn(n_features)

    target = data @ w_star
    target[target <= 0.0] = classes[0]
    target[target > 0.0] = classes[1]

    return data, target

datasets_path = os.getenv("LIBSVM_DIR")
datasets_params = {
    "webspam": {
        "train_path": f"{datasets_path}/webspam/webspam_train",
        "test_path": f"{datasets_path}/webspam/webspam_test",
        "n_features": 16_609_143, 
        },
    "news20.binary": {
        "train_path": f"{datasets_path}/news20/news20.binary",
        "n_features": 1_355_191, 
        },
    "a1a": {
        "train_path": f"{datasets_path}/a1a",
        "test_path": f"{datasets_path}/a1a.t",
        "n_features": 123,
    },
    "a8a": {
        "train_path": f"{datasets_path}/a8a",
        "test_path": f"{datasets_path}/a8a.t",
        "n_features": 123,
    },
    "a9a": {
        "train_path": f"{datasets_path}/a9a",
        "test_path": f"{datasets_path}/a9a.t",
        "n_features": 123,
    },
    "w1a": {
        "train_path": f"{datasets_path}/w1a",
        "test_path": f"{datasets_path}/w1a.t",
        "n_features": 300,
    },
    "w8a": {
        "train_path": f"{datasets_path}/w8a",
        "test_path": f"{datasets_path}/w8a.t",
        "n_features": 300,
    },
    "mushrooms": {
        "train_path": f"{datasets_path}/mushrooms",
        "n_features": 112,
    },
    "leu": {
        "train_path": f"{datasets_path}/leu",
        "test_path": f"{datasets_path}/leu.t",
        "n_features": 7129,
    },
    "real-sim": {
        "train_path": f"{datasets_path}/real-sim",
        "n_features": 20_958,
    },
    "rcv1.binary": {
        "train_path":  f"{datasets_path}/rcv1_train.binary",
        "test_path": f"{datasets_path}/rcv1_test.binary",
        "n_features": 47_236,
    },
    "colon-cancer": {
        "train_path": f"{datasets_path}/colon-cancer",
        "test_path": f"{datasets_path}/colon-cancer",
        "n_features": 2000,
    },
    "madelon": {
        "train_path": f"{datasets_path}/madelon",
        "test_path": f"{datasets_path}/madelon.t",
        "n_features": 500,
    },
    "gisette": {
        "train_path": f"{datasets_path}/gisette_scale",
        "test_path": f"{datasets_path}/gisette_scale.t",
        "n_features": 5000,
    },
    "duke": {
        "train_path": f"{datasets_path}/duke",
        "test_path": f"{datasets_path}/duke.tr",
        "n_features": 7129,
    },
    "diabetes_scale": {
        "train_path": f"{datasets_path}/diabetes_scale",
        "test_path": f"{datasets_path}/diabetes_scale",
        "n_features": 8,
    },
    "covtype.binary": {
        "train_path": f"{datasets_path}/covtype.libsvm.binary",
        "n_features": 54,
    },
    "covtype.binary.scale": {
        "train_path": f"{datasets_path}/covtype.libsvm.binary.scale",
        "n_features": 54,
    },
    "australian_scale": {
        "train_path": f"{datasets_path}/australian_scale",
        "test_path": f"{datasets_path}/australian_scale",
        "n_features": 14,
    },
    "breast-cancer_scale": {
        "train_path": f"{datasets_path}/breast-cancer_scale",
        "test_path": f"{datasets_path}/breast-cancer_scale",
        "n_features": 10,
    },
    "sonar_scale": {
        "train_path": f"{datasets_path}/sonar_scale",
        "test_path": f"{datasets_path}/sonar_scale",
        "n_features": 60,
    },
    "wine.scale": {
        "train_path": f"{datasets_path}/wine.scale",
        "test_path": f"{datasets_path}/wine.scale",
        "n_features": 13,
    },
    "sensorless.scale": {
        "train_path": f"{datasets_path}/Sensorless.scale.tr",
        "test_path": f"{datasets_path}/Sensorless.scale.val",
        "n_features": 48,
    },
    "sector.scale": {
        "train_path": f"{datasets_path}/sector.scale",
        "test_path": f"{datasets_path}/sector.t.scale",
        "n_features": 55_197,
    },
    "shuttle.scale": {
        "train_path": f"{datasets_path}/shuttle.scale",
        "test_path": f"{datasets_path}/shuttle.scale.t",
        "n_features": 9,
    },
    "protein": {
        "train_path": f"{datasets_path}/protein",
        "test_path": f"{datasets_path}/protein.t",
        "n_features": 357,
    },
    "usps": {
        "train_path": f"{datasets_path}/usps",
        "test_path": f"{datasets_path}/usps.t",
        "n_features": 256,
    },
    "dna.scale": {
        "train_path": f"{datasets_path}/dna.scale",
        "test_path": f"{datasets_path}/dna.scale.t",
        "n_features": 180,
    },
    "aloi.scale": {
        "train_path": f"{datasets_path}/aloi.scale",
        "n_features": 128,
    }
    
}


def get_libsvm(name: str, test_split: float = 0.0, seed: int = 0):
    
    test_data = None 
    test_target = None

    train_path = datasets_params[name].get("train_path")
    test_path = datasets_params[name].get("test_path")
    n_features = datasets_params[name]["n_features"]

    train_data, train_target = svmlight_loader.load_svmlight_file(train_path, n_features=n_features)
    
    if test_path is not None:
        test_data, test_target = svmlight_loader.load_svmlight_file(test_path, n_features=n_features)
    elif test_split > 0.0:
        print(f"Test data for `{name}` is not found. Splitting train into {(1 - test_split) * 100}% train and {test_split * 100}% test.")
        train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=test_split, random_state=seed)
        
    return train_data, train_target, test_data, test_target

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.01 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.01, 1.0]"%(x,))
    return x


def map_classes_to(target, new_classes):
    old_classes = np.unique(target)
    new_classes = np.sort(new_classes)
    
    if np.array_equal(old_classes, new_classes):
        return target
    
    assert np.unique(target).size == len(new_classes), \
        f"Old classes must match the number of new classes. " \
        f"Currently ({np.unique(target).size}) classes are being mapped to ({len(new_classes)}) new classes."

    mapping = {v: t for v, t in zip(old_classes, new_classes)}
    target = np.vectorize(mapping.get)(target)
    return target

TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

def get_MNIST(train_batch_size: int, test_batch_size: int, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.1307,), (0.3081,),
        ),
    ])

    train_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms)
    test_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_CIFAR10(train_batch_size: int, test_batch_size: int, size: tuple, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=size, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )

    test_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_CIFAR100(train_batch_size: int, test_batch_size: int, size: tuple, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=size, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR100(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )
    test_data = torchvision.datasets.CIFAR100(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_FashionMNIST(train_batch_size: int, test_batch_size: int, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(28, 28), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.5,), (0.5,),
        ),
    ])

    train_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )

    test_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader



@torch.inference_mode
def evaluate_classification_model(model: torch.nn.Module, criterion: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float, list[float], list[float]]:
    
    test_epoch_loss = 0.0
    
    test_batch_loss = []
    test_batch_acc = []
    
    total = 0
    correct = 0
    for i, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_target)
        test_epoch_loss += loss.item() * batch_data.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_target.size(0)
        batch_correct = (predicted == batch_target).sum().item()
        correct += batch_correct
        
        test_batch_loss.append(loss.item())
        test_batch_acc.append(batch_correct)
    
    test_epoch_loss = test_epoch_loss / len(test_loader.sampler)
    test_epoch_acc = correct / total
    return test_epoch_loss, test_epoch_acc, test_batch_loss, test_batch_acc




