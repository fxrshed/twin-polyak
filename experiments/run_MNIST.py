import os
import sys
import argparse
import time 
import datetime
from collections import defaultdict

import numpy as np
import neptune

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision.transforms import v2

import sps
import sls
from pt_methods import *

import utils

from dotenv import load_dotenv
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

def compute_topk_accuracy(output, target, k=5):
    """Compute top-k accuracy for the given outputs and targets."""
    with torch.no_grad():
        # Get the top-k indices
        _, topk_indices = output.topk(k, dim=1)
        # Check if targets are in the top-k predictions
        correct = topk_indices.eq(target.view(-1, 1).expand_as(topk_indices))
        # Compute accuracy
        return correct.any(dim=1).float().mean().item()


def eval_model(model: nn.Module, criterion: nn.Module, dataloader: DataLoader): 
    test_epoch_loss = 0.0
    total = 0
    correct = 0
    top3_acc = 0.0
    top5_acc = 0.0
    for i, (batch_data, batch_target) in enumerate(dataloader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_target)
        test_epoch_loss += loss.item() * batch_data.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_target.size(0)
        batch_correct = (predicted == batch_target).sum().item()
        batch_accuracy = batch_correct / batch_target.size(0)
        correct += batch_correct
        
        top3_acc += compute_topk_accuracy(outputs, batch_target, k=3) * batch_target.size(0)
        top5_acc += compute_topk_accuracy(outputs, batch_target, k=5) * batch_target.size(0)
        
    loss = test_epoch_loss / len(dataloader.sampler)
    top1_accuracy = correct / total
    top3_accuracy = top3_acc / total
    top5_accuracy = top5_acc / total

    return loss, (top1_accuracy, top3_accuracy, top5_accuracy)

optimizers_dict = {
    "SGD": torch.optim.SGD,
    "SPSMAX": sps.Sps,
    "SLS": sls.Sls,
    "DecSPS": DecSPS,
}

def train_optimizer(
    model_name: str,
    dataset: tuple[DataLoader, DataLoader],
    optimizer_name: str,
    n_epochs: int, 
    seed: int = 0,
    neptune_mode: str = "async",
    **optimizer_kwargs
                    ) -> dict:
    
    torch.manual_seed(seed)
    
    train_dataloader, test_dataloader = dataset
    
    num_classes = len(np.unique(train_dataloader.dataset.classes))
    
    if model_name == "LeNet5":
        model = LeNet5().to(device)
    
    optimizer = optimizers_dict[optimizer_name](model.parameters(), **optimizer_kwargs)
    if optimizer_name in ["SGD"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = defaultdict(list)
    
    run = neptune.init_run(
        mode=neptune_mode,
        tags=["multi-class-classification"]
    )
    
    run["dataset"] = {
        "name": "MNIST",
        "train_batch_size": train_dataloader.batch_size,
        "test_batch_size": test_dataloader.batch_size
    }
    
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    run["optimizer/parameters/name"] = optimizer_name
    run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer.defaults)
    run["model"] = model_name
    run["device"] = str(device)
    

    for epoch in range(n_epochs):
        
        model.eval()
        with torch.inference_mode():
            loss, accuracy = eval_model(model, criterion, test_dataloader)
            history["test/loss"].append(loss)
            history["test/accuracy/top1"].append(accuracy[0])
            history["test/accuracy/top3"].append(accuracy[1])
            history["test/accuracy/top5"].append(accuracy[2])
            
            run["test/loss"].append(loss)
            run["test/accuracu/top1"].append(accuracy[0])
            run["test/accuracy/top3"].append(accuracy[1])
            run["test/accuracy/top5"].append(accuracy[2])
            
            print(f"Epoch: {epoch} | Test loss: {loss} | Test accuracy: {accuracy[0]}")
        
        train_loss = 0.0
        model.train()
        for i, (batch_data, batch_target) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            def closure():
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_target)
                loss.backward()
                return loss

            if isinstance(optimizer, sps.Sps):
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.state["step_size"])
                run["lr"].append(optimizer.state["step_size"])
                train_loss += loss * batch_data.size(0)
            elif isinstance(optimizer, sls.Sls):
                def closure():
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_target)
                    return loss
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.state["step_size"])
                run["lr"].append(optimizer.state["step_size"])
                train_loss += loss.item() * batch_data.size(0)
            elif isinstance(optimizer, DecSPS):
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.step_size)
                run["lr"].append(optimizer.step_size)
                train_loss += loss.item() * batch_data.size(0)
            else:
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.param_groups[0]["lr"])
                run["lr"].append(optimizer.param_groups[0]["lr"])
                train_loss += loss.item() * batch_data.size(0)
        
        if optimizer_name in ["SGD"]:
            scheduler.step()
        
        history["train/loss"].append(train_loss / len(train_dataloader.sampler))
        run["train/loss"].append(train_loss / len(train_dataloader.sampler))

    
    run.stop()
    
    return history


def train_twin_polyak(
    model_name: str,
    dataset: tuple[DataLoader, DataLoader],
    n_epochs: int,
    seed: int = 0,
    neptune_mode: str = "async",
    **optimizer_kwargs) -> dict:
    
    torch.manual_seed(seed)
    
    train_dataloader, test_dataloader = dataset
    
    num_classes = len(np.unique(train_dataloader.dataset.classes))
    if model_name == "LeNet5":
        model = LeNet5
        
    model_x = model().to(device)
    model_y = model().to(device)

    optimizer_x = TwinPolyak(model_x.parameters(), **optimizer_kwargs)
    optimizer_y = TwinPolyak(model_y.parameters(), **optimizer_kwargs)
    
    criterion = nn.CrossEntropyLoss()

    history = defaultdict(list)
    
    run = neptune.init_run(
        mode=neptune_mode,
        tags=["multi-class-classification"]
    )
    
    run["dataset"] = {
        "name": "MNIST",
        "train_batch_size": train_dataloader.batch_size,
        "test_batch_size": test_dataloader.batch_size
    }
    
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    run["optimizer/parameters/name"] = "STP"
    run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer_x.defaults)
    run["model"] = model_name
    run["device"] = str(device)
    
    for epoch in range(n_epochs):
        
        model_x.eval()
        model_y.eval()
        with torch.inference_mode():
            loss_x, accuracy_x = eval_model(model_x, criterion, test_dataloader)
            loss_y, accuracy_y = eval_model(model_y, criterion, test_dataloader)

            history["test/loss_x"].append(loss_x)
            history["test/accuracy_x/top1"].append(accuracy_x[0])
            history["test/accuracy_x/top3"].append(accuracy_x[1])
            history["test/accuracy_x/top5"].append(accuracy_x[2])

            run["test/loss_x"].append(loss_x)
            run["test/accuracy_x/top1"].append(accuracy_x[0])
            run["test/accuracy_x/top3"].append(accuracy_x[1])
            run["test/accuracy_x/top5"].append(accuracy_x[2])
            
            history["test/loss_y"].append(loss_y)
            history["test/accuracy_y/top1"].append(accuracy_y[0])
            history["test/accuracy_y/top3"].append(accuracy_y[1])
            history["test/accuracy_y/top5"].append(accuracy_y[2])
            
            run["test/loss_y"].append(loss_y)
            run["test/accuracy_y/top1"].append(accuracy_y[0])
            run["test/accuracy_y/top3"].append(accuracy_y[1])
            run["test/accuracy_y/top5"].append(accuracy_y[2])

            if loss_x < loss_y:
                history["test/loss"].append(loss_x)
                history["test/accuracy/top1"].append(accuracy_x[0])
                history["test/accuracy/top3"].append(accuracy_x[1])
                history["test/accuracy/top5"].append(accuracy_x[2])

                run["test/loss"].append(loss_x)
                run["test/accuracy/top1"].append(accuracy_x[0])
                run["test/accuracy/top3"].append(accuracy_x[1])
                run["test/accuracy/top5"].append(accuracy_x[2])
            else:
                history["test/loss"].append(loss_y)
                history["test/accuracy/top1"].append(accuracy_y[0])
                history["test/accuracy/top3"].append(accuracy_y[1])
                history["test/accuracy/top5"].append(accuracy_y[2])
                
                run["test/loss"].append(loss_y)
                run["test/accuracy/top1"].append(accuracy_y[0])
                run["test/accuracy/top3"].append(accuracy_y[1])
                run["test/accuracy/top5"].append(accuracy_y[2])
                
            print(f"Epoch {epoch} | Test loss: {history["test/loss"][-1]} | Test accuracy: {history["test/accuracy/top1"][-1]}")
                
        train_loss = 0.0
        model_x.train()
        model_y.train()
        for i, (batch_data, batch_target) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            outputs_x = model_x(batch_data)
            outputs_y = model_y(batch_data)
            
            loss_x = criterion(outputs_x, batch_target)
            loss_y = criterion(outputs_y, batch_target)

            if loss_x > loss_y:
                optimizer_x.zero_grad()
                loss_x.backward()
                optimizer_x.step(loss_diff=(loss_x - loss_y).item())
                history["lr"].append(optimizer_x.step_size.item())
                run["lr"].append(optimizer_x.step_size.item())
                train_loss += loss_y.item() * batch_data.size(0)
            else:
                optimizer_y.zero_grad()
                loss_y.backward()
                optimizer_y.step(loss_diff=(loss_y - loss_x).item())
                history["lr"].append(optimizer_y.step_size.item())
                run["lr"].append(optimizer_y.step_size.item())
                train_loss += loss_x.item() * batch_data.size(0)
        
        train_loss = train_loss / len(train_dataloader.sampler)
        history["train/loss"].append(train_loss)
        run["train/loss"].append(train_loss)
        

    run.stop()
    
    return history




def main(
    model_name: str,
    optimizer_name: str,
    lr: float,
    n_epochs: int,
    save: bool = True,
    seed: int = 0,
    neptune_mode: str = "async",
    eta_max: float = 1.0,
    c_0: float = 1.0,
) -> None:
    
    ## MNIST
    TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

    transforms = v2.Compose([
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                (0.1307,), (0.3081,),
            ),
        ])

    train_batch_size = 512
    test_batch_size = 2048

    train_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms)
    test_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms)

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    dataset = (train_dataloader, test_dataloader)
    ###

    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
        
    for seed in seeds:
        if optimizer_name == "SGD":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_mode=neptune_mode,
                lr=lr,
            )
        elif optimizer_name == "SPSMAX":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_mode=neptune_mode,
                eta_max=eta_max,
            )
        elif optimizer_name == "SLS":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_mode=neptune_mode,
            )
        elif optimizer_name == "DecSPS":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_mode=neptune_mode,
                eta_max=eta_max,
                c_0=c_0,
            )
        elif optimizer_name in ["STP"]:
            history = train_twin_polyak(
                model_name=model_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_mode=neptune_mode,
            )
        
        
        if save:
            results = {"args": vars(args), **history}

            ## REMOVE ONCE METHODS WITH MOMENTUM ARE INCLUDED
            beta = 0.0
            
            if optimizer_name == "SPSMAX":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}".replace(".", "_")
            elif optimizer_name == "DecSPS":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}_{c_0}".replace(".", "_")
            elif optimizer_name == "STP-MA":
                optimizer_name_formatted = f"{optimizer_name}_{beta}".replace(".", "_")
            elif optimizer_name == "SPS-MA":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}_{beta}".replace(".", "_")
            elif optimizer_name == "SGD-MoMo":
                optimizer_name_formatted = f"{optimizer_name}_{beta}".replace(".", "_")
            else:
                optimizer_name_formatted = optimizer_name
                
            utils.save_results_dnn(
                results=history,
                model_name=model_name,
                dataset_name="MNIST",
                batch_size=train_batch_size,
                n_epochs=n_epochs,
                optimizer=optimizer_name_formatted,
                lr=lr,
                seed=seed,
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--model", type=str, choices=["LeNet5"])
    parser.add_argument("--optimizer", type=str, choices=["SGD", "SPSMAX", "SLS", "DecSPS", "STP"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--c_0", type=float, default=1.0)
    
    # parser.add_argument("--eps", type=float, default=1.0)
    # parser.add_argument("--train_batch_size", type=int, default=64)
    # parser.add_argument("--test_batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--neptune_mode", type=str, default="async", choices=["async", "debug", "offline", "read-only", "sync"])
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)
    
    start_time = time.time()

    main(model_name=args.model, 
         optimizer_name=args.optimizer, 
         lr=args.lr, 
         eta_max=args.eta_max,
         c_0=args.c_0,
         n_epochs=args.n_epochs, 
        #  train_batch_size=args.train_batch_size,
        #  test_batch_size=args.test_batch_size,
         save=args.save, 
         seed=args.seed,
         neptune_mode=args.neptune_mode)
        
    c = round(time.time() - start_time, 2)
    print(f"Run complete in {str(datetime.timedelta(seconds=c))} hrs:min:sec.")



