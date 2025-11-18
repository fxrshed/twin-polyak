from collections import defaultdict

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import neptune

import utils
from models import models_dict
from pt_methods import TwinPolyak, TwinPolyakMA


def train_twin_polyak(
    model_name: str,
    dataset: tuple[DataLoader, DataLoader],
    n_epochs: int,
    seed: int = 0,
    neptune_run = None,
    device: torch.device = 'cpu',
    **optimizer_kwargs) -> dict:
    
    torch.manual_seed(seed)
    
    train_dataloader, test_dataloader = dataset
    
    try:
        num_classes = len(np.unique(train_dataloader.dataset.classes))
    except Exception as e1:
        try:
            num_classes = len(np.unique(train_dataloader.dataset.labels))
        except Exception as e2:
            raise Exception(e2)
    
    model_x = models_dict[model_name](num_classes=num_classes).to(device)
    model_y = models_dict[model_name](num_classes=num_classes).to(device)

    optimizer_x = TwinPolyak(model_x.parameters(), **optimizer_kwargs)
    optimizer_y = TwinPolyak(model_y.parameters(), **optimizer_kwargs)
    
    criterion = torch.nn.CrossEntropyLoss()

    history = defaultdict(list)
    
    neptune_run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer_x.defaults)

    for epoch in range(n_epochs):
        
        model_x.eval()
        model_y.eval()
        with torch.inference_mode():
            loss_x, accuracy_x = utils.eval_model(model_x, criterion, test_dataloader)
            loss_y, accuracy_y = utils.eval_model(model_y, criterion, test_dataloader)

            history["test/loss_x"].append(loss_x)
            history["test/accuracy_x/top1"].append(accuracy_x[0])
            history["test/accuracy_x/top3"].append(accuracy_x[1])
            history["test/accuracy_x/top5"].append(accuracy_x[2])

            neptune_run["test/loss_x"].append(loss_x)
            neptune_run["test/accuracy_x/top1"].append(accuracy_x[0])
            neptune_run["test/accuracy_x/top3"].append(accuracy_x[1])
            neptune_run["test/accuracy_x/top5"].append(accuracy_x[2])
            
            history["test/loss_y"].append(loss_y)
            history["test/accuracy_y/top1"].append(accuracy_y[0])
            history["test/accuracy_y/top3"].append(accuracy_y[1])
            history["test/accuracy_y/top5"].append(accuracy_y[2])
            
            neptune_run["test/loss_y"].append(loss_y)
            neptune_run["test/accuracy_y/top1"].append(accuracy_y[0])
            neptune_run["test/accuracy_y/top3"].append(accuracy_y[1])
            neptune_run["test/accuracy_y/top5"].append(accuracy_y[2])

            if loss_x < loss_y:
                history["test/loss"].append(loss_x)
                history["test/accuracy/top1"].append(accuracy_x[0])
                history["test/accuracy/top3"].append(accuracy_x[1])
                history["test/accuracy/top5"].append(accuracy_x[2])

                neptune_run["test/loss"].append(loss_x)
                neptune_run["test/accuracy/top1"].append(accuracy_x[0])
                neptune_run["test/accuracy/top3"].append(accuracy_x[1])
                neptune_run["test/accuracy/top5"].append(accuracy_x[2])
            else:
                history["test/loss"].append(loss_y)
                history["test/accuracy/top1"].append(accuracy_y[0])
                history["test/accuracy/top3"].append(accuracy_y[1])
                history["test/accuracy/top5"].append(accuracy_y[2])
                
                neptune_run["test/loss"].append(loss_y)
                neptune_run["test/accuracy/top1"].append(accuracy_y[0])
                neptune_run["test/accuracy/top3"].append(accuracy_y[1])
                neptune_run["test/accuracy/top5"].append(accuracy_y[2])
                
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
                neptune_run["lr"].append(optimizer_x.step_size.item())
                train_loss += loss_y.item() * batch_data.size(0)
            else:
                optimizer_y.zero_grad()
                loss_y.backward()
                optimizer_y.step(loss_diff=(loss_y - loss_x).item())
                history["lr"].append(optimizer_y.step_size.item())
                neptune_run["lr"].append(optimizer_y.step_size.item())
                train_loss += loss_x.item() * batch_data.size(0)
        
        train_loss = train_loss / len(train_dataloader.sampler)
        history["train/loss"].append(train_loss)
        neptune_run["train/loss"].append(train_loss)
        
    return history


def train_twin_polyak_ma(
    model_name: str,
    dataset: tuple[DataLoader, DataLoader],
    n_epochs: int,
    seed: int = 0,
    neptune_run = None,
    device: torch.device = 'cpu',
    **optimizer_kwargs) -> dict:
    
    torch.manual_seed(seed)
    
    train_dataloader, test_dataloader = dataset

    try:
        num_classes = len(np.unique(train_dataloader.dataset.classes))
    except Exception as e1:
        try:
            num_classes = len(np.unique(train_dataloader.dataset.labels))
        except Exception as e2:
            raise Exception(e2)
    
    model_x = models_dict[model_name](num_classes=num_classes).to(device)
    model_y = models_dict[model_name](num_classes=num_classes).to(device)

    optimizer_x = TwinPolyakMA(model_x.parameters(), **optimizer_kwargs)
    optimizer_y = TwinPolyakMA(model_y.parameters(), **optimizer_kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = defaultdict(list)
    neptune_run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer_x.defaults)
    
    for epoch in range(n_epochs):
        
        model_x.eval()
        model_y.eval()
        with torch.inference_mode():
            loss_x, accuracy_x = utils.eval_model(model_x, criterion, test_dataloader)
            loss_y, accuracy_y = utils.eval_model(model_y, criterion, test_dataloader)
            
            history["test/loss_x"].append(loss_x)
            history["test/accuracy_x/top1"].append(accuracy_x[0])
            history["test/accuracy_x/top3"].append(accuracy_x[1])
            history["test/accuracy_x/top5"].append(accuracy_x[2])

            neptune_run["test/loss_x"].append(loss_x)
            neptune_run["test/accuracy_x/top1"].append(accuracy_x[0])
            neptune_run["test/accuracy_x/top3"].append(accuracy_x[1])
            neptune_run["test/accuracy_x/top5"].append(accuracy_x[2])
            
            history["test/loss_y"].append(loss_y)
            history["test/accuracy_y/top1"].append(accuracy_y[0])
            history["test/accuracy_y/top3"].append(accuracy_y[1])
            history["test/accuracy_y/top5"].append(accuracy_y[2])
            
            neptune_run["test/loss_y"].append(loss_y)
            neptune_run["test/accuracy_y/top1"].append(accuracy_y[0])
            neptune_run["test/accuracy_y/top3"].append(accuracy_y[1])
            neptune_run["test/accuracy_y/top5"].append(accuracy_y[2])

            if loss_x < loss_y:
                history["test/loss"].append(loss_x)
                history["test/accuracy/top1"].append(accuracy_x[0])
                history["test/accuracy/top3"].append(accuracy_x[1])
                history["test/accuracy/top5"].append(accuracy_x[2])

                neptune_run["test/loss"].append(loss_x)
                neptune_run["test/accuracy/top1"].append(accuracy_x[0])
                neptune_run["test/accuracy/top3"].append(accuracy_x[1])
                neptune_run["test/accuracy/top5"].append(accuracy_x[2])
            else:
                history["test/loss"].append(loss_y)
                history["test/accuracy/top1"].append(accuracy_y[0])
                history["test/accuracy/top3"].append(accuracy_y[1])
                history["test/accuracy/top5"].append(accuracy_y[2])
                
                neptune_run["test/loss"].append(loss_y)
                neptune_run["test/accuracy/top1"].append(accuracy_y[0])
                neptune_run["test/accuracy/top3"].append(accuracy_y[1])
                neptune_run["test/accuracy/top5"].append(accuracy_y[2])
                
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

            optimizer_x.zero_grad()
            optimizer_y.zero_grad()

            # Important note: loss.backward() is called in momentim_step
            # so we do not need to call it again 
            h_x = optimizer_x.momentum_step(loss_x)
            h_y = optimizer_y.momentum_step(loss_y)

            if h_x > h_y:
                # optimizer_x.zero_grad()
                # loss_x.backward()
                loss_diff = (h_x - h_y)
                optimizer_x.step(loss_diff=loss_diff.item())
                history["lr"].append(optimizer_x.step_size.item())
                neptune_run["lr"].append(optimizer_x.step_size.item())
                train_loss += loss_y.item() * batch_data.size(0)
            else:
                # optimizer_y.zero_grad()
                # loss_y.backward()
                loss_diff = (h_y - h_x)
                optimizer_y.step(loss_diff=loss_diff.item())
                history["lr"].append(optimizer_y.step_size.item())
                neptune_run["lr"].append(optimizer_y.step_size.item())
                train_loss += loss_x.item() * batch_data.size(0)
        
        train_loss = train_loss / len(train_dataloader.sampler)
        history["train/loss"].append(train_loss)
        neptune_run["train/loss"].append(train_loss)

    return history