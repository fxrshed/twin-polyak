from collections import defaultdict

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import neptune

from models import models_dict
import utils

import sls
import sps
from pt_methods import DecSPS, Momo

def train_optimizer(
    model_name: str,
    dataset: tuple[DataLoader, DataLoader],
    optimizer_name: str,
    n_epochs: int, 
    seed: int = 0,
    neptune_run = None,
    device: torch.device = "cpu",
    **optimizer_kwargs
                    ) -> dict:
    
    torch.manual_seed(seed)
    
    train_dataloader, test_dataloader = dataset
    try:
        num_classes = len(np.unique(train_dataloader.dataset.classes))
    except Exception as e1:
        try:
            num_classes = len(np.unique(train_dataloader.dataset.labels))
        except Exception as e2:
            raise Exception(e2)
            
    model = models_dict[model_name](num_classes=num_classes).to(device)
    
    optimizer = utils.optimizers_dict[optimizer_name](model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    history = defaultdict(list)

    neptune_run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer.defaults)

    for epoch in range(n_epochs):
        
        model.eval()
        with torch.inference_mode():
            loss, accuracy = utils.eval_model(model, criterion, test_dataloader)
            history["test/loss"].append(loss)
            history["test/accuracy/top1"].append(accuracy[0])
            history["test/accuracy/top3"].append(accuracy[1])
            history["test/accuracy/top5"].append(accuracy[2])
            
            neptune_run["test/loss"].append(loss)
            neptune_run["test/accuracy/top1"].append(accuracy[0])
            neptune_run["test/accuracy/top3"].append(accuracy[1])
            neptune_run["test/accuracy/top5"].append(accuracy[2])
            
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
                neptune_run["lr"].append(optimizer.state["step_size"])
                train_loss += loss * batch_data.size(0)
            elif isinstance(optimizer, sls.Sls):
                def closure():
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_target)
                    return loss
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.state["step_size"])
                neptune_run["lr"].append(optimizer.state["step_size"])
                train_loss += loss.item() * batch_data.size(0)
            elif isinstance(optimizer, DecSPS):
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.step_size)
                neptune_run["lr"].append(optimizer.step_size)
                train_loss += loss.item() * batch_data.size(0)
            elif isinstance(optimizer, Momo):
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.state["step_size_list"][-1])
                neptune_run["lr"].append(optimizer.state["step_size_list"][-1])
                train_loss += loss.item() * batch_data.size(0)
            else:
                loss = optimizer.step(closure=closure)
                history["lr"].append(optimizer.param_groups[0]["lr"])
                neptune_run["lr"].append(optimizer.param_groups[0]["lr"])
                train_loss += loss.item() * batch_data.size(0)
        
        history["train/loss"].append(train_loss / len(train_dataloader.sampler))
        neptune_run["train/loss"].append(train_loss / len(train_dataloader.sampler))

    
    return history