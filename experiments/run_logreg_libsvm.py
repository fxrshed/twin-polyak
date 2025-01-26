import os
from collections import defaultdict
import pickle
import argparse


import numpy as np
import scipy

import utils
from loss_functions import LogisticRegressionLoss
from methods import *

from dotenv import load_dotenv
load_dotenv()

optimizer_dict = {
    "SPSMAX": SPS,
    "SGD": SGD,
}


def train_loop(dataset: list[np.ndarray], 
               batch_size: int, 
               n_epochs: int,
               optimizer: BaseOptimizer,
               seed: int = 0,
               **optimizer_kwargs) -> dict: 

    np.random.seed(seed)

    train_data, train_target, test_data, test_target = dataset
    
    # params = np.zeros(train_data.shape[1])
    params = np.random.randn(train_data.shape[1])
    
    optim = optimizer(params, **optimizer_kwargs)

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=0.0)
    
    # logging 
    history = defaultdict(list)

    indices = np.arange(train_data.shape[0])
    
    # Train Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params, train_data, train_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["train/loss"].append(loss)
    history["train/acc"].append(acc)
    history["train/grad_norm_sq"].append(g_norm_sq)

    # Test Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params, test_data, test_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["test/loss"].append(loss)
    history["test/acc"].append(acc)
    history["test/grad_norm_sq"].append(g_norm_sq)
    
    for epoch in range(n_epochs):

        # Training 
        if batch_size != train_data.shape[0]: # If full batch then don't shuffle indices
            np.random.shuffle(indices)

        for idx in range(train_data.shape[0]//batch_size):

            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
        
            train_loss, train_grad, train_acc = loss_function.func_grad_acc(params, batch_data, batch_target)
            
            if optim.__class__.__name__ == "SLS":
                def closure(params):
                    return loss_function.func(params, batch_data, batch_target)
                optim.step(loss=train_loss, grad=train_grad, closure=closure)
            else:
                optim.step(loss=train_loss, grad=train_grad)
            
            g_norm_sq = np.linalg.norm(train_grad)**2
            history["train/batch/loss"].append(train_loss)
            history["train/batch/acc"].append(train_acc)
            history["train/batch/grad_norm_sq"].append(g_norm_sq)  

            history["lr"].append(optim.lr)


        # Train Evaluation 
        loss, grad, acc = loss_function.func_grad_acc(params, train_data, train_target)
        g_norm_sq = np.linalg.norm(grad)**2
        history["train/loss"].append(loss)
        history["train/acc"].append(acc)
        history["train/grad_norm_sq"].append(g_norm_sq)

        # Test Evaluation 
        loss, grad, acc = loss_function.func_grad_acc(params, test_data, test_target)
        g_norm_sq = np.linalg.norm(grad)**2
        history["test/loss"].append(loss)
        history["test/acc"].append(acc)
        history["test/grad_norm_sq"].append(g_norm_sq)
        
    return history


def twin_polyak(dataset: list[np.ndarray], 
               batch_size: int, 
               n_epochs: int,
               eps: float = 0.0,
               seed: int = 0,
               ) -> dict: 
    
    np.random.seed(seed)

    train_data, train_target, test_data, test_target = dataset

    # parameters
    params_x = np.random.randn(train_data.shape[1])
    params_y = np.random.randn(train_data.shape[1])

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=0.0)
    
    # logging 
    history = defaultdict(list)

    indices = np.arange(train_data.shape[0])
    
    # Train Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, train_data, train_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["train/loss"].append(loss)
    history["train/acc"].append(acc)
    history["train/grad_norm_sq"].append(g_norm_sq)
    
    # Test Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, test_data, test_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["test/loss"].append(loss)
    history["test/acc"].append(acc)
    history["test/grad_norm_sq"].append(g_norm_sq)
    
    for epoch in range(n_epochs):
    
        # Training 
        if batch_size != train_data.shape[0]:
            np.random.shuffle(indices)

        for idx in range(train_data.shape[0]//batch_size):
            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
            
            loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, batch_data, batch_target)
            loss_y, grad_y, acc_y  = loss_function.func_grad_acc(params_y, batch_data, batch_target)
    
            lr_x = np.minimum(( (loss_x - loss_y) / (0.5 * np.linalg.norm(grad_x)**2 + eps) ), np.inf) 
            lr_y = np.minimum(( (loss_y - loss_x) / (0.5 * np.linalg.norm(grad_y)**2 + eps) ), np.inf) 

            # Optimization step
            if loss_x > loss_y:
                params_x -= lr_x * grad_x
                lr = lr_x
            else:
                params_y -= lr_y * grad_y
                lr = lr_y         
            
            history["lr_x"].append(np.abs(lr_x))
            history["lr_y"].append(np.abs(lr_y))
            history["lr"].append(lr)
            
            
        # Train Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, train_data, train_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, train_data, train_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["train/loss"].append(loss)
        history["train/acc"].append(acc)
        history["train/grad_norm_sq"].append(g_norm_sq)
            
        # Test Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, test_data, test_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, test_data, test_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["test/loss"].append(loss)
        history["test/acc"].append(acc)
        history["test/grad_norm_sq"].append(g_norm_sq)

    return history



def main(seed: int, dataset_name: str, test_split: float, 
         batch_size: int, n_epochs: int, optimizer_name: str, 
         lr: float, eta_max: float, eps: float, save: bool):
    
    train_data, train_target, test_data, test_target = utils.get_libsvm(name=dataset_name, test_split=test_split)
    train_target = utils.map_classes_to(train_target, [-1.0, 1.0])
    test_target = utils.map_classes_to(test_target, [-1.0, 1.0])

    if test_split == 0.0:
        test_data, test_target = train_data, train_target

    dataset = train_data, train_target, test_data, test_target
    
    if batch_size == 0:
        bs = train_data.shape[0] # deterministic setting
        setting = "deterministic"
    else:
        bs = batch_size
        setting = "stochastic"
    
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    
    for seed in seeds:
        if optimizer_name == "SGD-L":
            L = (0.25 / train_data.shape[0]) * scipy.sparse.linalg.norm(train_data.T @ train_data, ord=2)
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                optimizer=SGD,
                seed=seed,
                lr=1/L,
            )
        elif optimizer_name == "SGD":
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                optimizer=SGD,
                seed=seed,
                lr=lr,
            )
        elif optimizer_name == "SPSMAX":
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                optimizer=SPS,
                seed=seed,
                eta_max=eta_max,
                eps=eps,
            )
            
        elif optimizer_name == "STP":
            hist = twin_polyak(dataset=dataset,
                       batch_size=bs,
                       n_epochs=n_epochs,
                       eps=eps,
                       seed=seed,
                       )

        if save:
            results = {"args": vars(args), **hist}
            
            if optimizer_name == "SPSMAX":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}".replace(".", "_")
            else:
                optimizer_name_formatted = optimizer_name
                
            utils.save_results(results=results, 
                               loss="logreg", 
                               setting=setting, 
                               dataset_name=dataset_name, 
                               batch_size=batch_size, # not `bs` here because of deterministic case
                               n_epochs=n_epochs, 
                               optimizer=optimizer_name_formatted, 
                               lr=f"{lr}".replace(".", "_"), 
                               seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from LibSVM datasets directory.")
    parser.add_argument("--test_split", type=float, default=0.0, help="train-test split ratio.")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--optimizer", type=str, choices=["STP", "SPSMAX", "SGD", "SGD-L"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eta_max", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args = parser.parse_args()
    # print(args)
    
    main(seed=args.seed, 
        dataset_name=args.dataset, 
        test_split=args.test_split,
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        optimizer_name=args.optimizer, 
        lr=args.lr, 
        eta_max=args.eta_max,
        eps=args.eps, 
        save=args.save,
        )