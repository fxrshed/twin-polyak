import os
from collections import defaultdict
import pickle
import argparse

import numpy as np
import scipy
import sklearn
import sklearn.datasets
import sklearn.model_selection

import utils
from loss_functions import LogisticRegressionLoss
from methods import *

from dotenv import load_dotenv
load_dotenv()


def train_loop(dataset: list[np.ndarray], 
               batch_size: int, 
               n_epochs: int,
               optimizer: BaseOptimizer,
               seed: int = 0,
               **optimizer_kwargs) -> dict: 
    
    train_data, train_target, test_data, test_target = dataset
    
    # parameters
    if batch_size == train_data.shape[0]:
        np.random.seed(seed)
    else:
        np.random.seed(0)
    params = np.random.randn(train_data.shape[1])
    
    optim = optimizer(params, **optimizer_kwargs)

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=0.0)
    
    # logging 
    history = defaultdict(list)

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
    
    np.random.seed(seed)
    indices = np.arange(train_data.shape[0])

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
        
    
    history["optimizer"].append(optim)
    return history


def main(seed: int, dataset_name: str, test_split: float, 
         batch_size: int, n_epochs: int, optimizer_name: str, 
         lr: float, eta_max: float, c_0: float, beta: float, eps: float, save: bool):
    
    if dataset_name in ["synthetic-interpolation", "synthetic-no-interpolation"]:
        np.random.seed(0)
        n = 500
        d = 100
        
        if dataset_name == "synthetic-no-interpolation":
            n_clusters_per_class = 2
            class_sep = 0.1
        else:
            n_clusters_per_class = 1
            class_sep = 3.0
            
        data, target = sklearn.datasets.make_classification(n_samples=n, n_features=d, n_redundant=0, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep, random_state=0)
        if test_split > 0.0:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=test_split, random_state=0)
        else:
            train_data, train_target = data, target
    else:
        train_data, train_target, test_data, test_target = utils.get_libsvm(name=dataset_name, test_split=test_split)
    
    if test_split == 0.0:
        test_data, test_target = train_data, train_target
        
    train_target = utils.map_classes_to(train_target, [-1.0, 1.0])
    test_target = utils.map_classes_to(test_target, [-1.0, 1.0])
    
    train_data, train_target = sklearn.preprocessing.normalize(train_data, norm='l2', axis=1), train_target
    test_data, test_target = sklearn.preprocessing.normalize(test_data, norm='l2', axis=1), test_target

    dataset = train_data, train_target, test_data, test_target
    
    if batch_size == 0:
        bs = train_data.shape[0]
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
            
        elif optimizer_name == "DecSPS":
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs, 
                optimizer=DecSPS,
                seed=seed,
                eps=eps,
                c_0=c_0,
                eta_max=eta_max,
            )
            
        elif optimizer_name == "SLS":
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                optimizer=SLS,
                seed=seed
            )
        elif optimizer_name == "STP-MA":
            hist = twin_polyak_ma(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                seed=seed,
                beta=beta,
                eps=eps
            )
        elif optimizer_name == "SPS-MA":
            hist = train_loop(
                dataset=dataset,
                batch_size=bs,
                n_epochs=n_epochs,
                optimizer=SPS_MA,
                seed=seed,
                betas=(beta, beta),
                eta_max=eta_max,
            )
        elif optimizer_name == "SGD-MoMo":
            hist = train_loop(
                dataset=dataset,
                batch_size=batch_size,
                n_epochs=n_epochs,
                optimizer=SGD_Momo,
                seed=seed,
                beta=beta,
                lr=lr,
                eps=eps
                )

        if save:
            results = {"args": vars(args), **hist}
            
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
    
    optimizers = ["STP", "SPSMAX", "SGD", "SGD-L", "DecSPS", "SLS", "STP-MA", "SPS-MA", "SGD-MoMo"]

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from LibSVM datasets directory.")
    parser.add_argument("--test-split", type=float, default=0.0, help="train-test split ratio.")
    parser.add_argument("--batch-size", type=int, default=0, help="Default 0 will run full batch.")
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--optimizer", type=str, choices=optimizers)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--c0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="Momentum parameter. Not applied if the method does not have momentum.")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args = parser.parse_args()

    main(seed=args.seed, 
        dataset_name=args.dataset, 
        test_split=args.test_split,
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        optimizer_name=args.optimizer, 
        lr=args.lr, 
        eta_max=args.eta_max,
        c_0=args.c0,
        beta=args.beta,
        eps=args.eps, 
        save=args.save,
        )