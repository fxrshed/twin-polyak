import os
import argparse
import time 
import datetime

import neptune

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import v2

import utils
from pt_twin_polyak import train_twin_polyak
from pt_run_optimizer import train_optimizer

from dotenv import load_dotenv
load_dotenv()

torch.set_num_threads(2) # COMMENT OUT IF CPU IS NOT LIMITED
os.environ["OMP_NUM_THREADS"] = "1" # !!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(
    model_name: str,
    optimizer_name: str,
    lr: float,
    n_epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    save: bool = True,
    seed: int = 0,
    neptune_mode: str = "async",
    eta_max: float = 1.0,
    c_0: float = 1.0,
) -> None:
    
    ## DATASET
    DATASET_NAME = "CIFAR10"
    TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(32, 32), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    test_transforms = v2.Compose([
        v2.Resize(size=(32, 32)),
        v2.CenterCrop((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=train_transforms
        )
    test_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=test_transforms
        )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    dataset = (train_dataloader, test_dataloader)
    ###

    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
        
    for seed in seeds:
        neptune_run = neptune.init_run(
        mode=neptune_mode,
        tags=["multi-class-classification"]
        )
        
        neptune_run["dataset"] = {
            "name": DATASET_NAME,
            "train_batch_size": train_dataloader.batch_size,
            "test_batch_size": test_dataloader.batch_size
        }
        
        neptune_run["n_epochs"] = n_epochs
        neptune_run["seed"] = seed
        neptune_run["optimizer/parameters/name"] = optimizer_name
        neptune_run["model"] = model_name
        neptune_run["device"] = str(device)

        if optimizer_name == "SGD":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
                lr=lr,
            )
        elif optimizer_name == "SPSMAX":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
                eta_max=eta_max,
            )
        elif optimizer_name == "SLS":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
            )
        elif optimizer_name == "DecSPS":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
                eta_max=eta_max,
                c_0=c_0,
            )
        elif optimizer_name in ["STP"]:
            history = train_twin_polyak(
                model_name=model_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
            )
        
        neptune_run.stop() 
        
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
                results=results,
                model_name=model_name,
                dataset_name=DATASET_NAME,
                batch_size=train_batch_size,
                n_epochs=n_epochs,
                optimizer=optimizer_name_formatted,
                lr=lr,
                seed=seed,
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--model", type=str, choices=["LeNet5", "WideResNet16-8"])
    parser.add_argument("--optimizer", type=str, choices=["SGD", "SPSMAX", "SLS", "DecSPS", "STP"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--c_0", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=2048)
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
         train_batch_size=args.train_batch_size,
         test_batch_size=args.test_batch_size,
         save=args.save, 
         seed=args.seed,
         neptune_mode=args.neptune_mode)
        
    c = round(time.time() - start_time, 2)
    print(f"Run complete in {str(datetime.timedelta(seconds=c))} hrs:min:sec.")



