import os
import argparse
import time 
import datetime
import random

import neptune

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import v2

import utils
from pt_twin_polyak import train_twin_polyak, train_twin_polyak_ma
from pt_run_optimizer import train_optimizer

from dotenv import load_dotenv
load_dotenv()

torch.set_num_threads(2) # COMMENT OUT IF CPU IS NOT LIMITED
os.environ["OMP_NUM_THREADS"] = "1" # !!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flip_labels_to_complements(dataset, corruption_prob=0.5, num_classes=10, seed=0):
    
    complements = {
        0: 1, 1: 0,
        2: 3, 3: 2,
        4: 5, 5: 4,
        6: 7, 7: 6,
        8: 9, 9: 8
    }
    
    random.seed(seed)
    targets = dataset.targets.clone()

    for idx in range(len(targets)):
        if random.random() < corruption_prob:
            targets[idx] = complements[int(targets[idx])]
    dataset.targets = targets
    return dataset

def main(
    model_name: str,
    optimizer_name: str,
    lr: float,
    beta: float,
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
    corruption_prob = 0.5
    DATASET_NAME = f"MNIST-noisy{corruption_prob}"
    TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

    transforms = v2.Compose([
            # v2.RandomRotation(10),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                (0.1307,), (0.3081,),
            ),
        ])

    train_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms)
    test_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms)
    
    
    train_data = flip_labels_to_complements(train_data, corruption_prob=corruption_prob)

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
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
                momentum=beta,
            )
        elif optimizer_name == "SPS":
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
        elif optimizer_name == "Momo":
            history = train_optimizer(
                model_name=model_name,
                optimizer_name=optimizer_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
                lr=lr, 
                beta=beta,
            )
        elif optimizer_name  == "STP":
            history = train_twin_polyak(
                model_name=model_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
            )
        elif optimizer_name == "STPm":
            history = train_twin_polyak_ma(
                model_name=model_name,
                dataset=dataset,
                n_epochs=n_epochs,
                seed=seed,
                neptune_run=neptune_run,
                device=device,
                beta=beta,
            )
        
        neptune_run.stop() 
        
        if save:
            results = {"args": vars(args), **history}

            if optimizer_name == "SPS":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}".replace(".", "_")
            elif optimizer_name == "DecSPS":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}_{c_0}".replace(".", "_")
            elif optimizer_name == "STPm":
                optimizer_name_formatted = f"{optimizer_name}_{beta}".replace(".", "_")
            elif optimizer_name == "SPSMA":
                optimizer_name_formatted = f"{optimizer_name}_{eta_max}_{beta}".replace(".", "_")
            elif optimizer_name == "Momo":
                optimizer_name_formatted = f"{optimizer_name}_{lr}_{beta}".replace(".", "_")
            elif optimizer_name == "SGD":
                optimizer_name_formatted = f"{optimizer_name}_{lr}_{beta}".replace(".", "_")
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
    parser.add_argument("--model", type=str, default="LeNet5", choices=["LeNet5", "SimpleCNN", "SimpleMLP"])
    parser.add_argument("--optimizer", type=str, choices=["SGD", "SPS", "SPSMA", "SLS", "DecSPS", "STP", "STPm", "Momo"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
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
         beta=args.beta,
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



