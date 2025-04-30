import argparse
import os 
import pickle

import numpy as np
import pandas as pd
import scipy
import sklearn

from loss_functions import LeastSquaresLoss
import utils

def main(dataset_name: str, 
         test_split: float,
         normalize: bool,
         verbose: bool):
    
    np.random.seed(0)

    train_data, train_target, test_data, test_target = utils.get_libsvm(name=dataset_name, test_split=test_split)

    if normalize:
        normalizer = sklearn.preprocessing.Normalizer()
        train_data = normalizer.fit_transform(train_data)
        test_data = normalizer.transform(test_data)

    oracle = LeastSquaresLoss()
    train_x_star = scipy.sparse.linalg.inv(train_data.T.dot(train_data)).dot(train_data.T.dot(train_target))
    train_f_star = oracle.func(train_x_star, train_data, train_target)
    
    test_x_star = scipy.sparse.linalg.inv(test_data.T.dot(test_data)).dot(test_data.T.dot(test_target))
    test_f_star = oracle.func(test_x_star, test_data, test_target)
    
    entry = {
        "dataset_name": dataset_name,
        "train/f_star": train_f_star,
        "test/f_star":  test_f_star,
        "train/x_star": train_x_star,
        "test/x_star": test_x_star
    }
    
    if verbose:
        print(f"Train f_stat: {entry["train/f_star"]}")
        print(f"Test f_star: {entry["test/f_star"]}")
        
    directory = "solver"
    filename = f"{directory}/{dataset_name}_sol.pkl"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, "wb") as file:
        pickle.dump(entry, file, protocol=pickle.HIGHEST_PROTOCOL)    
    
    print(f"Saved to {filename}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a LibSVM binary classification dataset.")
    parser.add_argument("--test-split", type=utils.restricted_float, default=0.0, help="Train split (e.g. 0.2 = 20%).")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False, help="Select to normalize the dataset before running solver.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    main(dataset_name=args.dataset, 
         test_split=args.test_split,
         normalize=args.normalize,
         verbose=args.verbose)