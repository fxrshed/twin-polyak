import argparse
import os 
import pickle

import numpy as np
import pandas as pd
import scipy
import sklearn

from npkit.loss import LogisticRegressionLoss

import utils

def solve_binary_libsvm(train_data, train_target, test_data, test_target):
    w = np.random.randn(train_data.shape[1])

    oracle = LogisticRegressionLoss()

    train_result = scipy.optimize.minimize(
        fun=oracle.loss,
        jac=oracle.grad,
        x0=w,
        args=(train_data, train_target),
        method="L-BFGS-B"
    )

    test_result = scipy.optimize.minimize(
        fun=oracle.loss,
        jac=oracle.grad,
        x0=w,
        args=(test_data, test_target),
        method="L-BFGS-B"
    )
    
    entry = {
        "train/f_star": oracle.func(train_result.x, train_data, train_target),
        "test/f_star": oracle.func(test_result.x, test_data, test_target),
        "train/x_star": train_result.x,
        "test/x_star": test_result.x
    }
    
    return entry

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

    train_target = utils.map_classes_to(train_target, [-1.0, 1.0])
    test_target = utils.map_classes_to(test_target, [-1.0, 1.0])

    entry = solve_binary_libsvm(train_data, train_target, test_data, test_target)
    
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
    
    return entry
    
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