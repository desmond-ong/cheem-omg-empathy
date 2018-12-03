""""Decision-level fusion through support vector regression (SVR)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score

import datasets

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def load_data(train_dir, test_dir, in_dirs, in_names=None):
    print("Loading data...")
    if in_names is None:
        in_names = [d.lower() for d in in_dirs]
    train_data = datasets.OMGFusion(
        in_names, [os.path.join(train_dir, d) for d in in_dirs],
        os.path.join(train_dir,"Annotations"))
    test_data = datasets.OMGFusion(
        in_names, [os.path.join(test_dir, d) for d in in_dirs],
        os.path.join(test_dir,"Annotations"))
    all_data = train_data.join(test_data)    
    print("Done.")
    return train_data, test_data, all_data

def main(train_data, test_data, args):
    # Concatenate sequences into train and test matrices
    X_train, y_train = zip(*train_data)
    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_test, y_test = zip(*test_data)
    X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    # Set up hyper-parameters for support vector regression
    params = {
        'gamma': ['auto'],
        'C': [0.1, 1.0, 10],
        'kernel':['rbf']
    }
    params = list(ParameterGrid(params))

    # Cross validate across hyper-parameters
    best_ccc = -1
    for p in params:
        print("Using parameters:", p)

        # Train SVR on training set
        print("Fitting SVR model...")
        model = svm.SVR(kernel=p['kernel'], C=p['C'], gamma=p['gamma'],
                        cache_size=1000)
        model.fit(X_train, y_train)

        # Predict and evaluate on test set
        print("Evaluating...")
        y_pred = model.predict(X_test)
        ccc = eval_ccc(y_test, y_pred)
        print('CCC:', ccc)

        # Save best parameters and model
        if ccc > best_ccc:
            best_ccc = ccc
            best_params = p 
            best_model = model

    print('Best CCC:', best_ccc)
    print('Best parameters:', best_params)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dirs', type=str, nargs='+', metavar='DIR',
                        help='paths to input features')
    parser.add_argument('--in_names', type=str, nargs='+', metavar='NAME',
                        help='names for input features')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: True)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--model_dir', type=str, default="./models",
                        help='path to save models')
    parser.add_argument('--pred_dir', type=str, default="./predictions",
                        help='path to save predictions')
    args = parser.parse_args()

    # Load data
    train_data, test_data, all_data =\
        load_data(args.train_dir, args.test_dir, args.in_dirs, args.in_names)

    # Normalize inputs
    if args.normalize:
        all_data.normalize()
    
    # Continue to rest of script
    main(train_data, test_data, args)
