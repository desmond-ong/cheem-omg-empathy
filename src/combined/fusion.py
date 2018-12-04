""""Decision-level fusion through support vector regression (SVR)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import joblib
import pandas as pd
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

def load_data(train_dir, test_dir, in_dirs, in_names):
    print("Loading data...")
    train_data = datasets.OMGFusion(
        in_names, [os.path.join(train_dir, d) for d in in_dirs],
        os.path.join(train_dir,"Annotations"))
    test_data = datasets.OMGFusion(
        in_names, [os.path.join(test_dir, d) for d in in_dirs],
        os.path.join(test_dir,"Annotations"))
    all_data = train_data.join(test_data)    
    print("Done.")
    return train_data, test_data, all_data

def train(train_data, test_data):
    # Concatenate training sequences into matrix
    X_train, y_train = zip(*train_data)
    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    y_train = y_train.flatten()

    # Set up hyper-parameters for support vector regression
    params = {
        'gamma': ['auto'],
        'C': [1e-3, 0.01, 0.03, 0.1, 0.3, 1.0],
        'kernel':['rbf']
    }
    params = list(ParameterGrid(params))

    # Cross validate across hyper-parameters
    print('---')
    best_ccc = -1
    for p in params:
        print("Using parameters:", p)

        # Train SVR on training set
        print("Fitting SVR model...")
        model = svm.SVR(kernel=p['kernel'], C=p['C'], gamma=p['gamma'],
                        cache_size=1000)
        model.fit(X_train, y_train)

        # Evaluate on test set
        ccc, predictions = evaluate(model, test_data)

        # Save best parameters and model
        if ccc > best_ccc:
            best_ccc = ccc
            best_params = p 
            best_model = model
            best_pred = predictions

    # Print best parameters
    print('---')
    print('Best CCC: {:0.3f}'.format(best_ccc))
    print('Best parameters:', best_params)

    return best_ccc, best_params, best_model, best_pred

def evaluate(model, test_data):
    ccc = 0
    predictions = []
    # Predict and evaluate on each test sequence
    print("Evaluating...")
    for i, (X_test, y_test) in enumerate(test_data):
        # Get original valence annotations
        y_test = test_data.val_orig[i].flatten()
        y_pred = model.predict(X_test)
        # Repeat and pad predictions to match original data length
        y_pred = np.repeat(y_pred, test_data.time_ratio)[:len(y_test)]
        l_diff = len(y_test) - len(y_pred)
        if l_diff > 0:
            y_pred = np.concatenate([y_pred, y_pred[-l_diff:]])
        ccc += eval_ccc(y_test, y_pred)
        predictions.append(y_pred)
    ccc /= len(test_data)
    print('CCC: {:0.3f}'.format(ccc))
    return ccc, predictions

def save_predictions(pred, dataset, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for p, subj, story in zip(pred, dataset.subjects, dataset.stories):
        df = pd.DataFrame(p, columns=['valence'])
        fname = "Subject_{}_Story_{}.csv".format(subj, story)
        df.to_csv(os.path.join(path, fname), index=False)
        
def main(train_data, test_data, args):
    if args.test is None:
        # Fit new model if test model is not provided
        ccc, params, model, pred = train(train_data, test_data)
        # Save best model
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        model_fname = "best_{}.save".format(",".join(args.in_names))
        joblib.dump(model, os.path.join(args.model_dir, "best.save"))
        # Save predictions of best model
        pred_dir = os.path.join(args.pred_dir, "pred_test")
        save_predictions(pred, test_data, pred_dir)
        return ccc
    else:
        # Load and test model on training and test set
        model = joblib.load(args.test)
        print("-Training-")
        ccc, pred = evaluate(model, train_data)
        pred_dir = os.path.join(args.pred_dir, "pred_train")
        save_predictions(pred, train_data, pred_dir)
        print("-Testing-")
        ccc, pred = evaluate(model, test_data)
        pred_dir = os.path.join(args.pred_dir, "pred_test")
        save_predictions(pred, test_data, pred_dir)
        return ccc
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dirs', type=str, nargs='+', metavar='DIR',
                        help='paths to input features')
    parser.add_argument('--in_names', type=str, nargs='+', metavar='NAME',
                        help='names for input features')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: True)')
    parser.add_argument('--test', type=str, default=None,
                        help='path to model to test (default: None)')
    parser.add_argument('--test_story', type=str, default="1",
                        help='story to use as test set (optional)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--model_dir', type=str, default="./fusion_models",
                        help='path to save models')
    parser.add_argument('--pred_dir', type=str, default="./fusion_pred",
                        help='path to save predictions')
    args = parser.parse_args()
    # Construct modality names if not provided
    if args.in_names is None:
        args.in_names = [os.path.basename(d).lower() for d in args.in_dirs]
    
    # Load data
    train_data, test_data, all_data =\
        load_data(args.train_dir, args.test_dir, args.in_dirs, args.in_names)
    
    # Normalize inputs
    if args.normalize:
        all_data.normalize()

    # Make new train/test split
    test_data, train_data = all_data.extract_story([args.test_story])
        
    # Continue to rest of script
    main(train_data, test_data, args)
