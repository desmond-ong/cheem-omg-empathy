""""Decision-level fusion through support vector regression (SVR)."""

import numpy as np
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def main(args):
    params = {
        'gamma': ['auto'],
        'C': [0.01,0.05,0.1,0.5,1,5,10,50,100,200],
        'kernel':['rbf']
    }

    params = list(ParameterGrid(params))

    best_ccc = -1
    for p in params:
        print("Using parameters:", p)

        # Train SVR on training set
        model = svm.SVR(kernel=p['kernel'], C=p['C'], gamma=p['gamma'])
        model.fit(X_train, y_train)

        # Predict and evaluate on test set
        y_pred = clf.predict(X_test)
        ccc = eval_ccc(y_test, y_pred)
        print('CCC:', ccc)

        if ccc > best_ccc:
            best_ccc = ccc
            best_params = p 
            best_model = model

    print('Best CCC:', best_ccc)
    print('Best parameters:', best_params)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, nargs='+',
                        help='input feature directories')
    parser.add_argument('--out', type=str, metavar='DIR', default="./predict",
                        help='output directory (default: ./predict)')
    args = parser.parse_args()
    main(args)

