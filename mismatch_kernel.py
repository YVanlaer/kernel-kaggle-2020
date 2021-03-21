"""Run kernel SVM using SpectrumKernel."""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from time import time
import joblib

from dataset import Dataset
from kernels import MismatchKernel
from estimators import KernelSVMEstimator, KernelRREstimator
from predict import predict


# Validate
def validate(args):
    for k in [0, 1, 2]:
        ds = Dataset(k=k)
        if k == 0:
            kernel = MismatchKernel(k=11, m=1)
        elif k == 1:
            kernel = MismatchKernel(k=10, m=1)
        else:
            kernel = MismatchKernel(k=11, m=1)
        if args.estimator == 'ksvm':
            est = KernelSVMEstimator(lbd=1e-6, kernel=kernel)
        elif args.estimator == 'krr':
            est = KernelRREstimator(lbd=1e-6, kernel=kernel)

        X_train, X_val, y_train, y_val = train_test_split(ds.X, ds.y,
                                                        test_size=0.2,
                                                        random_state=0)

        est.fit(X_train, y_train)
        y_pred = est.predict(X_val)

        acc = (y_pred == y_val).sum() / len(y_val)
        print(f'Accuracy dataset {k} is: {acc:.3g}')


# Grid search
def grid_search(args):
    ds = Dataset(k=2)

    if args.estimator == 'ksvm':
        estimator = KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=3, m=1))
    elif args.estimator == 'krr':
        estimator = KernelRRstimator(lbd=1e-6, kernel=MismatchKernel(k=3, m=1))
    kernels = []

    for k in range(4, 8):  #[5, 6, 7, 8]:
        for m in range(1, 4):
            kernels.append(MismatchKernel(k=k, m=m))

    param_grid = {
        'lbd': [1e-3],  #np.logspace(-6, -3, 2),
        'kernel': kernels,
    }
    # cv = ShuffleSplit(n_splits=5, random_state=0, test_size=0.2)
    cv = KFold(n_splits=5)

    with joblib.parallel_backend(backend='loky'):
        gscv = GridSearchCV(estimator, param_grid=param_grid, n_jobs=4, cv=cv,
                            refit=True, verbose=10, scoring='accuracy')

        gscv.fit(ds.X, ds.y)

    print(gscv.cv_results_)
    print(gscv.best_params_)
    print(gscv.best_score_)


# Predict
def get_predictions(args):
    lambdas = [1e-3, 1e-3, 1e-3]
    kernels = [
        MismatchKernel(k=11, m=0),
        MismatchKernel(k=7, m=1),
        MismatchKernel(k=10, m=0),
    ]
    predict(lambdas, kernels, args.estimator)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run computations on Mismatch kernel")
    parser.add_argument("--mode", type=int, help="Mode to use: 0 for validation, 1 for grid search, 2 for prediction")
    parser.add_argument("--estimator", type=str, help="Estimator to use. Available estimators are: ksvm, krr")
    args = parser.parse_args()

    assert args.estimator in ["ksvm", "krr"]

    if args.mode == 0:
        validate(args)
    elif args.mode == 1:
        grid_search(args)
    elif args.mode == 2:
        get_predictions(args)
    else:
        raise Exception("Wrong mode.")
