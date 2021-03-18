"""Run kernel SVM using SpectrumKernel."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from time import time
import joblib

from dataset import Dataset
from kernels import MismatchKernel
from estimators import KernelSVMEstimator
from predict import predict

# Validate
if __name__ == '__main__':
    for k in [0, 1, 2]:
        ds = Dataset(k=k)
        est = KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=3, m=2))

        X_train, X_val, y_train, y_val = train_test_split(ds.X, ds.y,
                                                        test_size=0.2,
                                                        random_state=0)

        est.fit(X_train, y_train)
        y_pred = est.predict(X_val)

        acc = (y_pred == y_val).sum() / len(y_val)
        print(f'Accuracy dataset {k} is: {acc:.3g}')


# Grid search
# if __name__ == '__main__':
#     ds = Dataset(k=2)

#     estimator = KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=3, m=1))
#     kernels = []

#     for k in range(4, 8):  #[5, 6, 7, 8]:
#         for m in range(1, 4):
#             kernels.append(MismatchKernel(k=k, m=m))

#     param_grid = {
#         'lbd': [1e-3],  #np.logspace(-6, -3, 2),
#         'kernel': kernels,
#     }
#     # cv = ShuffleSplit(n_splits=5, random_state=0, test_size=0.2)
#     cv = KFold(n_splits=5)

#     with joblib.parallel_backend(backend='loky'):
#         gscv = GridSearchCV(estimator, param_grid=param_grid, n_jobs=4, cv=cv,
#                             refit=True, verbose=10, scoring='accuracy')

#         gscv.fit(ds.X, ds.y)

#     print(gscv.cv_results_)
#     print(gscv.best_params_)
#     print(gscv.best_score_)


# Predict
# if __name__ == '__main__':
#     lambdas = [1e-3, 1e-3, 1e-3]
#     kernels = [
#         MismatchKernel(k=11, m=0),
#         MismatchKernel(k=6, m=0),
#         MismatchKernel(k=10, m=0),
#     ]
#     predict(lambdas, kernels)
