"""Run kernel SVM using SpectrumKernel."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from time import time
import joblib

from dataset import Dataset
from kernels import MismatchKernel
from estimators import KernelSVMEstimator


# Validate
if __name__ == '__main__':
    for k in [0, 1, 2]:
        ds = Dataset(k=k)
        est = KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=3, m=1))

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

#     estimator = KernelSVMEstimator(lbd=1e-6, kernel=SpectrumKernel(k=3))
#     kernels = []

#     for k in range(5, 13):  #[5, 6, 7, 8]:
#         kernels.append(SpectrumKernel(k=k))

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
#     print('Pred 1')
#     ds = Dataset(k=0)
#     est1 = KernelSVMEstimator(lbd=0.001, kernel=SpectrumKernel(k=11))
#     est1.fit(ds.X, ds.y)
#     y_pred1 = est1.predict(ds.X_test)
#     y_pred1 = pd.Series(y_pred1, index=ds.X_test.index, name='Bound')

#     print('Pred 2')
#     ds = Dataset(k=1)
#     est2 = KernelSVMEstimator(lbd=0.001, kernel=SpectrumKernel(k=6))
#     est2.fit(ds.X, ds.y)
#     y_pred2 = est2.predict(ds.X_test)
#     y_pred2 = pd.Series(y_pred2, index=ds.X_test.index, name='Bound')

#     print('Pred 3')
#     ds = Dataset(k=2)
#     est3 = KernelSVMEstimator(lbd=0.001, kernel=SpectrumKernel(k=10))
#     est3.fit(ds.X, ds.y)
#     y_pred3 = est3.predict(ds.X_test)
#     y_pred3 = pd.Series(y_pred3, index=ds.X_test.index, name='Bound')

#     y_pred = pd.concat([y_pred1, y_pred2, y_pred3], axis=0, verify_integrity=True)
#     y_pred = y_pred.astype(int)
#     y_pred.to_csv('y_pred.csv')
