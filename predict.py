"""Predict."""
import pandas as pd
from time import time

from dataset import Dataset
from estimators import KernelSVMEstimator


# Predict
def predict(lambdas, kernels):
    assert len(lambdas) == 3
    assert len(kernels) == 3

    start_time = time()

    print('Pred 1')
    ds = Dataset(k=0)
    est1 = KernelSVMEstimator(lbd=lambdas[0], kernel=kernels[0])
    est1.fit(ds.X, ds.y)
    y_pred1 = est1.predict(ds.X_test)
    y_pred1 = pd.Series(y_pred1, index=ds.X_test.index, name='Bound')

    print('Pred 2')
    ds = Dataset(k=1)
    est2 = KernelSVMEstimator(lbd=lambdas[1], kernel=kernels[1])
    est2.fit(ds.X, ds.y)
    y_pred2 = est2.predict(ds.X_test)
    y_pred2 = pd.Series(y_pred2, index=ds.X_test.index, name='Bound')

    print('Pred 3')
    ds = Dataset(k=2)
    est3 = KernelSVMEstimator(lbd=lambdas[2], kernel=kernels[2])
    est3.fit(ds.X, ds.y)
    y_pred3 = est3.predict(ds.X_test)
    y_pred3 = pd.Series(y_pred3, index=ds.X_test.index, name='Bound')

    y_pred = pd.concat([y_pred1, y_pred2, y_pred3], axis=0, verify_integrity=True)
    y_pred = y_pred.astype(int)
    y_pred.to_csv('y_pred.csv')

    print("Took {:.2f} seconds to compute the predictions.".format(time() - start_time))
