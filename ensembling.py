"""Predict with ensembling method (basic voting)."""
import pandas as pd
from time import time

from dataset import Dataset


def vote_predictions(predictions_list):
    all_predictions = np.array(predictions_list)    # Shape num_predictions x shape of each prediction
    num_samples = all_predictions.shape[1]
    return np.array([np.argmax(np.bincount(all_predictions[:, idx])) for idx in range(num_samples)])


# Predict with ensembling method
def ensembling_prediction(estimators1, estimators2, estimators3):
    start_time = time()

    print('Pred 1')
    ds = Dataset(k=0)
    y_predictions_1 = [estimator.fit(ds.X, ds.y).predict(ds.X_test) for estimator in estimators1]
    y_pred1 = vote_predictions(y_predictions_1)
    y_pred1 = pd.Series(y_pred1, index=ds.X_test.index, name='Bound')

    print('Pred 2')
    ds = Dataset(k=1)
    y_predictions_2 = [estimator.fit(ds.X, ds.y).predict(ds.X_test) for estimator in estimators2]
    y_pred2 = vote_predictions(y_predictions_2)
    y_pred2 = pd.Series(y_pred2, index=ds.X_test.index, name='Bound')

    print('Pred 3')
    ds = Dataset(k=2)
    y_predictions_3 = [estimator.fit(ds.X, ds.y).predict(ds.X_test) for estimator in estimators3]
    y_pred3 = vote_predictions(y_predictions_3)
    y_pred3 = pd.Series(y_pred3, index=ds.X_test.index, name='Bound')

    y_pred = pd.concat([y_pred1, y_pred2, y_pred3], axis=0, verify_integrity=True)
    y_pred = y_pred.astype(int)
    y_pred.to_csv('y_pred.csv')

    print("Took {:.2f} seconds to compute the predictions.".format(time() - start_time))
