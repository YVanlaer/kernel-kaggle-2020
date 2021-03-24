"""Predict with ensembling method (basic voting for csvs already computed)."""
import pandas as pd
import numpy as np


def vote_predictions(predictions_list):
    all_predictions = np.array(predictions_list)
    return np.argmax(np.bincount(all_predictions))


def vote_predictions_in_csvs(csvs, csv_name='y_pred.csv'):
    final_csv = csvs[0].copy()
    df = pd.concat(csvs, axis=1)['Bound']

    for row in df.iterrows():
        idx = row[0]
        vals = row[1].values
        voted_val = vote_predictions(vals)
        final_csv.loc[idx, 'Bound'] = voted_val

    final_csv.to_csv('y_pred_vote_a_posteriori.csv', index=False)


if __name__ == "__main__":
    csv_paths = [
        './y_pred_sum_2_best_kernels.csv',
        './y_pred_3_mismatch_kernels_ensemble.csv',
        './y_pred_2_mismatch_1_spectrum_kernels.csv'
    ]
    csvs = [pd.read_csv(path) for path in csv_paths]
