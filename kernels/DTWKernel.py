"""Implement the GaussianKernel class."""
import numpy as np
import pandas as pd
from scipy.spatial import distance
from dtaidistance import dtw
import hashlib

from .BaseKernel import BaseKernel


class DTWKernel(BaseKernel):
    """Implement a kernel seeing sequences as time series and dist as DTW."""
    # Store precomputed DTW matrices to save time
    dtw_matrices: dict = {}

    def __init__(self, var=1):
        """Init.

        Parameters:
        -----------

        """
        self.var = var

    @staticmethod
    def one_sequence_to_timeseries(sequence):
        """Convert a DNA sequence to a time series.

        Parameters:
        -----------
            sequence : str
                DNA sequence containing n letters.

        Returns:
        --------
            timeseries : np.array of shape (n,)

        """
        sequence = sequence.replace('A', '0')
        sequence = sequence.replace('T', '1')
        sequence = sequence.replace('C', '2')
        sequence = sequence.replace('G', '3')

        sequence = list(sequence)
        sequence = np.array(sequence).astype(float)

        return sequence

    @staticmethod
    def sequence_to_timeseries(sequences):
        return np.array([DTWKernel.one_sequence_to_timeseries(s) for s in sequences])

    @staticmethod
    def hash_sequences(Xs):
        X = pd.concat(Xs, axis=0)
        return hashlib.sha256(pd.util.hash_pandas_object(X).values).hexdigest()

    def register_dtw_matrix(self, matrix, X1, X2):
        h = self.hash_sequences([X1, X2])
        print(f'Registering DTW matrix for hash {h}')
        self.dtw_matrices[h] = matrix

    def get_dtw_matrix(self, X1, X2):
        h = self.hash_sequences([X1, X2])
        print(f'Retrieving DTW matrix for hash {h}', end='\t')
        return self.dtw_matrices.get(h, None)

    def __call__(self, X1, X2):
        print(f'Keys in dtw matrices: {len(self.dtw_matrices.keys())}')
        # Check if kernel matrix has already been computed
        dtw_matrix = self.get_dtw_matrix(X1, X2)

        if dtw_matrix is None:
            print('DTW matrix not found')

            X1_ = self.sequence_to_timeseries(X1)
            X2_ = self.sequence_to_timeseries(X2)
            n1 = X1_.shape[0]
            n2 = X2_.shape[0]

            X = np.concatenate([X1_, X2_], axis=0)

            dtw_matrix = dtw.distance_matrix_fast(X, parallel=True,
                                                  block=((0, n1), (n1, n1+n2)))
            dtw_matrix = dtw_matrix[:n1, n1:]

            self.register_dtw_matrix(dtw_matrix, X1, X2)
        else:
            print('DTW matrix found')

        K = np.exp(-np.power(dtw_matrix, 2)/(2*self.var))
        return K

    def __repr__(self):
        return f'DTWKernel({self.var})'
