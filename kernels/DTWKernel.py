"""Implement the GaussianKernel class."""
import numpy as np
from scipy.spatial import distance
from dtaidistance import dtw

from .BaseKernel import BaseKernel


class DTWKernel(BaseKernel):
    """Implement a kernel seeing sequences as time series and dist as DTW."""

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

    def __call__(self, X1, X2):
        X1 = self.sequence_to_timeseries(X1)
        X2 = self.sequence_to_timeseries(X2)
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        X = np.concatenate([X1, X2], axis=0)

        dist = dtw.distance_matrix_fast(X, parallel=True,
                                        block=((0, n1), (n1, n1+n2)))
        dist = dist[:n1, n1:]

        return np.exp(-np.power(dist, 2)/(2*self.var))

    def __repr__(self):
        return f'DTWKernel({self.var})'
