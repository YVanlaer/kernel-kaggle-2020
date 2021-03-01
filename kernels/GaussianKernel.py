"""Implement the GaussianKernel class."""
import numpy as np
from scipy.spatial import distance

from .BaseKernel import BaseKernel


class GaussianKernel(BaseKernel):
    """Implement the Gaussian kernel."""

    def __init__(self, var=1):
        """Init.

        Parameters:
        -----------
            var : float
                The kernel variance.

        """
        self.var = var

    def __call__(self, X1, X2):
        return np.exp(-np.power(distance.cdist(X1, X2), 2)/(2*self.var))
