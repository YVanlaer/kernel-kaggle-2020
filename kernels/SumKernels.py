"""Implement the SumKernels class."""
import numpy as np
from joblib import Memory

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)


class SumKernels(BaseKernel):
    """Sum several kernels."""

    def __init__(self, kernels):
        """Init.

        Parameters:
        -----------
            kernels : list
                List of kernels to sum

        """
        self.kernels = kernels
        self.get_kernel_matrix = memory.cache(self.get_kernel_matrix)

    @staticmethod
    def get_kernel_matrix(X1, X2, kernels):
        """May seem redundant with __call__ but is necessary for caching the result"""
        print('Start computation of kernels')
        all_kernels = [kernel(X1, X2) for kernel in kernels]
        K = 0
        for kernel in all_kernels:
            K += kernel

        return K


    def __call__(self, X1, X2, is_train = False, is_predict = False):
        """Create a kernel matrix which is a sum of other kernels given inputs."""
        return self.get_kernel_matrix(X1, X2, self.kernels)
