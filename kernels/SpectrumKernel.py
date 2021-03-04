"""Implement the SpectrumKernel class."""
import scipy.sparse
import numpy as np
from joblib import Memory

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)


class SpectrumKernel(BaseKernel):
    """Implement the k-spectrum kernel."""

    def __init__(self, k, n=4):
        """Init.

        Parameters:
        -----------
            k : int
                Size of the sliding window.
            n : int
                Size of the alphabet in the sequence. 4 for DNA sequence.

        """
        self.k = k
        self.n = n
        self.get_kernel_matrix = memory.cache(self.get_kernel_matrix)

    @staticmethod
    def sequence_to_int_sequence(sequence):
        """Convert a sequence of letters to a sequence of integers."""
        alphabet = list(set(list(sequence)))
        alphabet.sort()
        int_sequence = sequence

        for i, letter in enumerate(alphabet):
            int_sequence = int_sequence.replace(letter, str(i))

        return int_sequence

    @staticmethod
    def phi(sequence, k, n):
        """Create the embedding of a sequence y counting the k-sequences."""
        m = len(sequence)
        int_sequence = SpectrumKernel.sequence_to_int_sequence(sequence)
        count = scipy.sparse.dok_matrix((n**k, 1), dtype=int)
        idx_shape = tuple(n for _ in range(k))

        for i in range(m-k+1):
            subseq = int_sequence[i:i+k]
            idx = tuple(int(s) for s in tuple(subseq))
            idx = np.ravel_multi_index(idx, idx_shape)
            count[idx, 0] += 1

        return count.tocsc()

    @staticmethod
    def get_kernel_matrix(X1, X2, k, n):
        """May seem redundant with __call__ but is necessary for caching the result"""
        phis1 = [SpectrumKernel.phi(sequence, k, n) for sequence in X1]
        phis2 = [SpectrumKernel.phi(sequence, k, n) for sequence in X2]

        phi1 = scipy.sparse.hstack(phis1)
        phi2 = scipy.sparse.hstack(phis2)

        K = phi1.transpose().dot(phi2)

        return K

    def __call__(self, X1, X2):
        """Create a kernel matrix given inputs."""
        return self.get_kernel_matrix(X1, X2, self.k, self.n)
