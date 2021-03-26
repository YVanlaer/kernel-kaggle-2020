"""Implement the LocalAlignmentKernel class."""
import scipy.sparse
import numpy as np
import os
import sys
from joblib import Memory
from tqdm import tqdm
from numba import jit

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)


def blosum62():
    return np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, -2], [0, -1, -2, 5]])


def blast():
    return np.zeros((4, 4)) - 3 + 4 * np.identity(4)


@jit
def dynamic_compute(S, seq1, seq2, beta, d, e):
    n1, n2 = len(seq1), len(seq2)
    M = np.zeros((n1, n2))
    X, Y = np.zeros((n1, n2)), np.zeros((n1, n2))
    X2, Y2 = np.zeros((n1, n2)), np.zeros((n1, n2))

    for diag_idx in range(2, n1 + n2 - 1):
        i, j = diag_idx - 1, 1
        if diag_idx > n1:
            i, j = n1 - 1, j + diag_idx - n1

        while (j < n2 and i >= 1):
            x1_i = seq1[i]
            y2_j = seq2[j]

            M[i, j] = np.exp(beta * S[x1_i, y2_j]) * \
                (1 + X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])

            X[i, j] = np.exp(beta * d) * M[i - 1, j] + \
                np.exp(beta * e) * X[i - 1, j]

            Y[i, j] = np.exp(beta * d) * (M[i, j - 1] +
                                          X[i, j - 1]) + np.exp(beta * e) * Y[i, j - 1]

            X2[i, j] = M[i - 1, j] + X2[i - 1, j]
            Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]

            j += 1
            i -= 1

    return X2[-1, -1], Y2[-1, -1], M[-1, -1]


class LocalAlignmentKernel(BaseKernel):
    """Implement the (beta)-local alignment kernel."""

    def __init__(self, beta, d, e, n=4):
        """Init.

        Parameters:
        -----------
            beta : int
                Parameter of the exponential
            d:
            e:
            n : int
                Size of the alphabet in the sequence. 4 for DNA sequence.

        """
        self.beta = beta
        self.d = d
        self.e = e
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

        int_sequence_array = np.array(list(map(int, int_sequence)))

        return int_sequence_array

    @staticmethod
    def get_kernel_matrix(X1, X2, beta, d, e, is_train=False, is_predict=False):
        """May seem redundant with __call__ but is necessary for caching the result"""
        seqs_X1 = [LocalAlignmentKernel.sequence_to_int_sequence(
            sequence) for sequence in X1]
        seqs_X2 = [LocalAlignmentKernel.sequence_to_int_sequence(
            sequence) for sequence in X2]

        kernel_shape = (len(seqs_X1), len(seqs_X2))
        X2 = np.zeros(kernel_shape, dtype=float)
        Y2 = np.zeros(kernel_shape, dtype=float)
        M = np.zeros(kernel_shape, dtype=float)

        alphabet_size = int(max(max([list(seq) for seq in seqs_X1]))) + 1

        # Choose the similarity matrix of your choice: Identity, Blosum62 or BLAST
        # S = np.identity(alphabet_size)
        # S = blosum62()
        S = blast()

        for idx1, seq1 in tqdm(enumerate(seqs_X1), total=len(seqs_X1)):
            for idx2, seq2 in enumerate(seqs_X2):

                x2, y2, m = dynamic_compute(
                    S, seq1, seq2, beta, d, e)

                X2[idx1, idx2] = x2
                Y2[idx1, idx2] = y2
                M[idx1, idx2] = m

        K = 1 + X2 + Y2 + M
        K.data = np.log(K.data) / beta

        if is_train:
            eig_values = np.linalg.eigvalsh(K)
            smallest_eig_value = eig_values[0]
            if smallest_eig_value < 0:
                K -= (smallest_eig_value - 1e-10) * np.identity(len(seqs_X1))

        return K

    def __call__(self, X1, X2, is_train=False, is_predict=False):
        """Create a kernel matrix given inputs."""
        return self.get_kernel_matrix(X1, X2, self.beta, self.d, self.e, is_train=is_train, is_predict=is_predict)
