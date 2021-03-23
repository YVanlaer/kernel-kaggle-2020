"""Implement the LocalAlignmentKernel class."""
import scipy.sparse
import numpy as np
import os
import sys
from joblib import Memory
from tqdm import tqdm
from numba import jit
#from numba import float32
#from numba.experimental import jitclass

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)

# class TailRecurseException(BaseException):
#   def __init__(self, args, kwargs):
#     self.args = args
#     self.kwargs = kwargs
#
# def tail_call_optimized(g):
#   """
#   This function decorates a function with tail call
#   optimization. It does this by throwing an exception
#   if it is it's own grandparent, and catching such
#   exceptions to fake the tail call optimization.
#
#   This function fails if the decorated
#   function recurses in a non-tail context.
#   """
#   def func(*args, **kwargs):
#     f = sys._getframe()
#     if f.f_back and f.f_back.f_back \
#         and f.f_back.f_back.f_code == f.f_code:
#       raise TailRecurseException(args, kwargs)
#     else:
#       while 1:
#         try:
#           return g(*args, **kwargs)
#         except TailRecurseException as e:
#           args = e.args
#           kwargs = e.kwargs
#   func.__doc__ = g.__doc__
#   return func


def get_default_computation_matrix(h, w):
    # Depth of 2: depth 0 for value, depth 1 for computed or not
    M = np.zeros((h, w, 2))
    M[0, :, 1] = 1
    M[:, 0, 1] = 1
    return M


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

# @jitclass([('beta', float32), ('d', float32), ('e', float32), ('n', float32)])


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

    # @tail_call_optimized
    # def get_M(self, i, j):
    #     if i == 0 or j == 0:
    #         return 0
    #     elif self.M[i, j, 1] == 1:
    #         return self.M[i, j, 0]
    #     else:
    #         long_term = 1 + self.get_X(i-1, j-1) + self.get_Y(i-1, j-1) + self.get_M(i-1, j-1)
    #         M_term = np.exp(self.beta * self.S[i, j])
    #         self.M[i, j, 0] = M_term
    #         self.M[i, j, 1] = 1
    #         return M_term
    #
    # #@tail_call_optimized
    # def get_X(self, i, j):
    #     if i == 0 or j == 0:
    #         return 0
    #     elif self.X[i, j, 1] == 1:
    #         return self.X[i, j, 0]
    #     else:
    #         term_left = np.exp(self.beta * self.d) * self.get_M(i-1, j)
    #         term_right = np.exp(self.beta * self.e) * self.get_X(i-1, j)
    #         X_term = term_left + term_right
    #         self.X[i, j, 0] = X_term
    #         self.X[i, j, 1] = 1
    #         return X_term
    #
    # #@tail_call_optimized
    # def get_Y(self, i, j):
    #     if i == 0 or j == 0:
    #         return 0
    #     elif self.Y[i, j, 1] == 1:
    #         return self.Y[i, j, 0]
    #     else:
    #         term_left = np.exp(self.beta * self.d) * (self.get_M(i, j-1) + self.get_X(i, j-1))
    #         term_right = np.exp(self.beta * self.e) * self.get_Y(i, j-1)
    #         Y_term = term_left + term_right
    #         self.Y[i, j, 0] = Y_term
    #         self.Y[i, j, 1] = 1
    #         return Y_term
    #
    # #@tail_call_optimized
    # def get_X2(self, i, j):
    #     if i == 0 or j == 0:
    #         return 0
    #     elif self.X2[i, j, 1] == 1:
    #         return self.X2[i, j, 0]
    #     else:
    #         X2_term = self.get_M(i-1, j) + self.get_X2(i-1, j)
    #         self.X2[i, j, 0] = X2_term
    #         self.X2[i, j, 1] = 1
    #         return X2_term
    #
    # #@tail_call_optimized
    # def get_Y2(self, i, j):
    #     if i == 0 or j == 0:
    #         return 0
    #     elif self.Y2[i, j, 1] == 1:
    #         return self.Y2[i, j, 0]
    #     else:
    #         Y2_term = self.get_M(i, j-1) + self.get_X2(i, j-1) + self.get_Y2(i, j-1)
    #         self.Y2[i, j, 0] = Y2_term
    #         self.Y2[i, j, 1] = 1
    #         return Y2_term

    # @staticmethod
    # def phi(sequence, k, m, n):
    #     """Create the embedding of a sequence y counting the k-sequences."""
    #     int_sequence = LocalAlignmentKernel.sequence_to_int_sequence(sequence)
    #     count = scipy.sparse.dok_matrix((n**k, 1), dtype=int)
    #     idx_shape = tuple(n for _ in range(k))

    #     for i in range(len(sequence) - k + 1):
    #         subseq = int_sequence[i:i+k]
    #         all_mismatches_sequences = get_all_mismatches(subseq, m, n)

    #         for subseq_mismatch in all_mismatches_sequences:
    #             idx = tuple(int(s) for s in tuple(subseq_mismatch))
    #             idx = np.ravel_multi_index(idx, idx_shape)
    #             count[idx, 0] += 1

    #     return count.tocsc()

    @staticmethod
    def get_kernel_matrix(X1, X2, beta, d, e):
        """May seem redundant with __call__ but is necessary for caching the result"""
        seqs_X1 = [LocalAlignmentKernel.sequence_to_int_sequence(
            sequence) for sequence in X1]
        seqs_X2 = [LocalAlignmentKernel.sequence_to_int_sequence(
            sequence) for sequence in X2]

        kernel_shape = (len(seqs_X1), len(seqs_X2))
        X2 = scipy.sparse.dok_matrix(kernel_shape, dtype=float)
        Y2 = scipy.sparse.dok_matrix(kernel_shape, dtype=float)
        M = scipy.sparse.dok_matrix(kernel_shape, dtype=float)

        alphabet_size = int(max(max([list(seq) for seq in seqs_X1]))) + 1

        # S = np.ones((alphabet_size, alphabet_size)) - \
        #     np.identity(alphabet_size)
        S = np.identity(alphabet_size)

        for idx1, seq1 in tqdm(enumerate(seqs_X1), total=len(seqs_X1)):
            for idx2, seq2 in enumerate(seqs_X2):
                # seq_1_repeated = np.repeat(np.array(list(seq1))[:,np.newaxis], len(seq2), axis=1)
                # seq_2_repeated = np.tile(np.array(list(seq2)), (len(seq1), 1))
                # S = seq_1_repeated == seq_2_repeated

                x2, y2, m = dynamic_compute(
                    S, seq1, seq2, beta, d, e)

                X2[idx1, idx2] = x2
                Y2[idx1, idx2] = y2
                M[idx1, idx2] = m

        K = 1 + X2 + Y2 + M
        K = K.tocsc()
        K.data = np.log(K.data) / beta

        eig_values = scipy.sparse.linalg.eigsh(K, k=6)[0]
        smallest_eig_value = eig_values[0]
        print('eigenvalues before substraction', eig_values)
        print('smallest eigenvalues before substraction', smallest_eig_value)
        # K -= (smallest_eig_value - 1e-10) * scipy.sparse.identity(len(seqs_X1))
        print('eigenvalues before substraction',
              scipy.sparse.linalg.eigsh(K, k=6))

        # if allow_kernel_saving:
        #     LocalAlignmentKernel.save_sparse_matrix(K, beta, d, e)

        return K

    # @staticmethod
    # def get_sparse_matrix_file_name(beta, d, e):
    #     parent_folder = "sparse_matrices/local_alignment_kernels"
    #     file_name = f"{parent_folder}/local_alignment_beta_{beta}_d_{d}_e_{e}.npz"
    #     if not os.path.exists(parent_folder):
    #         os.makedirs(parent_folder)
    #     return file_name

    # @staticmethod
    # def check_exists_sparse_matrix_file_name(beta, d, e):
    #     file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(
    #         beta, d, e)
    #     return os.path.exists(file_name)

    # @staticmethod
    # def save_sparse_matrix(K, beta, d, e):
    #     matrix_file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(
    #         beta, d, e)
    #     scipy.sparse.save_npz(matrix_file_name, K)

    # @staticmethod
    # def load_sparse_matrix(beta, d, e):
    #     print("Loading sparse kernel...")
    #     matrix_file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(
    #         beta, d, e)
    #     K = scipy.sparse.load_npz(matrix_file_name)
    #     print("Kernel loaded.")
    #     return K

    def __call__(self, X1, X2):
        """Create a kernel matrix given inputs."""
        # if allow_file_loading and LocalAlignmentKernel.check_exists_sparse_matrix_file_name(self.beta, self.d, self.e):
        #    return LocalAlignmentKernel.load_sparse_matrix(self.beta, self.d, self.e)
        return self.get_kernel_matrix(X1, X2, self.beta, self.d, self.e)
