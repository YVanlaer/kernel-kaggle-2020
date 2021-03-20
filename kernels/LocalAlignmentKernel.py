"""Implement the LocalAlignmentKernel class."""
import scipy.sparse
import numpy as np
import os
from joblib import Memory
from tqdm import tqdm

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)

def get_default_computation_matrix(h, w):
    # Depth of 2: depth 0 for value, depth 1 for computed or not
    M = np.zeros((h, w, 2))
    M[0, :, 1] = 1
    M[:, 0, 1] = 1
    return M

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

        return int_sequence

    def get_M(self, i, j):
        if i == 0 or j == 0:
            return 0
        elif self.M[i, j, 1] == 1:
            return self.M[i, j, 0]
        else:
            long_term = 1 + self.get_X(i-1, j-1) + self.get_Y(i-1, j-1) + self.get_M(i-1, j-1)
            M_term = np.exp(self.beta * self.S[i, j])
            self.M[i, j, 0] = M_term
            self.M[i, j, 1] = 1
            return M_term

    def get_X(self, i, j):
        if i == 0 or j == 0:
            return 0
        elif self.X[i, j, 1] == 1:
            return self.X[i, j, 0]
        else:
            term_left = np.exp(self.beta * self.d) * self.get_M(i-1, j)
            term_right = np.exp(self.beta * self.e) * self.get_X(i-1, j)
            X_term = term_left + term_right
            self.X[i, j, 0] = X_term
            self.X[i, j, 1] = 1
            return X_term

    def get_Y(self, i, j):
        if i == 0 or j == 0:
            return 0
        elif self.Y[i, j, 1] == 1:
            return self.Y[i, j, 0]
        else:
            term_left = np.exp(self.beta * self.d) * (self.get_M(i, j-1) + self.get_X(i, j-1))
            term_right = np.exp(self.beta * self.e) * self.get_Y(i, j-1)
            Y_term = term_left + term_right
            self.Y[i, j, 0] = Y_term
            self.Y[i, j, 1] = 1
            return Y_term

    def get_X2(self, i, j):
        if i == 0 or j == 0:
            return 0
        elif self.X2[i, j, 1] == 1:
            return self.X2[i, j, 0]
        else:
            X2_term = self.get_M(i-1, j) + self.get_X2(i-1, j)
            self.X2[i, j, 0] = X2_term
            self.X2[i, j, 1] = 1
            return X2_term

    def get_Y2(self, i, j):
        if i == 0 or j == 0:
            return 0
        elif self.Y2[i, j, 1] == 1:
            return self.Y2[i, j, 0]
        else:
            Y2_term = self.get_M(i, j-1) + self.get_X2(i, j-1) + self.get_Y2(i, j-1)
            self.Y2[i, j, 0] = Y2_term
            self.Y2[i, j, 1] = 1
            return Y2_term

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


    def get_kernel_matrix(self, X1, X2, allow_kernel_saving=True):
        """May seem redundant with __call__ but is necessary for caching the result"""
        seqs_X1 = [LocalAlignmentKernel.sequence_to_int_sequence(sequence) for sequence in X1]
        seqs_X2 = [LocalAlignmentKernel.sequence_to_int_sequence(sequence) for sequence in X2]

        kernel_shape = (len(seqs_X1), len(seqs_X2))
        X2 = scipy.sparse.dok_matrix(kernel_shape, dtype=float)
        Y2 = scipy.sparse.dok_matrix(kernel_shape, dtype=float)
        M = scipy.sparse.dok_matrix(kernel_shape, dtype=float)

        for idx1, seq1 in tqdm(enumerate(seqs_X1), total=len(seqs_X1)):
            for idx2, seq2 in tqdm(enumerate(seqs_X2), total=len(seqs_X2)):
                self.S = scipy.sparse.dok_matrix((len(seq1) + 1, len(seq2) + 1), dtype=float)
                self.M = get_default_computation_matrix(len(seq1) + 1, len(seq2) + 1)
                self.X = get_default_computation_matrix(len(seq1) + 1, len(seq2) + 1)
                self.Y = get_default_computation_matrix(len(seq1) + 1, len(seq2) + 1)
                self.X2 = get_default_computation_matrix(len(seq1) + 1, len(seq2) + 1)
                self.Y2 = get_default_computation_matrix(len(seq1) + 1, len(seq2) + 1)

                for i1, char1 in enumerate(seq1):
                    for i2, char2 in enumerate(seq2):
                        self.S[i1, i2] = 1 if char1 == char2 else 0

                X2[idx1, idx2] = self.get_X2(len(seq1), len(seq2))
                Y2[idx1, idx2] = self.get_Y2(len(seq1), len(seq2))
                M[idx1, idx2] = self.get_M(len(seq1), len(seq2))

        K = 1 + X2 + Y2 + M
        K = K.tocsc()

        if allow_kernel_saving:
            LocalAlignmentKernel.save_sparse_matrix(K, self.beta, self.d, self.e)

        return K


    @staticmethod
    def get_sparse_matrix_file_name(beta, d, e):
        parent_folder = "sparse_matrices/local_alignment_kernels"
        file_name = f"{parent_folder}/local_alignment_beta_{beta}_d_{d}_e_{e}.npz"
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        return file_name

    @staticmethod
    def check_exists_sparse_matrix_file_name(beta, d, e):
        file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(beta, d, e)
        return os.path.exists(file_name)

    @staticmethod
    def save_sparse_matrix(K, beta, d, e):
        matrix_file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(beta, d, e)
        scipy.sparse.save_npz(matrix_file_name, K)

    @staticmethod
    def load_sparse_matrix(beta, d, e):
        print("Loading sparse kernel...")
        matrix_file_name = LocalAlignmentKernel.get_sparse_matrix_file_name(beta, d, e)
        K = scipy.sparse.load_npz(matrix_file_name)
        print("Kernel loaded.")
        return K

    def __call__(self, X1, X2, allow_file_loading=True, allow_kernel_saving=True, is_train=False, is_predict=False):
        """Create a kernel matrix given inputs."""
        if allow_file_loading and LocalAlignmentKernel.check_exists_sparse_matrix_file_name(self.beta, self.d, self.e):
            return LocalAlignmentKernel.load_sparse_matrix(self.beta, self.d, self.e)
        return self.get_kernel_matrix(X1, X2, allow_kernel_saving)
