"""Implement the MismatchKernel class."""
import scipy.sparse
import numpy as np
from joblib import Memory

from .BaseKernel import BaseKernel


memory = Memory('joblib_cache/', verbose=0)


def get_all_mismatches(sequence, m, n):
    if m == 0 or sequence == "":
        return [sequence]
    else:
        replace_head_sequences = [
            list(map(lambda seq: str(s) + seq,
                 get_all_mismatches(sequence[1:], m-1, n)))
            for s in range(n) if s != sequence[0]
        ]
        replace_head_sequences = [
            seq for sequence in replace_head_sequences for seq in sequence]  # Flatten

        keep_head_sequences = list(
            map(lambda seq: sequence[0] + seq,
                get_all_mismatches(sequence[1:], m, n)
                )
        )

        lower_mismatch_order_sequences = get_all_mismatches(sequence, m-1, n)

        all_sequences = [sequence] + replace_head_sequences + \
            keep_head_sequences + lower_mismatch_order_sequences
        return list(set(all_sequences))


class MismatchKernel(BaseKernel):
    """Implement the (k,m)-mismatch kernel."""

    def __init__(self, k, m, n=4):
        """Init.

        Parameters:
        -----------
            k : int
                Size of the sliding window.
            m : int
                Number of mismatches
            n : int
                Size of the alphabet in the sequence. 4 for DNA sequence.

        """
        self.k = k
        self.n = n
        self.m = m
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
    def phi(sequence, k, m, n):
        """Create the embedding of a sequence y counting the k-sequences."""
        int_sequence = MismatchKernel.sequence_to_int_sequence(sequence)
        count = scipy.sparse.dok_matrix((n**k, 1), dtype=int)
        idx_shape = tuple(n for _ in range(k))

        for i in range(len(sequence) - k + 1):
            subseq = int_sequence[i:i+k]
            all_mismatches_sequences = get_all_mismatches(subseq, m, n)

            for subseq_mismatch in all_mismatches_sequences:
                idx = tuple(int(s) for s in tuple(subseq_mismatch))
                idx = np.ravel_multi_index(idx, idx_shape)
                count[idx, 0] += 1

        return count.tocsc()

    @staticmethod
    def get_kernel_matrix(X1, X2, k, m, n):
        """May seem redundant with __call__ but is necessary for caching the result"""
        phis1 = [MismatchKernel.phi(sequence, k, m, n) for sequence in X1]
        phis2 = [MismatchKernel.phi(sequence, k, m, n) for sequence in X2]

        phi1 = scipy.sparse.hstack(phis1)
        phi2 = scipy.sparse.hstack(phis2)

        K = phi1.transpose().dot(phi2)

        return K

    def __call__(self, X1, X2, is_train=False, is_predict=False):
        """Create a kernel matrix given inputs."""
        return self.get_kernel_matrix(X1, X2, self.k, self.m, self.n)
