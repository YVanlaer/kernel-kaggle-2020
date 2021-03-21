"""Implement BaseKernel class."""
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    """Abstract class for creating kernels."""

    @abstractmethod
    def __call__(self, X1, X2, is_train=False, is_predict=False):
        """Create a kernel matrix given inputs."""
        pass

    def __repr__(self):
        d = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({d})'


# class SequenceKernel(BaseKernel):
#     """Implement common method to sequence based kernels."""
#     # Store precomputed DTW matrices to save time
#     dtw_matrices: dict = {}


#     @staticmethod
#     def hash_sequences(Xs):
#         array = np.concatenate(Xs, axis=0)
#         return hashlib.sha256(array).hexdigest()

#     def register_matrix(self, matrix, X1, X2):
#         h = self.hash_sequences([X1, X2])
#         print(f'Registering DTW matrix for hash {h}')
#         self.dtw_matrices[h] = matrix

#     def get_matrix(self, X1, X2):
#         h = self.hash_sequences([X1, X2])
#         print(f'Retrieving DTW matrix for hash {h}', end='\t')
#         return self.dtw_matrices.get(h, None)
