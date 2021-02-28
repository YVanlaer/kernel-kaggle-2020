"""Implement the Dataset class."""
import pandas as pd
from os.path import abspath, join


# Data folder
data_folder = abspath('machine-learning-with-kernel-methods-2021/')


class Dataset():
    """Handle data loading."""

    def __init__(self, k=0, root_dir=data_folder):
        if k < 0 or k > 2:
            raise ValueError(f'k must be in {{0, 1, 2}}, "{k}" given.')

        self.k = k
        self.root_dir = root_dir

        self._X = None  # Train set
        self._y = None  # Train target
        self._X_test = None  # Test set
        self._X_mat = None  # Train set (features precomputed)
        self._X_mat_test = None  # Test set (features precomputed)

    @property
    def X(self):
        """Load and access X."""
        if self._X is None:
            self._X = pd.read_csv(join(self.root_dir, f'Xtr{self.k}.csv'),
                                  index_col='Id', squeeze=True)
        return self._X

    @property
    def y(self):
        """Load and access y."""
        if self._y is None:
            self._y = pd.read_csv(join(self.root_dir, f'Ytr{self.k}.csv'),
                                  index_col='Id', squeeze=True)
        return self._y

    @property
    def X_test(self):
        """Load and access X_test."""
        if self._X_test is None:
            self._X_test = pd.read_csv(join(self.root_dir, f'Xte{self.k}.csv'),
                                       index_col='Id', squeeze=True)
        return self._X_test

    @property
    def X_mat(self):
        """Load and access X_mat."""
        if self._X_mat is None:
            self._X_mat = pd.read_csv(join(self.root_dir, f'Xtr{self.k}_mat100.csv'),
                                      sep=' ', header=None)
        return self._X_mat

    @property
    def X_mat_test(self):
        """Load and access X_mat_test."""
        if self._X_mat_test is None:
            self._X_mat_test = pd.read_csv(join(self.root_dir, f'Xte{self.k}_mat100.csv'),
                                           sep=' ', header=None)
        return self._X_mat_test
