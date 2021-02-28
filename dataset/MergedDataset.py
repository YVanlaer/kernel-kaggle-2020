
"""Implement the MergedDataset class."""
import pandas as pd

from .Dataset import Dataset, data_folder


class MergedDataset():
    """Handle data loading for the merged dataset."""

    def __init__(self, root_dir=data_folder):

        self.root_dir = root_dir

        self.datasets = [Dataset(k=k, root_dir=root_dir) for k in range(3)]

        self._X = None  # Train set
        self._y = None  # Train target
        self._X_test = None  # Test set
        self._X_mat = None  # Train set (features precomputed)
        self._X_mat_test = None  # Test set (features precomputed)

    @property
    def X(self):
        """Load and access X."""
        if self._X is None:
            self._X = pd.concat([ds.X for ds in self.datasets], axis=0,
                                verify_integrity=True)
        return self._X

    @property
    def y(self):
        """Load and access y."""
        if self._y is None:
            self._y = pd.concat([ds.y for ds in self.datasets], axis=0,
                                verify_integrity=True)
        return self._y

    @property
    def X_test(self):
        """Load and access X_test."""
        if self._X_test is None:
            self._X_test = pd.concat([ds.X_test for ds in self.datasets],
                                     axis=0, verify_integrity=True)
        return self._X_test

    @property
    def X_mat(self):
        """Load and access X_mat."""
        if self._X_mat is None:
            self._X_mat = pd.concat([ds.X_mat.set_index(ds.X.index) for ds in self.datasets],
                                    axis=0, verify_integrity=True)
        return self._X_mat

    @property
    def X_mat_test(self):
        """Load and access X_mat_test."""
        if self._X_mat_test is None:
            self._X_mat_test = pd.concat([ds.X_mat_test.set_index(ds.X_test.index) for ds in self.datasets],
                                         axis=0, verify_integrity=True)
        return self._X_mat_test
