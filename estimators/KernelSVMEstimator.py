"""Implement the KernelSVMEstimator class."""
import numpy as np
import cvxpy as cp

from .BaseEstimator import BaseEstimator


class KernelSVMEstimator(BaseEstimator):
    """Implement a sklearn compatible kernel SVM estimator."""

    def __init__(self, lbd, kernel):
        """Init.

        Parameters:
        -----------
            lbd : float
                Weight in the optimization problem.
            kernel : instance of BaseKernel
                The kernel to use.

        """
        self.kernel = kernel
        self.lbd = lbd

        self.X = None
        self.alpha = None
        self.labels = None

    def _transform_labels(self, y):
        """Transform labels into {-1, 1}."""
        labels = np.unique(y)
        if len(labels) > 2:
            raise ValueError(f'Only two classes are accepted in y, '
                             f'found: {labels}')

        self.labels = labels

        y = y.copy()
        y[y == labels[0]] = -1
        y[y == labels[1]] = 1

        return y

    def _inverse_transform_labels(self, y):
        """Transform labels from {-1, 1} to original ones."""
        y = y.copy()
        neg_idx = y < 0
        y[neg_idx] = self.labels[0]
        y[~neg_idx] = self.labels[1]

        return y

    def fit(self, X, y):
        K = self.kernel(X, X, is_train = True)
        n = K.shape[0]
        e = cp.Variable(n)
        alpha = cp.Variable(n)
        ones = np.ones((n, 1))

        # Set labels in {-1, 1}
        y = self._transform_labels(y)
        y = np.array(y)

        objective = cp.Minimize(1/n*ones.T@e + self.lbd*cp.quad_form(alpha, K))
        constraints = [e >= 0, y*K@alpha + e - 1 >= 0]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.X = X
        self.alpha = alpha.value

        return self

    def predict(self, X):
        K_test = self.kernel(self.X, X, is_predict = True)
        y_pred = self.alpha@K_test
        y_pred = self._inverse_transform_labels(y_pred)
        return y_pred
