"""Implement the KernelSVMEstimator class."""
import numpy as np
import cvxpy as cp


class KernelSVMEstimator():
    """Implement a sklearn compatible kernel SVM estimator."""

    def __init__(self, lbd, Kernel, kernel_params):
        """Init.

        Parameters:
        -----------
            lbd : float
                Weight in the optimization problem.
            Kernel : class inheriting from Kernel
                The kernel.
            kernel_params : dict
                The parameters given to the Kernel class when instanciating.

        """
        self.kernel = Kernel(**kernel_params)
        self.lbd = lbd

        self.X = None
        self.alpha = None

    def fit(self, X, y):
        K = self.kernel(X, X)

        n = K.shape[0]
        e = cp.Variable(n)
        alpha = cp.Variable(n)
        ones = np.ones((n, 1))

        # Set labels in {-1, 1}
        y = np.array(y)
        y[y == 0] = -1

        objective = cp.Minimize(1/n*ones.T@e + self.lbd*cp.quad_form(alpha, K))
        constraints = [e >= 0, y*K@alpha + e - 1 >= 0]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.X = X
        self.alpha = alpha.value

        return self

    def predict(self, X):
        K_test = self.kernel(self.X, X)
        y_pred = self.alpha@K_test
        return y_pred
