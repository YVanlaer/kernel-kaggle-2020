import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

from dataset import Dataset


# ds = Dataset(k=0)
# print(ds.X_mat)
# print(ds.X)
# print(ds.y)
# print(ds.X_val)


def gaussian_kernel(X, X2=None, var=1):
    if X2 is None:
        X2 = X
    return np.exp(-np.power(distance.cdist(X, X2), 2)/(2*var))


# Implement kernel SVM
def KernelSVM(K, y, lbd=1):
    """Implement kernel SVM.

    Parameters:
    -----------
        K : np.array of shape (n, n)
            Kernel
        y : np.array of shape (n,)
            Labels in {0, 1}

    Returns:
    --------
        alpha : np.array of shape (n,)
            The parameter to

    """
    assert K.shape[0] == K.shape[1]
    n = K.shape[0]
    e = cp.Variable(n)
    alpha = cp.Variable(n)
    ones = np.ones((n, 1))
    y = np.array(y)
    y[y == 0] = -1
    # y[y < 0] = -1
    print(y)

    objective = cp.Minimize(1/n*ones.T@e + lbd*cp.quad_form(alpha, K))

    constraints = [e >= 0, y*K@alpha + e - 1 >= 0]

    problem = cp.Problem(objective, constraints)

    problem.solve()

    print(e.value)
    print(alpha.value)

    return alpha.value


X_mats = []
X_mat_tests = []
X_tests = []
ys = []
for k in [0, 1, 2]:
    ds = Dataset(k=k)
    y = ds.y
    X_mat = ds.X_mat
    X_mat_test = ds.X_mat_test
    X_mat = X_mat.set_index(y.index)
    X_mat_test = X_mat_test.set_index(ds.X_test.index)
    X_mats.append(X_mat)
    X_mat_tests.append(X_mat_test)
    X_tests.append(ds.X_test)
    ys.append(y)

X_mat = pd.concat(X_mats, axis=0, verify_integrity=True)
X_mat_test = pd.concat(X_mat_tests, axis=0, verify_integrity=True)
X_test = pd.concat(X_tests, axis=0, verify_integrity=True)
y = pd.concat(ys, axis=0, verify_integrity=True)


# Prediction on test set
var = 0.1
K = gaussian_kernel(X_mat, var=var)
print(K)
alpha = KernelSVM(K, y, lbd=0.001)

K_test = gaussian_kernel(X_mat, X_mat_test, var=var)

y_pred = alpha@K_test

print(y_pred)

y_pred[y_pred >= 0] = 1
y_pred[y_pred < 0] = 0

y_pred = pd.Series(y_pred, index=X_test.index)

y_pred.to_csv('y_pred.csv')

exit()


# Prediction on validation set
X_train, X_val, y_train, y_val = train_test_split(X_mat, y,
                                                  test_size=0.2,
                                                  random_state=0)

# X_train, X_val, y_train, y_val = train_test_split(ds.X_mat, ds.y,
#                                                     test_size=0.2,
#                                                     random_state=0)


var = 0.1
K = gaussian_kernel(X_train, var=var)
print(K)
alpha = KernelSVM(K, y_train, lbd=0.001)

K_test = gaussian_kernel(X_train, X_val, var=var)

y_pred = alpha@K_test

print(y_pred)

y_pred[y_pred >= 0] = 1
y_pred[y_pred < 0] = 0
print(y_pred)

acc = (y_pred != y_val).sum() / len(y_val)
print("ACCURACY IS", acc)
