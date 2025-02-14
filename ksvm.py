"""Run kernel SVM with a Gaussian kernel."""
from sklearn.model_selection import train_test_split

from dataset import Dataset
from kernels import GaussianKernel
from estimators import KernelSVMEstimator

for k in [0, 1, 2]:
    ds = Dataset(k=k)
    est = KernelSVMEstimator(lbd=1e-6, kernel=GaussianKernel(var=0.001))

    X_train, X_val, y_train, y_val = train_test_split(ds.X_mat, ds.y,
                                                      test_size=0.2,
                                                      random_state=0)

    est.fit(X_train, y_train)
    y_pred = est.predict(X_val)

    acc = (y_pred == y_val).sum() / len(y_val)
    print(f'Accuracy dataset {k} is: {acc:.3g}')
