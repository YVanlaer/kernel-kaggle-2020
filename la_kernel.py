"""Run kernel SVM."""
from sklearn.model_selection import train_test_split

from dataset import Dataset, MergedDataset
from kernels import GaussianKernel, LocalAlignmentKernel
from estimators import KernelSVMEstimator

for k in [0, 1, 2]:
    ds = Dataset(k=k)
    kernel = LocalAlignmentKernel(beta=0.5, d=11, e=1)
    est = KernelSVMEstimator(lbd=1e-6, kernel=kernel)

    X_train, X_val, y_train, y_val = train_test_split(ds.X, ds.y,
                                                      test_size=0.2,
                                                      random_state=0)

    est.fit(X_train[:10], y_train[:10])
    y_pred = est.predict(X_val[:10])

    acc = (y_pred == y_val[:10]).sum() / len(y_val[:10])
    print(f'Accuracy dataset {k} is: {acc:.3g}')
