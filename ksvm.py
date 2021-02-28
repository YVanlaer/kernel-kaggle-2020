"""Execute kernel SVM."""
from sklearn.model_selection import train_test_split

from dataset import Dataset, MergedDataset
from kernels import GaussianKernel
from estimators import KernelSVMEstimator

for k in [0, 1, 2]:
    ds = Dataset(k=k)
    est = KernelSVMEstimator(lbd=0.001, kernel=GaussianKernel, kernel_params={'var': 0.1})

    X_train, X_val, y_train, y_val = train_test_split(ds.X_mat, ds.y,
                                                      test_size=0.2,
                                                      random_state=0)

    est.fit(X_train, y_train)
    y_pred = est.predict(X_val)

    acc = (y_pred == y_val).sum() / len(y_val)
    print(f'Accuracy dataset {k} is: {acc:.3g}')
