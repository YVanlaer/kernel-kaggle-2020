"""Run kernel SVM or KRR using SumKernels for summing several kernels."""
import argparse
from sklearn.model_selection import train_test_split

from dataset import Dataset
from kernels import MismatchKernel, SumKernels
from estimators import KernelSVMEstimator, KernelRREstimator
from predict import predict


# Validate
def validate(args):
    ds = Dataset(k=0)

    kernels = [
        MismatchKernel(k=12, m=1),
        MismatchKernel(k=11, m=1)
    ]
    kernel = SumKernels(kernels=kernels)

    if args.estimator == 'ksvm':
        est = KernelSVMEstimator(lbd=1e-6, kernel=kernel)
    elif args.estimator == 'krr':
        est = KernelRREstimator(lbd=1e-6, kernel=kernel)

    X_train, X_val, y_train, y_val = train_test_split(ds.X, ds.y,
                                                      test_size=0.2,
                                                      random_state=0)

    est.fit(X_train, y_train)
    y_pred = est.predict(X_val)

    acc = (y_pred == y_val).sum() / len(y_val)
    print(f'Accuracy dataset {0} is: {acc:.3g}')


# Predict
def get_predictions(args):
    lambdas = [1e-6, 1e-6, 1e-6]

    kernels = [
        SumKernels(kernels=[
            MismatchKernel(k=10, m=1),
            MismatchKernel(k=7, m=1)
        ]),
        SumKernels(kernels=[
            MismatchKernel(k=10, m=1),
            MismatchKernel(k=7, m=1)
        ]),
        SumKernels(kernels=[
            MismatchKernel(k=10, m=1),
            MismatchKernel(k=7, m=1)
        ]),
    ]
    predict(lambdas, kernels, args.estimator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run computations on Mismatch kernel")
    parser.add_argument("--mode", type=int, default=1,
                        help="Mode to use: 0 for validation, 1 for prediction")
    parser.add_argument("--estimator", type=str, default="krr",
                        help="Estimator to use. Available estimators are: ksvm, krr")
    args = parser.parse_args()

    assert args.estimator in ["ksvm", "krr"]

    if args.mode == 0:
        validate(args)
    elif args.mode == 1:
        get_predictions(args)
    else:
        raise Exception("Wrong mode.")
