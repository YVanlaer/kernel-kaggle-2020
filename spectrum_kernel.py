"""Run kernel SVM using SpectrumKernel."""
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import joblib

from dataset import Dataset
from kernels import SpectrumKernel
from estimators import KernelSVMEstimator, KernelRREstimator
from predict import predict


# Validate
def validate(args):
    for k in [0, 1, 2]:
        ds = Dataset(k=k)

        kernel = SpectrumKernel(k=3)

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
        print(f'Accuracy dataset {k} is: {acc:.3g}')


# Grid search
def grid_search(args):
    ds = Dataset(k=2)

    if args.estimator == 'ksvm':
        estimator = KernelSVMEstimator(
            lbd=1e-6, kernel=SpectrumKernel(k=3))
    elif args.estimator == 'krr':
        estimator = KernelRREstimator(
            lbd=1e-6, kernel=SpectrumKernel(k=3))
    kernels = []

    for k in range(5, 13):
        kernels.append(SpectrumKernel(k=k))

    param_grid = {
        'lbd': [1e-3],  # np.logspace(-6, -3, 2),
        'kernel': kernels,
    }

    cv = KFold(n_splits=5)

    with joblib.parallel_backend(backend='loky'):
        gscv = GridSearchCV(estimator, param_grid=param_grid, n_jobs=10, cv=cv,
                            refit=True, verbose=10, scoring='accuracy')

        gscv.fit(ds.X, ds.y)

    print(gscv.cv_results_)
    print(gscv.best_params_)
    print(gscv.best_score_)


# Predict
def get_predictions(args):
    lambdas = [1e-6, 1e-6, 1e-6]
    kernels = [
        SpectrumKernel(k=11),
        SpectrumKernel(k=6),
        SpectrumKernel(k=10),
    ]
    predict(lambdas, kernels, args.estimator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run computations on Spectrum kernel")
    parser.add_argument(
        "--mode", type=int, help="Mode to use: 0 for validation, 1 for grid search, 2 for prediction")
    parser.add_argument("--estimator", type=str,
                        help="Estimator to use. Available estimators are: ksvm, krr")
    args = parser.parse_args()

    assert args.estimator in ["ksvm", "krr"]

    if args.mode == 0:
        validate(args)
    elif args.mode == 1:
        grid_search(args)
    elif args.mode == 2:
        get_predictions(args)
    else:
        raise Exception("Wrong mode.")
