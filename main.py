"""Predict with ensembling method - Main function."""

from ensembling import ensembling_prediction
from estimators import KernelSVMEstimator
from kernels import MismatchKernel


if __name__ == '__main__':

    estimators1 = [
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=12, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=11, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=13, m=1)),
    ]

    estimators2 = [
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=10, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=7, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=9, m=1)),
    ]

    estimators3 = [
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=11, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=10, m=1)),
        KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=9, m=1)),
    ]

    # estimators1 = [
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=12, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=11, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=12, m=0)),
    # ]

    # estimators2 = [
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=10, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=7, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=10, m=0)),
    # ]

    # estimators3 = [
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=11, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=10, m=1)),
    #     KernelSVMEstimator(lbd=1e-6, kernel=MismatchKernel(k=11, m=0)),
    # ]

    csv_name = 'y_pred_3_mismatch_kernels_ensemble.csv'

    ensembling_prediction(
        estimators1=estimators1,
        estimators2=estimators2,
        estimators3=estimators3,
        csv_name=csv_name,
    )
