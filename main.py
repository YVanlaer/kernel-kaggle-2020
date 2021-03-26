"""Run KRR using SumKernels for 2 Mismatch kernels per dataset."""

#############################################################################
#                                                                           #
#   Script reproducing our best result for the Kernel Kaggle Challenge.     #
#                                                                           #
#############################################################################

from kernels import MismatchKernel, SumKernels
from predict import predict


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

predict(lambdas, kernels, "krr")
