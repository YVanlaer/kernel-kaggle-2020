"""Implement BaseKernel class."""
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    """Abstract class for creating kernels."""

    @abstractmethod
    def __call__(self, X1, X2):
        """Create a kernel matrix given inputs."""
        pass

    def __repr__(self):
        d = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({d})'
