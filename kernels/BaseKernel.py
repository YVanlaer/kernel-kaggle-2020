"""Implement BaseKernel class."""
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    """Abstract class for creating kernels."""

    @abstractmethod
    def __call__(self, X1, X2):
        """Create a kernel matrix given inputs."""
        pass
