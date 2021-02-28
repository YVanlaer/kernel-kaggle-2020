"""Implement the BaseEstimator class."""
from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
