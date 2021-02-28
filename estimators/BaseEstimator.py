"""Implement the BaseEstimator class."""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class BaseEstimator(ABC, BaseEstimator):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
