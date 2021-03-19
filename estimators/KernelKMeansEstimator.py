"""Implement the KernelSVMEstimator class."""
import numpy as np
import cvxpy as cp
import random as rd
from scipy.spatial import distance

from .BaseEstimator import BaseEstimator


class KernelKMeansEstimator(BaseEstimator):
    def __init__(self, nb_clusters, kernel, random_state=0):
        self.kernel = kernel
        self.number_clusters = nb_clusters
        self.random_state = random_state

        self.X = None
        self.labels = None

        self.centroids = None
        self.cluster_assignment = None

    def __repr__(self):
        return f"KernelKMeansEstimator(nb_clusters={self.number_clusters}, kernel={self.kernel}, random_state={self.random_state})"


    def initialize_cluster_assignment(self):
        n = self.X.shape[0]
        rd.seed(self.random_state)
        cluster_indices = rd.sample(range(n), self.number_clusters)
        self.centroids = self.X[cluster_indices]

        self.random_assign_clusters()
        self.assign_clusters()

    def random_assign_clusters(self):
        n = self.X.shape[0]
        
        cluster_assignment = []
        cluster = 0
        for i in range(n):
            cluster_assignment.append(cluster)
            cluster = (cluster + 1) % self.number_clusters
        self.cluster_assignment = cluster_assignment
        
    def update_centroids(self):
        for centroid_idx, _ in enumerate(self.centroids):
            points_to_consider = [point for id, point in enumerate(self.X) if self.cluster_assignment[id] == centroid_idx]
            points_to_consider = np.array(points_to_consider)
            new_centroid = np.mean(points_to_consider, axis=0)
            self.centroids[centroid_idx] = new_centroid

    def compute_distance(self, point_idx, centroid_idx):
        cluster = np.array([idx for idx, cluster_assigned in enumerate(self.cluster_assignment) if cluster_assigned == centroid_idx])
        cluster_size = cluster.shape[0]

        dist_point = self.K[point_idx, point_idx]
        dist_point_cluster = - 2 / cluster_size * np.sum(self.K[point_idx, cluster])
        dist_cluster_cluster = np.sum(self.K[cluster, cluster]) / np.power(cluster_size, 2)
        dist = dist_point + dist_point_cluster + dist_cluster_cluster
        
        return dist
        
    def assign_clusters(self):
        n = self.X.shape[0]
        cluster_assignment = []

        for idx in range(n):
            distances = [self.compute_distance(idx, centroid_idx) for centroid_idx in range(self.centroids.shape[0])]
            cluster_assigned = np.argmin(distances)
            cluster_assignment.append(cluster_assigned)

        self.cluster_assignment = cluster_assignment

    def fit(self, X):
        self.X = X.to_numpy()
        self.K = self.kernel(X.to_numpy(), X.to_numpy())
        self.initialize_cluster_assignment()

        old_cluster_assignment = None
        while old_cluster_assignment != self.cluster_assignment:
            old_cluster_assignment = self.cluster_assignment
            self.update_centroids()
            self.assign_clusters()
        
        self.labels = self.cluster_assignment

        return self

    def predict(self, X):
        K_test = self.kernel(self.X, X)
        y_pred = self.alpha@K_test
        y_pred = self._inverse_transform_labels(y_pred)
        return y_pred
