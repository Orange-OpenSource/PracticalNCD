# Software Name : PracticalNCD
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import numpy as np
import torch
import math

from models.NCD_Kmeans import k_means_pp


def batch_outer_difference_normed(x, batch_size=10):
    """
    Returns the 2-norm of all pairs of values in x.
    i.e. ||x_i-x_j||_2^2
    """
    diff = torch.empty((x.shape[0], x.shape[0]), device=x.device, dtype=torch.float32)

    n_batchs = math.ceil((x.shape[0]) / batch_size)
    start_index, end_index = 0, min(batch_size, x.shape[0])

    for batch_index in range(n_batchs):
        chunk = x[start_index:end_index][:, None] - x[None, :]
        diff[start_index:end_index] = torch.linalg.norm(chunk, dim=2, ord=2)

        start_index += batch_size
        end_index = min((end_index + batch_size), x.shape[0])

    return diff


def estimate_sigma(outer_diff, min_dist=0.5):
    """
    The largest difference in the minimum spanning tree must become min_dist after applying the gaussian kernel.
    Note: This method is affected by outliers. If there is an outlier, we can find a sigma that is too large.
    """
    A = outer_diff ** 2

    Tcsr = minimum_spanning_tree(A.cpu().numpy())

    max_squared_distance = Tcsr[Tcsr > 0].max()

    sigma = math.sqrt(- max_squared_distance / math.log(min_dist))

    return sigma


def get_adjacency_matrix(outer_diff, sigma):
    A = torch.exp(- outer_diff ** 2 / (2 * (sigma ** 2)))  # RBF kernel
    A = A - torch.eye(A.shape[0], device=outer_diff.device)  # The diagonal is currently 1s (exp(0)=1) so we set it to 0s
    return A


def get_laplacian(A, normed=True):
    m = A.clone()

    m.fill_diagonal_(0)

    w = m.sum(axis=0)

    if normed:
        isolated_node_mask = (w == 0)
        w = torch.where(isolated_node_mask, 1, w.sqrt())
        m /= w
        m /= w.unsqueeze(dim=-1)
        m *= -1
        m.fill_diagonal_(1)
    else:
        m *= -1
        m[range(len(m)), range(len(m))] = w

    return m


def get_spectral_embedding(x, n_components, min_dist, normed_laplacian=True):
    outer_diff = batch_outer_difference_normed(x)

    sigma = estimate_sigma(outer_diff, min_dist)

    A = get_adjacency_matrix(outer_diff, sigma)

    L = get_laplacian(A, normed=normed_laplacian)

    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    eigenvectors = eigenvectors[:, 1:n_components+1]

    return eigenvectors


class ncd_spectral_clustering:
    def __init__(self, n_new_clusters, n_components=None, n_init=10, min_dist=0.5, batch_size=10,
                 assign_labels="ncd_kmeans", normed_laplacian=True):
        super(ncd_spectral_clustering, self).__init__()
        """
        n_components: int, default=None
            Number of eigenvectors to use for the spectral embedding. If None, defaults to n_clusters.

        n_init: int, default=10
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.
            Only used if assign_labels='kmeans'.

        min_dist: int, default=0.5.
            Smallest distance in the minimum spanning tree (mst).
            This value is used to automatically estimate sigma in the RBF kernel.
            Having a minimum distance in the mst assures us that the tree will be fully connected.

        batch_size: int, default=10
            Batch size used in the computation of the outer difference of the data.
            Please reduce if you ecounter memory errors.
        """
        self.n_components = n_components
        self.n_new_clusters = n_new_clusters
        self.n_init = n_init
        self.min_dist = min_dist
        self.batch_size = batch_size
        self.assign_labels = assign_labels
        self.sigma = None
        self.normed_laplacian = normed_laplacian

    def fit_predict_simple(self, x):
        if self.n_components is None:
            self.n_components = self.n_new_clusters

        outer_diff = batch_outer_difference_normed(x, self.batch_size)

        self.sigma = estimate_sigma(outer_diff, self.min_dist)

        A = get_adjacency_matrix(outer_diff, self.sigma)

        L = get_laplacian(A, normed=self.normed_laplacian)

        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        eigenvectors = eigenvectors[:, 1:self.n_components + 1]

        assert eigenvectors.shape[0] == len(x)
        assert eigenvectors.shape[1] == self.n_components

        if self.assign_labels == "kmeans":
            km = KMeans(n_clusters=self.n_new_clusters, init='k-means++', n_init=10)
            km.fit(eigenvectors.cpu().numpy())
            return km.predict(eigenvectors.cpu().numpy())
        if self.assign_labels == "ncd_kmeans":
            kmpp = k_means_pp(pre_centroids=None, k_new_centroids=self.n_new_clusters)
            kmpp.fit(eigenvectors, tolerance=1e-10, n_iterations=1000, n_init=self.n_init)
            return kmpp.predict_unknown_data(eigenvectors).cpu().numpy()

    def fit_predict_known_and_unknown(self, x_unknown, x_known, y_known):
        if self.n_components is None:
            self.n_components = self.n_new_clusters + len(np.unique(y_known))

        x_full = torch.concat([x_unknown, x_known], axis=0)

        outer_diff = batch_outer_difference_normed(x_full, self.batch_size)

        self.sigma = estimate_sigma(outer_diff, self.min_dist)

        A = get_adjacency_matrix(outer_diff, self.sigma)

        L = get_laplacian(A, normed=self.normed_laplacian)

        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        eigenvectors = eigenvectors[:, 1:self.n_components + 1]

        assert eigenvectors.shape[0] == len(x_full)
        assert eigenvectors.shape[1] == self.n_components

        eigenvectors_known = eigenvectors[len(x_unknown):]
        eigenvectors_unknown = eigenvectors[:len(x_unknown)]

        # Known classes' centroids in the spectral embedding
        pre_centroids = torch.stack([eigenvectors_known[y_known == c].mean(axis=0) for c in np.unique(y_known)])

        kmpp = k_means_pp(pre_centroids=pre_centroids, k_new_centroids=self.n_new_clusters)
        kmpp.fit(eigenvectors_unknown, tolerance=1e-10, n_iterations=1000, n_init=self.n_init)
        return kmpp.predict_unknown_data(eigenvectors_unknown).cpu().numpy()
