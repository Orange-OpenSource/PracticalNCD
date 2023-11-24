# Software Name : PracticalNCD
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

import numpy as np
import random
import torch
import math


class k_means_pp:
    def __init__(self, pre_centroids=None, k_new_centroids=10):
        super(k_means_pp, self).__init__()

        self.pre_centroids = pre_centroids
        self.new_centroids = None
        self.k_new_centroids = k_new_centroids
        self.inertia = math.inf

    def init_new_centroids(self, x_unlab):
        """
        Using k-means++ to initialize the centroids.
        Each new centroid is chosen from the remaining data points with a probability.
        proportional to its squared distance from the points closest cluster center.
        """
        if self.pre_centroids is None:
            self.new_centroids = x_unlab[random.randint(0, len(x_unlab)-1)].unsqueeze(0)  # If there are no initial cluster center, choose one randomly (uniform random)
        else:
            self.new_centroids = torch.tensor([], device=x_unlab.device)

        while len(self.new_centroids) < self.k_new_centroids:
            if self.pre_centroids is None:
                all_centroids = self.new_centroids
            elif len(self.new_centroids) == 0:
                all_centroids = self.pre_centroids
            else:
                all_centroids = torch.cat([self.pre_centroids, self.new_centroids], dim=0)

            dist = torch.cdist(x_unlab, all_centroids, p=2)  # Pairwise distance of the data to every cluster
            d2, _ = torch.min(dist, dim=1)  # Keep the distance to the closest cluster center only
            prob = d2 / d2.sum()  # Define the probability of being chosen
            cumul_prob = torch.cumsum(prob, dim=0)
            r = np.random.rand()
            ind = (cumul_prob >= r).nonzero()[0][0]
            self.new_centroids = torch.cat((self.new_centroids, x_unlab[ind].unsqueeze(0)), dim=0)

    def make_centroids_converge(self, centroids, x, tolerance=1e-10, n_iterations=1000):
        """
        Make the new centroids converge using the base k-means algorithm.
        Convergence will stop if we either reach n_iterations or if the shift is smaller than the tolerance.
        """
        inertia = math.inf

        for it in range(n_iterations):
            new_centroids_previous_position = centroids.clone()

            # For each unlabeled point, get the dist to the closest cluster and the cluster index
            dist = torch.cdist(x, centroids, p=2)
            min_dist, labels = torch.min(dist, dim=1)
            inertia = min_dist.sum()

            for idx in range(len(centroids)):
                if idx in labels:  # Update the centroid only if there were points close to it
                    centroids[idx] = x[labels == idx].mean(dim=0)

            center_shift = torch.sum(torch.sqrt(torch.sum((centroids - new_centroids_previous_position) ** 2, dim=1)))
            if center_shift ** 2 < tolerance:
                # Stop the converge if the centroids don't move much
                break

        return inertia, centroids

    def fit(self, x, tolerance=1e-10, n_iterations=1000, n_init=10, update_centroids="unlab_only"):
        """
        For n_init executions, initialize and converge the new centroids.
        We keep the centroids that achieved the smallest inertia.
        """
        best_inertia = math.inf
        best_new_centroids = None

        assert update_centroids in ["unlab_only", "lab_only", "unlab_and_lab"], f"Unsupported value for update_centroids: {update_centroids}"

        for init in range(n_init):
            self.init_new_centroids(x)

            if update_centroids == "unlab_only":
                inertia, centroids = self.make_centroids_converge(self.new_centroids, x, tolerance=tolerance, n_iterations=n_iterations)
            elif update_centroids == "lab_only":
                inertia, centroids = self.make_centroids_converge(self.pre_centroids, x, tolerance=tolerance, n_iterations=n_iterations)
            elif update_centroids == "unlab_and_lab":
                inertia, centroids = self.make_centroids_converge(torch.cat([self.pre_centroids, self.new_centroids], dim=0), x, tolerance=tolerance, n_iterations=n_iterations)

            if inertia < best_inertia:
                best_inertia = inertia
                best_new_centroids = centroids

        # Take the centroids of the iteration that achieved the best inertia
        self.inertia = best_inertia
        if update_centroids == "unlab_only":
            self.new_centroids = best_new_centroids
        elif update_centroids == "lab_only":
            self.pre_centroids = best_new_centroids
        elif update_centroids == "unlab_and_lab":
            self.pre_centroids = best_new_centroids[:len(self.pre_centroids)]
            self.new_centroids = best_new_centroids[len(self.pre_centroids):]

    def predict_known_data(self, x_known):
        l_dist = torch.cdist(x_known, self.pre_centroids, p=2)
        l_mindist, l_labels = torch.min(l_dist, dim=1)
        return l_labels

    def predict_unknown_data(self, x_unknown):
        u_dist = torch.cdist(x_unknown, self.new_centroids, p=2)
        u_mindist, u_labels = torch.min(u_dist, dim=1)
        return u_labels
