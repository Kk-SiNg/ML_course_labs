"""
IC-272: Intro to ML Lab
Lab 9: Clustering with K-Means (Unsupervised Learning & K-Means++)
Rules: No scikit-learn, no scipy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab1_refresher import pairwise_distances

# ==============================================================================
# SETUP: Reproducible Initialization
# ==============================================================================

def random_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Selects K distinct points at random as starting centroids."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), K, replace=False)
    return X[idx].copy()


# ==============================================================================
# TASK 1: K-Means from Scratch
# ==============================================================================

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assigns each point in X to its nearest centroid.
    Uses pairwise_distances(X, centroids) of shape (n_samples, K).
    Returns: 1D array of cluster indices (shape: (n,))
    """
    D = pairwise_distances(X, centroids)
    labels = np.argmin(D, axis=1)
    return labels


def update_centroids(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Recomputes centroids as the mean of their assigned data points.
    If a cluster becomes empty, it retains its previous centroid location.
    """
    K = len(centroids)
    new_centroids = np.zeros_like(centroids)
    
    for k in range(K):
        cluster_mask = (labels == k)
        if np.any(cluster_mask):
            new_centroids[k] = np.mean(X[cluster_mask], axis=0)
        else:
            # If empty, keep old centroid to avoid NaN
            new_centroids[k] = centroids[k]
            
    return new_centroids


def distortion(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Computes total distortion J = sum_{i=1}^m ||x^(i) - mu_{c^(i)}||^2.
    The sum of squared Euclidean distances of each point to its assigned centroid.
    """
    assigned_centroids = centroids[labels] # Fancy indexing: shape (n, d)
    squared_dists = np.sum((X - assigned_centroids) ** 2, axis=1)
    return float(np.sum(squared_dists))


def kmeans(X: np.ndarray, init_centroids: np.ndarray, max_iter=100):
    """
    Alternates assignment and centroid update until convergence (np.allclose) or max_iter.
    Returns: final_centroids, final_labels, final_J, n_iterations
    """
    centroids = init_centroids.copy()
    
    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)
        
        # Convergence check: stop if centroids do not change
        if np.allclose(centroids, new_centroids):
            J = distortion(X, labels, new_centroids)
            return new_centroids, labels, J, iteration
            
        centroids = new_centroids
        
    J = distortion(X, labels, centroids)
    return centroids, labels, J, max_iter


# ==============================================================================
# TASK 3: Best of K-Means & The Elbow Method
# ==============================================================================

def best_of_kmeans(X: np.ndarray, K: int, n_init=10):
    """
    Runs K-means with n_init distinct random seeds (0 to n_init - 1).
    Returns the result that achieves the lowest distortion J.
    """
    best_J = float("inf")
    best_res = None
    
    for seed in range(n_init):
        init_c = random_init(X, K, seed)
        centroids, labels, J, iters = kmeans(X, init_c)
        if J < best_J:
            best_J = J
            best_res = (centroids, labels, J, iters)
            
    return best_res


def elbow_method(X: np.ndarray, max_k=10, n_init=10):
    """
    Computes best distortion J for K = 1, 2, ..., max_k.
    Plots J against K to identify the 'elbow' point.
    """
    K_range = list(range(1, max_k + 1))
    distortions = []
    
    for k in K_range:
        _, _, J, _ = best_of_kmeans(X, k, n_init=n_init)
        distortions.append(J)
        
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_range, distortions, marker="o", color="royalblue", linewidth=2)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Distortion J (SSE)")
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xticks(K_range)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return K_range, distortions


# ==============================================================================
# HOMEWORK H1: K-Means++ Initialization
# ==============================================================================

def kmeans_pp_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """
    K-Means++ smart centroid initialization:
    1. First centroid chosen uniformly at random.
    2. Each subsequent centroid chosen with probability proportional to D(x)^2,
       the squared distance from x to its closest centroid chosen so far.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    
    # First centroid chosen uniformly
    first_idx = rng.choice(n)
    centroids = [X[first_idx]]
    
    for _ in range(1, K):
        cur_centroids = np.array(centroids)
        # Distances from all points to chosen centroids: shape (n, len(centroids))
        dists = pairwise_distances(X, cur_centroids)
        # Shortest distance to any chosen centroid
        min_dists_sq = np.min(dists, axis=1) ** 2
        
        # Probabilities proportional to D(x)^2
        prob = min_dists_sq / np.sum(min_dists_sq)
        
        next_idx = rng.choice(n, p=prob)
        centroids.append(X[next_idx])
        
    return np.array(centroids)
