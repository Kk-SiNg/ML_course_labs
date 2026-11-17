"""
IC-272: Intro to ML Lab
Lab 6: k-Nearest Neighbours & The Curse of Dimensionality
Rules: No scikit-learn, no scipy.spatial.
Vectorized implementation without query row loops.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab1_refresher import pairwise_distances
from lab5_ridge_regression import standardise

# ==============================================================================
# TASK 1: Vectorised k-NN Classifier (No loop over query rows!)
# ==============================================================================

def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    """
    Predicts binary labels for all rows in X_query simultaneously.
    
    Python Concept:
    1. pairwise_distances(X_query, X_train) returns matrix D of shape (n_query, n_train).
    2. np.argsort(D, axis=1)[:, :k] sorts distances along each query row and takes first k columns.
    3. y_train[nearest_indices] uses fancy 2D indexing to get neighbor labels of shape (n_query, k).
    4. np.mean(..., axis=1) > 0.5 computes majority vote.
    """
    D = pairwise_distances(X_query, X_train) # Shape: (n_query, n_train)
    
    # Indices of the k closest training points for each query point
    nearest_indices = np.argsort(D, axis=1)[:, :k] # Shape: (n_query, k)
    
    # Retrieve the labels of the k neighbors
    neighbor_votes = y_train[nearest_indices] # Shape: (n_query, k)
    
    # Binary classification: if average vote > 0.5, predict 1, else 0
    predictions = (np.mean(neighbor_votes, axis=1) > 0.5).astype(int)
    return predictions


# ==============================================================================
# TASK 3: Choosing k by Cross-Validation
# ==============================================================================

def k_fold_cv_knn(X: np.ndarray, y: np.ndarray, k_neighbors: int, folds=5, seed=0) -> float:
    """
    5-Fold Cross Validation for k-NN strictly on the training set.
    
    CAUTION:
    The validation fold must NOT be present in X_train passed to knn_predict!
    Otherwise, every query point finds itself at distance 0, giving artificially 100% accuracy.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    fold_splits = np.array_split(indices, folds)
    accuracies = []
    
    for i in range(folds):
        val_idx = fold_splits[i]
        train_idx = np.concatenate([fold_splits[j] for j in range(folds) if j != i])
        
        # Standardise training fold, apply to val fold
        Z_tr, Z_val, _, _ = standardise(X[train_idx], X[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]
        
        preds = knn_predict(Z_tr, y_tr, Z_val, k=k_neighbors)
        acc = np.mean(preds == y_val)
        accuracies.append(acc)
        
    return float(np.mean(accuracies))


# ==============================================================================
# HOMEWORK: Distance-Weighted Voting & Decision Boundary
# ==============================================================================

def knn_predict_weighted(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int, eps=1e-5) -> np.ndarray:
    """
    Distance-weighted k-NN voting: weight_i = 1 / (d_i + eps).
    Points closer to query have greater vote influence.
    """
    D = pairwise_distances(X_query, X_train)
    nearest_indices = np.argsort(D, axis=1)[:, :k]
    
    # Extract distances of nearest k neighbors
    # Python Concept: np.take_along_axis gathers values along axis 1 using index array
    nearest_dists = np.take_along_axis(D, nearest_indices, axis=1)
    
    weights = 1.0 / (nearest_dists + eps) # Shape: (n_query, k)
    neighbor_labels = y_train[nearest_indices] # Shape: (n_query, k)
    
    # Weighted average: sum(w * y) / sum(w)
    weighted_votes = np.sum(weights * neighbor_labels, axis=1) / np.sum(weights, axis=1)
    return (weighted_votes > 0.5).astype(int)


def plot_decision_boundary(X_train: np.ndarray, y_train: np.ndarray, k: int):
    """
    Plots the decision boundary in 2D using np.meshgrid and contourf/imshow.
    Shows overfitting at k=1 (jagged islands) vs generalization at larger k (smooth).
    """
    x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    preds = knn_predict(X_train, y_train, grid_points, k=k)
    preds = preds.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=25)
    ax.set_title(f"k-NN Decision Boundary (k={k})")
    ax.set_xlabel("Vibration X")
    ax.set_ylabel("Vibration Y")
    plt.tight_layout()
    plt.show()
