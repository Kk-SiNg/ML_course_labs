"""
IC-272: Intro to ML Lab
Lab 5: Standardisation and Ridge Regression (L2 Regularization & Cross-Validation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab4_linear_regression import design_matrix, rmse

# ==============================================================================
# TASK 1: Standardisation (Leak-Free)
# ==============================================================================

def standardise(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardises features: z = (x - mu) / sigma.
    CRITICAL: mu and sigma are computed ONLY on the training set!
    The exact same mu and sigma are applied to the test set to prevent data leakage.
    Returns: Z_train, Z_test, mu, sigma
    """
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    
    # Avoid zero division if any constant column exists
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    
    Z_train = (X_train - mu) / sigma_safe
    Z_test = (X_test - mu) / sigma_safe
    
    return Z_train, Z_test, mu, sigma


# ==============================================================================
# TASK 2: Ridge Regression (Normal Equation with Intercept Exemption)
# ==============================================================================

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solves Ridge Regression: theta = (A^T A + lambda * I*)^(-1) A^T y
    
    IMPORTANT:
    The intercept must NOT be penalised! Penalising the intercept pulls predictions
    towards zero regardless of the target's baseline scale.
    We set I*[0, 0] = 0.0.
    """
    A = design_matrix(X)
    d = A.shape[1]
    
    # Create penalty matrix: identity matrix of size (d, d)
    penalty_matrix = np.eye(d)
    penalty_matrix[0, 0] = 0.0 # Switch off penalty for intercept!
    
    theta = np.linalg.solve(A.T @ A + lam * penalty_matrix, A.T @ y)
    return theta


# ==============================================================================
# TASK 3: Choosing Lambda via K-Fold Cross Validation
# ==============================================================================

def k_fold_cv(X: np.ndarray, y: np.ndarray, lam: float, k=5, seed=0) -> float:
    """
    Performs k-fold cross validation strictly on the training set rows.
    Returns: average validation RMSE across the k folds.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    # Python Concept: np.array_split divides indices into k chunks with balanced sizes
    folds = np.array_split(indices, k)
    fold_rmses = []
    
    for i in range(k):
        val_idx = folds[i]
        # Concatenate remaining folds for training
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Standardise fold train, apply fold mu and sigma to fold val
        Z_tr, Z_val, _, _ = standardise(X[train_idx], X[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Fit ridge and evaluate
        theta = fit_ridge(Z_tr, y_tr, lam)
        A_val = design_matrix(Z_val)
        y_val_pred = A_val @ theta
        
        fold_rmses.append(rmse(y_val, y_val_pred))
        
    return float(np.mean(fold_rmses))


# ==============================================================================
# HOMEWORK: Recovering Units, Ridge Path & Lasso with Soft-Thresholding
# ==============================================================================

def recover_original_units(theta_std: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    Converts coefficients learned on standardised features back to the original units:
      w_j = theta_j / sigma_j
      b   = theta_0 - sum_j (theta_j * mu_j / sigma_j)
    """
    theta_0 = theta_std[0]
    theta_j = theta_std[1:]
    
    w = theta_j / sigma
    b = theta_0 - np.sum(theta_j * mu / sigma)
    return np.r_[b, w]


def soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    """
    Vectorised soft-thresholding operator S(v, t) = sign(v) * max(|v| - t, 0).
    Used in Lasso (L1 regularization) to set small coefficients to EXACT zero.
    """
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def fit_lasso(X: np.ndarray, y: np.ndarray, lam: float, lr=0.05, n_iters=30000):
    """
    Fits Lasso Regression (L1 regularization) using proximal gradient descent:
    1. Take ordinary gradient step on MSE loss
    2. Apply soft-thresholding to all feature weights (EXCEPT intercept)
    """
    A = design_matrix(X)
    n, d = A.shape
    theta = np.zeros(d)
    
    for _ in range(n_iters):
        # 1. Gradient step on squared error: grad = (2 / n) * A.T @ (A @ theta - y)
        residuals = A @ theta - y
        grad = (2.0 / n) * (A.T @ residuals)
        theta_intermediate = theta - lr * grad
        
        # 2. Soft-threshold feature weights (index 1 to d-1), keeping intercept unregularized
        theta[0] = theta_intermediate[0]
        theta[1:] = soft_threshold(theta_intermediate[1:], t=lr * lam)
        
    return theta
