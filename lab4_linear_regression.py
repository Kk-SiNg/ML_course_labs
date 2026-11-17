"""
IC-272: Intro to ML Lab
Lab 4: Linear Regression from Scratch (Normal Equation, Gradient Descent & Diagnostics)
Rules: No sklearn, no np.polyfit, np.linalg.lstsq, or np.linalg.pinv.
Allowed: np.linalg.solve, np.linalg.inv.
RNG: np.random.default_rng(0).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# TASK 1: Simple Linear Regression (Closed Form)
# ==============================================================================

def train_test_split_scratch(X, y, test_frac=0.2, seed=0):
    """
    Splits data into train and test according to course specification:
    The FIRST int(n * test_frac) shuffled indices belong to the TEST set,
    and the remaining indices belong to the TRAINING set.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n)
    
    n_test = int(n * test_frac)
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]
    
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def fit_simple_lr(x: np.ndarray, y: np.ndarray):
    """
    Fits 1D simple linear regression: y = w*x + b using analytic formula:
      w = sum((x - x_bar) * (y - y_bar)) / sum((x - x_bar)^2)
      b = y_bar - w * x_bar
    """
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sum((x - x_bar) ** 2)
    
    w = numerator / denominator
    b = y_bar - w * x_bar
    return w, b


# ==============================================================================
# TASK 2: Multiple Linear Regression (Normal Equation)
# ==============================================================================

def design_matrix(X: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to an (n, d) feature matrix, returning (n, d + 1).
    Python Concept: np.c_[ones, X] concatenates along column axis.
    """
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves for theta in: (A.T @ A) @ theta = A.T @ y
    Uses np.linalg.solve instead of np.linalg.inv.
    
    WHY np.linalg.solve:
    1. Computing the explicit inverse (A.T @ A)^(-1) is numerically unstable and prone to rounding error.
    2. np.linalg.solve uses LU/Cholesky decomposition directly on (A.T @ A), which is both faster
       and guarantees lower numerical conditioning error.
    """
    A = design_matrix(X)
    theta = np.linalg.solve(A.T @ A, A.T @ y)
    return theta


# ==============================================================================
# TASK 3: Batch Gradient Descent
# ==============================================================================

def compute_loss(A: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Computes Mean Squared Error: J(theta) = (1 / n) * ||A*theta - y||^2"""
    n = len(y)
    residuals = A @ theta - y
    return float(np.sum(residuals ** 2) / n)


def fit_gradient_descent(X: np.ndarray, y: np.ndarray, lr: float, n_iters: int):
    """
    Performs batch gradient descent starting from theta = 0.
    Gradient: grad = (2 / n) * A.T @ (A @ theta - y)
    Update rule: theta = theta - lr * grad
    Returns: final theta, loss_history list
    """
    A = design_matrix(X)
    n, d = A.shape
    theta = np.zeros(d)
    loss_history = []
    
    for _ in range(n_iters):
        loss = compute_loss(A, y, theta)
        loss_history.append(loss)
        
        # Vectorized gradient calculation
        residuals = A @ theta - y
        grad = (2.0 / n) * (A.T @ residuals)
        
        theta = theta - lr * grad
        
    return theta, loss_history


# ==============================================================================
# TASK 4: Evaluation Metrics & Residual Diagnostics
# ==============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt((1 / m) * sum((y - y_hat)^2))"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R^2: 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot))


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plots residuals (y - y_pred) against predicted values y_pred.
    A well-fitted linear regression should show residuals randomly scattered around zero
    with no discernible curve, funnel (heteroscedasticity), or systematic pattern.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.7, color="purple", edgecolors="k")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Values (y_hat)")
    ax.set_ylabel("Residuals (y - y_hat)")
    ax.set_title("Residuals vs. Fitted Values")
    plt.tight_layout()
    plt.show()
