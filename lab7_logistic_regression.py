"""
IC-272: Intro to ML Lab
Lab 7: Logistic Regression from Scratch (Sigmoid, Maximum Likelihood & ROC/AUC)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab4_linear_regression import design_matrix
from lab5_ridge_regression import standardise
from lab1_refresher import binary_confusion_matrix, binary_metrics

# ==============================================================================
# TASK 1: Sigmoid and Log-Likelihood
# ==============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Vectorised sigmoid activation function: sigma(z) = 1 / (1 + exp(-z)).
    Numerically clipped to prevent overflow in exp(-z).
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def log_likelihood(X: np.ndarray, y: np.ndarray, theta: np.ndarray, eps=1e-12) -> float:
    """
    Computes total log-likelihood:
      ell(theta) = sum [ y * log(h + eps) + (1 - y) * log(1 - h + eps) ]
    where h = sigmoid(A @ theta).
    """
    A = design_matrix(X)
    h = sigmoid(A @ theta)
    
    # eps prevents log(0) = -inf
    ll = np.sum(y * np.log(h + eps) + (1.0 - y) * np.log(1.0 - h + eps))
    return float(ll)


# ==============================================================================
# TASK 2: Fitting by Gradient Ascent
# ==============================================================================

def fit_logistic(X: np.ndarray, y: np.ndarray, lr=0.1, n_iters=2000):
    """
    Fits logistic regression via Gradient Ascent (maximising likelihood):
      theta = theta + (lr / m) * A.T @ (y - sigmoid(A @ theta))
    Returns: final theta, list of log-likelihoods
    """
    A = design_matrix(X)
    m, d = A.shape
    theta = np.zeros(d)
    ll_history = []
    
    for _ in range(n_iters):
        ll = log_likelihood(X, y, theta)
        ll_history.append(ll)
        
        # Predicted probability
        h = sigmoid(A @ theta)
        
        # Gradient of log-likelihood: (1 / m) * A.T @ (y - h)
        grad = (1.0 / m) * (A.T @ (y - h))
        
        # Gradient ASCENT (PLUS SIGN!)
        theta = theta + lr * grad
        
    return theta, ll_history


# ==============================================================================
# TASK 3 & HOMEWORK H1: Thresholding, ROC Curve & Trapezoidal AUC
# ==============================================================================

def roc_curve_scratch(y_true: np.ndarray, p: np.ndarray):
    """
    Sweeps through sorted predicted probabilities as thresholds to compute:
      TPR = TP / (TP + FN)  (Recall)
      FPR = FP / (FP + TN)
    Returns: fpr_array, tpr_array sorted by increasing FPR.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    
    # Sort distinct probabilities in descending order
    thresholds = np.r_[1.1, np.sort(np.unique(p))[::-1], -0.1]
    
    fpr_list = []
    tpr_list = []
    
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        tpr = tp / total_pos if total_pos > 0 else 0.0
        fpr = fp / total_neg if total_neg > 0 else 0.0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        
    # Ensure points are ordered by increasing FPR for proper trapezoidal integration
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    order = np.argsort(fpr_arr)
    return fpr_arr[order], tpr_arr[order]


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Computes Area Under the ROC Curve using the trapezoidal rule:
      AUC = sum_i (FPR_{i+1} - FPR_i) * ((TPR_{i+1} + TPR_i) / 2)
    """
    diff_fpr = np.diff(fpr)
    avg_tpr = (tpr[:-1] + tpr[1:]) / 2.0
    return float(np.sum(diff_fpr * avg_tpr))


# ==============================================================================
# HOMEWORK H2: Decision Boundary Line Calculation
# ==============================================================================

def get_decision_boundary_line(theta: np.ndarray, mu: np.ndarray, sigma: np.ndarray, x1_vals: np.ndarray):
    """
    In standardised space: theta_0 + theta_1 * z1 + theta_2 * z2 = 0
    where z1 = (x1 - mu1) / sigma1, z2 = (x2 - mu2) / sigma2.
    
    Rearranging for unstandardised x2 (utilisation) as a function of x1 (income):
      w1 = theta_1 / sigma_1
      w2 = theta_2 / sigma_2
      b  = theta_0 - (theta_1 * mu1 / sigma1 + theta_2 * mu2 / sigma2)
      w1 * x1 + w2 * x2 + b = 0  =>  x2 = -(w1 * x1 + b) / w2
    """
    w1 = theta[1] / sigma[0]
    w2 = theta[2] / sigma[1]
    b = theta[0] - (theta[1] * mu[0] / sigma[0] + theta[2] * mu[1] / sigma[1])
    
    x2_vals = -(w1 * x1_vals + b) / w2
    return x2_vals
