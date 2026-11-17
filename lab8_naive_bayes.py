"""
IC-272: Intro to ML Lab
Lab 8: Gaussian & Categorical Naive Bayes (Log-Space Inference & Laplace Smoothing)
Rules: No scikit-learn, no scipy.stats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab7_logistic_regression import sigmoid, roc_curve_scratch, auc_trapezoid
from lab1_refresher import binary_confusion_matrix, binary_metrics

# ==============================================================================
# TASK 1: Gaussian Density & Fitting
# ==============================================================================

def gaussian_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Vectorised Gaussian (Normal) probability density function:
      P(x | mu, var) = 1 / sqrt(2 * pi * var) * exp( - (x - mu)^2 / (2 * var) )
    Works across 2D arrays (samples x features).
    """
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-((x - mu) ** 2) / (2.0 * var))
    return coeff * exponent


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray):
    """
    Fits Gaussian Naive Bayes:
    - Priors: P(y = c) = n_c / N
    - Means: mu_c = mean(X[y == c])
    - Variances: var_c = var(X[y == c]), divided by n_c (MLE, not n_c - 1)
    
    Returns:
      priors: (2,)
      means: (2, d)
      variances: (2, d)
    """
    classes = [0, 1]
    priors = np.array([np.mean(y == c) for c in classes])
    means = np.array([np.mean(X[y == c], axis=0) for c in classes])
    variances = np.array([np.var(X[y == c], axis=0) for c in classes])
    
    return priors, means, variances


# ==============================================================================
# TASK 2: Prediction in Log-Space (Underflow-Proof)
# ==============================================================================

def predict_proba_nb(X: np.ndarray, priors: np.ndarray, means: np.ndarray, variances: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Computes P(y = 1 | x) for all rows using log-likelihood to avoid floating-point underflow.
    
    Python Concept & Derivation:
      L_c = log P(y = c) + sum_alpha log P(x_alpha | y = c)
      P(y = 1 | x) = exp(L1) / (exp(L0) + exp(L1))
                   = 1 / (1 + exp(-(L1 - L0)))
                   = sigmoid(L1 - L0)
    This entirely prevents underflow to 0.0 without calculating large exponents!
    """
    # For class 0
    pdf_0 = gaussian_pdf(X, means[0], variances[0])
    L0 = np.log(priors[0]) + np.sum(np.log(pdf_0 + eps), axis=1)
    
    # For class 1
    pdf_1 = gaussian_pdf(X, means[1], variances[1])
    L1 = np.log(priors[1]) + np.sum(np.log(pdf_1 + eps), axis=1)
    
    # Stable probability via sigmoid of difference
    prob_class_1 = sigmoid(L1 - L0)
    return prob_class_1


# ==============================================================================
# HOMEWORK H1 & H2: Categorical Naive Bayes & Laplace Smoothing
# ==============================================================================

def fit_categorical_nb(emp: np.ndarray, y: np.ndarray, l: float, K=4):
    """
    Fits Categorical Naive Bayes with Laplace smoothing parameter l:
      theta_jc = (# of samples in class c with category j + l) / (n_c + l * K)
    
    When l = 0 (no smoothing), any unseen category in class c has probability 0,
    wiping out all other feature evidence by multiplying/adding -inf!
    When l = 1 (Laplace smoothing), every category gets at least a small non-zero probability.
    """
    categories = sorted(np.unique(emp))
    assert len(categories) == K, f"Expected {K} categories, found {len(categories)}"
    
    # Table of shape (2, K): Row 0 = Class 0, Row 1 = Class 1
    prob_table = np.zeros((2, K))
    
    for c in [0, 1]:
        mask_c = (y == c)
        n_c = np.sum(mask_c)
        emp_c = emp[mask_c]
        
        for j, cat in enumerate(categories):
            count_jc = np.sum(emp_c == cat)
            prob_table[c, j] = (count_jc + l) / (n_c + l * K)
            
    return prob_table, categories


def predict_proba_combined_nb(X: np.ndarray, emp: np.ndarray, priors: np.ndarray, 
                               means: np.ndarray, vars_: np.ndarray, 
                               cat_probs: np.ndarray, categories: list, eps=1e-12):
    """
    Combines Gaussian continuous features and Categorical discrete features in log-space:
      L_c = log(prior_c) + log P(continuous | c) + log P(categorical | c)
    """
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    emp_indices = np.array([cat_to_idx[e] for e in emp])
    
    # Continuous log-likelihoods
    L0_num = np.log(priors[0]) + np.sum(np.log(gaussian_pdf(X, means[0], vars_[0]) + eps), axis=1)
    L1_num = np.log(priors[1]) + np.sum(np.log(gaussian_pdf(X, means[1], vars_[1]) + eps), axis=1)
    
    # Categorical log-likelihoods
    L0_cat = np.log(cat_probs[0, emp_indices] + eps)
    L1_cat = np.log(cat_probs[1, emp_indices] + eps)
    
    L0 = L0_num + L0_cat
    L1 = L1_num + L1_cat
    
    return sigmoid(L1 - L0)
