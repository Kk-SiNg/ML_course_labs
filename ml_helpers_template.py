"""
================================================================================
IC-272: INTRODUCTION TO MACHINE LEARNING — MASTER HELPER TEMPLATE
Indian Institute of Technology Mandi

This file contains EVERY reusable helper function required across Labs 1 through 9.
You can copy-paste the sections you need directly into the top cell of your 
Jupyter Notebook, or import this entire module via:
    from ml_helpers_template import *
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: LAB 1 REUSABLES — Vectorization, Distances & Scratch Metrics
# ==============================================================================

def column_stats(X: np.ndarray):
    """
    Computes mean, std, min, and max down each column of a 2D array.
    Returns: four 1D arrays of shape (d,). NO loops used.
    """
    return np.mean(X, axis=0), np.std(X, axis=0), np.min(X, axis=0), np.max(X, axis=0)


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorised Euclidean distance between all rows in A (n, d) and B (m, d).
    Returns: (n, m) matrix where D[i, j] = ||A[i] - B[j]||_2.
    Uses broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d). NO loops used.
    """
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Binary confusion matrix for labels in {0, 1}. Treat class 1 as positive.
    Orientation:
      Rows = True Class (Row 0 = True 0, Row 1 = True 1)
      Cols = Predicted Class (Col 0 = Pred 0, Col 1 = Pred 1)
      [[TN, FP],
       [FN, TP]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def binary_metrics(cm: np.ndarray):
    """
    Computes Precision, Recall, and F1 from a 2x2 confusion matrix.
    Safe against zero denominators (returns 0.0 instead of NaN/crash).
    """
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Generalises confusion matrix to K classes (K inferred from data).
    Returns: cm (K x K), classes, per-class precisions, recalls, f1s, and macro_f1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[c2i[t], c2i[p]] += 1
        
    precisions, recalls, f1s = [], [], []
    for c in range(K):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        
    return cm, classes, np.array(precisions), np.array(recalls), np.array(f1s), np.mean(f1s)


# ==============================================================================
# SECTION 2: LAB 2 & 4 REUSABLES — Splitting & Simple Linear Regression
# ==============================================================================

def train_test_split_scratch(X, y, test_frac=0.2, seed=0):
    """
    Shuffles row indices using np.random.default_rng(seed).permutation.
    Convention: First int(n * test_frac) shuffled rows are TEST, rest are TRAIN.
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
    Fits 1D linear regression y = w*x + b using analytic closed-form formulas:
      w = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
      b = y_bar - w * x_bar
    """
    x_bar, y_bar = np.mean(x), np.mean(y)
    w = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    b = y_bar - w * x_bar
    return w, b


# ==============================================================================
# SECTION 3: LAB 4 REUSABLES — Design Matrix, Normal Equation, GD & Metrics
# ==============================================================================

def design_matrix(X: np.ndarray) -> np.ndarray:
    """Prepends a column of ones to an (n, d) feature matrix -> returns (n, d + 1)."""
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves Normal Equation: theta = (A^T A)^(-1) A^T y using np.linalg.solve.
    np.linalg.solve avoids explicit matrix inversion, ensuring numerical stability.
    """
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ y)


def fit_gradient_descent(X: np.ndarray, y: np.ndarray, lr=0.005, n_iters=20000):
    """
    Fits multiple linear regression via Batch Gradient Descent.
    Gradient: grad = (2 / n) * A^T @ (A @ theta - y)
    Update: theta = theta - lr * grad
    """
    A = design_matrix(X)
    n, d = A.shape
    theta = np.zeros(d)
    loss_history = []
    
    for _ in range(n_iters):
        residuals = A @ theta - y
        loss_history.append(float(np.sum(residuals ** 2) / n))
        grad = (2.0 / n) * (A.T @ residuals)
        theta = theta - lr * grad
        
    return theta, loss_history


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt((1 / m) * sum((y - y_hat)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R^2: 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot))


# ==============================================================================
# SECTION 4: LAB 5 REUSABLES — Standardisation, Ridge & Lasso
# ==============================================================================

def standardise(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardises features: z = (x - mu) / sigma.
    CRITICAL (Leak-Free): mu and sigma are computed ONLY on the training set!
    Returns: Z_train, Z_test, mu, sigma
    """
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    
    Z_train = (X_train - mu) / sigma_safe
    Z_test = (X_test - mu) / sigma_safe
    return Z_train, Z_test, mu, sigma


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Fits Ridge Regression: theta = (A^T A + lambda * I*)^(-1) A^T y.
    INTERCEPT IS EXEMPT FROM PENALTY: I*[0, 0] = 0.0.
    """
    A = design_matrix(X)
    d = A.shape[1]
    penalty_mat = np.eye(d)
    penalty_mat[0, 0] = 0.0  # Turn off penalty for intercept!
    return np.linalg.solve(A.T @ A + lam * penalty_mat, A.T @ y)


def k_fold_cv(X: np.ndarray, y: np.ndarray, lam: float, k=5, seed=0) -> float:
    """5-Fold Cross Validation for selecting lambda in Ridge Regression."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    fold_rmses = []
    
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        Z_tr, Z_val, _, _ = standardise(X[train_idx], X[val_idx])
        theta = fit_ridge(Z_tr, y[train_idx], lam)
        preds = design_matrix(Z_val) @ theta
        fold_rmses.append(rmse(y[val_idx], preds))
        
    return float(np.mean(fold_rmses))


def soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    """Vectorised soft-thresholding operator S(v, t) = sign(v) * max(|v| - t, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def fit_lasso(X: np.ndarray, y: np.ndarray, lam: float, lr=0.05, n_iters=30000):
    """Fits Lasso (L1) regression via proximal gradient descent."""
    A = design_matrix(X)
    n, d = A.shape
    theta = np.zeros(d)
    
    for _ in range(n_iters):
        grad = (2.0 / n) * (A.T @ (A @ theta - y))
        theta_step = theta - lr * grad
        theta[0] = theta_step[0]  # Intercept exempt
        theta[1:] = soft_threshold(theta_step[1:], t=lr * lam)
        
    return theta


# ==============================================================================
# SECTION 5: LAB 6 REUSABLES — k-Nearest Neighbours & k-NN Cross Validation
# ==============================================================================

def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorised k-NN classifier for binary labels (no query row loops!).
    Uses pairwise_distances, np.argsort(D, axis=1)[:, :k], and majority voting.
    """
    D = pairwise_distances(X_query, X_train)
    nearest_idx = np.argsort(D, axis=1)[:, :k]
    votes = y_train[nearest_idx]
    return (np.mean(votes, axis=1) > 0.5).astype(int)


def k_fold_cv_knn(X: np.ndarray, y: np.ndarray, k_neighbors: int, folds=5, seed=0) -> float:
    """5-Fold Cross Validation for selecting optimal k in k-NN classification."""
    rng = np.random.default_rng(seed)
    fold_splits = np.array_split(rng.permutation(len(y)), folds)
    accuracies = []
    
    for i in range(folds):
        val_idx = fold_splits[i]
        train_idx = np.concatenate([fold_splits[j] for j in range(folds) if j != i])
        
        Z_tr, Z_val, _, _ = standardise(X[train_idx], X[val_idx])
        preds = knn_predict(Z_tr, y[train_idx], Z_val, k=k_neighbors)
        accuracies.append(np.mean(preds == y[val_idx]))
        
    return float(np.mean(accuracies))


def knn_predict_weighted(X_train, y_train, X_query, k, eps=1e-5):
    """Distance-weighted k-NN voting: weight = 1 / (d + eps)."""
    D = pairwise_distances(X_query, X_train)
    nearest_idx = np.argsort(D, axis=1)[:, :k]
    nearest_d = np.take_along_axis(D, nearest_idx, axis=1)
    weights = 1.0 / (nearest_d + eps)
    weighted_votes = np.sum(weights * y_train[nearest_idx], axis=1) / np.sum(weights, axis=1)
    return (weighted_votes > 0.5).astype(int)


# ==============================================================================
# SECTION 6: LAB 7 REUSABLES — Logistic Regression, ROC Curve & AUC
# ==============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable vectorised sigmoid: 1 / (1 + exp(-clip(z, -500, 500)))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_likelihood(X: np.ndarray, y: np.ndarray, theta: np.ndarray, eps=1e-12) -> float:
    """Total log-likelihood for logistic regression."""
    A = design_matrix(X)
    h = sigmoid(A @ theta)
    return float(np.sum(y * np.log(h + eps) + (1.0 - y) * np.log(1.0 - h + eps)))


def fit_logistic(X: np.ndarray, y: np.ndarray, lr=0.1, n_iters=2000):
    """
    Fits Logistic Regression via Gradient ASCENT (maximising log-likelihood):
      theta = theta + (lr / m) * A^T @ (y - sigmoid(A @ theta))
    """
    A = design_matrix(X)
    m, d = A.shape
    theta = np.zeros(d)
    ll_history = []
    
    for _ in range(n_iters):
        ll_history.append(log_likelihood(X, y, theta))
        h = sigmoid(A @ theta)
        # Note the PLUS sign: Gradient Ascent!
        theta = theta + (lr / m) * (A.T @ (y - h))
        
    return theta, ll_history


def roc_curve_scratch(y_true: np.ndarray, p: np.ndarray):
    """
    Sweeps sorted unique probabilities to compute TPR and FPR arrays.
    Returns: fpr_array, tpr_array sorted by increasing FPR.
    """
    thresholds = np.r_[1.1, np.sort(np.unique(p))[::-1], -0.1]
    tpr_list, fpr_list = [], []
    pos_total = np.sum(y_true == 1)
    neg_total = np.sum(y_true == 0)
    
    for t in thresholds:
        yp = (p >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        tpr_list.append(tp / pos_total if pos_total > 0 else 0.0)
        fpr_list.append(fp / neg_total if neg_total > 0 else 0.0)
        
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Computes Area Under ROC Curve via trapezoidal integration."""
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))


# ==============================================================================
# SECTION 7: LAB 8 REUSABLES — Gaussian & Categorical Naive Bayes
# ==============================================================================

def gaussian_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Gaussian PDF evaluated element-wise."""
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-((x - mu) ** 2) / (2.0 * var))
    return coeff * exponent


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray):
    """
    Fits Gaussian Naive Bayes: computes class priors, means, and MLE variances (divided by n_c).
    Returns: priors (2,), means (2, d), variances (2, d).
    """
    classes = [0, 1]
    priors = np.array([np.mean(y == c) for c in classes])
    means = np.array([np.mean(X[y == c], axis=0) for c in classes])
    variances = np.array([np.var(X[y == c], axis=0) for c in classes])
    return priors, means, variances


def predict_proba_nb(X: np.ndarray, priors: np.ndarray, means: np.ndarray, variances: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Log-space inference converting to probability via sigmoid(L1 - L0) to prevent numerical underflow.
    """
    L0 = np.log(priors[0]) + np.sum(np.log(gaussian_pdf(X, means[0], variances[0]) + eps), axis=1)
    L1 = np.log(priors[1]) + np.sum(np.log(gaussian_pdf(X, means[1], variances[1]) + eps), axis=1)
    return sigmoid(L1 - L0)


def fit_categorical_nb(emp: np.ndarray, y: np.ndarray, l: float, K=4):
    """Fits Categorical Naive Bayes with Laplace smoothing parameter l."""
    cats = sorted(np.unique(emp))
    prob_table = np.zeros((2, K))
    for c in [0, 1]:
        emp_c = emp[y == c]
        n_c = len(emp_c)
        for j, cat in enumerate(cats):
            count_jc = np.sum(emp_c == cat)
            prob_table[c, j] = (count_jc + l) / (n_c + l * K)
    return prob_table, cats


# ==============================================================================
# SECTION 8: LAB 9 REUSABLES — K-Means & K-Means++
# ==============================================================================

def random_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Selects K distinct points at random without replacement as initial centroids."""
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), K, replace=False)].copy()


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assigns each point in X to its nearest centroid index using pairwise_distances."""
    return np.argmin(pairwise_distances(X, centroids), axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Recomputes centroids as mean of assigned points. Retains old centroid if cluster is empty."""
    K = len(centroids)
    new_c = np.zeros_like(centroids)
    for k in range(K):
        mask = (labels == k)
        if np.any(mask):
            new_c[k] = np.mean(X[mask], axis=0)
        else:
            new_c[k] = centroids[k]
    return new_c


def distortion(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Distortion J: total sum of squared Euclidean distances to assigned centroids."""
    return float(np.sum((X - centroids[labels]) ** 2))


def kmeans(X: np.ndarray, init_centroids: np.ndarray, max_iter=100):
    """
    Alternates assign_clusters and update_centroids until np.allclose or max_iter.
    Returns: centroids, labels, distortion J, iteration_count
    """
    centroids = init_centroids.copy()
    for it in range(1, max_iter + 1):
        labels = assign_clusters(X, centroids)
        new_c = update_centroids(X, labels, centroids)
        if np.allclose(centroids, new_c):
            return new_c, labels, distortion(X, labels, new_c), it
        centroids = new_c
    return centroids, labels, distortion(X, labels, centroids), max_iter


def best_of_kmeans(X: np.ndarray, K: int, n_init=10):
    """Runs K-means with seeds 0 to n_init - 1, returning result with lowest distortion J."""
    best_J = float("inf")
    best_res = None
    for seed in range(n_init):
        init_c = random_init(X, K, seed)
        c, l, J, it = kmeans(X, init_c)
        if J < best_J:
            best_J = J
            best_res = (c, l, J, it)
    return best_res


def elbow_method(X: np.ndarray, max_k=10, n_init=10):
    """Computes best distortion for K in 1..max_k and returns (K_range, distortions)."""
    K_range = list(range(1, max_k + 1))
    distortions = []
    for k in K_range:
        _, _, J, _ = best_of_kmeans(X, k, n_init=n_init)
        distortions.append(J)
    return K_range, distortions


def kmeans_pp_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """K-Means++ initialization picking centroids with probability proportional to D(x)^2."""
    rng = np.random.default_rng(seed)
    centroids = [X[rng.choice(len(X))]]
    for _ in range(1, K):
        cur_c = np.array(centroids)
        min_d_sq = np.min(pairwise_distances(X, cur_c), axis=1) ** 2
        probs = min_d_sq / np.sum(min_d_sq)
        centroids.append(X[rng.choice(len(X), p=probs)])
    return np.array(centroids)
