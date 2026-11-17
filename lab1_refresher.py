"""
IC-272: Intro to ML Lab
Lab 1: NumPy and Pandas Refresher & Classification Metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report

# ==============================================================================
# BLOCK 1: NumPy Arrays and Vectorization
# ==============================================================================

# Task 1.1: Column Summary (No loops allowed)
# Python Concept: axis=0 collapses rows, computing the statistic down each column.
def column_stats(X: np.ndarray):
    """
    Computes mean, standard deviation, minimum, and maximum of each column of a 2D array.
    Returns: four 1-D arrays of shape (d,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    return mean, std, col_min, col_max


# Task 1.2: Masking & Row Counts
# Python Concept: X.copy() prevents mutating original X. 
# Boolean condition X > t creates a boolean array of same shape.
def masking_tasks(X: np.ndarray, t: float):
    """
    1) Returns a copy of X where any element > t is clamped to t.
    2) Returns a 1-D array containing the count of elements > t for each row.
    """
    # Clamping without modifying original X
    X_clipped = np.where(X > t, t, X.copy())
    
    # (X > t) gives True (1) or False (0). sum(axis=1) sums across columns -> per row count
    row_counts = np.sum(X > t, axis=1)
    
    return X_clipped, row_counts


# Task 1.3: Pairwise Distances via Broadcasting (No loops allowed)
# Python Concept: 
# A has shape (n, d) -> A[:, np.newaxis, :] has shape (n, 1, d)
# B has shape (m, d) -> B[np.newaxis, :, :] has shape (1, m, d)
# Subtraction broadcasts to shape (n, m, d) representing all pairs (i, j, k)
def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance between all pairs of rows in A (n, d) and B (m, d).
    Returns: (n, m) matrix D where D[i, j] = sqrt(sum_k (A[i, k] - B[j, k])^2)
    """
    # (n, 1, d) - (1, m, d) -> (n, m, d)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    # Sum squares along the feature dimension (axis=2), then square root
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


# Task 1.4: Stretch - Closest row in B for each row in A
# Python Concept: argmin along axis=1 finds the column index of the minimum value for each row
def closest_indices(D: np.ndarray) -> np.ndarray:
    """
    Given distance matrix D of shape (n, m), returns (n,) array with the index
    of the closest row in B for each row in A.
    """
    return np.argmin(D, axis=1)


# ==============================================================================
# BLOCK 2: Classification Metrics from Scratch
# ==============================================================================

# Task 2.1: Binary Confusion Matrix
# Row = True Label (0, 1), Column = Predicted Label (0, 1)
# Row 0: True 0 -> [TN (pred 0), FP (pred 1)]
# Row 1: True 1 -> [FN (pred 0), TP (pred 1)]
def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Constructs a 2x2 confusion matrix:
      Orientation:
      Rows: True Class (Row 0 = True 0, Row 1 = True 1)
      Cols: Pred Class (Col 0 = Pred 0, Col 1 = Pred 1)
      [[TN, FP],
       [FN, TP]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Verify sum equals total samples
    assert (tp + fp + fn + tn) == len(y_true), "Counts do not sum to sample size!"
    
    return np.array([[tn, fp],
                     [fn, tp]], dtype=int)


# Task 2.2: Precision, Recall, F1 with Zero-Division Handling
# Python Concept: use simple conditionals or ternary expressions to prevent ZeroDivisionError/NaN
def binary_metrics(cm: np.ndarray):
    """
    Given binary confusion matrix [[TN, FP], [FN, TP]]:
    Computes precision, recall, and F1. Returns 0.0 if denominator is 0.
    """
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


# Task 2.3: Multiclass Confusion Matrix & Macro-F1
# Python Concept: np.unique finds all classes automatically.
# Class c TP is cm[c, c]; FP is col sum minus TP; FN is row sum minus TP.
def multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes K x K confusion matrix where K is inferred from unique classes.
    Rows = True Class, Columns = Predicted Class.
    Also computes per-class precision, recall, F1, and Macro-F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
        
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(K):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    macro_f1 = np.mean(f1s)
    return cm, classes, np.array(precisions), np.array(recalls), np.array(f1s), macro_f1


# Verification Helper
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Test Block 1
    X = rng.normal(size=(10, 4))
    m, s, mn, mx = column_stats(X)
    print("Column Stats Mean shape:", m.shape)
    
    A = rng.normal(size=(5, 3))
    B = rng.normal(size=(4, 3))
    D = pairwise_distances(A, B)
    # Naive verification
    D_naive = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            D_naive[i, j] = np.linalg.norm(A[i] - B[j])
    assert np.allclose(D, D_naive), "Pairwise distances mismatch!"
    print("Pairwise distances verified against double-loop successfully.")
    
    # Test Block 2
    y_t = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_p = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    cm = binary_confusion_matrix(y_t, y_p)
    sk_cm = sk_confusion_matrix(y_t, y_p)
    assert np.array_equal(cm, sk_cm), "Binary confusion matrix mismatch with sklearn!"
    print("Binary CM matches sklearn:\n", cm)
