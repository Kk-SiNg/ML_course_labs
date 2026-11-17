"""
IC-272: Intro to ML Lab
Lab 3: Visualization (Matplotlib Deep-Dive & Seeing Failure Modes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lab1_refresher import column_stats, pairwise_distances, multiclass_confusion_matrix, binary_confusion_matrix, binary_metrics

# ==============================================================================
# BLOCK 1: Visualizing Distributions, Distance Matrices, and Confusion Matrices
# ==============================================================================

def plot_five_distributions(X: np.ndarray, feature_names=None):
    """
    Plots a 2x3 grid of histograms, drawing mean (solid) and ±1 std (dashed) lines.
    Python Concept: fig, axes = plt.subplots(nrows, ncols) allows indexing axes[r, c].
    """
    n_features = X.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten() # Flattens 2D grid into 1D array of 6 axes for easy iteration
    
    means, stds, _, _ = column_stats(X)
    
    for i in range(n_features):
        ax = axes[i]
        col_data = X[:, i]
        ax.hist(col_data, bins=25, color="skyblue", edgecolor="black", alpha=0.7)
        
        # Draw mean and ±1 std lines
        ax.axvline(means[i], color="red", linestyle="-", linewidth=2, label=f"Mean: {means[i]:.2f}")
        ax.axvline(means[i] - stds[i], color="darkorange", linestyle="--", linewidth=1.5, label="±1 Std")
        ax.axvline(means[i] + stds[i], color="darkorange", linestyle="--", linewidth=1.5)
        
        name = feature_names[i] if feature_names is not None else f"Feature {i}"
        ax.set_title(f"Distribution of {name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
        
    # Hide unused 6th subplot if exactly 5 features
    if n_features < len(axes):
        axes[-1].axis("off")
        
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(A: np.ndarray, B: np.ndarray):
    """
    Displays pairwise distance matrix as an image (heatmap) and overlays nearest neighbors.
    Python Concept: 
      ax.imshow displays a 2D matrix where row index runs DOWN and column runs ACROSS.
      ax.scatter(x, y) expects x=column (horizontal) and y=row (vertical)!
    """
    D = pairwise_distances(A, B) # Shape: (n, m)
    n, m = D.shape
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Euclidean Distance", rotation=270, labelpad=15)
    
    # Overlay the nearest neighbour of each row
    nearest_cols = np.argmin(D, axis=1) # Shape: (n,)
    rows = np.arange(n)
    
    # Python Note: x=nearest_cols (col index), y=rows (row index)
    ax.scatter(nearest_cols, rows, color="red", marker="x", s=40, label="Nearest Neighbor")
    
    ax.set_title("Pairwise Distance Matrix with Nearest Neighbor Overlay")
    ax.set_xlabel("Index in B")
    ax.set_ylabel("Index in A")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_image(cm: np.ndarray, classes=None):
    """
    Plots a confusion matrix as a heatmap and annotates cell counts with ax.text.
    Python Concept: Loop over row i, col j and use ax.text(j, i, text).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    
    K = cm.shape[0]
    class_labels = classes if classes is not None else [f"Class {i}" for i in range(K)]
    
    # Annotate every cell with its count
    for i in range(K):
        for j in range(K):
            count = cm[i, j]
            # Change font color depending on cell darkness
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# BLOCK 2: Missingness and Imputation Visualizations
# ==============================================================================

def plot_grouped_missingness(df: pd.DataFrame, features, label_col="needs_service"):
    """
    Grouped bar chart showing the missing value rate of features across classes.
    Python Concept: Calculate bar positions using offsets: x - width/2, x + width/2.
    """
    classes = sorted(df[label_col].unique())
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i, c in enumerate(classes):
        sub_df = df[df[label_col] == c]
        missing_rates = [sub_df[f].isna().mean() for f in features]
        offset = (i - 0.5) * width
        ax.bar(x + offset, missing_rates, width, label=f"Class {c}")
        
    ax.set_ylabel("Missing Rate")
    ax.set_title("Missing Value Rate by Feature and Class")
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_imputation_distributions(original_series, global_imputed, group_imputed, global_med, class_meds):
    """
    Overlaid histograms demonstrating the distortion and 'spike' created by global imputation.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.hist(original_series.dropna(), bins=30, alpha=0.4, label="Original (Dropped NaNs)", color="gray", density=True)
    ax.hist(global_imputed, bins=30, alpha=0.4, label="Global Imputed", color="blue", density=True)
    ax.hist(group_imputed, bins=30, alpha=0.4, label="Group-wise Imputed", color="green", density=True)
    
    # Mark medians
    ax.axvline(global_med, color="blue", linestyle="--", linewidth=2, label=f"Global Median ({global_med:.1f})")
    for c, med in class_meds.items():
        ax.axvline(med, color="green", linestyle=":", linewidth=2, label=f"Class {c} Median ({med:.1f})")
        
    ax.set_title("Impact of Imputation Strategies on Distribution")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# BLOCK 3: Threshold Sweeps & Demonstrating Leakage
# ==============================================================================

def sweep_threshold(values: np.ndarray, y_true: np.ndarray, title="Threshold Sweep"):
    """
    Sweeps decision threshold t: predict 1 if values > t else 0.
    Computes precision, recall, and F1 across all thresholds.
    """
    thresholds = np.linspace(np.min(values), np.max(values), 100)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (values > t).astype(int)
        cm = binary_confusion_matrix(y_true, y_pred)
        p, r, f1 = binary_metrics(cm)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", color="blue")
    ax.plot(thresholds, recalls, label="Recall", color="green")
    ax.plot(thresholds, f1s, label="F1-Score", color="red", linewidth=2)
    ax.axvline(best_t, color="black", linestyle="--", label=f"Best t={best_t:.2f} (F1={best_f1:.2f})")
    
    ax.set_xlabel("Threshold (t)")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return thresholds, f1s, best_t, best_f1
