"""
IC-272: Intro to ML Lab
Lab 2: Pandas and a First ML Pipeline (Leak-free preprocessing)
"""

import numpy as np
import pandas as pd
from lab1_refresher import binary_confusion_matrix, binary_metrics

# ==============================================================================
# BLOCK 1: Data Inspection, Imputation, One-Hot Encoding, Train-Test Split
# ==============================================================================

def inspect_dataset(df: pd.DataFrame):
    """
    Inspects missing values, label distribution, and categorical feature distribution.
    Checks whether missing values are spread evenly across classes.
    """
    print("--- Missing Values Per Column ---")
    print(df.isna().sum())
    
    print("\n--- Distribution of Target ('needs_service') ---")
    print(df["needs_service"].value_counts(normalize=True))
    
    print("\n--- Distribution of 'machine_type' ---")
    print(df["machine_type"].value_counts())
    
    print("\n--- Missing Rate by Class ---")
    # Python Concept: groupby + isna().mean() calculates percentage of NaN per group
    missing_by_class = df.groupby("needs_service").apply(lambda g: g.isna().mean())
    print(missing_by_class[["temperature", "vibration", "pressure"]])
    return missing_by_class


def compare_imputations(df: pd.DataFrame):
    """
    Compares global median imputation vs group-wise median imputation.
    Python Concept: 
      - df['col'].median() -> single scalar
      - df.groupby('class')['col'].transform('median') -> expands each group's median back to row shape!
    """
    # Global median
    global_median = df["temperature"].median()
    df_global = df["temperature"].fillna(global_median)
    
    # Group-wise median
    # transform('median') produces a Series with the same index as df, matching each row's class median
    group_medians = df.groupby("needs_service")["temperature"].transform("median")
    df_group = df["temperature"].fillna(group_medians)
    
    print(f"\nGlobal Median: {global_median:.4f}")
    class_meds = df.groupby("needs_service")["temperature"].median()
    print("Class Medians:\n", class_meds)
    
    return df_global, df_group


def one_hot_encode_by_hand(series: pd.Series):
    """
    Constructs one-hot encoded binary columns without pd.get_dummies.
    Python Concept: (series == cat).astype(int) creates a 0/1 column.
    """
    categories = sorted(series.dropna().unique())
    ohe_dict = {}
    for cat in categories:
        col_name = f"{series.name}_{cat}"
        # Boolean comparison turns into integers (True -> 1, False -> 0)
        ohe_dict[col_name] = (series == cat).astype(int)
    
    custom_ohe = pd.DataFrame(ohe_dict, index=series.index)
    # Verification with pd.get_dummies
    auto_ohe = pd.get_dummies(series, prefix=series.name, dtype=int)
    # Reorder columns to ensure exact comparison
    auto_ohe = auto_ohe[custom_ohe.columns]
    assert np.array_equal(custom_ohe.values, auto_ohe.values), "OHE by hand does not match pd.get_dummies!"
    return custom_ohe


def train_test_split_scratch(X, y, test_size=0.2, seed=42):
    """
    Shuffles indices using seeded generator, then splits into train and test sets.
    Returns: X_train, X_test, y_train, y_test as NumPy arrays.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n)
    
    test_count = int(n * test_size)
    test_idx = shuffled_idx[:test_count]
    train_idx = shuffled_idx[test_count:]
    
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    """
    Stretch Task: Stratified split preserving the class proportions in both train and test.
    Python Concept: Indexing per class with np.where(y == c)[0], permuting each class independently.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    
    classes = np.unique(y_arr)
    train_indices = []
    test_indices = []
    
    for c in classes:
        c_idx = np.where(y_arr == c)[0]
        shuffled = rng.permutation(c_idx)
        n_test_c = int(len(c_idx) * test_size)
        test_indices.extend(shuffled[:n_test_c])
        train_indices.extend(shuffled[n_test_c:])
        
    train_idx = np.array(train_indices)
    test_idx = np.array(test_indices)
    
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


# ==============================================================================
# BLOCK 2: End-to-End Pipeline (NO DATA LEAKAGE)
# ==============================================================================

def run_pipeline(csv_path: str):
    """
    Implements the full preprocessing and evaluation pipeline:
    1. Load CSV
    2. Train-test split FIRST (avoids leaking test statistics)
    3. Compute medians on training set ONLY, then impute both train and test
    4. One-hot encode using training categories ONLY
    5. Evaluate majority class baseline on test set
    """
    df = pd.read_csv(csv_path)
    
    # 1. Separate features and target
    target_col = "needs_service"
    feature_cols = [c for c in df.columns if c != target_col]
    
    # 2. Split train and test before any imputation or encoding!
    train_df, test_df = df.iloc[:int(len(df)*0.8)].copy(), df.iloc[int(len(df)*0.8):].copy()
    # Alternatively use train_test_split_scratch on index:
    idx = np.random.default_rng(42).permutation(len(df))
    n_test = int(len(df) * 0.2)
    train_idx, test_idx = idx[n_test:], idx[:n_test]
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    # 3. Numeric imputation: compute medians strictly on train_df
    numeric_cols = ["runtime_hours", "temperature", "vibration", "pressure"]
    train_medians = train_df[numeric_cols].median()
    
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_medians) # Leak-free!
    
    # 4. Categorical encoding based on training categories
    train_categories = sorted(train_df["machine_type"].dropna().unique())
    for cat in train_categories:
        col = f"type_{cat}"
        train_df[col] = (train_df["machine_type"] == cat).astype(int)
        test_df[col] = (test_df["machine_type"] == cat).astype(int)
        
    train_df.drop(columns=["machine_type"], inplace=True)
    test_df.drop(columns=["machine_type"], inplace=True)
    
    # 5. Majority class baseline predictor
    # Python Concept: .mode()[0] gets the most frequent label
    majority_class = train_df[target_col].mode()[0]
    y_test_true = test_df[target_col].to_numpy()
    y_test_pred = np.full_like(y_test_true, fill_value=majority_class)
    
    # 6. Evaluation using Lab 1 scratch metrics
    cm = binary_confusion_matrix(y_test_true, y_test_pred)
    acc = np.mean(y_test_true == y_test_pred)
    prec, rec, f1 = binary_metrics(cm)
    
    print("\n--- Majority Class Baseline Results on Test Set ---")
    print(f"Majority Class Predicted: {majority_class}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Notice: High accuracy baseline, but 0 recall on minority class!")
    return acc, prec, rec, f1
