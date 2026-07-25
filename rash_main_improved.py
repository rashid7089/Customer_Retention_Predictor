"""
Improved customer-churn modeling pipeline for `rash_main.ipynb`.

This script fixes the main reasons the original notebook produced low
metrics, and adds training-vs-validation plots for every model.

Run with:
    .venv/bin/python rash_main_improved.py
"""

import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GridSearchCV,
    learning_curve,
    validation_curve,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_data_v1.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.125  # 0.125 * 0.80 = 0.10 of full data

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop index column that was written by pandas to_csv
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Raw data shape:", df.shape)
print("Class distribution:")
print(df["Exited"].value_counts(normalize=True))
print()

# ---------------------------------------------------------------------------
# 3. Preprocessing fixes
# ---------------------------------------------------------------------------
# FIX 1: Drop identifier columns that carry no signal.
# FIX 2: One-hot encode Geography (nominal) instead of label encoding.
# FIX 3: Keep Gender as binary (0/1) already correct.
# FIX 4: Add a few useful interaction/ratio features.

TARGET = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# Use only columns that exist in this dataset
DROP_COLS = [c for c in DROP_COLS if c in df.columns]

numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Gender",
]

# Feature engineering
df["BalancePerProduct"] = df["Balance"] / (df["NumOfProducts"] + 1e-6)
df["Age_Tenure"] = df["Age"] / (df["Tenure"] + 1)
df["SalaryToBalance"] = df["EstimatedSalary"] / (df["Balance"] + 1)
df["IsActive_x_Products"] = df["IsActiveMember"] * df["NumOfProducts"]

numeric_features += [
    "BalancePerProduct",
    "Age_Tenure",
    "SalaryToBalance",
    "IsActive_x_Products",
]

categorical_features = ["Geography"]

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

# Verify all expected columns exist
numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

print("Features used:", numeric_features + categorical_features)
print("Feature count:", X.shape[1])
print()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# ---------------------------------------------------------------------------
# 4. Train / validation / test split (stratified)
# ---------------------------------------------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))
print()

# ---------------------------------------------------------------------------
# 5. Model definitions
# ---------------------------------------------------------------------------
# We use class_weight='balanced' to handle the ~20% churn rate without
# throwing away data (the original notebook commented out undersampling).

models = {
    "LogisticRegression": {
        "pipe": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "params": {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
        },
    },
    "RandomForest": {
        "pipe": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "params": {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 5, 8, 12, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    },
    "SVM": {
        "pipe": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "classifier",
                    SVC(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        probability=True,
                    ),
                ),
            ]
        ),
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    },
    "DecisionTree": {
        "pipe": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "params": {
            "classifier__max_depth": [3, 5, 8, 12, 20, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
        },
    },
}

# ---------------------------------------------------------------------------
# 6. Helper functions
# ---------------------------------------------------------------------------
def get_best_threshold(y_true, y_prob):
    """Youden's J statistic: max(TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]


def evaluate(y_true, y_pred, y_prob=None, name="Model", print_header=True):
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    if print_header:
        print(f"--- {name} ---")
    print(f"  F1:      {f1:.4f}")
    print(f"  Recall:  {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    if auc is not None:
        print(f"  ROC-AUC: {auc:.4f}")
    return f1, recall, precision, auc


def plot_confusion(y_true, y_pred, name, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{name}.png")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Hyper-parameter tuning
# ---------------------------------------------------------------------------
results = {}
PLOTS_DIR = ROOT / "improved_results"
PLOTS_DIR.mkdir(exist_ok=True)

for model_name, cfg in models.items():
    print(f"\n===== Tuning {model_name} =====")
    search = GridSearchCV(
        estimator=cfg["pipe"],
        param_grid=cfg["params"],
        cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Validation set predictions
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_preds_default = best_model.predict(X_val)
    val_preds_best = (val_probs >= get_best_threshold(y_val, val_probs)[0]).astype(int)

    print("Validation (default threshold):")
    evaluate(y_val, val_preds_default, val_probs, name=model_name)
    print("Validation (best threshold):")
    evaluate(y_val, val_preds_best, val_probs, name=model_name)

    # Test set predictions (final unbiased evaluation)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds_default = best_model.predict(X_test)
    best_threshold, _, _ = get_best_threshold(y_val, val_probs)
    test_preds_best = (test_probs >= best_threshold).astype(int)

    print("Test (default threshold):")
    f1_d, rec_d, prec_d, auc_d = evaluate(
        y_test, test_preds_default, test_probs, name=model_name, print_header=False
    )
    print("Test (best threshold):")
    f1_b, rec_b, prec_b, auc_b = evaluate(
        y_test, test_preds_best, test_probs, name=model_name, print_header=False
    )

    results[model_name] = {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "val_default": {
            "f1": float(f1_score(y_val, val_preds_default)),
            "recall": float(recall_score(y_val, val_preds_default)),
            "precision": float(precision_score(y_val, val_preds_default)),
            "auc": float(roc_auc_score(y_val, val_probs)),
        },
        "val_best": {
            "f1": float(f1_score(y_val, val_preds_best)),
            "recall": float(recall_score(y_val, val_preds_best)),
            "precision": float(precision_score(y_val, val_preds_best)),
            "auc": float(roc_auc_score(y_val, val_probs)),
        },
        "test_default": {
            "f1": float(f1_d),
            "recall": float(rec_d),
            "precision": float(prec_d),
            "auc": float(auc_d),
        },
        "test_best": {
            "f1": float(f1_b),
            "recall": float(rec_b),
            "precision": float(prec_b),
            "auc": float(auc_b),
        },
    }

    plot_confusion(y_test, test_preds_best, model_name, PLOTS_DIR)

    # Save model
    import joblib

    joblib.dump(best_model, PLOTS_DIR / f"model_{model_name}.joblib")

print("\n")

# ---------------------------------------------------------------------------
# 8. Training vs validation plots (learning curves)
# ---------------------------------------------------------------------------
print("Generating training vs validation (learning) curves ...")

cv_for_curves = StratifiedShuffleSplit(
    n_splits=5, test_size=0.2, random_state=RANDOM_STATE
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, (model_name, cfg) in zip(axes, models.items()):
    best_estimator = results[model_name]["best_params"]

    # Rebuild the best pipeline so we can plot its learning curve
    best_pipe = cfg["pipe"].set_params(**best_estimator)

    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe,
        X_train,
        y_train,
        cv=cv_for_curves,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="tab:blue", label="Training F1")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="tab:blue",
    )
    ax.plot(train_sizes, val_mean, "o-", color="tab:orange", label="Validation F1")
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="tab:orange",
    )

    ax.set_title(f"Learning Curve: {model_name}")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Training vs Validation F1 (Learning Curves)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "learning_curves_all_models.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. Validation curves for a key hyperparameter of each model
# ---------------------------------------------------------------------------
print("Generating validation curves ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

validation_curve_configs = [
    ("LogisticRegression", "classifier__C", np.logspace(-3, 3, 7)),
    ("RandomForest", "classifier__max_depth", [3, 5, 8, 12, 20, 30, None]),
    ("SVM", "classifier__C", np.logspace(-2, 2, 5)),
    ("DecisionTree", "classifier__max_depth", [3, 5, 8, 12, 20, 30, None]),
]

for ax, (model_name, param_name, param_range) in zip(axes, validation_curve_configs):
    pipe = models[model_name]["pipe"]

    train_scores, val_scores = validation_curve(
        pipe,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv_for_curves,
        scoring="f1",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    x_labels = [str(p) if p is not None else "None" for p in param_range]
    x_pos = np.arange(len(param_range))

    ax.plot(x_pos, train_mean, "o-", color="tab:blue", label="Training F1")
    ax.fill_between(
        x_pos,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="tab:blue",
    )
    ax.plot(x_pos, val_mean, "o-", color="tab:orange", label="Validation F1")
    ax.fill_between(
        x_pos,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="tab:orange",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title(f"Validation Curve: {model_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Training vs Validation F1 (Validation Curves)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "validation_curves_all_models.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10. Summary bar chart
# ---------------------------------------------------------------------------
summary = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "Val_Default_F1": [results[m]["val_default"]["f1"] for m in results],
        "Val_Best_F1": [results[m]["val_best"]["f1"] for m in results],
        "Test_Default_F1": [results[m]["test_default"]["f1"] for m in results],
        "Test_Best_F1": [results[m]["test_best"]["f1"] for m in results],
    }
)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(summary))
width = 0.2

ax.bar(x - 1.5 * width, summary["Val_Default_F1"], width, label="Val (default th)", color="#2b5c8f")
ax.bar(x - 0.5 * width, summary["Val_Best_F1"], width, label="Val (best th)", color="#4a90c2")
ax.bar(x + 0.5 * width, summary["Test_Default_F1"], width, label="Test (default th)", color="#d95f02")
ax.bar(x + 1.5 * width, summary["Test_Best_F1"], width, label="Test (best th)", color="#f4a261")

ax.set_ylabel("F1 Score")
ax.set_title("Model Comparison: Validation vs Test F1")
ax.set_xticks(x)
ax.set_xticklabels(summary["Model"], rotation=15)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison_f1.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------------------
with open(PLOTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

summary.to_csv(PLOTS_DIR / "summary.csv", index=False)

print("\n===== SUMMARY (Test set, best threshold) =====")
print(summary.to_string(index=False))
print(f"\nAll plots and results saved in: {PLOTS_DIR}")
