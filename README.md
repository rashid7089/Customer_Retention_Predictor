# Methodology

[THE METHODOLOGY.md FILE!!](Methodology.md)

# Customer Retention Predictor

Predicts whether a bank customer will churn (leave the bank) using 10 customer features from the Churn Modelling dataset. Achieves **~86.8% accuracy** via a Random Forest classifier with hyperparameter tuning across multiple models.

## Dataset Preprocessing

The dataset was cleaned and prepared in [`preprocessing.ipynb`](preprocessing.ipynb):

- **Source:** `Churn_Modelling.csv` — 10,000 bank customers with 14 columns from Kaggle.
- **Drop non-predictive columns:** `Surname`, `RowNumber`, and `CustomerId` are removed because they carry no predictive value.
- **Encode categoricals:** `Gender` mapped to binary (Male=1, Female=0). `Geography` label-encoded to 0, 1, 2.
- **No missing values:** All 10,000 rows are complete across all 13 remaining columns.
- **Class distribution:** 7,963 retained (80%) vs 2,037 churned (20%). The team chose **not** to balance the dataset since downsampling would discard a large portion of data. Class imbalance is handled at training with `class_weight='balanced'`.
- **Correlation analysis:** Heatmap generated to confirm feature relationships. `Age` and `Balance` showed the strongest correlation with churn.
- **Export:** Cleaned data saved to `data/clean_data_v1.csv` (10,000 rows, 11 columns).

| Checkpoint | Value |
|---|---|
| Total samples | 10,000 |
| Retained (0) | 7,963 |
| Churned (1) | 2,037 |
| Features | 10 numeric attributes |

## Models & Configuration

Four classification models were trained with `GridSearchCV` in [`main.ipynb`](main.ipynb):

**Training setup:**
- Split: 70% train / 15% validation / 15% test (stratified)
- `GridSearchCV` with 5-fold cross-validation, scoring on accuracy
- `StandardScaler` applied via pipeline where needed

**Models explored:**

| Model | Key Hyperparameters Tested |
|---|---|
| Logistic Regression | `C`: 0.1, 1, 10; `penalty`: l1, l2; `solver`: liblinear, newton-cg; `class_weight`: balanced; `PolynomialFeatures(degree=3)` |
| SVM | `C`: 0.1, 1, 10; `kernel`: linear, rbf; `class_weight`: balanced |
| Decision Tree | `max_depth`: 3, 5, 7; `class_weight`: balanced, None |
| Random Forest | `n_estimators`: 50, 100, 200; `max_depth`: 10, 30, None; `class_weight`: balanced, None |

**Threshold tuning:** Each model was evaluated at both the default 0.5 threshold and the optimal threshold found using Youden's J statistic (ROC curve optimization).

## Results (Test Set)

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 48% | 75% | 59% | 79% |
| SVM | 49% | 77% | 60% | 79% |
| Decision Tree | 78% | 38% | 51% | 85% |
| Random Forest | 81% | 46% | 59% | 87% |

Random Forest achieved the highest accuracy (~87%), though Logistic Regression and SVM showed better recall for the minority churn class.

>  The project includes a full evaluation suite: classification reports, confusion matrices, ROC curves, and precision-recall curves for all models.
