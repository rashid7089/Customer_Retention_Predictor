# Introduction

This project builds a classification system to predict whether a bank customer will churn (close their account) based on demographic and financial features. We used the Churn Modelling dataset from Kaggle, trained multiple machine learning models, and compared them across various evaluation metrics.

# Team Division

The project was completed by a team of 4 members, each working on a separate branch:

- **all members** focused on **dataset finding** and **preprocessing** — cleaning the raw dataset and preparing it for training
- **3 members** focused on **training** — building and tuning the best classification models and then compare to each other
- **1 member** focused on **LLM integration** — integrating a large language model into the project

Each member pushed their work to separate branches and the final code was merged into the main branch.

# Dataset & Preprocessing

The preprocessing was done in [`preprocessing.ipynb`](preprocessing.ipynb).

## Dataset Overview

We used the `Churn_Modelling.csv` dataset containing **10,000 bank customers** with **14 columns**:

- `RowNumber` — row index (not useful)
- `CustomerId` — unique ID (not useful)
- `Surname` — customer last name (text, not useful)
- `CreditScore` — customer's credit score
- `Geography` — country (France, Spain, Germany)
- `Gender` — Male or Female
- `Age` — customer age
- `Tenure` — number of years with the bank
- `Balance` — account balance
- `NumOfProducts` — number of bank products used
- `HasCrCard` — whether the customer has a credit card (0 or 1)
- `IsActiveMember` — whether the customer is active (0 or 1)
- `EstimatedSalary` — estimated annual salary
- `Exited` — target variable: whether the customer churned (1) or stayed (0)

## Cleaning Steps

1. **Drop Surname** — The `Surname` column is text data with no predictive value, so it was removed.
2. **Encode Gender** — `Gender` was converted to binary: Male = 1, Female = 0.
3. **Encode Geography** — `Geography` was label-encoded using `sklearn.preprocessing.LabelEncoder`, mapping each country to 0, 1, or 2.
4. **Check for null values** — `.isna().sum()` confirmed that **no column contained any missing values**. All 10,000 rows were complete.

After these steps, the dataset had 13 columns (all numeric).

## Class Imbalance Decision

The class distribution was checked:

- **7,963 customers stayed** (Exited = 0) — 79.6%
- **2,037 customers churned** (Exited = 1) — 20.4%

The dataset is moderately imbalanced. However, we decided **not to balance the dataset** by downsampling. The reasoning was:

> Downsampling would mean discarding roughly 6,000 rows of data, which is a very large portion of the dataset. Instead, we chose to keep all data and handle the imbalance at training time using `class_weight='balanced'` in the models.

## Correlation Analysis

A correlation heatmap was generated to understand relationships between features and the target variable:

- **Age** showed the strongest positive correlation with churn (0.29)
- **Balance** also showed a correlation with churn (0.12)
- **IsActiveMember** showed a negative correlation with churn (-0.16)
- **RowNumber** and **CustomerId** had near-zero correlation with Exited, confirming they are identifier columns with no predictive signal

Based on this analysis, `RowNumber` and `CustomerId` were dropped from the dataset.

## Final Dataset

The cleaned dataset was exported to `data/clean_data_v1.csv` with **10,000 rows and 11 columns** (10 features + 1 target).

| Feature | Type |
|---|---|
| CreditScore | Integer |
| Geography | Integer (encoded: 0, 1, 2) |
| Gender | Integer (0 or 1) |
| Age | Integer |
| Tenure | Integer |
| Balance | Float |
| NumOfProducts | Integer |
| HasCrCard | Integer (0 or 1) |
| IsActiveMember | Integer (0 or 1) |
| EstimatedSalary | Float |
| Exited (target) | Integer (0 or 1) |

# Training

The main training pipeline is in [`main.ipynb`](main.ipynb). A separate training approach using StratifiedKFold is in [`training.ipynb`](training.ipynb).

## Data Split

The cleaned data was split into three sets using stratified sampling to preserve class proportions:

```python
x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7,
    random_state=42, stratify=y)

x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,
    random_state=42, stratify=y_temp)
```

| Split | Size | Percentage |
|---|---|---|
| Training | 7,000 | 70% |
| Validation | 1,500 | 15% |
| Testing | 1,500 | 15% |

## Models

Four classification models were defined with different pipelines and hyperparameter grids. Each model was tuned using `GridSearchCV` with 5-fold cross-validation, scoring on accuracy.

### Logistic Regression

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=3)),
    ("classifier", LogisticRegression(max_iter=5000, class_weight='balanced'))
])
```

**Hyperparameter grid:**
- `classifier__C`: 0.1, 1, 10
- `classifier__penalty`: l1, l2
- `classifier__solver`: liblinear, newton-cg

### SVM (Support Vector Machine)

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(probability=True, class_weight='balanced'))
])
```

**Hyperparameter grid:**
- `classifier__C`: 0.1, 1, 10
- `classifier__kernel`: linear, rbf

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

**Hyperparameter grid:**
- `max_depth`: 3, 5, 7
- `class_weight`: balanced, None

### Random Forest

```python
RandomForestClassifier(random_state=42)
```

**Hyperparameter grid:**
- `n_estimators`: 50, 100, 200
- `max_depth`: 10, 30, None
- `class_weight`: balanced, None

## Threshold Optimization

In addition to the default 0.5 decision threshold, each model was evaluated with an **optimal threshold** found using Youden's J statistic. The optimal threshold maximizes the difference between True Positive Rate (TPR) and False Positive Rate (FPR):

```python
fpr, tpr, thresholds = roc_curve(y_val, y_pred_probs)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
```

This gave us 8 evaluation variants: 4 models × 2 thresholds (default and optimal).

# Evaluation & Results

Each model was evaluated on the validation set using:
- Precision, Recall, F1-Score, Accuracy
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

A combined bar chart compared all models across thresholds. All ROC curves were plotted together for direct visual comparison.

## Validation Results

| Model | Threshold | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | Default (0.5) | 50.0% | 79.1% | 61.3% | 79.6% |
| SVM | Default (0.5) | 49.1% | 77.5% | 60.1% | 79.0% |
| Decision Tree | Default (0.5) | 78.0% | 38.2% | 51.3% | 85.2% |
| Random Forest | Default (0.5) | 81.4% | 45.8% | 58.6% | 86.8% |

## Test Results (Final Evaluation)

| Model | Threshold | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.5000 (default) | 75% | 59% | 79% |
| Logistic Regression | 0.5000 (best) | 75% | 59% | 79% |
| SVM | 0.5000 (default) | 75% | 59% | 79% |
| SVM | 0.2137 (best) | 75% | 59% | 79% |
| Decision Tree | 0.5000 (default) | 40% | 55% | 87% |
| Decision Tree | 0.3077 (best) | 67% | 63% | 84% |
| Random Forest | 0.5000 (default) | 46% | 59% | 87% |
| Random Forest | 0.1988 (best) | 77% | 58% | 78% |

## Key Observations

- **Decision Tree with the optimal threshold (0.3077)** gave the best overall balance — 63% F1-score with 67% recall and 84% accuracy, a massive improvement over its default-threshold performance (55% F1, 40% recall).
- **Random Forest** saw its default accuracy of 87% drop to 78% when the optimal threshold was applied, but its recall more than doubled from 46% to 77%, showing the threshold trade-off between precision and recall.
- **Logistic Regression** and **SVM** had the same optimal threshold as the default (0.5 for Logistic Regression, ~0.21 for SVM), confirming the default cutoff was already well-calibrated for these linear models.
- **SVM** and **Logistic Regression** tied at 79% accuracy / 59% F1 across both thresholds.

# LLM Integration

>  **Not yet completed** — content will be added once the LLM integration is finished.
