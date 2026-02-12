![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557C?logo=plotly&logoColor=white)

# ML Algorithms - From Scratch to Scikit-Learn

A hands-on collection of **15 machine learning algorithm implementations** covering both regression and classification techniques. Each algorithm is implemented step-by-step using scikit-learn, with real datasets, visualizations, and performance evaluation.

---

## Table of Contents

- [Regression Models (1-7)](#regression-models)
- [Classification Models (8-15)](#classification-models)
- [Algorithm Overview](#algorithm-overview)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Results Overview](#results-overview)
- [Author](#author)

---

## Regression Models

1. [Simple Linear Regression](./1_Simple_linear_regression) -- Predict salary from years of experience
2. [Multiple Linear Regression](./2_Multiple_linear_regression) -- Predict startup profit from R&D, admin, and marketing spend
3. [Polynomial Regression](./3_Polynomial_regression) -- Predict salary from position level using polynomial features
4. [Support Vector Regression](./4_Support_vector_regression) -- SVR with RBF kernel on position-salary data
5. [Decision Tree Regression](./5_Decision_tree_regression) -- Non-linear regression using decision trees
6. [Random Forest Regression](./6_Random_forest_regression) -- Ensemble of decision trees for salary prediction
7. [Compare Regression Models](./7_Compare_regression_models) -- Side-by-side comparison of all regression models

## Classification Models

8. [Logistic Regression](./8_Logistic_regression) -- Binary classification for iPhone purchase prediction
9. [K-Nearest Neighbors](./9_k_nearest_neighbor) -- KNN classifier with decision boundary visualization
10. [Support Vector Machine](./10_SVM) -- Linear SVM for purchase prediction
11. [Kernel SVM](./11_Kernel_svm) -- SVM with RBF kernel for non-linear classification
12. [Naive Bayes](./12_Naive_bayes) -- Gaussian Naive Bayes classifier
13. [Decision Tree Classifier](./13_Decision_tree_classifier) -- Entropy-based decision tree classification
14. [Random Forest Classifier](./14_Random_forest_classifier) -- Ensemble classification with random forests
15. [Compare Classification Algorithms](./15_Compare_classification_algorithms) -- 10-fold cross-validation comparison of all classifiers

---

## Algorithm Overview

| # | Algorithm | Type | Directory |
|---|-----------|------|-----------|
| 1 | Simple Linear Regression | Regression | [Link](./1_Simple_linear_regression) |
| 2 | Multiple Linear Regression | Regression | [Link](./2_Multiple_linear_regression) |
| 3 | Polynomial Regression | Regression | [Link](./3_Polynomial_regression) |
| 4 | Support Vector Regression | Regression | [Link](./4_Support_vector_regression) |
| 5 | Decision Tree Regression | Regression | [Link](./5_Decision_tree_regression) |
| 6 | Random Forest Regression | Regression | [Link](./6_Random_forest_regression) |
| 7 | Compare Regression Models | Regression | [Link](./7_Compare_regression_models) |
| 8 | Logistic Regression | Classification | [Link](./8_Logistic_regression) |
| 9 | K-Nearest Neighbors (KNN) | Classification | [Link](./9_k_nearest_neighbor) |
| 10 | Support Vector Machine (SVM) | Classification | [Link](./10_SVM) |
| 11 | Kernel SVM | Classification | [Link](./11_Kernel_svm) |
| 12 | Naive Bayes | Classification | [Link](./12_Naive_bayes) |
| 13 | Decision Tree Classifier | Classification | [Link](./13_Decision_tree_classifier) |
| 14 | Random Forest Classifier | Classification | [Link](./14_Random_forest_classifier) |
| 15 | Compare Classification Algorithms | Classification | [Link](./15_Compare_classification_algorithms) |

---

## Datasets

This project uses the following datasets:

| Dataset | Used In | Description |
|---------|---------|-------------|
| `Salary_Data.csv` | Simple Linear Regression (#1) | Years of experience vs. salary (30 samples) |
| `Position_Salaries.csv` | Polynomial Regression (#3), SVR (#4), Decision Tree (#5), Random Forest (#6), Compare (#7) | Position level vs. salary (10 samples) |
| `50_Startups.csv` | Multiple Linear Regression (#2) | R&D spend, administration, marketing spend, state, and profit for 50 startups |
| `iphone_purchase_records.csv` | Logistic Regression (#8), KNN (#9), SVM (#10), Kernel SVM (#11), Naive Bayes (#12), Decision Tree (#13), Random Forest (#14), Compare (#15) | Gender, age, salary, and purchase decision for iPhone buyers |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Samudyata/Ml-algorithms.git
cd Ml-algorithms
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## How to Run

Each algorithm is self-contained in its own directory. Navigate to any subdirectory and run the Python script:

```bash
# Example: Run Simple Linear Regression
cd 1_Simple_linear_regression
python simple_linear_regression.py

# Example: Run the classification comparison
cd 15_Compare_classification_algorithms
python compare_classification_algos.py
```

Each script will:
1. Load and preprocess the dataset
2. Train the model
3. Evaluate performance (MSE, R2, accuracy, confusion matrix, etc.)
4. Display visualizations (where applicable)

---

## Results Overview

### Regression Models -- Salary Prediction at Level 6.5

The regression comparison (#7) predicts the salary for a position at level 6.5 using the `Position_Salaries.csv` dataset:

| Regression Model | Predicted Salary at Level 6.5 |
|---|---|
| Linear Regression | ~$330,000 |
| Polynomial Regression (Degree 4) | ~$158,000 |
| Support Vector Regression (RBF) | ~$170,000 |
| Decision Tree Regression | ~$150,000 |
| Random Forest (300 Trees) | ~$160,000 |

Polynomial Regression and Random Forest provided the most accurate predictions, both landing close to the expected $160K range.

### Classification Models -- iPhone Purchase Prediction

The classification comparison (#15) uses 10-fold cross-validation on the `iphone_purchase_records.csv` dataset:

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 84.00% | 6.24% |
| K-Nearest Neighbors | 91.25% | 5.15% |
| Kernel SVM | 90.75% | 4.88% |
| Naive Bayes | 88.75% | 5.15% |
| Decision Tree | 85.00% | 7.07% |
| Random Forest | 88.75% | 4.51% |

K-Nearest Neighbors achieved the highest accuracy (91.25%), closely followed by Kernel SVM (90.75%) and Random Forest (88.75%).

---

## Author

**Samudyata**

---
