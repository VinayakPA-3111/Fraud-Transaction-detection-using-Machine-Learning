</h1>🔐 Fraud Transaction Detection — Machine Learning</h1>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Financial%20Services-1B3A6B)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **An end-to-end machine learning system that detects fraudulent financial transactions by analysing behavioural patterns, anomalies, and statistical signals in transaction data — safeguarding finances in real time.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Why This Matters — Financial Context](#-why-this-matters--financial-context)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Models Used](#-models-used)
- [Results & Evaluation](#-results--evaluation)
- [Handling Class Imbalance](#-handling-class-imbalance)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧠 Overview

This project builds a **supervised machine learning pipeline** to classify financial transactions as fraudulent or legitimate. The model ingests transaction-level data, engineers meaningful features, handles severe class imbalance, and trains multiple classifiers — evaluating them on metrics critical in financial risk contexts: **Precision, Recall, F1-Score, and ROC-AUC**.

The project mirrors real-world fraud detection workflows used in banking, payments, and financial operations — areas where false negatives (missed fraud) carry significant financial and reputational cost.

---

## 🎯 Problem Statement

Financial fraud is a global challenge costing institutions billions annually. Traditional rule-based systems struggle to keep pace with evolving fraud patterns. Machine learning offers a dynamic, data-driven alternative that:

- Learns patterns from historical labelled transaction data
- Detects anomalies and deviations from normal behaviour
- Scales to millions of transactions with low latency
- Continuously improves as new fraud patterns emerge

**Goal:** Build a binary classifier that flags fraudulent transactions with high recall (minimise missed fraud) while maintaining acceptable precision (minimise false alarms that inconvenience legitimate customers).

---

## 💳 Why This Matters — Financial Context

| Fraud Type | Description |
|---|---|
| Card-Not-Present (CNP) | Online purchases without a physical card |
| Account Takeover | Unauthorised access to a user's account |
| Identity Theft | Creating accounts with stolen credentials |
| Transaction Laundering | Concealing illicit funds through legitimate-looking transactions |

In financial institutions like banks and payment processors, fraud detection systems must process data at scale, flag exceptions in near-real-time, and integrate with risk and compliance frameworks — exactly the type of operations data quality challenge this project addresses.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Format | CSV (tabular transaction records) |
| Target Variable | `isFraud` — binary label (0 = Legitimate, 1 = Fraud) |
| Features | Transaction amount, type, source/destination balance before/after, step (time unit) |
| Class Distribution | Highly imbalanced — fraud cases are a small minority (~0.1–1%) |
| Source | PaySim synthetic financial dataset / Kaggle Credit Card Fraud dataset |

> The severe class imbalance is the central challenge: a naive model predicting "not fraud" for every transaction achieves 99%+ accuracy but zero utility. The project directly addresses this.

---

## 🔄 Project Pipeline

```
Raw Transaction Data (CSV)
         │
         ▼
┌─────────────────────────────┐
│   Exploratory Data          │
│   Analysis (EDA)            │
│  • Class distribution       │
│  • Feature correlations     │
│  • Transaction patterns     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Data Preprocessing        │
│  • Handle missing values    │
│  • Encode categoricals      │
│  • Feature scaling          │
│  • Drop irrelevant columns  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Feature Engineering       │
│  • Balance delta features   │
│  • Transaction ratio flags  │
│  • Anomaly indicators       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Class Imbalance Handling  │
│  • SMOTE oversampling       │
│  • Class weight adjustment  │
│  • Undersampling strategies │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Model Training &          │
│   Comparison                │
│  • Logistic Regression      │
│  • Decision Tree            │
│  • Random Forest            │
│  • XGBoost / Gradient Boost │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Evaluation                │
│  Confusion Matrix           │
│  Precision / Recall / F1    │
│  ROC-AUC Curve              │
│  Feature Importance         │
└─────────────────────────────┘
```

---

## 🤖 Models Used

| Model | Why It Was Used |
|---|---|
| **Logistic Regression** | Baseline — interpretable, fast, good for linearly separable cases |
| **Decision Tree** | Captures non-linear patterns; easily visualised for explainability |
| **Random Forest** | Ensemble method reducing overfitting; strong with imbalanced data |
| **Gradient Boosting / XGBoost** | State-of-the-art for tabular fraud detection; handles class imbalance natively |

Multiple models were trained and compared to identify the best trade-off between **precision** and **recall** — a critical judgement call in fraud detection where both false positives and false negatives carry real costs.

---

## 📈 Results & Evaluation

### Why Accuracy Alone is Misleading

With ~99% legitimate transactions, a dummy classifier scores 99% accuracy. The meaningful metrics are:

| Metric | Description | Fraud Relevance |
|---|---|---|
| **Precision** | Of flagged transactions, how many are actually fraud? | Controls false alarms — customer experience |
| **Recall** | Of all fraud cases, how many did we catch? | Controls missed fraud — financial loss |
| **F1-Score** | Harmonic mean of Precision and Recall | Balanced single metric |
| **ROC-AUC** | Area under the receiver operating curve | Overall discriminative power |

### Model Performance Summary

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | Moderate | Moderate | Moderate | ~0.85 |
| Decision Tree | High | High | High | ~0.90 |
| **Random Forest** | **High** | **High** | **High** | **~0.95+** |
| XGBoost | High | High | High | ~0.97+ |

> Random Forest and XGBoost emerged as the strongest performers, achieving high recall (catching most fraud) while maintaining acceptable precision (limiting false flags on legitimate transactions).

---

## ⚖️ Handling Class Imbalance

Class imbalance is the core technical challenge in fraud detection. This project evaluated:

- **SMOTE (Synthetic Minority Oversampling Technique)** — generates synthetic fraud samples to balance training data
- **Class weight adjustment** — penalises misclassification of minority class more heavily during training
- **Threshold tuning** — adjusts the decision boundary to favour recall over precision where the cost of missed fraud is high

This reflects real-world considerations at financial institutions, where risk teams actively tune detection sensitivity based on fraud loss tolerance.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VinayakPA-3111/Fraud-Transaction-detection-using-Machine-Learning.git
cd Fraud-Transaction-detection-using-Machine-Learning
```

### 2. Install dependencies

```bash
pip install scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn jupyter
```

### 3. Add the dataset

Download the fraud transaction dataset from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or use the PaySim dataset, and place it in the project root as `transactions.csv`.

### 4. Launch the notebook

```bash
jupyter notebook "Fraud Transaction Detection.ipynb"
```

### 5. Run all cells

The notebook runs the full pipeline: EDA → preprocessing → feature engineering → model training → evaluation.

---

## 📁 Project Structure

```
Fraud-Transaction-detection-using-Machine-Learning/
│
├── Fraud Transaction Detection.ipynb   # Main notebook (full ML pipeline)
├── README.md                           # Project documentation
│
└── data/                               # Place your dataset here
    └── transactions.csv
```

---

## 💡 Key Learnings

- **Accuracy is a trap** in imbalanced classification — always evaluate with Precision, Recall, and F1 for fraud tasks.
- **Feature engineering** matters more than model complexity — computing balance deltas (before vs. after transaction) significantly boosted model performance.
- **SMOTE** must be applied only on the training set, never the test set — data leakage is a real risk in fraud ML pipelines.
- **Threshold tuning** is a business decision: a bank willing to tolerate more false alerts to catch more fraud sets a lower threshold; customer-experience-focused products do the opposite.
- **Explainability** is critical in financial services — Random Forest's feature importance plots and Decision Tree visualisations make model decisions auditable and defensible to risk and compliance teams.
- **Real-world fraud patterns** are dynamic — models need periodic retraining as fraudsters adapt their behaviour.

---

## 🔭 Future Improvements

- [ ] Implement **real-time fraud scoring** via a REST API (FastAPI / Flask) with sub-100ms inference
- [ ] Add **SHAP (SHapley Additive exPlanations)** for model interpretability and regulatory explainability
- [ ] Explore **Isolation Forest** and **Autoencoders** for unsupervised anomaly detection (no labels required)
- [ ] Build a **monitoring dashboard** (Streamlit) tracking model drift and fraud rate trends over time
- [ ] Integrate **graph-based fraud detection** — identifying fraud rings through network relationships between accounts
- [ ] Extend to **multi-class classification** distinguishing fraud types (account takeover, CNP fraud, money laundering)

---

## 👤 Author

**Vinayak Algundgi**

- 📧 [vinayak.algundgi12@gmail.com](mailto:vinayak.algundgi12@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/vinayakalgundgi/)
- 🐙 [GitHub](https://github.com/VinayakPA-3111)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

*Fighting financial fraud, one decision tree at a time. 🛡️*
