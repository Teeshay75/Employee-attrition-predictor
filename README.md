# 🧠 Intelligent Employee Attrition Prediction

> Gradient Boosting classifier with SMOTENC-based class balancing and adaptive threshold optimisation — deployed as an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project predicts employee attrition using the **IBM HR Analytics dataset** (1,470 records, 35 features). The pipeline addresses the inherent class imbalance (84:16) using **SMOTENC oversampling + random undersampling**, trains a **Gradient Boosting Classifier**, and applies **adaptive threshold optimisation** to maximise accuracy while maintaining a minimum recall of 80%.

| Metric | Value |
|---|---|
| Accuracy | **87.41%** |
| Recall (Attrition) | **81.76%** |
| Precision (Attrition) | **92%** |
| F1-Score (Attrition) | **87%** |
| Decision Threshold | **0.52** |

---

## 🗂️ Project Structure

```
employee_attrition_prediction/
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Original IBM HR dataset
│   ├── balanced_attrition_1470.csv              # Balanced dataset (SMOTENC + undersampling)
│   ├── balanced_attrition_1470 (1).csv          # Alternate balanced split
│   ├── synthetic_attrition_60_40.csv            # Synthetic 60/40 split variant
│   └── synthetic_attrition_correlated.csv       # Synthetic correlated variant
│
├── model/                                        # Auto-generated after training
│   ├── model.pkl                                 # Trained Gradient Boosting model
│   ├── columns.pkl                               # Feature column names
│   ├── threshold.pkl                             # Optimised decision threshold
│   └── feature_importance.csv                   # Feature importance scores
│
├── app.py                                        # Streamlit frontend dashboard
├── train_model.py                                # Model training & threshold tuning
├── requirements.txt                              # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

### 1. Class Balancing (SMOTENC)
The original dataset has a severe **84:16 class imbalance**. A two-step resampling strategy maintains the original row count of 1,470:
- **SMOTENC**: Oversamples minority class (Attrition = Yes) from 237 → 650
- **RandomUnderSampler**: Undersamples majority class (Attrition = No) from 1,233 → 820
- Final split: **820 No : 650 Yes (56:44)**

### 2. Model Training
A `GradientBoostingClassifier` is trained with:
- `n_estimators = 300`
- `learning_rate = 0.05`
- `max_depth = 3`
- 80/20 stratified train-test split

### 3. Threshold Optimisation
The decision threshold is swept from **0.10 to 0.90** in steps of 0.01. The optimal threshold is selected as:

**τ\* = argmax Accuracy(τ)  subject to  Recall(τ) ≥ 0.80**

Selected threshold: **0.52**

---

## 🖥️ Streamlit Dashboard Features

| Page | Description |
|---|---|
| 🎯 Predict Risk | Enter individual employee details and get a real-time attrition probability gauge, risk label, and retention recommendations |
| 📊 Analytics | Multi-tab dashboard — attrition by department/role/travel, satisfaction analysis, compensation insights |
| 🔍 SHAP Explainability | Global beeswarm and bar charts showing feature-level model explanations |
| 📈 Feature Importance | Interactive bar chart of the top N model predictors |
| 📁 Batch Prediction | Upload a CSV of employees and download a scored output with HIGH / MEDIUM / LOW risk labels |

---

## 🏆 Top 10 Predictive Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | StockOptionLevel | 15.79% |
| 2 | OverTime_Yes | 12.73% |
| 3 | MaritalStatus_Single | 6.22% |
| 4 | JobInvolvement | 6.07% |
| 5 | RelationshipSatisfaction | 5.51% |
| 6 | MonthlyIncome | 5.35% |
| 7 | JobRole_Sales Executive | 4.26% |
| 8 | JobSatisfaction | 3.84% |
| 9 | JobLevel | 3.83% |
| 10 | DailyRate | 2.87% |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/employee_attrition_prediction.git
cd employee_attrition_prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

This will generate the `model/` directory containing `model.pkl`, `columns.pkl`, `threshold.pkl`, and `feature_importance.csv`.

### Run the Dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
shap
matplotlib
seaborn
plotly
```

> Install all at once: `pip install -r requirements.txt`

---

## 📊 Dataset

- **Source**: [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Records**: 1,470 employees
- **Features**: 35 (demographics, compensation, satisfaction, career history)
- **Target**: `Attrition` (Yes / No)

---

## 🎓 Academic Context

**Course**: CBS3006 — Machine Learning
**Project**: End of Project Journal (EPJ) — Review II
**Institution**: Vellore Institute of Technology

---

## 👥 Author


Teeshay Singh Sahni 


---

## 📄 License

This project is licensed under the MIT License.
