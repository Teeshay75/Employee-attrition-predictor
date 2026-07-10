# train_model.py

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv("data/balanced_attrition_1470.csv")

df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -------------------------------
# 2. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Train Gradient Boosting Model
# -------------------------------

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Gradient Boosting Model Training Complete!\n")

# -------------------------------
# 4. Threshold Tuning
# -------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

best_threshold = None
best_accuracy = 0
best_recall = 0
best_pred = None

MIN_RECALL = 0.80  # recall floor

for threshold in np.arange(0.10, 0.91, 0.01):
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall_class1 = report['1']['recall']
    accuracy = accuracy_score(y_test, y_pred)

    # Only consider thresholds that satisfy minimum recall
    if recall_class1 < MIN_RECALL:
        continue

    # Among valid thresholds, pick the one with highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_recall = recall_class1
        best_threshold = threshold
        best_pred = y_pred

if best_pred is None:
    print("No threshold met the minimum recall requirement. Lowering floor to 0.75...")
    MIN_RECALL = 0.75
    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        recall_class1 = report['1']['recall']
        accuracy = accuracy_score(y_test, y_pred)
        if recall_class1 < MIN_RECALL:
            continue
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_recall = recall_class1
            best_threshold = threshold
            best_pred = y_pred

cm = confusion_matrix(y_test, best_pred)

print("Selected Threshold:", round(best_threshold, 2))
print("Accuracy:", round(best_accuracy, 4))
print("Recall (Attrition):", round(best_recall, 4))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, best_pred))

# -------------------------------
# 5. Feature Importance
# -------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# -------------------------------
# 6. Save Model & Threshold
# -------------------------------

if not os.path.exists("model"):
    os.makedirs("model")

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))
pickle.dump(best_threshold, open("model/threshold.pkl", "wb"))
feature_importance.to_csv("model/feature_importance.csv", index=False)

print("\nModel, Columns, Threshold, and Feature Importance saved successfully!")