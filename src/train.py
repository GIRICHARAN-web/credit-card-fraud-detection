
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    average_precision_score
)
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Separate input features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]


#3. Split into training (60%), validation (20%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# 4. Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2],
}

results = []

for params in ParameterGrid(param_grid):
    candidate = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    candidate.fit(X_train, y_train)

    val_probabilities = candidate.predict_proba(X_val)[:, 1]

    ap = average_precision_score(
        y_val, val_probabilities
    )

    results.append((ap, params))

    print(f"Validation AP: {ap:.4f} | {params}")

results.sort(key=lambda item: item[0], reverse=True)

print("\nBest configuration:")
print("Validation AP:", results[0][0])
print("Parameters:", results[0][1])

best_params = results[0][1]

best_model = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

# Test-set probabilities
best_test_probabilities = best_model.predict_proba(X_test)[:, 1]

print(
    "\nTuned Random Forest Test Average Precision:",
    average_precision_score(y_test, best_test_probabilities)
)


# 5. Create and train the model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model using training data
model.fit(X_train_scaled, y_train)

# Scale validation data
X_val_scaled = scaler.transform(X_val)
val_probabilities = model.predict_proba(X_val_scaled)[:, 1]

print("\nValidation Average Precision:",
      average_precision_score(y_val, val_probabilities))


# Train a Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Get fraud probabilities for validation data
rf_val_probabilities = rf_model.predict_proba(X_val)[:, 1]

# Evaluate Average Precision
rf_ap = average_precision_score(
    y_val,
    rf_val_probabilities
)

print("\nRandom Forest Validation Average Precision:", rf_ap)

for threshold in [0.3, 0.5, 0.7]:
    rf_val_predictions = (
        rf_val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        rf_val_predictions,
        labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"\nRandom Forest Threshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"False alarms: {fp}")
    print(f"Missed frauds: {fn}")

# Final evaluation: Random Forest
# Threshold selected using validation results
chosen_threshold = 0.3

rf_test_probabilities = rf_model.predict_proba(X_test)[:, 1]

rf_test_predictions = (
    rf_test_probabilities >= chosen_threshold
).astype(int)

print("\nRandom Forest Final Test Results")
print("Threshold:", chosen_threshold)

print(confusion_matrix(
    y_test, rf_test_predictions, labels=[0, 1]
))

print(classification_report(
    y_test, rf_test_predictions, zero_division=0
))

print("Test Average Precision:",
      average_precision_score(y_test, rf_test_probabilities))


print("Random forest comparison with the logistic regresion\n");

# Compare different thresholds on validation data
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    val_predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_predictions, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"False alarms: {fp}")
    print(f"Missed frauds: {fn}")

# Final evaluation at the default threshold on the untouched test set
X_test_scaled = scaler.transform(X_test)
test_probabilities = model.predict_proba(X_test_scaled)[:, 1]
chosen_threshold = 0.7

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

print(f"\nFinal test results at threshold {chosen_threshold}:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(
    y_test, test_predictions, zero_division=0
))
print("Test Average Precision:",
      average_precision_score(y_test, test_probabilities))


# Logistic Regression probabilities
lr_test_probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Random Forest probabilities
rf_test_probabilities = rf_model.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Calculate precision-recall curves
lr_precision, lr_recall, _ = precision_recall_curve(
    y_test, lr_test_probabilities
)

rf_precision, rf_recall, _ = precision_recall_curve(
    y_test, rf_test_probabilities
)

# Plot both models
plt.figure(figsize=(8, 6))

plt.plot(
    lr_recall, lr_precision,
    label="Logistic Regression"
)

plt.plot(
    rf_recall, rf_precision,
    label="Random Forest"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Fraud Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()