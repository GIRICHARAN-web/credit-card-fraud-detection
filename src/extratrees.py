import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


extra_model = ExtraTreesClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra_model.fit(X_train, y_train)

extra_val_probabilities = extra_model.predict_proba(X_val)[:, 1]

extra_val_ap = average_precision_score(
    y_val,
    extra_val_probabilities
)

print("Extra Trees Validation Average Precision:", extra_val_ap)

# Final evaluation of Extra Trees
extra_test_probabilities = extra_model.predict_proba(X_test)[:, 1]

extra_test_ap = average_precision_score(
    y_test,
    extra_test_probabilities
)

print("Extra Trees Test Average Precision:", extra_test_ap)


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
)

# Predict fraud probabilities on the test set
extra_test_probabilities = extra_model.predict_proba(X_test)[:, 1]

# Convert probabilities into predictions using threshold 0.3
extra_test_predictions = (
    extra_test_probabilities >= 0.3
).astype(int)

# Evaluate the model
print("Test Average Precision:",
      average_precision_score(y_test, extra_test_probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, extra_test_predictions))

print("\nClassification Report:")
print(classification_report(y_test, extra_test_predictions, digits=4))


import joblib
from pathlib import Path

model_path = Path("models/extra_trees_model.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(extra_model, model_path)

print(f"Model saved to: {model_path}")