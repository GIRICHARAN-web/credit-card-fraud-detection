
import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
model_path = Path("models/extra_trees_model.joblib")
model = joblib.load(model_path)

# Load one transaction from the dataset for a test prediction
data_path = Path("data/creditcard.csv")
df = pd.read_csv(data_path)

# Select one transaction, excluding the target column
fraud_rows = df[df["Class"] == 1]
transaction = fraud_rows.drop(columns=["Class"]).iloc[[0]]

# Apply the threshold used in our evaluation
threshold = 0.3

def predict_transaction(transaction, threshold=0.3):
    """
    Predict whether a transaction may be fraudulent.

    transaction must be a pandas DataFrame containing
    the 30 input features in the same order as training.
    """
    probability = model.predict_proba(transaction)[0, 1]
    prediction = int(probability >= threshold)

    return probability, prediction

fraud_probability, prediction = predict_transaction(transaction)

print(f"Fraud probability: {fraud_probability:.4f}")
print(f"Prediction: {'Fraud' if prediction == 1 else 'Legitimate'}")


actual_class = int(df.loc[transaction.index[0], "Class"])

print(f"Actual class: {'Fraud' if actual_class == 1 else 'Legitimate'}")