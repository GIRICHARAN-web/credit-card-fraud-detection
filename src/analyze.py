
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Count legitimate and fraudulent transactions
class_counts = df["Class"].value_counts()

print("Transaction counts:")
print(class_counts)

# Calculate the percentage of each class
print("\nTransaction percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

# Plot the counts
class_counts.plot(kind="bar")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
