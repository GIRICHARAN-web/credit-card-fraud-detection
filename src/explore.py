import pandas as pd

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print("Dataset shape:", df.shape)

# Display column information
df.info()

# Count missing values
print("Missing values:", df.isnull().sum().sum())

# Count legitimate and fraudulent transactions
print(df["Class"].value_counts())