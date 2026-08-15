import pandas as pd

# Load Excel dataset
df = pd.read_excel("../data/customer_churn.xlsx")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nExited distribution:")
print(df["Exited"].value_counts())