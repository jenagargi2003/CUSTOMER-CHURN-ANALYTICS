import pandas as pd

# Load dataset
df = pd.read_excel("../data/customer_churn.xlsx")

print("Original shape:", df.shape)

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove unnecessary columns
df_clean = df.drop(columns=["RowNumber", "Surname"])

print("\nColumns after cleaning:")
print(df_clean.columns.tolist())

print("\nCleaned dataset shape:", df_clean.shape)

# Save cleaned dataset
df_clean.to_csv("../data/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully.")
