import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_customer_churn.csv")

# -----------------------------
# Basic Churn Analysis
# -----------------------------

churn_counts = df["Exited"].value_counts()

print("Churn Counts:")
print(churn_counts)

churn_rate = df["Exited"].mean() * 100

print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

# -----------------------------
# Churn by Geography
# -----------------------------

geo_churn = df.groupby("Geography")["Exited"].mean() * 100

print("\nChurn Rate by Geography:")
print(geo_churn)

plt.figure(figsize=(7, 5))
sns.barplot(x=geo_churn.index, y=geo_churn.values)
plt.title("Churn Rate by Geography")
plt.xlabel("Geography")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# Churn by Gender
# -----------------------------

gender_churn = df.groupby("Gender")["Exited"].mean() * 100

print("\nChurn Rate by Gender:")
print(gender_churn)

plt.figure(figsize=(7, 5))
sns.barplot(x=gender_churn.index, y=gender_churn.values)
plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# Churn by Active Membership
# -----------------------------

active_churn = df.groupby("IsActiveMember")["Exited"].mean() * 100

print("\nChurn Rate by Active Membership:")
print(active_churn)

plt.figure(figsize=(7, 5))
sns.barplot(x=active_churn.index, y=active_churn.values)
plt.title("Churn Rate by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# Churn by Number of Products
# -----------------------------

product_analysis = df.groupby("NumOfProducts").agg(
    Customers=("CustomerId", "count"),
    Churned=("Exited", "sum"),
    Churn_Rate=("Exited", "mean")
)

product_analysis["Churn_Rate"] *= 100

print("\nChurn Analysis by Number of Products:")
print(product_analysis)

# -----------------------------
# Age Distribution
# -----------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Age",
    hue="Exited",
    bins=30,
    kde=True
)

plt.title("Age Distribution by Churn Status")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# -----------------------------
# Credit Score vs Churn
# -----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Exited",
    y="CreditScore"
)

plt.title("Credit Score Distribution by Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.show()
