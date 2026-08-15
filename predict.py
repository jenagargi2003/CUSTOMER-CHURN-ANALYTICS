import pandas as pd
import joblib

# -----------------------------------
# 1. Load cleaned dataset
# -----------------------------------

df = pd.read_csv("../data/cleaned_customer_churn.csv")

# Keep CustomerId for final results
customer_ids = df["CustomerId"]

# -----------------------------------
# 2. Prepare features
# -----------------------------------

X = df.drop(columns=["Exited", "CustomerId"])

# -----------------------------------
# 3. Load trained model
# -----------------------------------

model = joblib.load("../models/churn_model.pkl")

# -----------------------------------
# 4. Predict churn probability
# -----------------------------------

churn_probability = model.predict_proba(X)[:, 1]

# -----------------------------------
# 5. Create prediction dataset
# -----------------------------------

results = df.copy()

results["Churn_Probability"] = churn_probability

# Convert probability to percentage
results["Churn_Probability"] = (
    results["Churn_Probability"] * 100
).round(2)

# -----------------------------------
# 6. Assign Risk Level
# -----------------------------------

def risk_level(probability):

    if probability >= 70:
        return "High"

    elif probability >= 40:
        return "Medium"

    else:
        return "Low"


results["Risk_Level"] = results["Churn_Probability"].apply(
    risk_level
)

# -----------------------------------
# 7. Save predictions
# -----------------------------------

results.to_csv(
    "../data/customer_churn_predictions.csv",
    index=False
)

# -----------------------------------
# 8. Display summary
# -----------------------------------

print("\nPrediction completed successfully.")

print("\nRisk Level Distribution:")
print(results["Risk_Level"].value_counts())

print("\nTop 10 High-Risk Customers:")
print(
    results[
        [
            "CustomerId",
            "Geography",
            "Age",
            "Balance",
            "IsActiveMember",
            "Churn_Probability",
            "Risk_Level"
        ]
    ]
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
    .head(10)
)