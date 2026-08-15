import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# -----------------------------------
# 1. Load cleaned dataset
# -----------------------------------

df = pd.read_csv("../data/cleaned_customer_churn.csv")

# -----------------------------------
# 2. Separate features and target
# -----------------------------------

X = df.drop(columns=["Exited"])
y = df["Exited"]

# CustomerId is kept in the dataset for identification,
# but should NOT be used as an ML feature.
X = X.drop(columns=["CustomerId"])

# -----------------------------------
# 3. Identify column types
# -----------------------------------

categorical_features = [
    "Geography",
    "Gender"
]

numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# -----------------------------------
# 4. Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)

# -----------------------------------
# 5. Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------
# 6. Logistic Regression
# -----------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# -----------------------------------
# 7. Random Forest
# -----------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

# Save the trained Random Forest model
joblib.dump(
    random_forest_model,
    "../models/churn_model.pkl"
)

print("\nRandom Forest model saved successfully.")

# -----------------------------------
# 8. Evaluation Function
# -----------------------------------

def evaluate_model(name, y_true, predictions, probabilities):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(
        accuracy_score(y_true, predictions), 4
    ))

    print("Precision:", round(
        precision_score(y_true, predictions), 4
    ))

    print("Recall   :", round(
        recall_score(y_true, predictions), 4
    ))

    print("F1 Score :", round(
        f1_score(y_true, predictions), 4
    ))

    print("ROC-AUC  :", round(
        roc_auc_score(y_true, probabilities), 4
    ))

    print("\nClassification Report:")
    print(classification_report(y_true, predictions))


# -----------------------------------
# 9. Evaluate Models
# -----------------------------------

evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)
