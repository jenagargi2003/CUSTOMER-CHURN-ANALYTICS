# Customer Churn Analytics & Prediction

An end-to-end customer churn analytics and machine learning project that analyzes customer behavior, identifies churn patterns, predicts customer churn probability, and classifies customers into risk levels.

## Project Overview

Customer churn is a major business challenge for banks and financial institutions. This project analyzes 10,000 customer records to understand why customers leave and uses machine learning to identify customers who are most likely to churn.

The project combines **Python, Machine Learning, MySQL, and Power BI** to create a complete analytics and prediction solution.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- MySQL
- Power BI
- Jupyter/VS Code

## Project Workflow

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Churn Pattern Analysis
4. Machine Learning Model Training
5. Customer Churn Probability Prediction
6. Customer Risk Classification
7. MySQL Data Storage and SQL Analysis
8. Interactive Power BI Dashboard

## Machine Learning

Two classification models were trained and evaluated:

- Logistic Regression
- Random Forest

### Random Forest Performance

- Accuracy: 86.15%
- Precision: 77.54%
- Recall: 44.96%
- F1 Score: 56.92%
- ROC-AUC: 85.47%

Random Forest performed better overall and was used for customer churn prediction.

## Key Findings

- Overall churn rate: **20.37%**
- Total customers: **10,000**
- Churned customers: **2,037**
- High-risk customers identified: **1,386**
- Germany showed the highest churn rate among the three countries.
- Inactive members had a significantly higher churn rate than active members.
- Customers with certain product combinations showed substantially higher churn risk.

## Customer Risk Prediction

The machine learning model generates a churn probability for each customer and assigns a risk level:

- **Low Risk**
- **Medium Risk**
- **High Risk**

This allows businesses to identify customers who may require proactive retention strategies.

## SQL Analysis

Customer prediction data was stored in **MySQL** for further analysis and querying.

SQL was used to analyze:

- Customer churn
- Churn by geography
- Risk levels
- High-risk customers
- Customer segmentation

## Power BI Dashboard

The Power BI dashboard contains two pages:

### Page 1 — Customer Churn Overview

- Total Customers
- Churned Customers
- Overall Churn Rate
- High-Risk Customers
- Churn Rate by Geography
- Customer Risk Distribution
- Churn Rate by Gender
- Active vs Inactive Member Analysis

### Page 2 — Customer Risk Intelligence

- Predicted Churn Probability by Number of Products
- Predicted Churn Probability by Age
- Top High-Risk Customers
- Customer-level churn probability and risk classification

## Dashboard Screenshots

### Page 1 — Customer Churn Overview

![Customer Churn Dashboard](screenshots/dashboard1.png)

### Page 2 — Customer Risk Intelligence

![Customer Risk Intelligence](screenshots/dashboard2.png)

