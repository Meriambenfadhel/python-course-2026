# python-course-2026
Introduction in Python Winter term 2025/2026

# Author
Meriam Ben Fadhel 
# Churn Predictor
A Python CLI tool for customer churn analysis and prediction.

# Features
- Load and clean customer churn dataset
- Train a logistic regression model
- Evaluate model performance with classification metrics
- shows basic dataset information
- displays churn distribution
- 
# OUTPUT 
The program provides the following outputs:
Number of rows and columns in the dataset
Churn distribution (Yes vs No)
Model accuracy
Confusion matrix
Classification report

It also saves a plot of the churn distribution to `outputs/churn_distribution.png`.

# Dataset
The project uses the Telco Customer Churn dataset.
It includes customer information such as:
Demographics
Contract type
Monthly charges
Service usage

The target variable is: Churn (Yes / No)

# Installation
Run the following commands to set up and execute the project:
```bash
uv venv
.venv\Scripts\activate
uv pip install -e .
Run the program using:
uv run -m churn_predictor
.
---


