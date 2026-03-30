import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_model(df: pd.DataFrame) -> None:
    df = df.copy()

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"])
    df = pd.get_dummies(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print("=" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nInterpretation:")
    if accuracy_score(y_test, y_pred) >= 0.75:
        print(" The model achieves a reasonable overall accuracy.")
    else:
        print(" The model performance is limited and could be improved.")

    print("Recall for churned customers is lower than for non-churned customers.")
    print("This means the model detects non-churners better than churners.")
