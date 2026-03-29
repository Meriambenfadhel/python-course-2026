from churn_predictor.data import load_data
from churn_predictor.model import train_model


def main() -> None:
    df = load_data()

    print("Churn Predictor")
    print("=" * 30)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    churn_counts = df["Churn"].value_counts()
    print("\nChurn distribution:")
    for label, count in churn_counts.items():
        print(f"- {label}: {count}")

    print("\nTraining model...")
    train_model(df)
