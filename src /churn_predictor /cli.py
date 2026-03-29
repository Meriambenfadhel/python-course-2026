
from churn_predictor.data import load_data
from churn_predictor.model import train_model


def main() -> None:
    df = load_data()

    print("Churn Predictor")
    print("=" * 30)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nTraining model...")
    train_model(df)
