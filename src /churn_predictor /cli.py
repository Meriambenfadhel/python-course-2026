from pathlib import Path
import matplotlib.pyplot as plt

from churn_predictor.data import load_data
from churn_predictor.model import train_model


def save_churn_distribution_plot(df):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(6, 4))
    churn_counts.plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_distribution.png")
    plt.close()


def main():
    df = load_data()

    print("Churn Predictor")
    print("=" * 30)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    churn_counts = df["Churn"].value_counts()
    print("\nChurn distribution:")
    for label, count in churn_counts.items():
        print(f"- {label}: {count}")

    save_churn_distribution_plot(df)
    print("\nSaved plot to outputs/churn_distribution.png")

    print("\nTraining model...")
    train_model(df)
