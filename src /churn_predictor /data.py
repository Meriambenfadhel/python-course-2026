from pathlib import Path
import pandas as pd


def load_data():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "telco_churn.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")

    df = pd.read_csv(file_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    return df
