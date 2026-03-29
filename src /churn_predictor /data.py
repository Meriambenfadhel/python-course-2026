 
from pathlib import Path
import pandas as pd


def load_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "telco_churn.csv"

    df = pd.read_csv(file_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    return df
