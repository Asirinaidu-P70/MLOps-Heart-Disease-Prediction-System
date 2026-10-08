from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
OUT = Path("data/heart.csv")
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 100:
        print(f"Dataset already exists: {OUT}")
        return
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        dataset = archive.read("processed.cleveland.data")

    df = pd.read_csv(
        BytesIO(dataset),
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    df["target"] = (pd.to_numeric(df["target"], errors="raise") > 0).astype(int)
    df.to_csv(OUT, index=False)
    print(f"Downloaded {len(df)} rows and {len(df.columns)} columns to {OUT}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
