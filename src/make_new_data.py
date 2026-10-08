from pathlib import Path
import pandas as pd

SOURCE = Path("data/heart.csv")
OUT = Path("data/new_data.csv")


def main() -> None:
    df = pd.read_csv(SOURCE)
    # Create a small monitoring sample from later records and shift two
    # numeric features so the drift report has something to detect.
    new_df = df.tail(min(60, len(df))).copy()
    for col, amount in [("age", 8), ("chol", 35), ("trestbps", 10)]:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce") + amount
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OUT, index=False)
    print(f"Created {OUT} with {len(new_df)} rows")


if __name__ == "__main__":
    main()
