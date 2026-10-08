from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE = Path("reports/reference_data.csv")
CURRENT = Path("data/new_data.csv")
OUT = Path("reports/drift_report.html")


def main() -> None:
    if not REFERENCE.exists():
        raise FileNotFoundError("Run training first so reports/reference_data.csv exists.")
    if not CURRENT.exists():
        raise FileNotFoundError("Run: python src/make_new_data.py")

    reference = pd.read_csv(REFERENCE)
    current = pd.read_csv(CURRENT)

    # Keep only the columns used by training.
    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common].apply(pd.to_numeric, errors="coerce")
    current = current[common].apply(pd.to_numeric, errors="coerce")

    report = Report(metrics=[DataDriftPreset()])
    report.run(current_data=current, reference_data=reference)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(OUT))
    print(f"Drift report saved to {OUT}")


if __name__ == "__main__":
    main()
