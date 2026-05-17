import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")


def generate_risk_labels(file_path):

    ticker = file_path.stem.replace("_features", "")

    print(f"Generating labels for {ticker}...")

    df = pd.read_csv(file_path)

    # -----------------------------
    # Volatility Quantiles
    # -----------------------------
    low_threshold = df["Rolling_Volatility"].quantile(0.33)

    high_threshold = df["Rolling_Volatility"].quantile(0.66)

    # -----------------------------
    # Risk Classification Function
    # -----------------------------
    def classify_risk(volatility):

        if volatility <= low_threshold:
            return 0   # Low Risk

        elif volatility <= high_threshold:
            return 1   # Medium Risk

        else:
            return 2   # High Risk

    # Create target label
    df["Risk_Label"] = df["Rolling_Volatility"].apply(classify_risk)

    # Save updated dataset
    output_path = PROCESSED_DATA_PATH / f"{ticker}_labeled.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":

    csv_files = PROCESSED_DATA_PATH.glob("*_features.csv")

    for file_path in csv_files:
        generate_risk_labels(file_path)

    print("Risk label generation completed.")