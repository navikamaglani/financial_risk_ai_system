import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import joblib

PROCESSED_DATA_PATH = Path("data/processed")


# -----------------------------
# Load All Labeled Data
# -----------------------------
def load_all_data():

    csv_files = PROCESSED_DATA_PATH.glob("*_labeled.csv")

    dataframes = []

    for file_path in csv_files:

        df = pd.read_csv(file_path)

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


# -----------------------------
# Main Training Pipeline
# -----------------------------
def train_model():

    df = load_all_data()

    print("Combined dataset shape:")
    print(df.shape)

    # -----------------------------
    # Feature Columns
    # -----------------------------
    feature_columns = [
        "Daily_Return",
        "Rolling_Volatility",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_High",
        "BB_Low",
        "Momentum_10"
    ]

    X = df[feature_columns]

    y = df["Risk_Label"]

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Train Model
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
    joblib.dump(model, "models/risk_classifier.pkl")

    print("\nModel saved successfully.")

if __name__ == "__main__":

    train_model()

