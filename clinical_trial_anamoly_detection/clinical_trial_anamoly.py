import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json


def load_clinical_data(file_path):
    """Load clinical trial data from CSV."""
    df = pd.read_csv(file_path)
    return df


def detect_anomalies(df, contamination=0.05):
    """Detect anomalies using Isolation Forest."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=['trial_id']))

    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model.fit_predict(X_scaled)
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 'Yes' if x == -1 else 'No')
    return df


def save_anomalies(df, output_file):
    """Save detected anomalies to a JSON file."""
    anomalies = df[df['is_anomaly'] == 'Yes'].to_dict(orient='records')
    with open(output_file, "w") as f:
        json.dump(anomalies, f, indent=4)
    print(f"Anomalies saved to {output_file}")


def main():
    file_path = "data.csv"  # Input CSV file with clinical trial data
    output_file = "clinical_trial_anomalies.json"

    df = load_clinical_data(file_path)
    df_with_anomalies = detect_anomalies(df)
    save_anomalies(df_with_anomalies, output_file)

    print("Anomaly detection completed.")


if __name__ == "__main__":
    main()
