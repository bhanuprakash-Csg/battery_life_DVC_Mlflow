"""
Module for preprocessing battery analytics data for model training and storing results.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
from data_loading import load_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def select_features(df: pd.DataFrame, input_features: Optional[List[str]]) -> List[str]:
    """Select valid numerical features for preprocessing."""
    all_numerical_columns = [
        'elv_spy', 'speed', 'soc', 'amb_temp', 'regenwh', 'motor_pwrw',
        'aux_pwr100w', 'motor_temp', 'torque_nm', 'rpm', 'capacity',
        'ref_consumption', 'wind_mph', 'wind_kph', 'wind_degree',
        'frontal_wind', 'veh_deg', 'totalvehicles', 'speedavg', 'max_speed',
        'radius', 'step', 'accelerationm/s²', 'actualbatterycapacitywh',
        'speedm/s', 'speedfactor', 'totalenergyconsumedwh',
        'totalenergyregeneratedwh', 'lon', 'lat', 'alt', 'slope_deg',
        'completeddistancekm', 'mwh', 'remainingrangekm', 'year', 'month',
        'week', 'day', 'hour'
    ]

    if input_features and 'soh' in input_features:
        raise ValueError("'soh' cannot be included in selected features; it is the target variable")

    used_features = input_features if input_features else [c for c in all_numerical_columns if c in df.columns]

    # Remove duplicates
    seen = set()
    used_features = [x for x in used_features if not (x in seen or seen.add(x))]

    if not used_features:
        raise ValueError("No valid features available for preprocessing")

    return used_features

def preprocess_data(
    df: pd.DataFrame,
    sequence_length: int = 30,
    input_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "processed_data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[str]]:
    """Preprocess data for model training, create train-test splits, and store results."""
    try:
        if 'timestamp_data_utc' in df.columns:
            df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'], errors='coerce')
        else:
            logger.warning("Column 'timestamp_data_utc' not found in dataset")

        used_features = select_features(df, input_features)
        logger.info("Selected features: %s", used_features)

        df = df.dropna(subset=['timestamp_data_utc'])
        df[used_features] = df[used_features].fillna(df[used_features].mean())
        df['soh'] = df['soh'].fillna(df['soh'].mean())

        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        df[used_features] = feature_scaler.fit_transform(df[used_features])
        df['soh'] = target_scaler.fit_transform(df[['soh']])

        # Create sequences
        sequences, targets = [], []
        for i in range(len(df) - sequence_length):
            sequences.append(df[used_features].iloc[i:i + sequence_length].values)
            targets.append(df['soh'].iloc[i + sequence_length])

        x_data = np.array(sequences)
        y_data = np.array(targets)

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=random_state
        )

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(x_train, os.path.join(output_dir, 'x_train.pkl'))
        joblib.dump(x_test, os.path.join(output_dir, 'x_test.pkl'))
        joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
        joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
        joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
        joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        joblib.dump(used_features, os.path.join(output_dir, 'used_features.pkl'))
        logger.info("Stored preprocessed data in %s", output_dir)

        return x_train, x_test, y_train, y_test, feature_scaler, target_scaler, used_features

    except Exception as e:
        logger.error("Error in preprocess_data: %s", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess battery analytics data")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save processed data")
    args = parser.parse_args()

    try:
        df = load_data(args.input_csv)
        preprocess_data(df, output_dir=args.output_dir)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        exit(1)
