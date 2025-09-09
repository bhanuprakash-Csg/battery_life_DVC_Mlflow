"""
LSTM-based Time Series Forecasting Module with MLflow and DVC integration.
Handles loading preprocessed data, model creation/training, evaluation, and forecasting.
"""

import os
import logging
import joblib
import yaml
import numpy as np
from tensorflow import keras
from typing import Tuple, List
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from model_evaluation import evaluate_model

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Load parameters from params.yaml
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

DATA_DIR = params["data"]["results_dir"].strip()
MODEL_DIR = params["data"]["model_dir"].strip()

LSTM1_UNITS = params["model"]["units_lstm1"]
LSTM2_UNITS = params["model"]["units_lstm2"]
DROPOUT = params["model"]["dropout"]
DENSE_UNITS = params["model"]["dense_units"]
LEARNING_RATE = params["model"]["learning_rate"]
LOSS = params["model"]["loss"]
OPTIMIZER = params["model"]["optimizer"]
METRICS = params["model"]["metrics"]

EPOCHS = params["training"]["epochs"]
BATCH_SIZE = params["training"]["batch_size"]
VAL_SPLIT = params["training"]["validation_split"]
VERBOSE = params["training"]["verbose"]
STRATEGY = params["training"]["strategy"]

FORECAST_WEEKS = params["forecast"]["forecast_weeks"]

# ----------------------------
# MLflow hosted server setup
# ----------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"  # hosted MLflow server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "LSTM_TimeSeries_Forecasting"
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# Load preprocessed data
# ----------------------------
def load_preprocessed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object, List[str]]:
    try:
        x_train = joblib.load(os.path.join(DATA_DIR, 'x_train.pkl'))
        x_test = joblib.load(os.path.join(DATA_DIR, 'x_test.pkl'))
        y_train = joblib.load(os.path.join(DATA_DIR, 'y_train.pkl'))
        y_test = joblib.load(os.path.join(DATA_DIR, 'y_test.pkl'))
        feature_scaler = joblib.load(os.path.join(DATA_DIR, 'feature_scaler.pkl'))
        target_scaler = joblib.load(os.path.join(DATA_DIR, 'target_scaler.pkl'))
        used_features = joblib.load(os.path.join(DATA_DIR, 'used_features.pkl'))
        logger.info("Loaded preprocessed data from %s", DATA_DIR)
        return x_train, x_test, y_train, y_test, feature_scaler, target_scaler, used_features
    except FileNotFoundError as e:
        logger.error("Preprocessed data not found in %s: %s", DATA_DIR, e)
        raise

# ----------------------------
# Create LSTM model
# ----------------------------
def create_lstm_model(seq_length: int, feature_count: int) -> keras.Sequential:
    if feature_count < 1:
        raise ValueError("At least one feature is required")
    if seq_length < 1:
        raise ValueError("Timesteps must be positive")

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=LSTM1_UNITS, return_sequences=True, input_shape=(seq_length, feature_count)))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.LSTM(units=LSTM2_UNITS))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(units=DENSE_UNITS, activation='relu'))
    model.add(keras.layers.Dense(units=1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS
    )
    logger.info("Created LSTM model with input shape (%s, %s)", seq_length, feature_count)
    return model

# ----------------------------
# Load or train model with MLflow
# ----------------------------
def load_or_train_model(seq_length: int, feature_count: int, train_data: np.ndarray, train_target: np.ndarray, strategy: str = STRATEGY) -> keras.Sequential:
    if train_data.shape[1:] != (seq_length, feature_count):
        raise ValueError(f"Expected train_data shape (*, {seq_length}, {feature_count}), got {train_data.shape}")

    model_path = os.path.join(MODEL_DIR, strategy, 'lstm_model.keras')

    with mlflow.start_run(run_name=f"LSTM_{strategy}"):
        # Log parameters
        mlflow.log_params({
            "strategy": strategy,
            "seq_length": seq_length,
            "feature_count": feature_count,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "val_split": VAL_SPLIT,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "loss": LOSS
        })

        if strategy == "recommended" and os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            logger.info("Loaded existing LSTM model from %s", model_path)

            # Log model with input_example and signature
            # input_example = train_data[:1].tolist()
            # signature = infer_signature(train_data[:1], model.predict(train_data[:1]))
            mlflow.keras.log_model(model, name="lstm_model", signature=infer_signature(train_data[:1], model.predict(train_data[:1])), input_example=None)
            return model

        # Train new model
        model = create_lstm_model(seq_length, feature_count)
        history = model.fit(
            train_data, train_target,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT,
            verbose=VERBOSE
        )

        # Log metrics per epoch
        for epoch, (loss, mae) in enumerate(zip(history.history['loss'], history.history['mae']), start=1):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_mae", mae, step=epoch)
        if 'val_loss' in history.history:
            for epoch, (v_loss, v_mae) in enumerate(zip(history.history['val_loss'], history.history['val_mae']), start=1):
                mlflow.log_metric("val_loss", v_loss, step=epoch)
                mlflow.log_metric("val_mae", v_mae, step=epoch)

        os.makedirs(os.path.join(MODEL_DIR, strategy), exist_ok=True)
        model.save(model_path)

        # Log model to MLflow with signature
        # input_example = train_data[:1]
        # signature = infer_signature(input_example, model.predict(input_example))
        mlflow.keras.log_model(model, name="lstm_model", signature=infer_signature(train_data[:1], model.predict(train_data[:1])), input_example=None)
        logger.info("Trained and saved LSTM model to %s", model_path)
        return model

# ----------------------------
# Evaluate model and forecast
# ----------------------------
def evaluate_and_forecast(model: keras.Sequential, x_test: np.ndarray, y_test: np.ndarray, target_scaler, forecast_weeks: int = FORECAST_WEEKS):
    if x_test.shape[0] == 0 or y_test.shape[0] == 0:
        raise ValueError("Test data or target is empty")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Mismatch in test data lengths: x_test ({x_test.shape[0]}), y_test ({y_test.shape[0]})")

    with mlflow.start_run(run_name="Evaluation_Forecast", nested=True):
        y_pred = model.predict(x_test).flatten()

        # Fix tuple issue from evaluate_model
        mae, mse = evaluate_model(y_test, y_pred)
        metrics = {"mae": mae, "mse": mse}

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        logger.info("Logged MAE=%0.4f and MSE=%0.4f to MLflow", mae, mse)

        # Forecasting
        forecasted_soh = []
        last_sequence = x_test[-1:]
        for _ in range(forecast_weeks):
            next_pred = model.predict(last_sequence).flatten()[0]
            forecasted_soh.append(next_pred)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[:, -1, 0] = next_pred

        mlflow.log_metric("forecast_weeks", forecast_weeks)
        mlflow.log_metric("last_forecast", forecasted_soh[-1])
        return metrics, (y_test, y_pred), np.array(forecasted_soh)

# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":
    try:
        # Load preprocessed data
        x_train, x_test, y_train, y_test, feature_scaler, target_scaler, used_features = load_preprocessed_data()
        timesteps = x_train.shape[1]
        n_features = x_train.shape[2]

        # Loop over strategies
        for strategy in ["recommended", "retrained"]:
            # Load or train model
            model = load_or_train_model(timesteps, n_features, x_train, y_train, strategy=strategy)
            
            # Evaluate and forecast
            metrics, (y_test_vals, y_pred_vals), forecasted = evaluate_and_forecast(model, x_test, y_test, target_scaler)
            
            # Log results
            logger.info(
                "Strategy: %s | Metrics: %s | Forecasted values (last 5): %s",
                strategy, metrics, forecasted[-5:]
            )

    except FileNotFoundError as fnf_error:
        logger.error("File not found: %s", fnf_error)
    except ValueError as val_error:
        logger.error("Value error: %s", val_error)
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
