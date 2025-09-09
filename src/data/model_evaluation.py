"""
Module for evaluating machine learning models for SOH prediction.
"""

import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow  # Added MLflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Compute evaluation metrics for model predictions and log them to MLflow.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Tuple containing MAE and MSE.

    Raises:
        ValueError: If y_true or y_pred is empty or lengths mismatch.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty true or predicted values")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatch in lengths: y_true ({len(y_true)}), y_pred ({len(y_pred)})")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # Log metrics to MLflow
    try:
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        logger.info("Logged MAE=%.4f and MSE=%.4f to MLflow", mae, mse)
    except Exception as e:
        logger.warning("Failed to log metrics to MLflow: %s", e)

    return mae, mse

if __name__ == "__main__":
    logger.info("Evaluation module ready for use")

    # Example test run with MLflow
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    with mlflow.start_run(run_name="eval_test"):
        evaluate_model(y_true, y_pred)
