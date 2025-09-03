import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_data(s3_path: str) -> pd.DataFrame:
    """
    Load dataset from S3 bucket.
    Args:
        s3_path: S3 URI (e.g. "s3://my-bucket/data/file.csv")
    Returns:
        Pandas DataFrame
    """
    try:
        df = pd.read_csv(s3_path, storage_options={"anon": False})
        logger.info("Loaded data from %s: shape=%s, columns=%s", s3_path, df.shape, list(df.columns))
        return df
    except Exception as e:
        logger.error("Error loading data from %s: %s", s3_path, e)
        raise ValueError(f"Error loading data: {e}") from e

if __name__ == "__main__":
    try:
        # Example: S3 URI
        s3_csv = "s3://my-bucket/final_merge_batches/synthetic_weeeekkly_data_150weeks_corrected.csv"
        test_df = load_data(s3_csv)
        logger.info("Test load successful: shape=%s", test_df.shape)
    except ValueError as e:
        logger.error("Test failed: %s", e)
