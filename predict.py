import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from model import BalancedMLP

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_logging(log_dir: Path):
    """Sets up console logging."""
    log_dir.mkdir(exist_ok=True)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[BalancedMLP, StandardScaler, List[str]]:
    """
    Loads a complete model checkpoint.
    """
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found at: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path)
    logger.info(f"Loading model checkpoint from {checkpoint_path}...")

    model = BalancedMLP(
        input_dim=checkpoint["input_dim"], dropout_rate=checkpoint["dropout_rate"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = checkpoint["scaler"]
    feature_names = checkpoint["feature_names"]

    logger.info(
        f"Model loaded successfully. Expecting {len(feature_names)} input features."
    )
    return model, scaler, feature_names


def preprocess_new_data(
    df: pd.DataFrame, feature_names: List[str], scaler: StandardScaler
) -> torch.Tensor:
    """
    Preprocesses new data for inference using the training configuration.
    """
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns in the new data: {missing_cols}")
        sys.exit(1)

    X = df[feature_names].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.warning(
                f"NaNs found in column '{col}'. Filled with median value ({median_val:.4f})."
            )

    X_scaled = scaler.transform(X.values)
    return torch.tensor(X_scaled, dtype=torch.float32)


def run_inference(model: BalancedMLP, data_tensor: torch.Tensor) -> np.ndarray:
    """
    Runs the model on the preprocessed data to get predictions.
    """
    logger.info("Performing inference on new data...")
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = torch.sigmoid(logits.squeeze())
        predictions = (probabilities >= 0.5).float().cpu().numpy()
    logger.info(f"Inference complete. Generated {len(predictions)} predictions.")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference using a trained model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/predict_config.json",
        help="Path to the prediction configuration JSON file.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (overrides config).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the new data CSV file (overrides config).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the results CSV file (overrides config).",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(Path("logs"))

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    paths = config["paths"]

    model_checkpoint_path = Path(args.model_path or paths["model_checkpoint"])
    new_data_path = Path(args.data_path or paths["new_data_csv"])
    results_dir = Path(paths["results_dir"])

    if args.output_path:
        output_file_path = Path(args.output_path)
    else:
        results_dir.mkdir(exist_ok=True)
        output_file_path = results_dir / paths.get(
            "output_filename", "benchmark_results.csv"
        )

    model, scaler, feature_names = load_checkpoint(model_checkpoint_path)

    try:
        new_df = pd.read_csv(new_data_path)
        logger.info(
            f"Successfully loaded new data for prediction from: {new_data_path.resolve()}"
        )
    except FileNotFoundError:
        logger.error(f"The new data file was not found at: {new_data_path}")
        sys.exit(1)

    preprocessed_tensor = preprocess_new_data(new_df, feature_names, scaler)
    predictions = run_inference(model, preprocessed_tensor)

    results_df = new_df.copy()
    results_df["predicted_state"] = predictions.astype(int)

    desired_columns = [
        "material_id",
        "Composition",
        "band_gap",
        "crystal_system",
        "State",
        "predicted_state",
    ]
    final_columns = [col for col in desired_columns if col in results_df.columns]

    if not final_columns:
        logger.error(
            "None of the desired output columns were found. Saving all columns instead."
        )
        final_results_df = results_df
    else:
        logger.info(f"Filtering output to include columns: {final_columns}")
        final_results_df = results_df[final_columns]

    final_results_df.to_csv(output_file_path, index=False)
    logger.info(f"✅ Prediction results saved successfully to: {output_file_path}")
